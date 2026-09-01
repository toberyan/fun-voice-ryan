# 活跃会话与悬浮临时转写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不让任何神经模型随登录常驻的前提下，把现有“一次按住、松键后整段识别”的管道改为可控的 Nano 活跃会话：首次使用冷加载，成功后默认热驻留 8 分钟；录音期间显示不抢焦点的 X11 悬浮状态和经过 POC 验证的临时转写；只有风险文本才按需运行 Qwen3.5-0.8B，随后异步恢复 Nano。

**Architecture:** `VoiceDaemon` 拆出只管理状态、焦点和桌面提交的 `ActiveSessionController`。新的单一 `XpuScheduler` 是唯一可以启动、停止或向 Nano/Qwen/CAM++ 提交工作的组件，使用 session id + generation 丢弃过期结果。`PipeWireRecorder` 提供只在内存有效期内可借用的音频窗口；`X11TransientOverlay` 只显示状态/临时文本，不改变焦点、不写剪贴板。默认先交付稳定的热会话和 UI；连续尾段 Nano 仅在本机 XPU POC 成功后开启。

**Tech Stack:** Python 3.12、python-xlib、PipeWire `pw-record`、Fun-ASR-Nano-2512、FSMN-VAD、Qwen3.5-0.8B、PyTorch/vLLM XPU、systemd --user、pytest、ruff、mypy。

**Approved design:** [`2026-09-01-active-session-product-architecture-design.md`](../specs/2026-09-01-active-session-product-architecture-design.md). This plan supersedes the active-lifecycle and correction behavior in earlier plans; it does not replace their already-delivered X11, focus-guard, Fcitx, XPU-only, or on-demand-Qwen safety guarantees.

## Global Constraints

- 所有模型工作只允许 `xpu:0`。Nano/SenseVoice/CAM++/Qwen 绝不 CPU/CUDA 回退；SenseVoice 仍仅处理 Nano `model_load` 或 `oom`。
- 登录只启动轻量 daemon。Nano、SenseVoice、Qwen、CAM++ 都不得因 daemon、overlay 或指标初始化而加载。
- 悬浮窗不请求键盘焦点、不向目标窗口发事件、不写剪贴板、不保存文本；完成、取消、失败时立即清空画面内容。
- 最终上屏仍只能走“剪贴板备份 → 重新取焦点 → 比较录音起始焦点 → Fcitx token → 必要时 XTEST”的既有顺序。临时或稳定段绝不写入目标应用。
- 调度器必须保证 Nano/SenseVoice 与 Qwen 不同时持有 XPU。后台富化比任何录音任务低优先级，且可取消。
- 日志、通知、metrics、诊断和 POC 报告不得含音频、转写、临时文本、窗口身份、Fcitx token、session id 或模型原始异常。
- 本计划完成前，连续尾段保持关闭；不得用 SenseVoice 或第二个 Nano worker 冒充实时字幕。

## Task 1: 建立活跃会话、资源策略和调度的无模型契约

**Files:**

- Modify: `src/fun_voice/contracts.py`
- Modify: `src/fun_voice/config.py`
- Modify: `src/fun_voice/metrics.py`
- Modify: `scripts/config.example.toml`
- Modify: `tests/test_contracts.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_metrics.py`
- Create: `src/fun_voice/session.py`
- Create: `tests/test_session.py`

**Interfaces:**

- Add `DaemonState.PREPARING`, `FINALIZING`, `CORRECTING`, `REHYDRATING`, `ENRICHING`, and `ACTIVE_IDLE`; keep `IDLE`, `RECORDING`, `COMMITTING`, `ERROR` only as compatible names while callers migrate.
- Produce immutable `SessionKey(session_id: str, generation: int)` and `ModelTaskKind(final_tail | stable_segment | provisional_tail | correction | enrichment)`; stale `(session_id, generation)` work must have no side effect.
- Produce `ActiveSessionConfig(policy: Literal["memory_saver", "balanced", "sustained"], active_idle_seconds, worker_failsafe_idle_seconds, provisional_enabled)` and `ResourcePolicy` with fixed windows `120/480/1800` seconds. `sustained` is rejected unless the injected power probe reports AC.
- Produce `ActiveSessionController` as a pure, clock-injected state machine. It emits actions (`start_nano`, `show_overlay`, `finalize`, `begin_correction`, `rehydrate`, `enqueue_enrichment`, `stop_models`) rather than importing desktop, systemd, PipeWire, or a model runtime.

- [ ] **Step 1: 写出状态/配置/隐私的失败测试**

  在 `tests/test_session.py` 用 `FakeClock` 和 action collector 覆盖完整状态表：冷态按住发出 `start_nano` 与 `PREPARING`；成功的非空最终结果进入 `ACTIVE_IDLE` 并从完成时刻刷新 480 秒；120/480/1800 秒边界准确停止 Nano；锁屏、资源压力、连续模型错误和显式省内存立即停止；过期 generation 的 `nano_ready`/`enrichment_ready` 被忽略。加入：电池上的 `sustained` 被降为 `balanced` 并只返回固定枚举原因。

  在 config/metric tests 加：任意非 `xpu:0`、窗口值偏离固定策略、或 `worker_failsafe_idle_seconds < 1800` 均失败。另在启动 gate 测试中证明：未通过 POC 时，即使配置 `provisional_enabled=true` 也必须拒绝启用临时转写。metrics 只接受固定的 `session_policy`、`session_transition`、`risk_gate`、`nano_rehydration`、`background_enrichment` 枚举和整数时间；尝试写入 `text` 字段必须继续失败。

- [ ] **Step 2: 运行 RED 测试**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_contracts.py tests/test_config.py tests/test_metrics.py tests/test_session.py -q`

  Expected: FAIL，因为新状态、策略、state machine 与受限指标字段尚不存在；失败不可来自模型导入或真实 X11。

- [ ] **Step 3: 最小实现纯契约与配置解析**

  在 `contracts.py` 以冻结 dataclass/enum 实现上述值对象。`SessionKey` 不得实现会把 id 写入日志的 `__repr__`；使用 opaque id 仅作进程内相等比较。`session.py` 只接收 `clock`、`resource_probe` 和 action sink；转换失败返回固定 `ErrorCode`，不保存任何会话内容。

  将当前 `InferenceConfig.idle_unload_seconds` 重命名为 `worker_failsafe_idle_seconds`（保留旧 TOML key 仅一个发布周期，并在读取时映射；禁止在日志提示旧值），默认值改为 1800。活跃窗口只能由 `ActiveSessionConfig` 和 controller 的明确 `stop_models` 控制，不能依靠 worker 请求结束后的 120 秒隐式卸载。将示例配置写成：

  ```toml
  [active_session]
  policy = "balanced"             # memory_saver | balanced | sustained
  # 120 / 480 / 1800 秒窗口为固定产品策略，不接受任意秒数。
  provisional_enabled = false      # 仅本机 incremental POC 报告通过后才可改 true
  ```

  在 `MetricsLedger` 中只增加批准的字段和固定枚举白名单；统计仅输出计数/P50/P95。

- [ ] **Step 4: 运行局部 GREEN、lint 与类型检查**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_contracts.py tests/test_config.py tests/test_metrics.py tests/test_session.py -q && .venv/bin/ruff check src/fun_voice/contracts.py src/fun_voice/config.py src/fun_voice/metrics.py src/fun_voice/session.py tests/test_contracts.py tests/test_config.py tests/test_metrics.py tests/test_session.py && .venv/bin/mypy src/fun_voice/contracts.py src/fun_voice/config.py src/fun_voice/metrics.py src/fun_voice/session.py`

  Expected: PASS；测试能够证明 controller 与配置不导入 `torch`、`funasr`、`Xlib` 或 systemd。

- [ ] **Step 5: 提交契约层**

  ```bash
  git add src/fun_voice/contracts.py src/fun_voice/config.py src/fun_voice/metrics.py \
    src/fun_voice/session.py scripts/config.example.toml \
    tests/test_contracts.py tests/test_config.py tests/test_metrics.py tests/test_session.py
  git commit -m "feat: define active voice session policy"
  ```

## Task 2: 用唯一 XPU Scheduler 替换分散的 worker 生命周期调用

**Files:**

- Create: `src/fun_voice/scheduler.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/worker.py`
- Modify: `src/fun_voice/xpu_lease.py`
- Modify: `src/fun_voice/config.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_worker.py`
- Create: `tests/test_scheduler.py`

**Interfaces:**

- Produce `XpuScheduler(submit, start_profile, stop_profile, health_profile, clock)` with one dispatcher thread and a heap ordered exactly `final_tail`, `stable_segment`, `provisional_tail`, `correction`, `enrichment`.
- `submit(key, kind, fn) -> TaskHandle` never runs a model task on the caller thread. `TaskHandle.cancel()` and `cancel_before(kind)` only cancel queued work; a started decode is marked stale and its result discarded after it returns.
- `run_correction(key, profile, fn)` must call `stop_profile(profile)` and verify `health_profile(profile).state in {inactive, failed}` before invoking Qwen. Uncertain release returns raw text with fixed `skipped_lease` status.
- The worker gains a `health` reply field `lifecycle: loading | ready | inactive | failed` but keeps its existing same-UID socket and message size bounds. Worker’s 1800-second failsafe remains a final safety net, not the active-session timer.

- [ ] **Step 1: 写出 scheduler 竞争与优先级的失败测试**

  Create fakes whose task bodies block on events. Assert: a final tail queued after provisional work runs first; two Nano `generate` calls never overlap; new recording cancels pending enrichment; a post-cancel task can finish physically but cannot update its completion collector; and a correction callback is never called unless fake `stop_profile` then `health_profile` report `inactive`/`failed`.

  Add daemon tests demonstrating that it no longer invokes `default_start_worker_service`, `default_stop_worker_service`, `nano_preloader`, or `XpuLeaseCoordinator` from arbitrary pipeline branches. Add worker protocol tests for unknown/invalid `lifecycle` and no text/audio in health responses.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_scheduler.py tests/test_daemon.py tests/test_worker.py -q`

  Expected: FAIL because `XpuScheduler` and lifecycle verification do not yet exist.

- [ ] **Step 3: Implement serial ownership and cancellation**

  Implement `scheduler.py` with a condition-protected priority heap; all callbacks are invoked outside its lock. The scheduler owns the model-profile transition table, including `nano`, `sensevoice`, `qwen`, `enrichment`; it must reject a Qwen request while *any* ASR profile is `loading`/`ready`, and reject a second non-Nano model while Nano is active. It passes `SessionKey` into completion callbacks and checks current generation before publishing anything.

  Move service start/stop wrappers behind a `ModelProfileSupervisor` protocol in `daemon.py`; adapt existing `SocketWorkerClient` and `XpuLeaseCoordinator` rather than deleting tested socket behavior. Add a bounded same-UID `health` round trip before and after service stop. Do not infer release from VRAM counters. Update `WorkerHealth` codec with a fixed lifecycle enum and ensure failed start never claims ready.

  Preserve existing Nano→SenseVoice fallback, but route fallback through the scheduler and only on `worker.model_load`/`worker.oom`. No normal transcription may be redirected to SenseVoice.

- [ ] **Step 4: Run focused regression and static checks**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_scheduler.py tests/test_daemon.py tests/test_worker.py tests/test_worker_protocol.py tests/test_end_to_end_fakes.py -q && .venv/bin/ruff check src/fun_voice/scheduler.py src/fun_voice/daemon.py src/fun_voice/worker.py src/fun_voice/xpu_lease.py tests/test_scheduler.py tests/test_daemon.py tests/test_worker.py && .venv/bin/mypy src/fun_voice/scheduler.py src/fun_voice/daemon.py src/fun_voice/worker.py src/fun_voice/xpu_lease.py`

  Expected: PASS; fakes prove exactly one model task at a time and no hidden model loading during daemon construction.

- [ ] **Step 5: 提交调度层**

  ```bash
  git add src/fun_voice/scheduler.py src/fun_voice/daemon.py src/fun_voice/worker.py \
    src/fun_voice/xpu_lease.py src/fun_voice/config.py \
    tests/test_scheduler.py tests/test_daemon.py tests/test_worker.py tests/test_worker_protocol.py tests/test_end_to_end_fakes.py
  git commit -m "feat: serialize active voice model work"
  ```

## Task 3: 添加不抢焦点的 X11 临时状态层，并接入热会话状态机

**Files:**

- Create: `src/fun_voice/overlay.py`
- Modify: `src/fun_voice/desktop.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/session.py`
- Modify: `tests/test_desktop.py`
- Create: `tests/test_overlay.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_end_to_end_fakes.py`

**Interfaces:**

- Produce `OverlayModel(phase, stable_text="", provisional_text="", level: int | None = None)` and `OverlayController.show(model)`, `clear()`, `close()`.
- Produce `X11TransientOverlay` on a display owned by the overlay UI thread. It creates an override-redirect X11 window, selects no keyboard events, never calls focus/raise/input injection APIs, and draws only supplied in-memory strings.
- `VoiceDaemon` receives an `OverlayController` dependency. All state changes call it through `ActiveSessionController` actions; direct notification messages remain fixed-category fallbacks for unavailable overlay/capture/model errors.

- [ ] **Step 1: 写出 overlay 和 daemon 集成的失败测试**

  Build a fake X11 window/display that records create/map/unmap/draw/close calls. Assert `show(PREPARING)` maps a window without any focus call; `show(RECORDING)` redraws level/duration without text; stable content is dark and provisional content light in the render model; `clear()` overwrites/unmaps and drops all text references; and `close()` is idempotent.

  Add daemon tests for: cold press displays `PREPARING` before Nano completes; hot press immediately displays `RECORDING`; release displays `FINALIZING`, then `CORRECTING` only when selected later by Risk Gate; commit/focus rejection/error clears the overlay. Assert fake clipboard/Fcitx/injector receive no provisional text.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_overlay.py tests/test_desktop.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q`

  Expected: FAIL because overlay types and state action wiring do not exist.

- [ ] **Step 3: Implement the minimal X11 overlay and nonblocking state wiring**

  Implement `overlay.py` with a UI-command queue. The daemon only places immutable `OverlayModel` values onto it; the UI thread owns all Xlib objects. Use a small override-redirect window positioned near the pointer/root work area, `event_mask=0`, no `set_input_focus`, no clipboard selections and no `XTest`. Never log the model fields. If X11 creation fails, use a no-op controller and a fixed `overlay.unavailable` metric; do not fail audio input.

  Refactor `VoiceDaemon.start_if_idle()` to create one `SessionKey`, capture focus, start recording, then ask the controller/scheduler to prepare Nano. Release must return from the X11 listener quickly: it seals recording and schedules `FINALIZING`; it no longer performs ASR while holding the daemon state lock. Existing `stop()` and auto-stop tests retain their observable single-finalization behavior.

- [ ] **Step 4: Run local GREEN and static checks**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_overlay.py tests/test_desktop.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q && .venv/bin/ruff check src/fun_voice/overlay.py src/fun_voice/desktop.py src/fun_voice/daemon.py src/fun_voice/session.py tests/test_overlay.py tests/test_desktop.py tests/test_daemon.py && .venv/bin/mypy src/fun_voice/overlay.py src/fun_voice/desktop.py src/fun_voice/daemon.py src/fun_voice/session.py`

- [ ] **Step 5: 提交界面与状态接线**

  ```bash
  git add src/fun_voice/overlay.py src/fun_voice/desktop.py src/fun_voice/daemon.py \
    src/fun_voice/session.py tests/test_overlay.py tests/test_desktop.py \
    tests/test_daemon.py tests/test_end_to_end_fakes.py
  git commit -m "feat: show transient active voice status"
  ```

## Task 4: 引入内存音频借用、VAD endpoint 和增量 Nano 的硬件 POC 门

**Files:**

- Modify: `src/fun_voice/capture.py`
- Modify: `src/fun_voice/contracts.py`
- Modify: `src/fun_voice/nano_runtime.py`
- Modify: `src/fun_voice/worker.py`
- Modify: `src/fun_voice/scheduler.py`
- Create: `src/fun_voice/incremental_poc.py`
- Create: `scripts/run-incremental-nano-poc.sh`
- Modify: `tests/test_capture.py`
- Modify: `tests/test_worker.py`
- Modify: `tests/test_worker_protocol.py`
- Create: `tests/test_incremental_poc.py`

**Interfaces:**

- Produce reference-counted `AudioLease(CaptureArtifact, release)` and `LiveCaptureView.snapshot(start_ms, end_ms) -> AudioLease`. A lease is an anonymous memfd/temporary in-memory handle, has no stable path, and remains valid until its final `release()`.
- The Nano worker gains `detect_vad` and `transcribe_window` requests over its existing private socket. New live-audio requests transfer one duplicated lease descriptor with `SCM_RIGHTS` plus bounded JSON metadata `(SessionKey, source_start_ms, source_end_ms)`; the worker never accepts an audio pathname for these operations and closes the received descriptor after the request.
- Produce a POC report with fixed non-text fields: `ready`, Nano/VAD device evidence, segment counts, duplicate-boundary count, final-text equality/CER aggregate, peak XPU memory and deadlock/timeout outcome.

- [ ] **Step 1: 写出 capture/worker/POC 的失败测试**

  Extend recorder fakes so writer chunks arrive while recording. Assert overlapping snapshots have exactly the expected sample boundaries, `release()` is idempotent, a second recording cannot invalidate a retained lease, and cleanup/daemon shutdown closes every unreferenced memfd without leaving a named file. For worker protocol, assert `SCM_RIGHTS` passes exactly one descriptor, `detect_vad` only returns ordered time ranges, `transcribe_window` preserves source offsets, unknown ids/ranges/ancillary layouts fail with a fixed protocol code, and health/error responses contain no audio path.

  POC unit tests must use a fake runner and prove failure reports cannot set `provisional_enabled`; a passed fake report must be revision/device-bound and less than the configured freshness window. Do not unit-test model text by embedding user transcription in an artifact.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_capture.py tests/test_worker.py tests/test_worker_protocol.py tests/test_incremental_poc.py -q`

  Expected: FAIL because leases, live windows, worker operations and POC report validation are absent.

- [ ] **Step 3: Implement leases and a disabled-by-default POC harness**

  Refactor `PipeWireRecorder` so `stop()` still returns the existing full `CaptureArtifact`, but its backing file is managed by a small ref-count object. A live snapshot copies only the bounded requested PCM range to a new anonymous `/dev/shm` temporary file; it never exposes recorder memory directly across threads. Extend the worker/client transport for the two new live operations with `sendmsg`/`recvmsg` and `SCM_RIGHTS`; retain the existing JSON-line path protocol only for the compatibility whole-artifact `transcribe` operation. Existing 10-minute RAM threshold and 60-second private shards remain unchanged; snapshots of a long recording read shard bytes then immediately close them after writing the anonymous lease.

  Add Nano runtime methods that run the already-loaded VAD on supplied samples and execute one window through the existing serialized `_generate_lock`; do not instantiate a second runtime. Worker serializes these results with source offsets and rejects any device other than `xpu:0` before processing.

  `scripts/run-incremental-nano-poc.sh` invokes the file-backed module (never Python stdin) against a user-owned local test corpus. It must compare final full-utterance transcription with the merged incremental sequence, measure duplicate boundary rate and deadlock/timeout behavior, and write only aggregate JSON under the private runtime directory. A missing/corrupt/stale/revision-mismatched report means disabled.

- [ ] **Step 4: Run deterministic checks, then the real XPU gate**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_capture.py tests/test_worker.py tests/test_worker_protocol.py tests/test_incremental_poc.py -q && .venv/bin/ruff check src/fun_voice/capture.py src/fun_voice/contracts.py src/fun_voice/nano_runtime.py src/fun_voice/worker.py src/fun_voice/incremental_poc.py tests/test_capture.py tests/test_worker.py tests/test_worker_protocol.py tests/test_incremental_poc.py && .venv/bin/mypy src/fun_voice/capture.py src/fun_voice/contracts.py src/fun_voice/nano_runtime.py src/fun_voice/worker.py src/fun_voice/incremental_poc.py`

  Then run: `scripts/run-incremental-nano-poc.sh`

  Expected hardware gate: all parameters/runtimes report `xpu`; final merge does not regress the private corpus beyond the approved CER tolerance; no duplicate boundary survives final merge; and no timeout/deadlock occurs under final-tail preemption. If any gate fails, keep `provisional_enabled=false` and ship only state/VAD UI.

- [ ] **Step 5: 提交 lease 与 POC 门**

  ```bash
  git add src/fun_voice/capture.py src/fun_voice/contracts.py src/fun_voice/nano_runtime.py \
    src/fun_voice/worker.py src/fun_voice/scheduler.py src/fun_voice/incremental_poc.py \
    scripts/run-incremental-nano-poc.sh tests/test_capture.py tests/test_worker.py \
    tests/test_worker_protocol.py tests/test_incremental_poc.py
  git commit -m "feat: gate incremental Nano transcription by XPU poc"
  ```

## Task 5: 在通过 POC 后启用稳定段/推测尾段队列与安全拼接

**Files:**

- Modify: `src/fun_voice/session.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/scheduler.py`
- Modify: `src/fun_voice/overlay.py`
- Modify: `src/fun_voice/nano_runtime.py`
- Modify: `tests/test_session.py`
- Modify: `tests/test_scheduler.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_end_to_end_fakes.py`

**Interfaces:**

- Produce `LiveSegmentPlanner`: VAD seals a stable segment only after 400–800 ms silence; continuous speech emits a 1.5-second provisional tail with a fixed overlap; release emits exactly one final unsealed tail.
- `SegmentAssembler.accept(key, kind, source_range, text)` returns a display model and, only for finalization, the one final merged raw text/ordered segments. It tracks audio ranges, never compares or logs text, and cannot deliver provisional text to the committer.
- The feature is enabled only when `ActiveSessionConfig.provisional_enabled` is true *and* `IncrementalPocGate.is_approved()` matches local model revision/device; otherwise daemon uses whole-utterance final ASR and overlay shows no speculative text.

- [ ] **Step 1: 写出 queue/merge 的失败测试**

  Use marker texts in pure in-memory tests to assert queue priority `final > stable > provisional`, late provisional result is dropped after a newer stable/final range exists, overlaps are reconciled only by audio range and the approved deterministic boundary rule, and finalization has no duplicate/missing segment range. Verify a 399 ms silence does not seal, 400 ms does, and 800 ms remains valid. Assert release while a provisional request is queued cancels it and schedules final tail first.

  End-to-end fakes must assert overlay has provisional text before release only when gate approved; fake Fcitx/clipboard/XTEST observe exactly one final text after release; and fallback to whole-utterance ASR when the POC gate is false.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_session.py tests/test_scheduler.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q`

- [ ] **Step 3: Implement ordered planning and conservative final assembly**

  Feed bounded capture snapshots to `detect_vad` at a fixed cadence only while Nano is ready. `LiveSegmentPlanner` stores offsets/lease handles, not raw PCM or text. It asks the scheduler for stable/provisional work in the approved order. On scheduler backlog, it first cancels provisional tasks, never stable/final tasks. Overlay receives stable text in its persistent color and only the newest provisional tail in its replaceable color.

  At release, cancel every provisional task, seal the pending audio tail, wait only for final/stable work belonging to the current key, then assemble in source-time order. If a required stable request fails or becomes stale, discard partial display output and execute the existing full-artifact Nano transcription once; do not concatenate uncertain text. The whole-utterance result remains the only text eligible for Risk Gate, result storage and commit.

- [ ] **Step 4: Run local GREEN and integration checks**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_session.py tests/test_scheduler.py tests/test_daemon.py tests/test_end_to_end_fakes.py tests/test_capture.py tests/test_worker_protocol.py -q && .venv/bin/ruff check src/fun_voice/session.py src/fun_voice/daemon.py src/fun_voice/scheduler.py src/fun_voice/overlay.py src/fun_voice/nano_runtime.py tests/test_session.py tests/test_scheduler.py tests/test_daemon.py tests/test_end_to_end_fakes.py && .venv/bin/mypy src/fun_voice/session.py src/fun_voice/daemon.py src/fun_voice/scheduler.py src/fun_voice/overlay.py src/fun_voice/nano_runtime.py`

- [ ] **Step 5: 提交受 POC 保护的临时转写**

  ```bash
  git add src/fun_voice/session.py src/fun_voice/daemon.py src/fun_voice/scheduler.py \
    src/fun_voice/overlay.py src/fun_voice/nano_runtime.py tests/test_session.py \
    tests/test_scheduler.py tests/test_daemon.py tests/test_end_to_end_fakes.py
  git commit -m "feat: render POC-gated live Nano segments"
  ```

## Task 6: 用 Risk Gate 控制 Qwen，并完成热会话回暖、资源回收与验收

**Files:**

- Create: `src/fun_voice/risk_gate.py`
- Modify: `src/fun_voice/corrector.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/session.py`
- Modify: `src/fun_voice/metrics.py`
- Modify: `src/fun_voice/config.py`
- Modify: `src/fun_voice/selftest.py`
- Modify: `docs/operations.md`
- Modify: `docs/acceptance-checklist.md`
- Create: `tests/test_risk_gate.py`
- Modify: `tests/test_corrector.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**

- Produce pure `RiskGate.decide(raw_text, protected_terms) -> RiskDecision(hit: bool, reason: Literal["punctuation", "term", "mixed_technical", "explicit_polish", "none"])`; reason is an allowlisted category, never a matched token.
- Conditions are exactly: missing terminal punctuation plus continuous dictation shape; local homophone/term candidate; code/path/version/mixed-language boundary anomaly; or explicit per-session `polish`. No hit means Qwen process is never started.
- After Qwen exits, `ActiveSessionController` schedules `REHYDRATING` only if its active deadline remains valid and resource probe is healthy. Rehydrate failure returns to `ACTIVE_IDLE`/`IDLE` with a fixed metric and never delays an already-committed final text.

- [ ] **Step 1: 写出 Risk Gate 和 lifecycle 的失败测试**

  Table-test Mandarin prose, `get commit`/`py test`, paths, version-like values, protected commands and already punctuated simple text. Tests may use short synthetic literals but must assert only `reason` categories reach metrics. Add a daemon fake where corrector raises timeout/OOM/invalid output: every case commits raw Nano text and schedules only allowed rehydration after Qwen completion. Add no-risk test asserting the corrector runner has zero calls and Nano remains hot.

  Add active-window tests: a normal successful Nano result refreshes 480 seconds; Qwen releases Nano before starting, then rehydrates asynchronously; lock/resource pressure during correction prevents rehydration; explicit memory saver stops worker at 120 seconds. Verify `selftest` reports boolean/enum readiness only, no model text or process command details.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_risk_gate.py tests/test_corrector.py tests/test_daemon.py tests/test_metrics.py -q`

- [ ] **Step 3: Implement deterministic gate and post-Qwen rehydration**

  Make `VoiceDaemon` ask `RiskGate` after Nano finalization and before it asks scheduler for correction. Preserve `corrector.py`’s one-request Transformers XPU process and its envelope/similarity/protected-token validation; do not add a warm Qwen service or any alternative model. Qwen’s accepted text alone becomes `final_text`; raw Nano output remains the fallback for all errors or rejected candidates.

  Emit only fixed risk/correction/rehydration enum metrics and timing. Arrange final commit before rehydration. Rehydration uses the scheduler’s Nano preload action and a synthetic warmup, never user audio. Update operations/acceptance documentation with default/profile windows, POC flag behavior, resource/lock handling, and manual checks for cold first use, hot P95 ≤3 seconds on no-risk short phrases, worker absence at login and XPU-only model checks.

- [ ] **Step 4: Run full quality gate**

  Run: `PYTHONPATH=src .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy src`

  Then perform the documented local manual acceptance: restart daemon and confirm no worker/Qwen is active; hold/release `Super+C` for cold and hot input; check no-risk short input does not spawn Qwen; force a risk sample and confirm Nano is inactive before Qwen; lock screen/resource-pressure fake path unloads models; and inspect metrics for aggregates only.

- [ ] **Step 5: 提交首期活跃会话产品**

  ```bash
  git add src/fun_voice/risk_gate.py src/fun_voice/corrector.py src/fun_voice/daemon.py \
    src/fun_voice/session.py src/fun_voice/metrics.py src/fun_voice/config.py \
    src/fun_voice/selftest.py docs/operations.md docs/acceptance-checklist.md \
    tests/test_risk_gate.py tests/test_corrector.py tests/test_daemon.py tests/test_metrics.py
  git commit -m "feat: add active Nano session and conditional Qwen"
  ```
