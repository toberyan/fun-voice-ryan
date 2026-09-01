# 结构化结果、时间戳与本地身份富化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在活跃会话产品稳定后，向同一 UID 的本地消费者提供仅内存、10 分钟/8 条上限的结构化识别结果：保留 Nano 原文和最终校正文本、段级时间、token 时间戳、说话人聚类以及可信的本地身份匹配。桌面输入路径仍只使用最终校正文本；富化永远不能拖慢或改写已提交的输入。

**Architecture:** 在 [`2026-09-01-active-session-overlay-implementation.md`](2026-09-01-active-session-overlay-implementation.md) 的 `XpuScheduler` 上增加最低优先级的 `EnrichmentCoordinator`。提交后，`ResultStore` 先发布 `final_ready`，用同一个 `result_id` 更新为 `enriched` 或 `enrichment_cancelled`。音频由引用计数匿名 memfd 临时保活，只到富化完成/取消/超时；CAM++ 和身份匹配只在 scheduler 空闲的 XPU 串行任务中运行。身份资料只持久化 Secret Service 密钥加密的 embedding centroid，从不保存音频、原文、最终文本或结果 JSON。

**Tech Stack:** Python 3.12、Fun-ASR-Nano-2512、FunASR CAM++、PyTorch XPU、ModelScope 本地快照、`cryptography` AES-GCM、`secretstorage`、Unix-domain socket/SO_PEERCRED、pytest、ruff、mypy。

**Prerequisite:** 活跃会话计划的 Task 1–2（`SessionKey`、`XpuScheduler`、中央 model supervisor）必须先合入；Task 4 的 `AudioLease` 是富化投入运行前的硬前提。若增量 Nano POC 失败，不影响本计划，因为富化可以只使用松键后的最终音频。

## Global Constraints

- 所有对音频或 embedding 的模型运算必须在 `xpu:0`；无 CPU/CUDA 回退。若 CAM++/时间戳接口不能给出 XPU 证据，本计划功能保持关闭而非改走 FunASR CPU helper、NumPy/SciPy clustering 或 cloud API。
- `ResultStore` 的 TTL 固定 600 秒、容量固定 8。daemon 退出、TTL 到期、显式 clear、或容量淘汰时必须从 RAM 删除；不得写缓存、临时 JSON、sqlite、日志或 crash artifact。
- desktop commit socket 只可访问 `final_text` 的提交动作；结果/身份管理分别有受限 socket，均在 `0700` runtime dir、socket `0600`、每连接用 `SO_PEERCRED` 限制当前 UID。
- `result_id`、profile id、profile display name、embedding、音频句柄、文本、词级时间戳均不得进入日志、metrics、通知或 self-test；API 返回是唯一允许返回这些字段的路径。
- 一次新 `Super+C` 必须取消未开始的富化，释放它持有的 `AudioLease`，并优先录音；它不能取消或改变已经上屏的 `final_text`。
- 身份自动标注需要最佳分数超过接受阈值且领先第二候选至少 margin；否则仅返回匿名 `speaker_cluster`。绝不从低置信度匹配推断名称。

## Task 1: 定义可验证、不可变的结构化结果和富化状态契约

**Files:**

- Modify: `src/fun_voice/contracts.py`
- Modify: `src/fun_voice/config.py`
- Modify: `tests/test_contracts.py`
- Modify: `tests/test_config.py`

**Interfaces:**

- Produce frozen/slots dataclasses `TokenTimestamp`, `ResultSegment`, `CorrectionProvenance`, `ResultTranscript`, `VoiceResult`, and `EnrichmentState`.
- `VoiceResult` contains `result_id`, created/expires monotonic times, `raw_text`, `final_text`, correction status/fixed reason, engine/model revision, ordered `ResultSegment`s, timing/fixed errors and `state: final_ready | enriching | enriched | enrichment_cancelled | enrichment_failed`.
- `ResultSegment` contains `start_ms`, `end_ms`, raw/final segment text, `tokens`, anonymous `speaker_cluster`, optional trusted `speaker_profile_id`, and fixed provenance. Timestamps are either `available`, `unavailable`, or `approximate`; a caller cannot label an estimate as available.
- `VoiceResult.finalize_enrichment(segments, state, error) -> VoiceResult` accepts an ordered immutable sequence of `ResultSegment` and may update only token timestamps, speakers, identity fields, state and fixed error codes. It must reject a raw/final text or correction change.

- [ ] **Step 1: 写出失败的契约测试**

  Add constructor/codec tests for reversed ranges, non-monotonic segments, token outside segment, duplicate profile ids, unknown enum, raw/final text mutation after enrichment, and a successful transition `final_ready → enriching → enriched`. Add stale/cancelled transitions that preserve already-known fields. Test wire codecs reject arbitrary keys and cap token/segment counts before allocating an unbounded structure.

  Assert `repr(MetricsLedger.summary())` cannot gain values from a `VoiceResult`, and config rejects non-`xpu:0` timestamp/CAM++ device settings, non-fixed result TTL/capacity, or identity settings which ask to persist audio/text.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_contracts.py tests/test_config.py -q`

  Expected: FAIL because rich result types and immutable enrichment transitions do not exist.

- [ ] **Step 3: Implement bounded immutable contracts**

  Add only explicit `to_wire()` / `from_wire()` methods instead of generic `asdict()`. `VoiceResult` must distinguish API output (can contain text) from metrics output (never receives a `VoiceResult`). Create ids with `secrets.token_urlsafe`/UUID within the daemon only; do not put them in exceptions. Keep existing `Transcription.text`/`segments` backward-compatible while adding a structured payload or conversion helper at the daemon boundary.

  Add config fields with locked values for `result_ttl_seconds=600`, `result_max_entries=8`, enrichment timeout and accepted profile calibration ranges. Do not expose model name/device alternatives: Nano/CAM++ remain pinned and XPU-only.

- [ ] **Step 4: Run GREEN and static checks**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_contracts.py tests/test_config.py -q && .venv/bin/ruff check src/fun_voice/contracts.py src/fun_voice/config.py tests/test_contracts.py tests/test_config.py && .venv/bin/mypy src/fun_voice/contracts.py src/fun_voice/config.py`

- [ ] **Step 5: 提交结果契约**

  ```bash
  git add src/fun_voice/contracts.py src/fun_voice/config.py tests/test_contracts.py tests/test_config.py
  git commit -m "feat: define immutable enriched voice results"
  ```

## Task 2: 实现 owner-only、仅内存的 Result Store 与查询接口

**Files:**

- Create: `src/fun_voice/results.py`
- Modify: `src/fun_voice/config.py`
- Modify: `src/fun_voice/contracts.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `pyproject.toml`
- Create: `tests/test_results.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_install_scripts.py`

**Interfaces:**

- Add `results_socket: Path` to `RuntimePaths` and console tool `fun-voice-result` with only `latest`, `get <id>`, `list`, `clear` commands.
- Produce `ResultStore(clock, max_entries=8, ttl_seconds=600)` with `publish_final`, `begin_enrichment`, `finish_enrichment`, `cancel`, `latest`, `get`, `list_metadata`, `clear`.
- Produce `ResultSocketServer` and `ResultSocketClient`; protocol accepts one bounded JSON line, checks `SO_PEERCRED`, and accepts only `latest`, `get`, `list`, `clear`. A malformed/foreign request receives no text response and no log echo.

- [ ] **Step 1: 写出 broker/socket 的失败测试**

  With a fake monotonic clock, assert TTL boundary 600/601 seconds, FIFO eviction at 8 entries, daemon shutdown clear, and `clear` immediately releases all results. `latest` and `get` return an exact result snapshot but an unknown/expired id returns a fixed `not_found` status. Verify a returned object cannot mutate store state.

  Test UDS setup uses directory `0700`, socket `0600`, rejects other UID before decoding, rejects a >64 KiB command, rejects unknown fields/operations, and does not start an additional model or write a file. In daemon fakes, final commit first publishes structured `final_ready`; Fcitx/clipboard receives only the `final_text` field.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_results.py tests/test_daemon.py tests/test_install_scripts.py -q`

- [ ] **Step 3: Implement store and separate socket lifecycle**

  Use a locked `OrderedDict`/deque plus injected clock; purge expired entries before every public operation. Store copies only immutable contracts. Start `ResultSocketServer` with the daemon but keep it separate from `daemon.sock`, Fcitx and worker sockets. It must use the same safe socket unlink/permission rules, close on daemon shutdown and retain no request body.

  Add the CLI only as an owner-side inspection/control tool; it prints result payload only to the invoking terminal and never logs it. Installation deploys the console script but must not enable a new service. No `result_id` is added to aggregate metrics.

- [ ] **Step 4: Run GREEN and protocol/static checks**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_results.py tests/test_daemon.py tests/test_install_scripts.py tests/test_contracts.py -q && .venv/bin/ruff check src/fun_voice/results.py src/fun_voice/daemon.py src/fun_voice/contracts.py tests/test_results.py tests/test_daemon.py tests/test_install_scripts.py && .venv/bin/mypy src/fun_voice/results.py src/fun_voice/daemon.py src/fun_voice/contracts.py`

- [ ] **Step 5: 提交结果 API**

  ```bash
  git add src/fun_voice/results.py src/fun_voice/config.py src/fun_voice/contracts.py \
    src/fun_voice/daemon.py pyproject.toml tests/test_results.py tests/test_daemon.py tests/test_install_scripts.py
  git commit -m "feat: publish bounded local voice results"
  ```

## Task 3: 让 CaptureArtifact 的生命周期可被后台富化安全借用

**Files:**

- Modify: `src/fun_voice/capture.py`
- Modify: `src/fun_voice/contracts.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/session.py`
- Modify: `tests/test_capture.py`
- Modify: `tests/test_daemon.py`

**Interfaces:**

- Extend the `AudioLease` introduced in the active-session plan with `retain(owner)`, `release(owner)`, `close_all()`, `deadline`, and an opaque `lease_id` that is never exposed through the result API.
- `PipeWireRecorder.stop()` creates exactly one full-utterance lease. `VoiceDaemon` transfers its finalizer reference to `EnrichmentCoordinator` only after successful final commit; all non-enrichment exits release it immediately.
- A lease timeout, cancellation, final enrichment completion, new recording preemption and daemon shutdown all use the same idempotent `release` path.

- [ ] **Step 1: 写出生命周期失败测试**

  Assert final commit without enrichment closes the backing FD after desktop commit; final commit with queued enrichment leaves exactly one retained handle; new hotkey cancels queued enrichment and closes it; worker failure/capture error/Qwen error/focus rejection all close it; daemon shutdown closes it once even when a scheduler callback races. For long input fakes, prove temporary shard files are removed once merged and retained content remains only anonymous memfd, never a named path.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_capture.py tests/test_daemon.py -q`

- [ ] **Step 3: Implement reference transfer, not copying or persistence**

  Keep the existing privacy behavior: short recordings live only in RAM and long source shards are removed immediately after materialization. The ref-count wrapper owns the final anonymous backing file, guarding count/deadline under a lock. `VoiceDaemon._cleanup()` must release only its owner reference; it must not blindly call recorder-wide cleanup while an enrichment lease exists. No enrichment work receives a `/proc/<daemon-pid>/fd/<fd>` string after its lease closes.

- [ ] **Step 4: Run GREEN/static checks**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_capture.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q && .venv/bin/ruff check src/fun_voice/capture.py src/fun_voice/contracts.py src/fun_voice/daemon.py src/fun_voice/session.py tests/test_capture.py tests/test_daemon.py && .venv/bin/mypy src/fun_voice/capture.py src/fun_voice/contracts.py src/fun_voice/daemon.py src/fun_voice/session.py`

- [ ] **Step 5: 提交富化音频保活**

  ```bash
  git add src/fun_voice/capture.py src/fun_voice/contracts.py src/fun_voice/daemon.py \
    src/fun_voice/session.py tests/test_capture.py tests/test_daemon.py
  git commit -m "feat: retain capture audio only for active enrichment"
  ```

## Task 4: 用本机 XPU POC 验证并实现 Nano 时间戳输出

**Files:**

- Create: `src/fun_voice/timestamps.py`
- Modify: `src/fun_voice/nano_runtime.py`
- Modify: `src/fun_voice/worker.py`
- Modify: `src/fun_voice/contracts.py`
- Create: `src/fun_voice/timestamp_poc.py`
- Create: `scripts/run-nano-timestamp-poc.sh`
- Create: `tests/test_timestamps.py`
- Modify: `tests/test_worker_protocol.py`
- Modify: `tests/test_enhanced_poc_script.py`

**Interfaces:**

- `NanoTimestampAdapter` parses only the documented local Nano response fields. It returns ordered `TokenTimestamp` values or `unavailable`; it never fabricates character timings from string length.
- Worker operation `enrich_timestamps` consumes a live `AudioLease` capability and known raw segments, returns a bounded result plus `timestamp_status`, and uses the Nano/CAM++ XPU POC-approved path only.
- POC report contains model revision, package versions, all parameter devices, token monotonicity/count coverage, aggregate alignment error against locally owned reference samples, time/memory, and `ready`. It contains no sample text/audio/file path.

- [ ] **Step 1: 写出 adapter 和 gate 的失败测试**

  Unit-test response parsing for native token/timestamp field variants, malformed/non-monotonic data, gap/overlap policy, UTF-8 multi-codepoint tokens and a missing timestamp field. Assert missing/invalid output yields `unavailable`, not guessed timestamps. Gate tests reject CPU device evidence, wrong model revision, failed aggregate coverage, stale report, or an unavailable native feature.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_timestamps.py tests/test_worker_protocol.py tests/test_enhanced_poc_script.py -q`

- [ ] **Step 3: Implement native-only adapter and POC**

  Inspect the installed FunASR-Nano revision’s local API during implementation and isolate its undocumented/output-shape adaptation in `timestamps.py`; no response structure leaks into `daemon.py`. First invoke it in the file-backed POC against the private benchmark. Require all Nano/VAD parameters and generated tensors used for alignment to report `xpu`. If current Nano cannot emit trustworthy native timing, publish `timestamp_status="unavailable"`, record a fixed POC failure category and stop here—do not activate a CPU `forced_align()` fallback.

- [ ] **Step 4: Run deterministic checks and real hardware gate**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_timestamps.py tests/test_worker_protocol.py tests/test_enhanced_poc_script.py -q && .venv/bin/ruff check src/fun_voice/timestamps.py src/fun_voice/nano_runtime.py src/fun_voice/worker.py src/fun_voice/timestamp_poc.py tests/test_timestamps.py tests/test_worker_protocol.py && .venv/bin/mypy src/fun_voice/timestamps.py src/fun_voice/nano_runtime.py src/fun_voice/worker.py src/fun_voice/timestamp_poc.py`

  Then run: `scripts/run-nano-timestamp-poc.sh`

  Expected: only a passing report matching the installed model revision permits the runtime feature. A failed report still leaves final text and result API available with explicit `unavailable` timing.

- [ ] **Step 5: 提交时间戳门**

  ```bash
  git add src/fun_voice/timestamps.py src/fun_voice/nano_runtime.py src/fun_voice/worker.py \
    src/fun_voice/contracts.py src/fun_voice/timestamp_poc.py scripts/run-nano-timestamp-poc.sh \
    tests/test_timestamps.py tests/test_worker_protocol.py tests/test_enhanced_poc_script.py
  git commit -m "feat: gate Nano token timestamps by XPU poc"
  ```

## Task 5: 实现可抢占的 CAM++ 说话人聚类与结果富化

**Files:**

- Create: `src/fun_voice/diarization.py`
- Create: `src/fun_voice/enrichment.py`
- Modify: `src/fun_voice/scheduler.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/results.py`
- Modify: `src/fun_voice/preflight.py`
- Modify: `src/fun_voice/enhanced_poc.py`
- Create: `scripts/run-camplus-enrichment-poc.sh`
- Create: `tests/test_diarization.py`
- Create: `tests/test_enrichment.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**

- `CamplusEmbedder` loads a local CAM++ snapshot only in an XPU worker action; `embed(lease, segments) -> embeddings` verifies all inspectable parameters are XPU.
- `SpeakerClusterer` is deterministic and bounded. It returns anonymous labels such as `speaker_0` and `speaker_1`, never display names, and is permitted only when the CAM++ POC passes. It must not import/use CPU SciPy/sklearn clustering runtime paths.
- `EnrichmentCoordinator.enqueue(key, result_id, lease)` publishes `enriching`, queues one low-priority task, then updates ResultStore to `enriched`, `enrichment_cancelled`, or `enrichment_failed`; every outcome releases the lease.

- [ ] **Step 1: 写出 scheduler/enrichment 的失败测试**

  Use fake timestamps/CAM++/identity matcher. Assert `final_ready` is visible before enrichment begins; a new recording cancels queued work and changes only enrichment state; a running result with stale generation cannot update a newer result; final/stable Nano work precedes enrichment; and every completion/cancel/timeout releases the lease. Assert a POC failure produces `identity.unavailable`/unavailable fields rather than a second ASR or CPU clustering.

  Add CAM++ adapter tests for device mismatch, bad embedding dimensions, short/low-quality segment rejection, deterministic cluster ordering and no profile name in diagnostic/metric output.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_diarization.py tests/test_enrichment.py tests/test_daemon.py tests/test_metrics.py -q`

- [ ] **Step 3: Implement serial, cancellable post-commit enrichment**

  Extend `XpuScheduler` with its already-defined `enrichment` priority rather than starting a parallel process. Before CAM++ starts, it verifies Nano/SenseVoice/Qwen all inactive. The default implementation runs timestamps then CAM++ serially, records only duration/status enums, and checks session generation before each ResultStore mutation. Apply a hard timeout from config; `finally` always releases the AudioLease.

  `scripts/run-camplus-enrichment-poc.sh` must measure standalone and sequential Nano→CAM++ peak memory, representative embedding/cluster quality, device proof and cancellation latency. Concurrent Nano+CAM++ residency may only be considered in a later plan after this POC passes; it is explicitly not enabled here.

- [ ] **Step 4: Run local checks and real CAM++ POC**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_diarization.py tests/test_enrichment.py tests/test_daemon.py tests/test_metrics.py tests/test_results.py -q && .venv/bin/ruff check src/fun_voice/diarization.py src/fun_voice/enrichment.py src/fun_voice/scheduler.py src/fun_voice/daemon.py src/fun_voice/results.py tests/test_diarization.py tests/test_enrichment.py tests/test_daemon.py && .venv/bin/mypy src/fun_voice/diarization.py src/fun_voice/enrichment.py src/fun_voice/scheduler.py src/fun_voice/daemon.py src/fun_voice/results.py`

  Then run: `scripts/run-camplus-enrichment-poc.sh`

  Expected: passing XPU-only/quality/memory/cancellation report enables CAM++; otherwise ResultStore keeps raw/final text and timestamp status but returns anonymous/unavailable speaker fields.

- [ ] **Step 5: 提交后台富化**

  ```bash
  git add src/fun_voice/diarization.py src/fun_voice/enrichment.py src/fun_voice/scheduler.py \
    src/fun_voice/daemon.py src/fun_voice/results.py src/fun_voice/preflight.py \
    src/fun_voice/enhanced_poc.py scripts/run-camplus-enrichment-poc.sh \
    tests/test_diarization.py tests/test_enrichment.py tests/test_daemon.py tests/test_metrics.py
  git commit -m "feat: enrich local voice results after commit"
  ```

## Task 6: 添加加密身份资料、质量门控和显式注册管理

**Files:**

- Create: `src/fun_voice/identity.py`
- Modify: `src/fun_voice/diarization.py`
- Modify: `src/fun_voice/enrichment.py`
- Modify: `src/fun_voice/results.py`
- Modify: `src/fun_voice/config.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `pyproject.toml`
- Create: `tests/test_identity.py`
- Modify: `tests/test_enrichment.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_install_scripts.py`

**Interfaces:**

- Produce `IdentityVault` backed by a `0700` profile directory whose records are `0600`, encrypted with AES-256-GCM using a per-user key retrieved/created via Secret Service. Record plaintext schema is only `{profile_id, display_name, encrypted_embedding, model_revision, created_at, threshold_version}`; audio/text/result/session data is rejected by schema.
- Produce owner-only identity operations over a dedicated `identity.sock` (or a capability-separated result subserver): `begin_enrollment`, `submit_enrollment_utterance`, `enrollment_status`, `list_profiles`, `disable`, `reenroll`, `delete`, `delete_all`. `list_profiles` exposes names only to the owner terminal; no profile data appears in desktop socket/metrics.
- Enrollment accepts 3–5 freshly recorded utterances. `EnrollmentQualityGate` checks bounded duration, SNR/energy, speech coverage and embedding outlier distance; it persists only the final centroid after enough valid samples.
- `IdentityMatcher.match(cluster_embedding)` returns a profile id only if score ≥ accept threshold and best-minus-second ≥ margin; otherwise it returns anonymous.

- [ ] **Step 1: 写出 crypto/quality/matching 的失败测试**

  Use deterministic fake Secret Service/key provider to prove persisted bytes contain neither a profile name nor a known float-vector encoding; successful decrypt round-trip produces only allowed fields. Test wrong AAD/key, corrupted data, restrictive directory/file modes, and unavailable Secret Service: every case fails closed and leaves no partial plaintext record.

  Test 2, 3, 5, 6 enrollment samples; poor SNR/too-short/outlier rejection; centroid deterministic aggregation; disabled/deleted/re-enrolled profiles; threshold and margin boundary/tie behavior; and that low-confidence matching returns `speaker_profile_id=None`. Assert incoming audio/strings cannot be handed to `IdentityVault` APIs.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_identity.py tests/test_enrichment.py tests/test_config.py tests/test_install_scripts.py -q`

- [ ] **Step 3: Implement fail-closed vault and explicit enrollment flow**

  Keep profile metadata directory separate from `$XDG_RUNTIME_DIR`; create it with explicit `0700` and atomic temp-file/rename writes restricted to that exact directory. AES-GCM AAD binds `profile_id`, schema and CAM++ model revision. Retrieve an opaque 256-bit key through `secretstorage`; do not create a fallback key file, environment variable or CPU/cloud backend. On an unavailable Secret Service, show only fixed `identity.unavailable` and leave profile matching off.

  Enrollment is user-initiated and uses ordinary capture, scheduler and CAM++ task pathways. It must show a fixed overlay progress count, never display/retain utterance text, and release each enrollment AudioLease as soon as its embedding/quality result is produced. Matching runs inside the same post-commit enrichment task after anonymous clustering; it receives embeddings, not audio/text.

- [ ] **Step 4: Run GREEN/static checks**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_identity.py tests/test_enrichment.py tests/test_config.py tests/test_install_scripts.py -q && .venv/bin/ruff check src/fun_voice/identity.py src/fun_voice/diarization.py src/fun_voice/enrichment.py src/fun_voice/results.py src/fun_voice/config.py tests/test_identity.py tests/test_enrichment.py && .venv/bin/mypy src/fun_voice/identity.py src/fun_voice/diarization.py src/fun_voice/enrichment.py src/fun_voice/results.py src/fun_voice/config.py`

- [ ] **Step 5: 提交本地身份能力**

  ```bash
  git add src/fun_voice/identity.py src/fun_voice/diarization.py src/fun_voice/enrichment.py \
    src/fun_voice/results.py src/fun_voice/config.py src/fun_voice/daemon.py pyproject.toml \
    tests/test_identity.py tests/test_enrichment.py tests/test_config.py tests/test_install_scripts.py
  git commit -m "feat: add encrypted local voice identities"
  ```

## Task 7: 以私有基准、运行手册和升级门完成产品验收

**Files:**

- Modify: `src/fun_voice/benchmark.py`
- Modify: `src/fun_voice/metrics.py`
- Modify: `src/fun_voice/selftest.py`
- Modify: `docs/operations.md`
- Modify: `docs/acceptance-checklist.md`
- Create: `docs/quality-gates.md`
- Modify: `tests/test_benchmark.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_selftest.py`

**Interfaces:**

- Extend the local-only benchmark manifest to support a user-deletable 20–50 sample set and emit only aggregate CER, punctuation F1, protected-term retention, timestamp coverage/error, speaker cluster quality, identity false accept/reject, hot/cold P50/P95, peak XPU memory and fixed failure counts.
- `selftest` exposes only readiness/status booleans and revision fingerprints, never actual result data/profile names. It checks result/identity sockets with same UID and checks no model process is active immediately after daemon login start.
- `docs/quality-gates.md` defines pass/fail thresholds and mandatory re-run triggers for FunASR/vLLM/PyTorch XPU/Qwen/CAM++/model revision changes.

- [ ] **Step 1: 写出失败的 benchmark/observability 测试**

  Test benchmark aggregate computation with synthetic counts only; reject a manifest that points outside its private benchmark root, includes unbounded samples, or asks to export raw predictions. Verify reports omit original/final text, paths, profile names and result ids. Add metric tests for enrichment/identity counters and check unknown sensitive fields still fail. Add selftest fakes for expired result socket, foreign UID, unavailable Secret Service and inactive models at login.

- [ ] **Step 2: Run RED tests**

  Run: `PYTHONPATH=src .venv/bin/pytest tests/test_benchmark.py tests/test_metrics.py tests/test_selftest.py -q`

- [ ] **Step 3: Implement aggregate quality gates and operator workflow**

  Keep individual sample files entirely local and make benchmark output metrics-only. Do not add automatic uploads. Document a user procedure to create/delete the private corpus, enroll/revoke identities, inspect/clear ResultStore, reproduce an XPU POC, choose active-session profile and recover safely from a failed model/Secret Service. Define blocking gates: XPU-only proof, privacy/socket permission checks, no desktop injection from provisional text, hot no-risk P95 ≤3 seconds, no unapproved quality regression, and calibrated false-accept limits.

- [ ] **Step 4: Run final verification and manual acceptance**

  Run: `PYTHONPATH=src .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy src`

  Then perform documented real-device checks: daemon starts with all model workers absent; produce a result and query it only through owner result CLI; wait 600 seconds/clear and confirm disappearance; exercise hotkey during queued enrichment and confirm cancellation; register 3–5 utterances, test confident/ambiguous speaker behavior, disable and delete the profile; and run all three XPU POC scripts before enabling their corresponding flags.

- [ ] **Step 5: 提交验收闭环**

  ```bash
  git add src/fun_voice/benchmark.py src/fun_voice/metrics.py src/fun_voice/selftest.py \
    docs/operations.md docs/acceptance-checklist.md docs/quality-gates.md \
    tests/test_benchmark.py tests/test_metrics.py tests/test_selftest.py
  git commit -m "docs: add local voice quality gates"
  ```
