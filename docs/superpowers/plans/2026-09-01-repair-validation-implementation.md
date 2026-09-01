# Repair Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复配置接线、POC 严格性和真实 DDE 按住证据，使本地语音输入助手的发布检查与已确认设计一致。

**Architecture:** TOML 由 daemon 与 Worker 共同读取，Fcitx 超时在 daemon 边界完成 ms→s 转换。POC 对齐 NanoRuntime 的完整结果契约；daemon 通过同 UID socket 暴露无敏感内容的进程内按住标记供 selftest 使用。

**Tech Stack:** Python 3.12、pytest、ruff、mypy、Bash、FunASR Nano、Fcitx5 C++ addon、systemd user service。

## Global Constraints

- 仅支持 Deepin DDE X11；不得读取 `/dev/input` 或承诺 Wayland。
- 推理设备固定为 `xpu:0`；禁止 CPU/CUDA 自动回退或替换模型后端。
- 原样拼接模型文本；不得加入词典、正则或 LLM 后处理。
- 不持久化音频或转写文本；诊断、日志和报告不得包含两者。
- 保留用户当前未提交的 POC 与文档修改，使用最小 `apply_patch` 修改。

---

### Task 1: 接线并收紧 TOML 配置

**Files:**
- Modify: `src/fun_voice/config.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/worker.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_worker_protocol.py`
- Modify: `scripts/config.example.toml`
- Modify: `docs/operations.md`

**Interfaces:**
- Consumes: `Config.fcitx_commit_timeout_ms: int` and `Config.inference: InferenceConfig`.
- Produces: `build_fcitx_factory(cfg)` passes `cfg.fcitx_commit_timeout_ms / 1000` to `FcitxClient`; `worker.main()` passes TOML XPU parameters to `load_nano_runtime`.

- [ ] **Step 1: Write the failing tests**

```python
def test_load_config_uses_canonical_fcitx_timeout_ms(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[input_method]\ncommit_timeout_ms = 500\n")
    assert load_config(path).fcitx_commit_timeout_ms == 500

def test_build_fcitx_factory_converts_milliseconds_to_seconds() -> None:
    factory = build_fcitx_factory(Config(fcitx_commit_timeout_ms=500))
    assert factory.keywords == {"timeout": 0.5}
```

Add a `worker.main` test with patched `config.load_config`, `load_nano_runtime`, and `serve`; assert the captured loader kwargs contain `device="xpu:0"`, `dtype="bf16"`, `gpu_memory_utilization=0.35`, and `enforce_eager=True`. Add a rejecting test for `device="cpu"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_daemon.py tests/test_worker_protocol.py -q`

Expected: failure because the existing timeout is `0.5` and Worker does not load TOML.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class Config:
    fcitx_commit_timeout_ms: int = 500

def build_fcitx_factory(cfg: config.Config) -> Callable[[], FcitxClient]:
    return functools.partial(FcitxClient, timeout=cfg.fcitx_commit_timeout_ms / 1000)
```

Parse `input_method.commit_timeout_ms` as a positive integer, reject any non-`xpu:0` `inference.device`, and have `worker.main()` load `Config` before it calls `load_nano_runtime`. CLI options remain explicit overrides but receive the same XPU validation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_daemon.py tests/test_worker_protocol.py -q`

Expected: exit 0.

- [ ] **Step 5: Update user-facing configuration documentation**

Keep only effective TOML keys in `scripts/config.example.toml`, document `commit_timeout_ms = 500`, and replace the obsolete “没有独立配置文件” paragraph in `docs/operations.md`.

### Task 2: 使 XPU POC 拒绝部分分段结果

**Files:**
- Modify: `src/fun_voice/preflight.py`
- Modify: `tests/test_preflight.py`
- Modify: `scripts/run-nano-xpu-poc.sh`
- Modify: `docs/xpu-poc.md`

**Interfaces:**
- Consumes: `vad.detect(samples, sample_rate) -> list[tuple[int, int]]` and `engine.generate(slices, max_new_tokens=...)`.
- Produces: `check_decode` passes only if every sorted VAD segment has exactly one dictionary result with a string `text`.

- [ ] **Step 1: Write failing tests**

```python
def test_check_decode_fails_when_engine_omits_a_vad_segment() -> None:
    result = check_decode("decode_60s", _SegmentedEngine(["only-one"]),
                          _FakeVad([(0, 100), (300, 400)]), "long.wav",
                          max_new_tokens=256, min_segments=2)
    assert result.status == STATUS_FAIL
    assert result.detail["error_class"] == "ModelOutputError"
```

Add cases for excess results and a `{"text": None}` result. Add a shell-level verification that a real report has four nonempty source records after the POC script completes.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preflight.py -q`

Expected: the omitted-result case incorrectly passes before the implementation.

- [ ] **Step 3: Write minimal implementation**

```python
if not isinstance(results, list) or len(results) != len(slices):
    raise ModelOutputError("model result count does not match VAD segments")
if any(not isinstance(item, dict) or not isinstance(item.get("text"), str)
       for item in results):
    raise ModelOutputError("malformed model result")
```

Set `umask 077` before `REPORT_DIR`/`SAMPLES_DIR` creation. Append `{"source": ..., "language": ..., "duration_s": ...}` to `comp` inside the sample-source loop so `sample_composition.short.sources` and `.long.sources` are nonempty, while preserving the no-path/no-text report contract.

- [ ] **Step 4: Run tests and POC syntax check**

Run: `uv run pytest tests/test_preflight.py -q && bash -n scripts/run-nano-xpu-poc.sh`

Expected: exit 0.

### Task 3: 以无敏感内存标记验证真实 DDE 按住触发

**Files:**
- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/selftest.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_selftest.py`
- Modify: `docs/acceptance-checklist.md`
- Modify: `tests/manual/test_dde_press_release.md`

**Interfaces:**
- Consumes: a same-UID daemon request `{"op":"diagnostics"}`.
- Produces: `{"status":"ok","held_trigger_seen": bool}` and a selftest `bridge_hold_timing` pass only after `held_trigger_seen` is true.

- [ ] **Step 1: Write failing tests**

```python
def test_diagnostics_marks_a_start_observed_while_c_is_down() -> None:
    harness = Harness()
    assert harness.daemon.diagnostics() == {"held_trigger_seen": False}
    _started(harness)
    assert harness.daemon.diagnostics() == {"held_trigger_seen": True}

def test_bridge_timing_fails_without_real_held_trigger() -> None:
    assert check_bridge_timing(lambda: False).status == STATUS_FAIL
```

Add an IPC handler test for the diagnostics response and a passing selftest case for `lambda: True`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_daemon.py tests/test_selftest.py -q`

Expected: failure because diagnostics and the injected probe do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
self._held_trigger_seen = False
if c_down:
    self._held_trigger_seen = True

def diagnostics(self) -> dict[str, bool]:
    return {"held_trigger_seen": self._held_trigger_seen}
```

Teach `dispatch` and `DaemonRequestHandler` to preserve mapping results in the response, add a bounded same-UID daemon probe in `selftest.py`, and require both fake bridge mapping and the live boolean for `bridge_hold_timing`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_daemon.py tests/test_selftest.py -q`

Expected: exit 0.

### Task 4: 卫生、格式和完整回归

**Files:**
- Modify: `tests/test_daemon.py`
- Modify: `docs/xpu-poc.md`
- Modify: `docs/operations.md`
- Modify: `docs/acceptance-checklist.md`
- Modify: `tests/manual/test_dde_press_release.md`

- [ ] **Step 1: Fix the existing ruff import failure**

Run: `uv run ruff check tests/test_daemon.py --fix`

Expected: import block reformatted without behavioral change.

- [ ] **Step 2: Update stale wording**

Document real TOML behavior, explain that POC measurements vary per run, and require a manual held Super+C once after daemon start before selftest can pass its DDE timing item.

- [ ] **Step 3: Remove confirmed historical runtime samples**

Before deletion, require `/run/user/1000/fun-voice-ryan/investigate-samples` to be owned by the current user and contain only the three known POC files. Then remove precisely that directory and confirm the runtime directory retains only report and sockets.

- [ ] **Step 4: Execute final verification**

Run:

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src
cmake --build build/fcitx5-fun-voice --parallel 2
ctest --test-dir build/fcitx5-fun-voice --output-on-failure
scripts/run-nano-xpu-poc.sh --skip-model-download
PYTHONPATH=src .venv/bin/fun-voice-selftest --format json
git diff --check
```

Expected: all automated gates pass. The last selftest can report a DDE-timing failure until a human holds Super+C once in the live DDE X11 session; after that action it must pass without retaining audio or text.
