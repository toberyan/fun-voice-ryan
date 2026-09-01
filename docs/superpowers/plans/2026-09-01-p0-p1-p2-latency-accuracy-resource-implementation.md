# P0/P1/P2 Accuracy, Latency, and XPU Resource Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add private local measurements, preload Nano during an effective push-to-talk recording, and serialize ASR/Qwen XPU residency.

**Architecture:** `metrics.py` owns bounded, text-free stage timings. The worker gains one `preload` operation that materializes its existing `LazyTranscriber` after PipeWire capture starts. The daemon owns a lease coordinator that confirms the producing ASR service has exited before invoking the one-request Qwen3.5-0.8B corrector.

**Tech Stack:** Python 3.12, pytest, PipeWire, systemd user services, Unix sockets, FunASR/vLLM XPU, Transformers XPU.

## Global Constraints

- Models use only `xpu:0`; do not introduce CPU fallback or another correction model.
- Nano remains primary; SenseVoiceSmall is only a Nano `model_load` / `oom` fallback and is never preloaded.
- Login loads no neural model. A preload happens only after capture has started and daemon state is `RECORDING`.
- No metric, report, socket reply, notification, or journal entry includes text, audio, paths, window identity, or Fcitx tokens.
- Qwen is fixed to `Qwen/Qwen3.5-0.8B`, BF16, non-thinking, and starts only after the active ASR profile is inactive.

---

### Task 1: Bounded private metric ledger and daemon metrics endpoint

**Files:**

- Create: `src/fun_voice/metrics.py`
- Create: `tests/test_metrics.py`
- Modify: `src/fun_voice/daemon.py:498-516,672-718`
- Modify: `tests/test_daemon.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SessionMetric:
    sequence: int
    capture_duration_ms: int | None = None
    preload_ms: int | None = None
    asr_ms: int | None = None
    correction_ms: int | None = None
    commit_ms: int | None = None
    end_to_end_ms: int | None = None
    asr_profile: str | None = None
    nano_preload: str = "not_requested"
    correction: str = "disabled"
    error_code: str | None = None
    nano_was_stopped_for_qwen: bool = False

class MetricsLedger:
    def begin(self) -> int: ...
    def record(self, sequence: int, **updates: object) -> None: ...
    def summary(self) -> dict[str, object]: ...
```

- [x] **Step 1: Write failing tests**

```python
def test_metrics_summary_has_percentiles_but_no_sensitive_values() -> None:
    ledger = MetricsLedger(max_entries=2)
    a = ledger.begin(); ledger.record(a, asr_ms=20, asr_profile="nano")
    b = ledger.begin(); ledger.record(b, asr_ms=40, asr_profile="sensevoice")
    assert ledger.summary()["asr_ms"] == {"p50": 30, "p95": 39}
    assert "text" not in repr(ledger.summary())

def test_daemon_metrics_operation_returns_aggregate_only() -> None:
    assert Harness().daemon.dispatch({"op": "metrics"}) == {"count": 0}
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_metrics.py tests/test_daemon.py -q`

Expected: FAIL because `MetricsLedger` and `metrics` dispatch are absent.

- [x] **Step 3: Implement**

Use a deque of 128 `SessionMetric` objects and nearest-rank integer percentiles. Record only fixed enums and stage durations. Add `{"op":"metrics"}` to `VoiceDaemon.dispatch`; its reply is the aggregate summary, never rows. Instrument capture completion, preload completion, worker request, corrector request, and commit with `time.monotonic()`.

- [x] **Step 4: Verify GREEN**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_metrics.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q && .venv/bin/ruff check src/fun_voice/metrics.py src/fun_voice/daemon.py tests/test_metrics.py && .venv/bin/mypy src`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/fun_voice/metrics.py src/fun_voice/daemon.py tests/test_metrics.py tests/test_daemon.py
git commit -m "feat: add private voice pipeline metrics"
```

### Task 2: Aggregate-only local benchmark CLI

**Files:**

- Create: `src/fun_voice/benchmark.py`
- Create: `tests/test_benchmark.py`
- Modify: `pyproject.toml`
- Modify: `docs/operations.md`

**Interfaces:**

```python
@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    audio: str
    reference: str
    terms: tuple[str, ...] = ()

def score_text(reference: str, candidate: str, terms: Sequence[str]) -> dict[str, float | int]: ...
def aggregate_scores(rows: Sequence[tuple[str, Mapping[str, float | int]]]) -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing score and privacy tests**

```python
def test_score_text_calculates_cer_terms_and_punctuation_without_text() -> None:
    score = score_text("运行 git commit。", "运行 git commit", ("git", "commit"))
    assert score["cer"] > 0
    assert score["term_exact"] == 1.0
    assert score["punctuation_f1"] == 0.0
    assert "运行" not in repr(score)

def test_aggregate_groups_category_without_audio_or_reference() -> None:
    report = aggregate_scores([("mixed", {"cer": 0.1, "term_exact": 1.0})])
    assert report["categories"]["mixed"]["count"] == 1
    assert "audio" not in repr(report)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_benchmark.py -q`

Expected: FAIL because the benchmark module is absent.

- [ ] **Step 3: Implement**

Add `fun-voice-benchmark --manifest PATH [--output PATH]`. Manifest JSONL is user-owned and untracked; each line contains `category`, `audio`, `reference`, optional `terms`. Use character Levenshtein CER, configured term exactness, punctuation precision/recall/F1 and wall-clock cold/warm request durations. Recognized/reference text lives only during scoring. The report has category aggregates, counts and percentiles only; write it only at explicit `--output` with mode `0600`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_benchmark.py -q && .venv/bin/ruff check src/fun_voice/benchmark.py tests/test_benchmark.py && .venv/bin/mypy src`

```bash
git add src/fun_voice/benchmark.py tests/test_benchmark.py pyproject.toml docs/operations.md
git commit -m "feat: add local aggregate accuracy benchmark"
```

### Task 3: Worker preload protocol and recording-phase Nano loading

**Files:**

- Modify: `src/fun_voice/worker.py:66-119,179-245`
- Modify: `src/fun_voice/daemon.py:200-390,538-600`
- Modify: `tests/test_worker_protocol.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_end_to_end_fakes.py`

**Interfaces:**

```python
class LazyTranscriber:
    def preload(self) -> WorkerHealth: ...

class SocketWorkerClient:
    def preload(self) -> None: ...

class VoiceDaemon:
    def __init__(..., nano_preloader: Callable[[], None] | None = None, ...): ...
```

- [ ] **Step 1: Write failing behavior tests**

```python
def test_preload_constructs_lazy_runtime_once_then_transcribe_reuses_it() -> None:
    loader = CountingLoader(FakeRuntime())
    worker = Worker(LazyTranscriber(loader, device="xpu:0"))
    assert worker.handle({"id": "p", "op": "preload"})["model_ready"] is True
    worker.handle({"id": "t", "op": "transcribe", "audio": "/tmp/a", "sample_rate": 16000})
    assert loader.calls == 1

def test_daemon_requests_nano_preload_only_after_recording_starts() -> None:
    calls: list[str] = []
    daemon = Harness(nano_preloader=lambda: calls.append("preload")).daemon
    assert daemon.start_if_idle() == "started"
    wait_until(lambda: calls == ["preload"])
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_worker_protocol.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q`

Expected: FAIL because `preload` does not exist.

- [ ] **Step 3: Implement**

Worker `preload` calls only `LazyTranscriber._get_runtime()` and returns XPU/model readiness without audio. `SocketWorkerClient.preload` starts its assigned template if needed and sends that operation. Once daemon state is `RECORDING`, create one daemon preload thread. Its exception is captured as a metric enum; it never blocks hotkey handling or release. Wire only `nano_worker.preload` in `main`; never preload `FallbackWorkerClient` or SenseVoice. Preserve the single-threaded worker server so preload and transcribe serialize model access.

- [ ] **Step 4: Verify GREEN and commit**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_worker_protocol.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q && .venv/bin/ruff check src/fun_voice/worker.py src/fun_voice/daemon.py && .venv/bin/mypy src`

```bash
git add src/fun_voice/worker.py src/fun_voice/daemon.py tests/test_worker_protocol.py tests/test_daemon.py tests/test_end_to_end_fakes.py
git commit -m "feat: preload Nano during recording"
```

### Task 4: Live Qwen controls and protected technical tokens

**Files:**

- Modify: `src/fun_voice/config.py`
- Modify: `src/fun_voice/corrector.py`
- Modify: `scripts/config.example.toml`
- Modify: `tests/test_config.py`
- Modify: `tests/test_corrector.py`

**Interfaces:**

```python
class EnhancedInferenceConfig:
    correction_max_source_characters: int = 512
    correction_max_new_tokens: int = 512
    correction_timeout_seconds: int = 30
    correction_protected_terms: tuple[str, ...] = ()

def extract_protected_tokens(raw_text: str, configured_terms: Sequence[str]) -> tuple[str, ...]: ...
def validate_correction(raw_text: str, corrected_text: str, protected_terms: Sequence[str] = ()) -> str: ...
```

- [ ] **Step 1: Write failing tests**

```python
def test_live_qwen_limits_are_loaded_from_config() -> None:
    value = validate_enhanced_inference_config(EnhancedInferenceConfig())
    assert value.correction_timeout_seconds == 30
    assert value.correction_max_new_tokens == 512

def test_changed_protected_command_is_rejected() -> None:
    with pytest.raises(CorrectionError, match="invalid_output"):
        validate_correction("运行 git commit --amend", "运行 get commit --amend")
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_config.py tests/test_corrector.py -q`

Expected: FAIL because these controls and preservation validation are absent.

- [ ] **Step 3: Implement**

Remove ignored Transformers-Qwen `gpu_memory_utilization` and `max_model_len` fields. Parse and validate the three live limits plus an optional list of configured protected terms. Extract ordered unique URL, path, backtick code, option, version, `snake_case`, `CamelCase` and configured spans. Candidate validation requires every protected token at strictly increasing positions. Pass generation/timeout settings to the Qwen child; any failure remains a raw-text fallback.

- [ ] **Step 4: Verify GREEN and commit**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_config.py tests/test_corrector.py tests/test_daemon.py -q && .venv/bin/ruff check src/fun_voice/config.py src/fun_voice/corrector.py && .venv/bin/mypy src`

```bash
git add src/fun_voice/config.py src/fun_voice/corrector.py scripts/config.example.toml tests/test_config.py tests/test_corrector.py
git commit -m "feat: bound and protect Qwen correction"
```

### Task 5: Serialize producing ASR worker and Qwen with a lease

**Files:**

- Create: `src/fun_voice/xpu_lease.py`
- Create: `tests/test_xpu_lease.py`
- Modify: `src/fun_voice/contracts.py`
- Modify: `src/fun_voice/nano_runtime.py`
- Modify: `src/fun_voice/daemon.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_end_to_end_fakes.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class Transcription:
    text: str
    segments: tuple[Segment, ...] = ()
    request_id: str | None = None
    engine: Literal["nano", "sensevoice"] = "nano"

class XpuLeaseCoordinator:
    def release_asr_for_qwen(self, profile: Literal["nano", "sensevoice"]) -> bool: ...
```

- [ ] **Step 1: Write failing mutual-exclusion tests**

```python
def test_qwen_lease_stops_the_producing_asr_profile() -> None:
    calls: list[str] = []
    lease = XpuLeaseCoordinator(stop_service=lambda profile: calls.append(profile) or True)
    assert lease.release_asr_for_qwen("nano") is True
    assert calls == ["nano"]

def test_daemon_skips_qwen_and_commits_raw_when_release_is_unconfirmed() -> None:
    corrector = FakeCorrector("git commit")
    harness = Harness(corrector=corrector, xpu_lease=RejectingLease())
    harness.daemon.start_if_idle(); harness.daemon.stop()
    assert corrector.calls == []
    assert harness.clipboard.writes == ["get commit"]
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_xpu_lease.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q`

Expected: FAIL because daemon invokes Qwen while ASR remains warm.

- [ ] **Step 3: Implement**

Have `SenseVoiceRuntime` return `Transcription(engine="sensevoice")`; Nano defaults to `"nano"`. Implement `default_stop_worker_service(profile) -> bool`: stop exactly `fun-voice-worker@<profile>.service`, then poll its `ActiveState` until only `inactive` or `failed` is observed, with a 30-second limit. The lease uses this callback. Right before `corrector.correct`, daemon acquires the lease for `transcription.engine`; if false it records `correction="skipped_lease"` and commits raw text. It must never start Qwen after failed confirmation or invoke SenseVoice as a correction fallback.

- [ ] **Step 4: Verify full suite and actual lifecycle**

Run: `PYTHONPATH=src .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy src && bash -n scripts/install-user.sh scripts/uninstall-user.sh && git diff --check`

Expected: PASS.

For one real capture, inspect only non-sensitive lifecycle evidence:

```bash
journalctl --user -u fun-voice-daemon.service -b --no-pager -n 80
systemctl --user is-active 'fun-voice-worker@nano.service'
pgrep -af 'VLLM::EngineCore|fun_voice.corrector' || true
```

Expected: Qwen follows inactive Nano/SenseVoice, then exits.

- [ ] **Step 5: Commit**

```bash
git add src/fun_voice/xpu_lease.py src/fun_voice/contracts.py src/fun_voice/nano_runtime.py src/fun_voice/daemon.py tests/test_xpu_lease.py tests/test_daemon.py tests/test_end_to_end_fakes.py
git commit -m "feat: serialize ASR and Qwen XPU residency"
```

### Task 6: Publish benchmark and lifecycle operating guidance

**Files:**

- Modify: `README.md`
- Modify: `docs/operations.md`
- Modify: `docs/acceptance-checklist.md`
- Modify: `docs/xpu-poc.md`
- Modify: `tests/test_install_scripts.py`

- [ ] **Step 1: Write a failing documentation contract test**

```python
def test_operations_document_benchmark_preload_and_serial_qwen() -> None:
    text = (ROOT / "docs/operations.md").read_text(encoding="utf-8")
    assert "fun-voice-benchmark" in text
    assert "预加载" in text
    assert "停止 Nano" in text
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_install_scripts.py -q`

Expected: FAIL until the new operating flow is documented.

- [ ] **Step 3: Implement and verify**

Document manifest ownership, aggregate-only output, baseline collection order, P1 first-use behavior, ASR/Qwen swap and raw fallback. Correct stale POC text from `.35/4096` to active Nano `.15/1536`.

Run: `PYTHONPATH=src .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy src && git diff --check`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/operations.md docs/acceptance-checklist.md docs/xpu-poc.md tests/test_install_scripts.py
git commit -m "docs: explain measured on-demand tuning"
```
