# Measured Nano Warmup and Correction Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute local voice-pipeline latency safely, shift Nano's first inference compilation into the recording window, and expose fixed Qwen rejection categories without changing desktop fallback behavior.

**Architecture:** Worker responses carry only bounded stage durations, never input or output payloads beyond the existing transcription result. The daemon aggregates those durations in its in-memory metrics ledger and separately measures the ASR-worker shutdown required before Qwen. Nano preload loads the runtime and runs a one-second synthetic, in-memory XPU inference under the existing engine lock. The isolated Qwen child returns bounded timing and an allow-listed reason enum; parent/daemon retain the external error code and commit raw ASR text on any failure.

**Tech Stack:** Python 3.11, FunASR Nano/SenseVoice, PyTorch XPU, Transformers Qwen3.5-0.8B, systemd user workers, pytest, ruff, mypy.

## Global Constraints

- Use only `xpu:0`; Nano remains primary, SenseVoiceSmall only handles Nano `model_load`/`oom`, and Qwen is exactly `Qwen/Qwen3.5-0.8B`.
- Do not record, return from metrics, log, or persist audio, transcription/correction text, paths, focus data, Fcitx tokens, model exception details, or raw child stderr.
- Keep models unloaded at daemon boot, preserve the one-model XPU lease, and add neither CPU/CUDA fallback nor an alternate correction model.
- Metrics remain an owner-only aggregate of at most 128 in-memory rows; all stage values are non-negative integer milliseconds and all categorical values are fixed allow-lists.
- Any preload, warmup, timing-decoding, correction, or diagnostics failure must still retain the existing raw-ASR desktop commit fallback.

---

### Task 1: Add private timing contracts and Nano worker warmup

**Files:**
- Modify: `src/fun_voice/contracts.py:119-145`
- Modify: `src/fun_voice/nano_runtime.py:266-411,561-604`
- Modify: `src/fun_voice/worker.py:66-193,227-262`
- Test: `tests/test_worker.py:39-272`
- Test: `tests/test_worker_protocol.py:107-159,235-290`

**Interfaces:**
- Produces `AsrStageTiming(audio_load_ms, vad_ms, generate_ms)` and `PreloadTiming(worker_elapsed_ms, runtime_load_ms, warmup_ms, warmup_status)` immutable contracts with optional non-negative stages.
- Produces `NanoRuntime.warmup() -> int`, which generates from a fixed 16 kHz, one-second float32 zero array and never calls VAD.
- Produces worker `preload` frames with `elapsed_ms`, `runtime_load_ms`, `warmup_ms`, and `warmup_status`; successful transcription frames carry `timing_ms`.
- Consumes existing `NanoRuntime._run_asr()` serialization lock, so real user audio and warmup can never call `engine.generate` concurrently.

- [ ] **Step 1: Write the failing contract/warmup tests**

```python
def test_nano_warmup_generates_synthetic_audio_without_vad() -> None:
    engine = FakeEngine(results=[{"text": ""}])
    vad = FakeVad(regions=[(0, 100)])
    runtime = NanoRuntime(engine=engine, vad=vad)  # type: ignore[arg-type]

    elapsed_ms = runtime.warmup()

    assert elapsed_ms >= 0
    assert len(engine.calls) == 1
    assert engine.calls[0][0].shape == (16_000,)
    assert vad.calls == []


def test_preload_response_exposes_only_duration_stages() -> None:
    response = Worker(LazyTranscriber(lambda: WarmableRuntime(), device="xpu:0")).handle(
        {"id": "p", "op": "preload"}
    )

    assert response["status"] == "ok"
    assert response["warmup_status"] == "ready"
    assert isinstance(response["elapsed_ms"], int)
    assert "audio" not in repr(response)
    assert "你好" not in repr(response)
```

- [ ] **Step 2: Run the focused tests and verify they fail for the missing interfaces**

Run: `pytest tests/test_worker.py::test_nano_warmup_generates_synthetic_audio_without_vad tests/test_worker_protocol.py::test_preload_response_exposes_only_duration_stages -v`

Expected: FAIL because `NanoRuntime` has no `warmup` and the preload response has no stage fields.

- [ ] **Step 3: Implement the smallest wire-safe contracts and runtime measurements**

```python
@dataclass(frozen=True)
class AsrStageTiming:
    audio_load_ms: int | None = None
    vad_ms: int | None = None
    generate_ms: int | None = None


@dataclass(frozen=True)
class PreloadTiming:
    worker_elapsed_ms: int | None = None
    runtime_load_ms: int | None = None
    warmup_ms: int | None = None
    warmup_status: Literal["not_requested", "ready", "failed"] = "not_requested"
```

Measure audio load around `_load_audio_samples`, VAD around `vad.detect`, and model generation around `_run_asr`; attach only `AsrStageTiming` and `worker_elapsed_ms` to `Transcription`. Add `NanoRuntime.warmup()` that calls `_run_asr([np.zeros(16_000, dtype=np.float32)], self.default_timeout)` and discards all output. Let `LazyTranscriber.preload()` measure first runtime construction, invoke an optional `warmup()` exactly once after it succeeds, retain a usable runtime when warmup fails, and return `PreloadTiming`. Serialize these scalar fields in worker responses and validate/parse their types in `SocketWorkerClient`.

- [ ] **Step 4: Run focused worker and protocol suites**

Run: `pytest tests/test_worker.py tests/test_worker_protocol.py -q`

Expected: PASS; warmup uses no VAD and existing lazy preload/transcribe reuse behavior remains intact.

- [ ] **Step 5: Commit the independently testable worker change**

```bash
git add src/fun_voice/contracts.py src/fun_voice/nano_runtime.py src/fun_voice/worker.py tests/test_worker.py tests/test_worker_protocol.py
git commit -m "feat: warm Nano and expose private worker timing"
```

### Task 2: Aggregate diagnostic stages in the daemon without payload retention

**Files:**
- Modify: `src/fun_voice/metrics.py:15-173`
- Modify: `src/fun_voice/daemon.py:203-220,361-498,638-831`
- Modify: `src/fun_voice/xpu_lease.py:1-32`
- Test: `tests/test_metrics.py:1-45`
- Test: `tests/test_daemon.py:430-470` and metric-focused cases
- Test: `tests/test_end_to_end_fakes.py` socket worker frame cases

**Interfaces:**
- Consumes `Transcription.timing`, `Transcription.worker_elapsed_ms`, and `PreloadTiming` from Task 1.
- Produces metric timings `preload_worker_ms`, `preload_runtime_load_ms`, `preload_warmup_ms`, `asr_worker_ms`, `asr_queue_transport_ms`, `asr_audio_load_ms`, `asr_vad_ms`, `asr_generate_ms`, and `asr_release_ms`.
- Produces fixed `nano_warmup` counts (`not_requested`, `ready`, `failed`) alongside existing `nano_preload` state.

- [ ] **Step 1: Write the failing aggregation tests**

```python
def test_metrics_aggregate_stage_durations_but_reject_payload_fields() -> None:
    ledger = MetricsLedger()
    row = ledger.begin()
    ledger.record(row, asr_worker_ms=12, asr_generate_ms=7, nano_warmup="ready")

    assert ledger.summary()["asr_worker_ms"] == {"p50": 12, "p95": 12}
    assert ledger.summary()["nano_warmup"] == {"ready": 1}
    with pytest.raises(ValueError, match="unsupported"):
        ledger.record(row, worker_response={"text": "secret"})


def test_daemon_records_worker_stages_and_transport_gap() -> None:
    daemon = make_daemon(
        worker=FakeWorker(Transcription(
            text="你好", worker_elapsed_ms=20,
            timing=AsrStageTiming(audio_load_ms=3, vad_ms=5, generate_ms=9),
        )),
    )
    # complete a fake hold-to-talk session
    assert daemon.metrics_summary()["asr_audio_load_ms"]["p50"] == 3
```

- [ ] **Step 2: Run those tests and verify they fail because the fields are unknown/unrecorded**

Run: `pytest tests/test_metrics.py tests/test_daemon.py -q`

Expected: FAIL with unsupported stage fields or missing summary keys.

- [ ] **Step 3: Implement validated aggregation and daemon attribution**

Add the eleven stage names to `_TIMING_FIELDS`, `nano_warmup` to the enumerated fields, and update `SessionMetric`, `summary`, and `_validate_updates` consistently. In `VoiceDaemon._preload_nano`, consume a `PreloadTiming` result and record total plus available component fields. In `_transcribe_and_commit`, record worker stages, and calculate `asr_queue_transport_ms = max(0, daemon_asr_wall_ms - worker_elapsed_ms)`. Add `XpuLeaseCoordinator.last_release_ms`, measured with `perf_counter` in a `finally`, and record it before Qwen starts. Treat missing/malformed peer timing as absent, never as an exception; no response dictionary may be stored in metrics.

- [ ] **Step 4: Run daemon/metrics/fake integration tests**

Run: `pytest tests/test_metrics.py tests/test_daemon.py tests/test_end_to_end_fakes.py -q`

Expected: PASS; summary exposes only P50/P95 and allow-listed histograms, while raw text/audio/path representations are absent.

- [ ] **Step 5: Commit the daemon diagnostic change**

```bash
git add src/fun_voice/metrics.py src/fun_voice/daemon.py src/fun_voice/xpu_lease.py tests/test_metrics.py tests/test_daemon.py tests/test_end_to_end_fakes.py
git commit -m "feat: attribute voice pipeline latency stages"
```

### Task 3: Classify Qwen rejection safely and expose correction stage timings

**Files:**
- Modify: `src/fun_voice/contracts.py:119-145`
- Modify: `src/fun_voice/corrector.py:55-385`
- Modify: `src/fun_voice/daemon.py:746-782`
- Modify: `src/fun_voice/metrics.py:15-173`
- Test: `tests/test_corrector.py:1-115`
- Test: `tests/test_daemon.py` correction fallback cases

**Interfaces:**
- Produces `CorrectionTiming(model_load_ms, generate_ms, validate_ms)` and `CorrectionError(code, reason, timing)`.
- Produces allow-listed `correction_rejection` values: `envelope_missing`, `envelope_malformed`, `output_empty`, `output_too_long`, `similarity`, `protected_token`, `input_too_large`, `model_load`, `oom`, `device`, `protocol`, `no_output`, `generation`, `timeout`, `unavailable`, and `internal`.
- Retains all existing external `correction.*` error codes and `OnDemandQwenCorrector.correct(raw_text) -> str` API.

- [ ] **Step 1: Write failing parser/validator/IPC tests**

```python
def test_correction_rejection_reason_preserves_generic_error_code() -> None:
    with pytest.raises(CorrectionError) as caught:
        parse_correction_output("git commit")

    assert caught.value.code == "correction.invalid_output"
    assert caught.value.reason == "envelope_missing"


def test_qwen_parent_preserves_child_stage_timing_on_raw_fallback() -> None:
    corrector = OnDemandQwenCorrector(
        command=("qwen",),
        runner=lambda *_: json.dumps({
            "status": "error", "error_code": "correction.invalid_output",
            "error_reason": "similarity",
            "timing_ms": {"model_load_ms": 4, "generate_ms": 8, "validate_ms": 1},
        }),
    )
    with pytest.raises(CorrectionError) as caught:
        corrector.correct("get commit")
    assert caught.value.reason == "similarity"
    assert caught.value.timing.generate_ms == 8
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_corrector.py -q`

Expected: FAIL because `CorrectionError` has no `reason`/`timing` and the child frame has no timing contract.

- [ ] **Step 3: Implement bounded timing/rejection propagation**

Define the timing dataclass in `contracts.py`, and make `CorrectionError` carry only the stable code, an allow-listed reason, and optional scalar timing. Time Qwen model loading, generation/decode, and envelope/semantic validation in the child. Map parsing and validation branches to fixed reasons while keeping `correction.invalid_output`; ensure model/runtime categories receive their fixed reason. Print only `{status, error_code, error_reason, timing_ms}` on errors and `{status, text, timing_ms}` on success. In the parent, strictly validate the returned scalar map and reason allow-list before storing `last_timing` or re-raising. In the daemon, record the three timing fields and a `correction_rejection` histogram on any correction fallback; no failure changes raw-text commit behavior.

- [ ] **Step 4: Run focused correction and daemon tests**

Run: `pytest tests/test_corrector.py tests/test_daemon.py -q`

Expected: PASS; envelope, similarity, and protected-token failures produce distinct fixed reasons yet the fake desktop commits raw ASR text.

- [ ] **Step 5: Commit the correction diagnostics change**

```bash
git add src/fun_voice/contracts.py src/fun_voice/corrector.py src/fun_voice/daemon.py src/fun_voice/metrics.py tests/test_corrector.py tests/test_daemon.py
git commit -m "feat: classify Qwen correction rejections"
```

### Task 4: Document operation and perform full verification

**Files:**
- Modify: `docs/operations.md` (or the existing operational reference that documents `metrics`)
- Modify: `README.md` only if it links to the metrics endpoint
- Test: full `tests/` suite and static checks

**Interfaces:**
- Documents that metrics are aggregate-only, which stage keys may be returned, and how to interpret a `nano_warmup` or `correction_rejection` count without exposing session content.

- [ ] **Step 1: Add an operation test/checklist that prevents sensitive metric examples**

```text
Metrics documentation names aggregate keys only. It contains no real audio path,
transcription, corrected sentence, Fcitx token, or model stderr excerpt.
```

- [ ] **Step 2: Update the metrics operational reference**

Document `preload_runtime_load_ms` versus `preload_warmup_ms`, `asr_queue_transport_ms`, `asr_generate_ms`, `asr_release_ms`, the three Qwen stage fields, and that P50/P95/histograms are memory-only aggregates. State that a failed warmup never blocks a real recording and a `correction_rejection` result retains raw ASR text.

- [ ] **Step 3: Run the complete verification set**

Run: `pytest -q`

Expected: PASS.

Run: `ruff check src tests`

Expected: `All checks passed!`

Run: `mypy src`

Expected: `Success: no issues found`.

- [ ] **Step 4: Validate the installed editable service uses this workspace**

Run: `systemctl --user restart fun-voice-daemon.service && systemctl --user is-active fun-voice-daemon.service`

Expected: `active` after startup completes; no model worker is active until a hold-to-talk session.

- [ ] **Step 5: Commit operational documentation**

```bash
git add docs README.md
git commit -m "docs: explain warmup and latency metrics"
```

## Plan Self-Review

- Spec coverage: Task 1 implements synthetic warmup and the private worker protocol; Task 2 implements all ASR/preload/lease measurements and bounded aggregation; Task 3 implements all Qwen timings and rejection enums; Task 4 documents and verifies the full path.
- Placeholder scan: no unfinished implementation markers are used; every task lists concrete files, APIs, tests, commands, and expected outcomes.
- Type consistency: `AsrStageTiming` and `PreloadTiming` originate in `contracts.py`, flow worker to `SocketWorkerClient` to `VoiceDaemon`; `CorrectionTiming` and `CorrectionError` originate in the corrector contract and flow to the daemon/metrics ledger.
