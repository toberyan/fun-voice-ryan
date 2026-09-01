# Structured Voice Result, Identity, and Qwen3.5 Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the local X11 voice assistant with XPU-only token timestamps, diarization and registered local speaker identity, a memory-only structured-result API, and validated Qwen3.5-0.8B correction for Fcitx and clipboard output.

**Architecture:** Keep `VoiceDaemon` responsible for recording, focus-safe desktop output and in-memory result retention. Extend the warm ASR Worker with XPU VAD/Nano/alignment/diarization; run Qwen3.5 in an isolated user service behind a private socket. Raw acoustic facts remain immutable; the corrector can supply only validated `corrected_text` candidates, while the daemon publishes the complete result to `results.sock` and injects only `final_text`.

**Tech Stack:** Python 3.12, PyTorch XPU, vLLM XPU, FunASR Nano/CAM++, ModelScope, `cryptography` AES-GCM, Secret Service via `secretstorage`, Unix-domain sockets, systemd user services, pytest/ruff/mypy.

## Global Constraints

- Use only `xpu:0` for all VAD, Nano, CTC alignment, CAM++, diarization/matching and Qwen3.5 model work. A device mismatch is an error, never a CPU fallback.
- Keep existing X11 `Super+C`, focus guard, PipeWire cleanup, Fcitx primary injection and XTEST fallback intact.
- Persist neither recordings, raw/corrected text, result JSON nor speaker vectors. The sole durable biometric data is an AES-256-GCM encrypted profile record in a `0700` directory and `0600` database.
- `final_text` only goes to Fcitx and CLIPBOARD. Raw text and structure are retrievable only from same-uid local sockets, retained for 10 minutes and at most 8 results.
- Do not invoke FunASR's CPU `forced_align()` or its NumPy/SciPy/scikit-learn CAM++ clustering path at runtime.
- Qwen correction must preserve the requested unit ids/order/count and be accepted only by deterministic validation. Bad output, timeout and OOM use raw text with a safe status; they never invoke CPU correction.
- Runtime network access is forbidden after installation/model download. Pin all model revisions and XPU package versions in the POC report.
- Keep logs, notifications, health reports and test assertions free of audio, raw text, corrected text, terms, profile names, vectors and result ids.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/fun_voice/contracts.py` | Immutable structures and codecs shared by daemon, worker and result APIs. |
| `src/fun_voice/config.py` | Enhanced config, model/runtime paths and XPU-only validation. |
| `src/fun_voice/results.py` | Bounded in-memory `ResultBroker`, same-uid `results.sock` server and CLI client. |
| `src/fun_voice/correction.py` | Sentinel request builder, parser, deterministic validator and Corrector socket client. |
| `src/fun_voice/corrector.py` | Warm Qwen3.5-0.8B vLLM XPU service and its private socket protocol. |
| `src/fun_voice/alignment.py` | XPU-only CTC forced alignment and token-time projection. |
| `src/fun_voice/diarization.py` | XPU CAM++ feature adapter, in-recording clustering and profile matching. |
| `src/fun_voice/identity.py` | Secret Service backed AES-GCM profile store and `identity.sock` controller. |
| `src/fun_voice/nano_runtime.py` | Preserve Nano intermediate outputs and assemble immutable raw structured results. |
| `src/fun_voice/worker.py` | Enhanced worker operations: transcribe and enrollment sample extraction. |
| `src/fun_voice/daemon.py` | Corrector call, result publication, final-text injection and enrollment state. |
| `src/fun_voice/preflight.py` | Enhanced XPU hard gates and version evidence. |
| `src/fun_voice/selftest.py` | Live socket/service checks for the enhanced deployment. |
| `systemd/fun-voice-corrector.service` | Isolated warm Qwen3.5 user service. |
| `scripts/run-enhanced-xpu-poc.sh` | Repeatable model/device POC that writes the enhanced report. |
| `scripts/install-user.sh`, `scripts/uninstall-user.sh` | Install/remove service and model cache without touching profile data unless purge is explicitly requested. |
| `tests/test_*.py` | Pure unit, protocol and fake end-to-end coverage mapped below. |

## Task 1: Establish the Enhanced XPU Contract and POC Gate

**Files:**

- Modify: `pyproject.toml`
- Modify: `scripts/create-xpu-env.sh`
- Create: `scripts/run-enhanced-xpu-poc.sh`
- Modify: `src/fun_voice/config.py`
- Modify: `tests/test_config.py`
- Create: `tests/test_enhanced_poc_script.py`

**Interfaces:**

- Produces `EnhancedInferenceConfig` and `validate_enhanced_inference_config(config) -> EnhancedInferenceConfig`.
- Produces `scripts/run-enhanced-xpu-poc.sh`, which must write `${XDG_RUNTIME_DIR}/fun-voice-ryan/enhanced-poc-report.json` only when all gates pass.

- [ ] **Step 1: Write failing config and POC-script tests**

```python
def test_enhanced_inference_rejects_non_xpu() -> None:
    with pytest.raises(ConfigError, match="correction.device must be 'xpu:0'"):
        validate_enhanced_inference_config(
            EnhancedInferenceConfig(correction_device="cpu")
        )

def test_enhanced_poc_script_uses_qwen35_text_only_and_no_cpu_fallback() -> None:
    text = (ROOT / "scripts/run-enhanced-xpu-poc.sh").read_text()
    assert "Qwen/Qwen3.5-0.8B" in text
    assert "language_model_only" in text
    assert 'device="cpu"' not in text
    assert "fallback_to_cpu" not in text
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_config.py tests/test_enhanced_poc_script.py -q`

Expected: FAIL because the enhanced config, test script and validation do not exist.

- [ ] **Step 3: Add locked enhanced configuration and runtime dependencies**

Add direct dependencies `cryptography` and `secretstorage` to `pyproject.toml`; do not put the existing XPU stack into normal `uv sync` resolution. Add these exact typed settings:

```python
@dataclass(frozen=True)
class EnhancedInferenceConfig:
    enabled: bool = False
    result_ttl_seconds: int = 600
    result_max_entries: int = 8
    correction_model: str = "Qwen/Qwen3.5-0.8B"
    correction_device: str = XPU_DEVICE
    correction_dtype: str = "bf16"
    correction_max_model_len: int = 4096
    correction_enable_thinking: bool = False
    identity_enabled: bool = False
    identity_device: str = XPU_DEVICE

def validate_enhanced_inference_config(value: EnhancedInferenceConfig) -> EnhancedInferenceConfig:
    if value.correction_device != XPU_DEVICE:
        raise ConfigError("correction.device must be 'xpu:0'")
    if value.identity_device != XPU_DEVICE:
        raise ConfigError("speaker_identity.device must be 'xpu:0'")
    if value.correction_model != "Qwen/Qwen3.5-0.8B":
        raise ConfigError("correction.model must be 'Qwen/Qwen3.5-0.8B'")
    if not 1 <= value.result_max_entries <= 8 or value.result_ttl_seconds != 600:
        raise ConfigError("enhanced result retention is fixed at 8 entries / 600 seconds")
    return value
```

Make `create-xpu-env.sh` install the pinned XPU extra requirements before installing `cryptography`/`secretstorage`, and make the POC script print only model revision, package versions, device and memory metrics. The POC must explicitly check that Qwen3.5 is loaded from a local ModelScope snapshot in text-only/non-thinking mode, every parameter/decoder reports `xpu`, and no fallback report is accepted.

- [ ] **Step 4: Run the focused tests and static checks**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_config.py tests/test_enhanced_poc_script.py -q && .venv/bin/ruff check src/fun_voice/config.py tests/test_config.py tests/test_enhanced_poc_script.py && .venv/bin/mypy src/fun_voice/config.py`

Expected: PASS.

- [ ] **Step 5: Run the real hardware POC before enabling any service**

Run: `scripts/run-enhanced-xpu-poc.sh`

Expected: exit `0`, a report with `ready: true`, Qwen3.5/Nano/CAM++/CTC device evidence of `xpu`, and no transcription or identity payloads. If it fails, stop implementation on hardware integration; fix the specific POC failure without adding CPU fallback.

- [ ] **Step 6: Commit the isolated gate**

```bash
git add pyproject.toml scripts/create-xpu-env.sh scripts/run-enhanced-xpu-poc.sh \
  src/fun_voice/config.py tests/test_config.py tests/test_enhanced_poc_script.py
git commit -m "feat: add enhanced XPU preflight configuration"
```

## Task 2: Define Immutable Enhanced Result Contracts

**Files:**

- Modify: `src/fun_voice/contracts.py`
- Modify: `tests/test_contracts.py`

**Interfaces:**

- Produces `TokenTiming`, `Speaker`, `TranscriptionUnit`, `CorrectionEdit`, `CorrectionInfo`, `RawStructuredResult` and `StructuredResult`.
- `RawStructuredResult.with_final(final_text, correction) -> StructuredResult` is the sole constructor for a final result.
- Produces `ProfileVector`, `MatchCalibration` and `EnrollmentSample` for the worker/identity boundary; their `to_wire()` values are permitted only on the existing same-uid Worker socket and must never be written to a result.

- [ ] **Step 1: Write failing contract tests**

```python
def test_final_result_keeps_raw_acoustic_facts_immutable() -> None:
    raw = RawStructuredResult(
        raw_text="get commit",
        units=(TranscriptionUnit("u1", 0, 100, "speaker_0", "get commit", ()),),
        speakers=(Speaker("speaker_0", None, None, "unknown"),),
        duration_ms=100,
    )
    result = raw.with_final("git commit", CorrectionInfo.accepted("Qwen/Qwen3.5-0.8B", ()))
    assert result.raw_text == "get commit"
    assert result.units[0].raw_text == "get commit"
    assert result.final_text == "git commit"

def test_token_time_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="start_ms"):
        TokenTiming(text="今", start_ms=2, end_ms=1, confidence=0.9)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_contracts.py -q`

Expected: FAIL because the enhanced types and finalization method do not exist.

- [ ] **Step 3: Implement frozen, validated structures and explicit JSON codecs**

Use `@dataclass(frozen=True, slots=True)`. Give every public object `to_wire()` and `from_wire()` methods; reject unknown enum status, duplicate unit ids, non-monotonic unit order, token ranges outside their unit, and a corrected unit count/order that differs from raw. Keep `Transcription` backwards-compatible by adding `structured: RawStructuredResult | None = None`, not by replacing the existing `text`/`segments` fields.

```python
@dataclass(frozen=True, slots=True)
class TranscriptionUnit:
    id: str
    start_ms: int
    end_ms: int
    speaker: str
    raw_text: str
    tokens: tuple[TokenTiming, ...]
    corrected_text: str | None = None

@dataclass(frozen=True, slots=True)
class RawStructuredResult:
    raw_text: str
    units: tuple[TranscriptionUnit, ...]
    speakers: tuple[Speaker, ...]
    duration_ms: int
    timing_status: Literal["available", "approximate", "unavailable"]
    identity_status: Literal["available", "locked", "unavailable"]
    def with_final(self, final_text: str, correction: CorrectionInfo) -> StructuredResult: ...

@dataclass(frozen=True, slots=True)
class CorrectionInfo:
    status: Literal["accepted", "rejected", "unavailable"]
    model: str | None
    reason: str | None
    corrected_units: tuple[str, ...] = ()
    def final_text_or(self, fallback: str) -> str: ...

@dataclass(frozen=True, slots=True)
class StructuredResult:
    raw: RawStructuredResult
    final_text: str
    correction: CorrectionInfo
    result_id: str | None = None
```

- [ ] **Step 4: Run all contract tests and type checks**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_contracts.py -q && .venv/bin/ruff check src/fun_voice/contracts.py tests/test_contracts.py && .venv/bin/mypy src/fun_voice/contracts.py`

Expected: PASS.

- [ ] **Step 5: Commit the result boundary**

```bash
git add src/fun_voice/contracts.py tests/test_contracts.py
git commit -m "feat: define structured transcription contracts"
```

## Task 3: Implement Memory-Only Results API and CLI

**Files:**

- Create: `src/fun_voice/results.py`
- Modify: `src/fun_voice/config.py`
- Modify: `src/fun_voice/contracts.py`
- Modify: `pyproject.toml`
- Create: `tests/test_results.py`
- Modify: `tests/test_config.py`

**Interfaces:**

- Produces `ResultBroker(max_entries: int = 8, ttl_seconds: int = 600)`.
- Produces `ResultSocketServer(paths: RuntimePaths, broker: ResultBroker)` and `ResultSocketClient(path: Path)`.
- Adds `results_socket: Path` to `RuntimePaths` and `fun-voice-result = "fun_voice.results:main"` to console scripts.

- [ ] **Step 1: Write failing broker/socket tests**

```python
def test_broker_expires_and_evicts_without_persisting(monkeypatch) -> None:
    clock = FakeClock()
    broker = ResultBroker(clock=clock, max_entries=2, ttl_seconds=600)
    first = broker.publish(_result("1"))
    broker.publish(_result("2")); broker.publish(_result("3"))
    assert broker.get(first.result_id) is None
    clock.advance(601)
    assert broker.latest() is None

def test_result_socket_rejects_other_uid() -> None:
    server = ResultSocketServer(_paths(), ResultBroker(), peer_uid=lambda _sock: 999)
    assert server.authorize(_FakeSocket()) is False
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_results.py -q`

Expected: FAIL because `results.py`, the path and CLI are absent.

- [ ] **Step 3: Implement bounded store, UID authorization and JSON-line operations**

Use `OrderedDict[str, tuple[float, StructuredResult]]` guarded by `threading.Lock`. `publish()` must assign `secrets.token_hex(16)` through `dataclasses.replace(result, result_id=...)` and return that stored value; callers never create or log a result id. Purge expired entries before every `publish`, `latest`, `get` and `list_metadata`. Create a `0600` Unix socket only after `resolve_runtime_dir()` has validated ownership. Support exactly `result.latest`, `result.get` and `result.list`; use the existing `encode_message`/`decode_message` with `WORKER_RESPONSE_MAX_BYTES`. `result.list` returns id/status/creation metadata only.

```python
class ResultBroker:
    def publish(self, result: StructuredResult) -> StructuredResult: ...
    def latest(self) -> StructuredResult | None: ...
    def get(self, result_id: str) -> StructuredResult | None: ...
    def list_metadata(self) -> tuple[ResultMetadata, ...]: ...
```

Ensure CLI stdout is the requested JSON only; errors go to stderr and return non-zero. Do not add a file cache, a network listener or a subscribe operation.

- [ ] **Step 4: Run focused tests and the existing protocol suite**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_results.py tests/test_contracts.py tests/test_worker_protocol.py -q && .venv/bin/ruff check src/fun_voice/results.py tests/test_results.py && .venv/bin/mypy src/fun_voice/results.py`

Expected: PASS.

- [ ] **Step 5: Commit the API surface**

```bash
git add pyproject.toml src/fun_voice/config.py src/fun_voice/contracts.py \
  src/fun_voice/results.py tests/test_results.py tests/test_config.py
git commit -m "feat: add ephemeral structured result socket"
```

## Task 4: Build and Verify the Pure Correction Protocol

**Files:**

- Create: `src/fun_voice/correction.py`
- Create: `tests/test_correction.py`

**Interfaces:**

- Produces `build_correction_prompt(raw: RawStructuredResult, terms: tuple[str, ...]) -> str`.
- Produces `parse_correction_envelope(text: str, unit_ids: tuple[str, ...]) -> tuple[str, ...]`.
- Produces `validate_correction(raw, candidate_units, terms, max_edit_ratio) -> CorrectionInfo`.

- [ ] **Step 1: Write failing parser/validator tests**

```python
def test_parser_requires_exact_unit_order_and_no_free_text() -> None:
    with pytest.raises(CorrectionProtocolError, match="unit order"):
        parse_correction_envelope("[[UNIT:u2]]b[[/UNIT]][[UNIT:u1]]a[[/UNIT]]", ("u1", "u2"))

def test_validator_accepts_known_term_but_rejects_semantic_rewrite() -> None:
    raw = _raw("get commit")
    assert validate_correction(raw, ("git commit",), ("git", "commit"), 0.35).status == "accepted"
    assert validate_correction(raw, ("明天取消发布",), ("git",), 0.35).status == "rejected"
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_correction.py -q`

Expected: FAIL because no correction module exists.

- [ ] **Step 3: Implement a deterministic envelope protocol**

Use only `[[UNIT:<id>]]...[[/UNIT]]` blocks. Reject nested markers, duplicate/missing ids, free text, empty content, unexpected ids, control characters and malformed UTF-8. Build the prompt with the exact correction rules: preserve id/order/count, no commentary, do not alter speaker/time/tokens, and only correct punctuation/case/spacing or terms present in the supplied list. Implement edit ratio with `difflib.SequenceMatcher`; enforce individual and total ratios supplied by configuration constants. Return a rejected `CorrectionInfo` with a stable reason rather than throwing for a valid-but-unacceptable candidate.

```python
def validate_correction(
    raw: RawStructuredResult,
    corrected_units: tuple[str, ...],
    allowed_terms: tuple[str, ...],
    max_edit_ratio: float,
) -> CorrectionInfo:
    if len(corrected_units) != len(raw.units):
        return CorrectionInfo.rejected("unit_count")
    # Verify ratios and protected terms before creating deterministic edits.
```

- [ ] **Step 4: Run correction tests and static checks**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_correction.py tests/test_contracts.py -q && .venv/bin/ruff check src/fun_voice/correction.py tests/test_correction.py && .venv/bin/mypy src/fun_voice/correction.py`

Expected: PASS.

- [ ] **Step 5: Commit the validator before adding a model**

```bash
git add src/fun_voice/correction.py tests/test_correction.py
git commit -m "feat: add validated correction envelope"
```

## Task 5: Implement the Isolated Qwen3.5 Corrector Service

**Files:**

- Create: `src/fun_voice/corrector.py`
- Modify: `src/fun_voice/correction.py`
- Modify: `src/fun_voice/config.py`
- Modify: `pyproject.toml`
- Create: `systemd/fun-voice-corrector.service`
- Create: `tests/test_corrector.py`
- Modify: `tests/test_config.py`

**Interfaces:**

- Produces console command `fun-voice-corrector`.
- Produces `SocketCorrectorClient.correct(raw: RawStructuredResult, terms: tuple[str, ...], timeout: float) -> CorrectionInfo`.
- Corrector protocol: `{"id":"...","op":"correct","units":[...],"terms":[...]}` → `{"id":"...","status":"ok","envelope":"..."}`.

- [ ] **Step 1: Write failing service/client tests with a fake engine**

```python
def test_corrector_rejects_cpu_model_before_serving() -> None:
    with pytest.raises(DeviceMismatchError, match="expected xpu"):
        assert_corrector_xpu(_Engine(device_type="cpu"))

def test_socket_client_maps_invalid_envelope_to_rejected_result(server) -> None:
    server.reply({"status": "ok", "envelope": "explanation"})
    assert SocketCorrectorClient(server.path).correct(_raw(), (), 1).status == "rejected"
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_corrector.py -q`

Expected: FAIL because the service and XPU guard do not exist.

- [ ] **Step 3: Implement warm text-only engine loading and private serving**

Resolve the ModelScope snapshot before creating vLLM. Construct the engine with the exact enhanced config: Qwen3.5 model, BF16, `max_model_len=4096`, text-only/language-model-only and non-thinking chat template. Use `SamplingParams(temperature=0.0, max_tokens=512)`; do not enable tools, visual input, MTP or network downloading during a request. Inspect the loaded decoder/model parameters and fail startup unless all report `xpu`.

```python
class CorrectorRuntime:
    def correct(self, prompt: str) -> str: ...

def load_corrector_runtime(config: EnhancedInferenceConfig) -> CorrectorRuntime:
    snapshot = qwen35_snapshot_path(config)
    engine = LLM(model=str(snapshot), dtype="bfloat16", max_model_len=config.correction_max_model_len,
                 language_model_only=True, enforce_eager=True)
    assert_corrector_xpu(engine)
    return CorrectorRuntime(engine)
```

The systemd unit must use `Restart=on-failure`, private runtime socket paths, no TCP port and no model text in journal output. The response builder returns only the envelope, never a model explanation. Client code always runs the pure parser and validator from Task 4.

- [ ] **Step 4: Run fake-service tests and a real single-prompt XPU smoke test**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_corrector.py tests/test_correction.py -q && .venv/bin/ruff check src/fun_voice/corrector.py src/fun_voice/correction.py tests/test_corrector.py && .venv/bin/mypy src/fun_voice/corrector.py src/fun_voice/correction.py`

Then run: `fun-voice-corrector --smoke-test`

Expected: fake tests PASS; smoke test returns only device/model/revision/latency metrics and exits `0` only with XPU evidence.

- [ ] **Step 5: Commit the isolated corrector**

```bash
git add pyproject.toml src/fun_voice/config.py src/fun_voice/correction.py \
  src/fun_voice/corrector.py systemd/fun-voice-corrector.service \
  tests/test_corrector.py tests/test_config.py
git commit -m "feat: add Qwen35 XPU correction service"
```

## Task 6: Add XPU-Only CTC Alignment

**Files:**

- Create: `src/fun_voice/alignment.py`
- Modify: `src/fun_voice/nano_runtime.py`
- Create: `tests/test_alignment.py`
- Modify: `tests/test_worker.py`

**Interfaces:**

- Produces `XpuCtcAligner.align(log_probs, target_ids, *, window_start_ms, vad_start_ms, vad_end_ms) -> tuple[TokenTiming, ...]`.
- Changes Nano's internal ASR result seam from `list[str]` to `list[AsrSegmentResult]`, retaining generated tokens, encoder outputs and actual window start.

- [ ] **Step 1: Write failing alignment tests**

```python
def test_alignment_rejects_cpu_logits() -> None:
    with pytest.raises(DeviceMismatchError, match="log_probs"):
        XpuCtcAligner().align(_cpu_logits(), _xpu_targets(), window_start_ms=0, vad_start_ms=0, vad_end_ms=100)

def test_alignment_offsets_and_clamps_token_times() -> None:
    tokens = _align_with_fake_xpu([[0.0, 0.1], [0.1, 0.3]], window_start_ms=900, vad_start_ms=1000, vad_end_ms=1150)
    assert [(t.start_ms, t.end_ms) for t in tokens] == [(1000, 1000), (1000, 1150)]
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_alignment.py -q`

Expected: FAIL because `XpuCtcAligner` and retained intermediate data are absent.

- [ ] **Step 3: Implement XPU dynamic-programming alignment**

Implement blank-expanded CTC targets and a Viterbi forward table with XPU torch operations. Verify `log_probs.device.type == target_ids.device.type == "xpu"` before allocating the table. Keep backpointers as XPU integer tensors, then transfer only final token indices/times/scores to host. Do not call `torchaudio.functional.forced_align` or `.cpu()`.

```python
class XpuCtcAligner:
    def align(self, log_probs: Any, target_ids: Any, *, window_start_ms: int,
              vad_start_ms: int, vad_end_ms: int) -> tuple[TokenTiming, ...]:
        _require_xpu(log_probs, "log_probs")
        _require_xpu(target_ids, "target_ids")
        score, backptr = _viterbi_ctc_xpu(log_probs, target_ids)
        return _project_token_times(score, backptr, window_start_ms, vad_start_ms, vad_end_ms)
```

Modify Nano runtime to retain only the required encoder output until alignment completes, then delete it in `finally`. On one-segment alignment failure, return that segment with `timing_status="unavailable"`; do not invent timings and do not fail text recognition.

- [ ] **Step 4: Run alignment, worker and regression tests**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_alignment.py tests/test_worker.py tests/test_preflight.py -q && .venv/bin/ruff check src/fun_voice/alignment.py src/fun_voice/nano_runtime.py tests/test_alignment.py && .venv/bin/mypy src/fun_voice/alignment.py src/fun_voice/nano_runtime.py`

Expected: PASS. Separately compare a fixed non-sensitive fixture with the CPU golden reference in the POC; runtime tests must assert no CPU call path.

- [ ] **Step 5: Commit device-resident alignment**

```bash
git add src/fun_voice/alignment.py src/fun_voice/nano_runtime.py \
  tests/test_alignment.py tests/test_worker.py
git commit -m "feat: add XPU CTC token alignment"
```

## Task 7: Implement XPU Diarization and Conservative Profile Matching

**Files:**

- Create: `src/fun_voice/diarization.py`
- Modify: `src/fun_voice/nano_runtime.py`
- Create: `tests/test_diarization.py`

**Interfaces:**

- Produces `CamplusXpuEmbedder.embed(windows: list[numpy.ndarray]) -> XpuEmbeddings`.
- Produces `XpuDiarizer.assign(embeddings, windows) -> tuple[SpeakerSpan, ...]`.
- Produces `match_profiles_xpu(centers, profiles, calibration) -> tuple[Speaker, ...]`.

- [ ] **Step 1: Write failing XPU-only diarization tests**

```python
def test_embedder_rejects_cpu_model() -> None:
    with pytest.raises(DeviceMismatchError, match="CAM\+\+"):
        CamplusXpuEmbedder(_cpu_camplus()).embed([_speech_window()])

def test_matching_prefers_unknown_when_margin_is_small() -> None:
    profiles = _profiles([[1.0, 0.0], [0.99, 0.01]])
    match = match_profiles_xpu(_xpu([[1.0, 0.0]]), profiles, _calibration(threshold=.7, margin=.1))
    assert match[0].match == "unknown"
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_diarization.py -q`

Expected: FAIL because XPU CAM++ adapters and matcher are absent.

- [ ] **Step 3: Implement device-resident feature, cluster and match stages**

Download/load CAM++ explicitly on `xpu:0`; assert parameters, feature batch and returned embeddings all stay on XPU. Create 1.5 s / 0.75 s windows. Implement affinity with normalized matrix multiplication and use a torch-only clustering strategy. First attempt XPU eigengap/spectral clustering; when that unsupported operation fails in the POC, select the documented XPU-only threshold-agglomeration implementation, not a CPU fallback. Smooth adjacent spans and map their greatest temporal overlap to `TranscriptionUnit`.

```python
def match_profiles_xpu(centers: Any, profiles: tuple[ProfileVector, ...], calibration: MatchCalibration) -> tuple[Speaker, ...]:
    _require_xpu(centers, "speaker centers")
    reference = torch.stack([profile.vector for profile in profiles])
    _require_xpu(reference, "profile vectors")
    similarity = F.normalize(centers, dim=-1) @ F.normalize(reference, dim=-1).T
    return _accept_only_clear_matches(similarity, profiles, calibration)
```

Return `unknown_overlap` when overlapping windows cannot be confidently assigned. Never create an identity profile from a normal transcription.

- [ ] **Step 4: Run XPU fake tests and runtime regression tests**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_diarization.py tests/test_worker.py -q && .venv/bin/ruff check src/fun_voice/diarization.py tests/test_diarization.py && .venv/bin/mypy src/fun_voice/diarization.py`

Expected: PASS.

- [ ] **Step 5: Commit diarization boundary**

```bash
git add src/fun_voice/diarization.py src/fun_voice/nano_runtime.py tests/test_diarization.py
git commit -m "feat: add XPU diarization and profile matching"
```

## Task 8: Build the Encrypted Identity Store and Identity Socket

**Files:**

- Create: `src/fun_voice/identity.py`
- Modify: `src/fun_voice/config.py`
- Modify: `pyproject.toml`
- Create: `tests/test_identity.py`

**Interfaces:**

- Produces `SecretServiceKeyProvider.get_or_create() -> bytes`.
- Produces `EncryptedProfileStore.create(label, vectors, calibration) -> ProfileSummary` and list/rename/delete methods.
- `EncryptedProfileStore.load_match_vectors() -> tuple[ProfileVectorWire, ...]` is internal to the same-uid daemon/Worker path. It returns decrypted vectors only in memory; Worker converts them to `xpu:0` before `match_profiles_xpu()` and no identity socket operation returns them.
- Produces `IdentitySocketServer` and console command `fun-voice-identity`.

- [ ] **Step 1: Write failing encryption and authorization tests**

```python
def test_profile_ciphertext_cannot_be_opened_with_modified_aad(tmp_path) -> None:
    store = EncryptedProfileStore(tmp_path / "profiles.sqlite3", FakeKeyProvider(b"k" * 32))
    profile = store.create("Ryan", _vectors(), _calibration())
    store._connection.execute("UPDATE profiles SET profile_id = 'other' WHERE profile_id = ?", (profile.id,))
    with pytest.raises(ProfileIntegrityError):
        store.list()

def test_identity_server_requires_same_uid() -> None:
    assert IdentitySocketServer(_paths(), _store(), peer_uid=lambda _: os.getuid() + 1).authorize(_FakeSocket()) is False
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_identity.py -q`

Expected: FAIL because the identity store and socket do not exist.

- [ ] **Step 3: Implement Secret Service keying and AES-GCM records**

Create the data directory `0700`, database `0600` and table `profiles(profile_id TEXT PRIMARY KEY, ciphertext BLOB NOT NULL)`. Generate a 32-byte random master key only through the Secret Service provider. Encrypt one canonical JSON payload with `AESGCM.encrypt(nonce, payload, aad)` where AAD is `f"{uid}:{profile_id}:v1"`. Keep only encrypted fields, including label. Treat an unavailable or locked Secret Service as `IdentityLockedError`; do not write any key to disk.

```python
def _aad(uid: int, profile_id: str) -> bytes:
    return f"{uid}:{profile_id}:v1".encode("ascii")

def _encrypt(key: bytes, profile_id: str, payload: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, payload, _aad(os.getuid(), profile_id))
```

The identity socket supports `profile.list`, `profile.rename`, `profile.delete`, `enroll.begin`, `enroll.status` and `enroll.cancel`. `profile.list` returns id/label only, never a vector. Delete requires `{ "confirm": true }`; wrong uid, bad JSON, unavailable key and invalid state get stable error codes.

- [ ] **Step 4: Run identity tests, permission checks and static checks**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_identity.py tests/test_config.py -q && .venv/bin/ruff check src/fun_voice/identity.py tests/test_identity.py && .venv/bin/mypy src/fun_voice/identity.py`

Expected: PASS.

- [ ] **Step 5: Commit biometric persistence in isolation**

```bash
git add pyproject.toml src/fun_voice/config.py src/fun_voice/identity.py tests/test_identity.py
git commit -m "feat: add encrypted local speaker profiles"
```

## Task 9: Assemble Enhanced Worker Operations and Enrollment Samples

**Files:**

- Modify: `src/fun_voice/nano_runtime.py`
- Modify: `src/fun_voice/worker.py`
- Modify: `src/fun_voice/contracts.py`
- Modify: `tests/test_worker.py`
- Modify: `tests/test_worker_protocol.py`

**Interfaces:**

- `NanoRuntime.transcribe(...) -> Transcription` now sets `structured` whenever enhanced mode is enabled.
- Worker operation `"enroll_sample"` consumes `audio`/`sample_rate` and returns an in-memory encoded quality-approved speaker vector only to the daemon.
- Enhanced `"transcribe"` accepts an optional same-uid `profile_vectors` payload. It must decode each item with `ProfileVector.from_wire_xpu(..., device="xpu:0")`; vectors are discarded with other per-request tensors and never appear in `structured`.
- Worker operation `"health"` adds non-sensitive `enhanced_ready`, `alignment_ready`, `diarization_ready` and `device` fields.

- [ ] **Step 1: Write failing worker protocol tests**

```python
def test_transcribe_response_carries_raw_structure_but_not_final_text() -> None:
    response = _worker(_runtime_with_structure()).handle(_transcribe_request())
    assert response["structured"]["raw_text"] == "get commit"
    assert "final_text" not in response["structured"]

def test_enroll_sample_requires_valid_speech_and_never_logs_vector(caplog) -> None:
    response = _worker(_runtime()).handle({"id": "e1", "op": "enroll_sample", "audio": "/tmp/a.wav", "sample_rate": 16000})
    assert response["status"] == "ok"
    assert "embedding" not in caplog.text
```

- [ ] **Step 2: Run worker tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_worker.py tests/test_worker_protocol.py -q`

Expected: FAIL because enhanced worker fields and enrollment operation are absent.

- [ ] **Step 3: Wire runtime stages without altering raw facts**

Make runtime order explicit: `VAD → Nano → XpuCtcAligner → CamplusXpuEmbedder → XpuDiarizer → RawStructuredResult`. Preserve current direct concatenation as `raw_text`; generate stable `u1...` ids only after original time ordering. Decrypt the daemon-supplied match vectors only in memory and transfer them to `xpu:0` before the cosine-matching operation. `enroll_sample` calls the same CAM++ feature/quality path but does not run correction or normal text injection. Encode its vector only over the existing same-uid worker socket and delete temporary arrays after response construction.

```python
if op == "enroll_sample":
    sample = self.runtime.enrollment_sample(audio, sample_rate=sample_rate)
    return {"id": request_id, "status": "ok", "vector": sample.to_wire(), "quality": sample.quality}
```

Reject non-XPU enhanced runtime at worker startup. Update response size tests for the structured payload, not by weakening the 4 MiB cap.

- [ ] **Step 4: Run focused and full worker/protocol suites**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_worker.py tests/test_worker_protocol.py tests/test_contracts.py -q && .venv/bin/ruff check src/fun_voice/nano_runtime.py src/fun_voice/worker.py tests/test_worker.py tests/test_worker_protocol.py && .venv/bin/mypy src/fun_voice/nano_runtime.py src/fun_voice/worker.py`

Expected: PASS.

- [ ] **Step 5: Commit enhanced worker protocol**

```bash
git add src/fun_voice/contracts.py src/fun_voice/nano_runtime.py src/fun_voice/worker.py \
  tests/test_worker.py tests/test_worker_protocol.py
git commit -m "feat: return structured XPU transcription results"
```

## Task 10: Integrate Correction, Result Publication and Enrollment into the Daemon

**Files:**

- Modify: `src/fun_voice/daemon.py`
- Modify: `src/fun_voice/contracts.py`
- Modify: `src/fun_voice/results.py`
- Modify: `src/fun_voice/identity.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_end_to_end_fakes.py`

**Interfaces:**

- Adds `CorrectorClient` and `ResultPublisher` protocols to `daemon.py`.
- Adds daemon-owned `EnrollmentController` which consumes 3–5 `enroll_sample` replies and calls `EncryptedProfileStore.create` only after explicit completion.
- `VoiceDaemon._transcribe_and_commit()` publishes one `StructuredResult` before output and injects `final_text` only.
- Adds `_commit_final(text: str, session: CaptureSession) -> None`, which invokes the existing focus-safe Fcitx and CLIPBOARD mechanisms and accepts no structured or raw result object.

- [ ] **Step 1: Write failing daemon behavior tests**

```python
def test_daemon_commits_and_mirrors_corrected_final_text_only() -> None:
    daemon, fcitx, clipboard, broker = _daemon_with_raw("get commit", corrected="git commit")
    daemon.start_if_idle(); daemon.stop()
    assert fcitx.commits == ["git commit"]
    assert clipboard.values == ["git commit"]
    assert broker.latest().raw_text == "get commit"

def test_invalid_correction_uses_raw_text_and_publishes_rejection() -> None:
    daemon, fcitx, _, broker = _daemon_with_invalid_correction()
    daemon.start_if_idle(); daemon.stop()
    assert fcitx.commits == ["get commit"]
    assert broker.latest().correction.status == "rejected"

def test_enrollment_capture_never_commits_text() -> None:
    daemon, fcitx, _, _ = _daemon_in_enrollment_mode()
    daemon.start_if_idle(); daemon.stop()
    assert fcitx.commits == []
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_daemon.py tests/test_end_to_end_fakes.py -q`

Expected: FAIL because correction/result/enrollment seams are absent.

- [ ] **Step 3: Add pure seams and state transitions**

Instantiate `SocketCorrectorClient` and `ResultSocketServer` in daemon `main`; inject fakes in tests. After Worker returns raw structured output, call Corrector with a bounded timeout, build the final structure, publish it, then invoke current focus-safe commit and clipboard code with only `final_text`. When focus changed, publish normally and copy only `final_text`; do not commit.

```python
raw = transcription.structured
correction = self._corrector.correct(raw, self._terms(), timeout=CORRECTION_TIMEOUT_SECONDS)
result = raw.with_final(correction.final_text_or(raw.raw_text), correction)
self._results.publish(result)
self._commit_final(result.final_text, session)
```

Add enrollment as a separate controller state, not a new global hotkey state: the next 3–5 valid captures call Worker `enroll_sample`, notify progress without content, and call the encrypted store only after enough quality-approved samples. Cancel, focus change, capture error or worker error clears in-memory samples and performs no normal injection.

- [ ] **Step 4: Run daemon/end-to-end suites and privacy assertions**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_daemon.py tests/test_end_to_end_fakes.py tests/test_results.py tests/test_identity.py -q && .venv/bin/ruff check src/fun_voice/daemon.py tests/test_daemon.py tests/test_end_to_end_fakes.py && .venv/bin/mypy src/fun_voice/daemon.py`

Expected: PASS; include assertions that fake logs/notifications never receive raw/final text or profile labels.

- [ ] **Step 5: Commit desktop integration**

```bash
git add src/fun_voice/daemon.py src/fun_voice/contracts.py src/fun_voice/results.py \
  src/fun_voice/identity.py tests/test_daemon.py tests/test_end_to_end_fakes.py
git commit -m "feat: publish structured results and corrected input"
```

## Task 11: Extend Preflight, Self-Test, Installation and Documentation

**Files:**

- Modify: `src/fun_voice/preflight.py`
- Modify: `src/fun_voice/selftest.py`
- Modify: `scripts/install-user.sh`
- Modify: `scripts/uninstall-user.sh`
- Modify: `scripts/config.example.toml`
- Modify: `README.md`
- Modify: `docs/operations.md`
- Modify: `docs/acceptance-checklist.md`
- Modify: `tests/test_preflight.py`
- Modify: `tests/test_selftest.py`
- Modify: `tests/test_install_scripts.py`

**Interfaces:**

- `run_preflight()` emits the original nine checks plus named enhanced checks: `xpu_vad`, `xpu_ctc_alignment`, `camplus_xpu`, `xpu_diarization`, `qwen35_xpu`, `corrector_isolation`, `profile_store_security`.
- `fun-voice-selftest` requires the enhanced report, Worker health, Corrector health and `results.sock` only when `[enhanced].enabled=true`.

- [ ] **Step 1: Write failing preflight/installation tests**

```python
def test_enhanced_preflight_refuses_cpu_alignment() -> None:
    report = run_enhanced_preflight(_fake_components(alignment_device="cpu"))
    assert report.check("xpu_ctc_alignment").status == STATUS_FAIL
    assert report.ready is False

def test_installer_requires_ready_enhanced_report_when_enabled(tmp_path) -> None:
    result = run_install(tmp_path, config="[enhanced]\nenabled = true\n", enhanced_report={"ready": False})
    assert result.returncode != 0
    assert "enhanced XPU preflight" in result.stderr
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_preflight.py tests/test_selftest.py tests/test_install_scripts.py -q`

Expected: FAIL because the enhanced checks and service installation gate do not exist.

- [ ] **Step 3: Implement hard gates and safe user-service lifecycle**

Make checks inspect actual module/tensor/decoder devices, probe correction response syntax, compare timestamps against the golden fixture, and verify profile storage permission/key availability using a fake provider in normal unit tests. Store only check status and device/version/memory details in reports. Installer must download models only during explicit install/update, write the Corrector service, daemon socket paths and console scripts, then start Worker → Corrector → Daemon. Uninstaller stops/removes all services and runtime sockets; it retains encrypted profiles unless `--purge` is explicitly confirmed.

- [ ] **Step 4: Run all automated verification**

Run: `PYTHONPATH=src .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy src && cmake --build build/fcitx && ctest --test-dir build/fcitx --output-on-failure`

Expected: all Python tests pass, static checks pass, Fcitx build and CTest pass.

- [ ] **Step 5: Update operational instructions with exact user flows**

Document: model install/POC, `fun-voice-result latest`, identity registration/list/delete, Corrector health, private-data boundary, the raw fallback notification, profile purge confirmation and the manual X11 test. Do not put a real profile name, recording, vector or transcript in documentation.

- [ ] **Step 6: Commit operational completion**

```bash
git add src/fun_voice/preflight.py src/fun_voice/selftest.py scripts/install-user.sh \
  scripts/uninstall-user.sh scripts/config.example.toml README.md docs/operations.md \
  docs/acceptance-checklist.md tests/test_preflight.py tests/test_selftest.py \
  tests/test_install_scripts.py
git commit -m "feat: gate enhanced XPU voice services"
```

## Task 12: Perform a Real XPU and Privacy Acceptance Run

**Files:**

- Modify: `docs/acceptance-checklist.md`
- Create: `tests/manual/test-structured-identity-correction.md`

**Interfaces:**

- Produces a completed, local-only acceptance record with pass/fail evidence categories, never transcript or biometric payloads.

- [ ] **Step 1: Add a failing acceptance checklist item for every hard boundary**

```markdown
- [ ] `fun-voice-preflight --enhanced` reports every enhanced check as pass and each model device as `xpu`.
- [ ] `fun-voice-result latest` returns structure while `xclip -o` contains only `final_text`.
- [ ] An unregistered voice yields `unknown`; a deleted profile cannot be matched afterward.
- [ ] Killing Corrector results in raw-text fallback without taking down Nano Worker or injecting into a changed focus window.
```

- [ ] **Step 2: Run the complete automated suite before manual testing**

Run: `PYTHONPATH=src .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy src`

Expected: PASS before touching the live desktop.

- [ ] **Step 3: Execute the real model/device verification**

Run: `scripts/run-enhanced-xpu-poc.sh && fun-voice-selftest --format json`

Expected: both return ready/pass evidence for Nano, VAD, CTC, CAM++, diarization, Qwen3.5, Corrector and result socket; do not capture the JSON result payload in shell history or committed logs.

- [ ] **Step 4: Execute controlled desktop and identity scenarios**

1. Register one consenting profile with three short, preselected non-sensitive samples; confirm only encrypted profile data exists and no audio file remains.
2. In a text editor, speak a preselected Mandarin/code-mixed test sample; verify the focused editor and CLIPBOARD get the same `final_text` and `fun-voice-result latest` exposes the raw/structured form. Keep the sample and resulting text out of the repository, shell history and acceptance record.
3. Use an unregistered speaker; verify `unknown`, not a guessed profile label.
4. Delete the profile with confirmation; repeat and verify it is unknown.
5. Switch focus while transcribing; verify no injection into the new window and CLIPBOARD contains only `final_text`.
6. Stop Corrector; verify a safe raw fallback and that the next Nano transcription remains available after Corrector restarts.

- [ ] **Step 5: Record only pass/fail and metric categories in the checklist**

Record model revisions, XPU memory ranges, latency ranges, timestamp error range, accepted/rejected correction counts and identity precision/false-accept metrics. Do not record sample phrases, text output, profile labels, ids, audio paths or vectors.

- [ ] **Step 6: Commit the acceptance procedure**

```bash
git add docs/acceptance-checklist.md tests/manual/test-structured-identity-correction.md
git commit -m "test: document enhanced voice acceptance"
```

## Plan Self-Review

| Design requirement | Implemented by |
| --- | --- |
| Qwen3.5-0.8B text-only XPU service | Tasks 1 and 5 |
| Raw immutable structure and corrected final text | Tasks 2, 4, 9 and 10 |
| `results.sock`, TTL 10 min, 8 entries, same uid | Task 3 |
| XPU token timestamps without upstream CPU aligner | Task 6 |
| XPU CAM++ / cluster / conservative speaker identification | Task 7 |
| Local encrypted, consent-only persistent identities | Task 8 and Task 10 enrollment state |
| Fcitx/clipboard final-only and focus safety | Task 10 |
| No CPU fallback and POC/install block | Tasks 1 and 11 |
| Privacy, model/service failure and real acceptance | Tasks 10–12 |

The implementation must not start hardware/service integration until Task 1's real POC passes. Each task has a focused failing test, a named production boundary, a verification command and a one-purpose commit. No task depends on implicit APIs: all cross-task types and methods are declared above.
