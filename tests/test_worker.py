"""Unit tests for the Nano runtime orchestration and worker message dispatch.

These tests use fake VAD and fake Nano (ASR) runtime components; they never
import torch / vllm / funasr, so they run in milliseconds.
"""

from __future__ import annotations

import time
import wave
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fun_voice.contracts import ErrorCode, Segment, Transcription, WorkerHealth
from fun_voice.nano_runtime import (
    VAD_OVERLAP_MS,
    AudioFormatError,
    DeviceMismatchError,
    EmptySpeechError,
    FsmnVadSegmenter,
    InferenceTimeoutError,
    ModelOutputError,
    NanoRuntime,
    NanoRuntimeError,
    OomError,
    VllmError,
    _slice_windows,
    check_engine_devices,
)
from fun_voice.worker import VERSION, Worker

# --- Fakes ------------------------------------------------------------------


class FakeVad:
    """Fake FSMN-VAD returning a caller-supplied list of (start_ms, end_ms)."""

    def __init__(self, regions: list[tuple[int, int]]) -> None:
        self.regions = regions
        self.detect_calls: list[tuple[int, int]] = []

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        self.detect_calls.append((len(samples), sample_rate))
        return list(self.regions)


class FakeEngine:
    """Fake Nano ASR engine returning one text per input slice, in order."""

    def __init__(
        self,
        texts: list[str] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.texts = texts if texts is not None else []
        self.error = error
        self.inputs: list[tuple[list[np.ndarray], int]] = []

    def generate(
        self, inputs: list[np.ndarray], max_new_tokens: int = 512
    ) -> list[dict[str, Any]]:
        self.inputs.append((list(inputs), max_new_tokens))
        if self.error is not None:
            raise self.error
        return [
            {"key": f"sample_{i}", "text": text}
            for i, text in enumerate(self.texts)
        ]


class FakeFsmnVadModel:
    """Fake raw FunASR FSMN-VAD returning ``[{"key", "value": [[s, e], ...]}]``."""

    def __init__(self, result: list[dict[str, Any]]) -> None:
        self.result = result

    def generate(self, input: Any, cache: Any, is_final: Any) -> list[dict[str, Any]]:
        return self.result


class SlowEngine:
    """Fake ASR engine that blocks long enough to trip a small timeout."""

    def __init__(self, delay: float = 0.3, text: str = "late") -> None:
        self.delay = delay
        self.text = text

    def generate(
        self, inputs: list[np.ndarray], max_new_tokens: int = 512
    ) -> list[dict[str, Any]]:
        time.sleep(self.delay)
        return [{"key": "sample_0", "text": self.text}]


class FlakyRuntime:
    """Raises once, then succeeds — for the "keeps serving after error" path."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(
        self, audio: str, *, sample_rate: int = 16000, timeout: float | None = None
    ) -> Transcription:
        self.calls += 1
        if self.calls == 1:
            raise OomError("out of memory")
        return Transcription(text="ok", segments=(Segment(0, 100, "ok"),))

    def health(self) -> WorkerHealth:
        return WorkerHealth(
            version="test", xpu_ready=True, model_ready=True, device="xpu:0"
        )

    def close(self) -> None:
        pass

def _runtime(engine: FakeEngine, vad: FakeVad) -> NanoRuntime:
    return NanoRuntime(engine=engine, vad=vad)  # type: ignore[arg-type]


def _silence(seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


# --- VAD / orchestration ----------------------------------------------------


def test_vad_empty_result_raises_empty_speech() -> None:
    runtime = _runtime(FakeEngine(texts=[]), FakeVad([]))
    with pytest.raises(EmptySpeechError):
        runtime.transcribe_samples(_silence())
    # No ASR call is made when VAD finds nothing.
    assert not runtime.engine.inputs  # type: ignore[attr-defined]


def test_multiple_segments_preserve_strict_time_order() -> None:
    # VAD returns segments out of audio-time order; the runtime must sort them.
    engine = FakeEngine(texts=["a", "b", "c"])
    vad = FakeVad([(300, 400), (0, 100), (600, 700)])
    runtime = _runtime(engine, vad)
    result = runtime.transcribe_samples(_silence())

    assert [seg.start_ms for seg in result.segments] == [0, 300, 600]
    assert [seg.end_ms for seg in result.segments] == [100, 400, 700]
    # Texts map onto segments in the same (time-sorted) order.
    assert [seg.text for seg in result.segments] == ["a", "b", "c"]
    assert result.text == "abc"


def test_final_text_is_char_exact_concatenation() -> None:
    # CJK + punctuation + astral emoji: the join must insert/delete nothing.
    texts = ["你好", "，世界！", "😀"]
    engine = FakeEngine(texts=texts)
    vad = FakeVad([(0, 100), (200, 300), (400, 500)])
    runtime = _runtime(engine, vad)
    result = runtime.transcribe_samples(_silence())

    assert result.text == "你好，世界！😀"
    # Character-by-character equality (no space/separator injected, none dropped).
    assert list(result.text) == list("你好，世界！😀")
    assert [seg.text for seg in result.segments] == texts


def test_runtime_tracks_last_error_category() -> None:
    engine = FakeEngine(error=RuntimeError("boom"))
    vad = FakeVad([(0, 100)])
    runtime = _runtime(engine, vad)
    with pytest.raises(VllmError):
        runtime.transcribe_samples(_silence())
    assert runtime.health().last_error == ErrorCode("worker", "vllm")

def test_slice_windows_apply_fixed_overlap_and_clamp() -> None:
    overlap = int(VAD_OVERLAP_MS * 16000 / 1000)  # 4000 samples at 16 kHz
    assert overlap == 4000
    windows = _slice_windows([(1000, 2000)], total_samples=32000, sample_rate=16000)
    assert windows == [(16000 - overlap, 32000)]  # end clamped to total
    windows = _slice_windows(
        [(1000, 2000), (3000, 4000)], total_samples=64000, sample_rate=16000
    )
    assert windows == [(12000, 36000), (44000, 64000)]


def test_segment_text_is_empty_string_when_model_outputs_empty() -> None:
    engine = FakeEngine(texts=["", "ok"])
    vad = FakeVad([(0, 100), (200, 300)])
    runtime = _runtime(engine, vad)
    result = runtime.transcribe_samples(_silence())
    assert result.text == "ok"
    assert [seg.text for seg in result.segments] == ["", "ok"]


# --- FSMN-VAD adapter (real parse path) -------------------------------------


def test_fsmn_vad_detect_parses_segments() -> None:
    model = FakeFsmnVadModel([{"key": "k", "value": [[100, 200], [300, 400]]}])
    assert FsmnVadSegmenter(model).detect(_silence(), 16000) == [(100, 200), (300, 400)]


def test_fsmn_vad_detect_empty_value_returns_empty() -> None:
    model = FakeFsmnVadModel([{"key": "k", "value": []}])
    assert FsmnVadSegmenter(model).detect(_silence(), 16000) == []


def test_fsmn_vad_detect_missing_value_key_raises() -> None:
    model = FakeFsmnVadModel([{"key": "k"}])
    with pytest.raises(ModelOutputError):
        FsmnVadSegmenter(model).detect(_silence(), 16000)


def test_fsmn_vad_detect_empty_result_list_raises() -> None:
    with pytest.raises(ModelOutputError):
        FsmnVadSegmenter(FakeFsmnVadModel([])).detect(_silence(), 16000)


def test_fsmn_vad_detect_malformed_segment_raises() -> None:
    model = FakeFsmnVadModel([{"key": "k", "value": [[100]]}])  # 1-element segment
    with pytest.raises(ModelOutputError):
        FsmnVadSegmenter(model).detect(_silence(), 16000)


# --- ASR error taxonomy -----------------------------------------------------


def test_runtime_oom_maps_to_oom_error() -> None:
    runtime = _runtime(
        FakeEngine(error=RuntimeError("XPU out of memory")), FakeVad([(0, 100)])
    )
    with pytest.raises(OomError):
        runtime.transcribe_samples(_silence())


def test_runtime_model_no_output_raises() -> None:
    runtime = _runtime(FakeEngine(texts=[]), FakeVad([(0, 100)]))
    with pytest.raises(ModelOutputError):
        runtime.transcribe_samples(_silence())


def test_runtime_timeout_raises_then_next_request_succeeds() -> None:
    # A timed-out generate keeps holding the generate lock until it finishes;
    # the next request must wait for the lock and then succeed (Minor 4).
    runtime = NanoRuntime(
        engine=SlowEngine(), vad=FakeVad([(0, 100)]), default_timeout=0.05
    )
    with pytest.raises(InferenceTimeoutError):
        runtime.transcribe_samples(_silence(), timeout=0.05)
    result = runtime.transcribe_samples(_silence(), timeout=2.0)
    assert result.text == "late"


# --- Device guard -----------------------------------------------------------


def _engine_on(device_type: str) -> SimpleNamespace:
    mod = SimpleNamespace(device=SimpleNamespace(type=device_type))
    return SimpleNamespace(audio_encoder=mod, audio_adaptor=mod, embed_tokens=mod)


def test_check_engine_devices_accepts_xpu() -> None:
    check_engine_devices(_engine_on("xpu"))


def test_check_engine_devices_rejects_non_xpu() -> None:
    with pytest.raises(DeviceMismatchError):
        check_engine_devices(_engine_on("cpu"))


def test_transcribe_rejects_non_xpu_before_asr(tmp_path) -> None:
    engine = FakeEngine(texts=["x"])
    # Attach non-XPU modules so the path-level guard fires.
    engine.audio_encoder = SimpleNamespace(device=SimpleNamespace(type="cpu"))  # type: ignore[attr-defined]
    engine.audio_adaptor = SimpleNamespace(device=SimpleNamespace(type="xpu"))  # type: ignore[attr-defined]
    engine.embed_tokens = SimpleNamespace(device=SimpleNamespace(type="xpu"))  # type: ignore[attr-defined]
    vad = FakeVad([(0, 100)])
    runtime = _runtime(engine, vad)
    wav_path = tmp_path / "a.wav"
    _write_wav(wav_path, np.zeros(1600, dtype=np.int16))

    with pytest.raises(DeviceMismatchError):
        runtime.transcribe(str(wav_path))
    assert not engine.inputs  # rejected before any ASR call


def _write_wav(path: Any, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())


# --- Worker dispatch --------------------------------------------------------


class FakeRuntime:
    """Fake Transcriber for Worker dispatch tests."""

    def __init__(
        self,
        transcription: Transcription | None = None,
        *,
        error: BaseException | None = None,
        health: WorkerHealth | None = None,
    ) -> None:
        self.transcription = transcription or Transcription(text="", segments=())
        self.error = error
        self.health_result = health or WorkerHealth(
            version="test", xpu_ready=True, model_ready=True, device="xpu:0"
        )
        self.calls: list[tuple[str, int, float | None]] = []

    def transcribe(
        self, audio: str, *, sample_rate: int = 16000, timeout: float | None = None
    ) -> Transcription:
        self.calls.append((audio, sample_rate, timeout))
        if self.error is not None:
            raise self.error
        return self.transcription

    def health(self) -> WorkerHealth:
        return self.health_result

    def close(self) -> None:
        pass


def _worker(runtime: FakeRuntime | None = None) -> Worker:
    return Worker(runtime or FakeRuntime())


def test_worker_transcribe_success_shape() -> None:
    runtime = FakeRuntime(
        transcription=Transcription(text="你好", segments=(Segment(0, 100, "你好"),))
    )
    worker = _worker(runtime)
    response = worker.handle(
        {"id": "u1", "op": "transcribe", "audio": "/tmp/a.wav", "sample_rate": 16000}
    )
    assert response["status"] == "ok"
    assert response["id"] == "u1"
    assert response["text"] == "你好"
    assert response["segments"] == [{"start_ms": 0, "end_ms": 100, "text": "你好"}]
    assert response["elapsed_ms"] >= 0
    assert response["error_code"] is None


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (EmptySpeechError("no speech"), "worker.empty_speech"),
        (OomError("out of memory"), "worker.oom"),
        (VllmError("ValueError"), "worker.vllm"),
        (ModelOutputError("no output"), "worker.no_output"),
        (AudioFormatError("bad audio"), "worker.format"),
        (InferenceTimeoutError("slow"), "worker.timeout"),
        (DeviceMismatchError("not xpu"), "worker.device"),
    ],
)
def test_worker_maps_runtime_errors_to_codes(
    error: NanoRuntimeError, expected_code: str
) -> None:
    runtime = FakeRuntime(error=error)
    response = _worker(runtime).handle(
        {"id": "u1", "op": "transcribe", "audio": "/tmp/a.wav", "sample_rate": 16000}
    )
    assert response["status"] == "error"
    assert response["error_code"] == expected_code
    assert response["text"] == ""
    assert response["segments"] == []


def test_worker_keeps_serving_after_error() -> None:
    worker = _worker(FlakyRuntime())  # type: ignore[arg-type]
    first = worker.handle(
        {"id": "u1", "op": "transcribe", "audio": "/tmp/a.wav", "sample_rate": 16000}
    )
    assert first["status"] == "error"
    assert first["error_code"] == "worker.oom"
    second = worker.handle(
        {"id": "u2", "op": "transcribe", "audio": "/tmp/a.wav", "sample_rate": 16000}
    )
    assert second["status"] == "ok"
    assert second["text"] == "ok"


def test_worker_passes_timeout_ms_to_runtime() -> None:
    runtime = FakeRuntime()
    _worker(runtime).handle(
        {
            "id": "u1",
            "op": "transcribe",
            "audio": "/tmp/a.wav",
            "sample_rate": 16000,
            "timeout_ms": 1500,
        }
    )
    _audio, _sr, timeout = runtime.calls[0]
    assert timeout == 1.5


def test_worker_default_sample_rate() -> None:
    runtime = FakeRuntime()
    _worker(runtime).handle({"id": "u1", "op": "transcribe", "audio": "/tmp/a.wav"})
    _audio, sample_rate, _timeout = runtime.calls[0]
    assert sample_rate == 16000


def test_worker_health_response() -> None:
    runtime = FakeRuntime(
        health=WorkerHealth(
            version="x",
            xpu_ready=True,
            model_ready=True,
            device="xpu:0",
            last_error=ErrorCode("worker", "oom"),
        )
    )
    response = _worker(runtime).handle({"op": "health"})
    assert response["status"] == "ok"
    assert response["version"] == VERSION
    assert response["model_ready"] is True
    assert response["xpu_ready"] is True
    assert response["device"] == "xpu:0"
    assert response["last_error"] == "worker.oom"


def test_worker_health_never_carries_audio_or_text() -> None:
    response = _worker().handle({"op": "health"})
    assert "audio" not in response
    assert "text" not in response
    assert "path" not in response
    assert "segments" not in response


def test_worker_unknown_op_returns_protocol_error() -> None:
    response = _worker().handle({"op": "bogus"})
    assert response["status"] == "error"
    assert response["error_code"] == "worker.protocol"


def test_worker_missing_id_returns_protocol_error() -> None:
    response = _worker().handle({"op": "transcribe", "audio": "/tmp/a.wav"})
    assert response["status"] == "error"
    assert response["error_code"] == "worker.protocol"


def test_worker_missing_audio_returns_protocol_error() -> None:
    response = _worker().handle({"id": "u1", "op": "transcribe"})
    assert response["status"] == "error"
    assert response["error_code"] == "worker.protocol"


def test_worker_invalid_sample_rate_returns_protocol_error() -> None:
    response = _worker().handle(
        {"id": "u1", "op": "transcribe", "audio": "/tmp/a.wav", "sample_rate": -1}
    )
    assert response["status"] == "error"
    assert response["error_code"] == "worker.protocol"
