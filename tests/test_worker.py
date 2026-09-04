"""Unit tests for the Nano runtime orchestration and worker message dispatch.

These tests use fake VAD and fake Nano (ASR) runtime components; they never
import torch / funasr, so they run in milliseconds.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest

from fun_voice import nano_runtime as nano_mod
from fun_voice.config import InferenceConfig
from fun_voice.contracts import ErrorCode, Segment, Transcription, WorkerHealth
from fun_voice.nano_runtime import (
    MAX_LIVE_PCM_BYTES,
    MAX_LIVE_WINDOW_MS,
    VAD_MAX_SINGLE_SEGMENT_TIME_MS,
    VAD_OVERLAP_MS,
    AudioFormatError,
    DeviceMismatchError,
    EmptySpeechError,
    FsmnVadSegmenter,
    InferenceTimeoutError,
    LiveAudioProtocolError,
    ModelLoadError,
    ModelOutputError,
    NanoRuntime,
    NanoRuntimeError,
    NativeNanoEngine,
    OomError,
    VllmError,
    _slice_windows,
    check_engine_devices,
)
from fun_voice.runtime_selection import RuntimeSelection
from fun_voice.worker import VERSION, LazyTranscriber, Worker

# --- Fakes ------------------------------------------------------------------


class FakeVad:
    """Fake FSMN-VAD returning a caller-supplied list of (start_ms, end_ms)."""

    def __init__(self, regions: list[tuple[int, int]]) -> None:
        self.regions = regions
        self.detect_calls: list[tuple[int, int]] = []

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        self.detect_calls.append((len(samples), sample_rate))
        return list(self.regions)


class FakeModule:
    def __init__(self, device_type: str, dtype: str = "torch.bfloat16") -> None:
        self.device = SimpleNamespace(type=device_type)
        self.dtype = dtype


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
        self.generate_kwargs: dict[str, Any] | None = None

    def generate(
        self, input: Any, cache: Any, is_final: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.generate_kwargs = {
            "input": input,
            "cache": cache,
            "is_final": is_final,
            **kwargs,
        }
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


class FakeNativeNanoModel:
    """Fake native FunASR wrapper used without importing model dependencies."""

    def __init__(self, dtype: str = "torch.bfloat16") -> None:
        module = SimpleNamespace(
            device=SimpleNamespace(type="xpu"), dtype=dtype
        )
        parameter = SimpleNamespace(
            device=SimpleNamespace(type="xpu"),
            dtype=dtype,
            is_floating_point=lambda: True,
        )
        module.parameters = lambda: iter([parameter])
        llm_model = SimpleNamespace(get_input_embeddings=lambda: module)
        llm_model.parameters = lambda: iter([parameter])
        self.model = SimpleNamespace(
            audio_encoder=module,
            audio_adaptor=module,
            llm=SimpleNamespace(model=llm_model),
        )
        self.parameters = lambda: iter([parameter])
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append(kwargs)
        return [{"key": "sample_0", "text": "native"}]


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

def _selection(backend: Literal["cuda", "xpu", "cpu"] = "xpu") -> RuntimeSelection:
    if backend == "cpu":
        device = "cpu"
        dtype = "float32"
        primary = "sensevoice"
        fallback = None
        enhanced_enabled = False
        speaker_enabled = False
        revisions = {"sensevoice": "master", "vad": "master"}
    else:
        device = f"{backend}:0"
        dtype = "bf16"
        primary = "nano"
        fallback = "sensevoice"
        enhanced_enabled = True
        speaker_enabled = True
        revisions = {
            "nano": "master",
            "sensevoice": "master",
            "vad": "master",
            "qwen": "master",
            "campplus": "master",
        }
    return RuntimeSelection(
        schema_version=1,
        backend=backend,
        python=Path("/selected-runtime/bin/python"),
        device=device,
        dtype=dtype,
        primary_asr_profile=primary,
        fallback_asr_profile=fallback,
        enhanced_enabled=enhanced_enabled,
        speaker_enabled=speaker_enabled,
        model_revisions=revisions,
        probe_status="pass",
        selected_at=1,
    )


def _inference(selection: RuntimeSelection) -> InferenceConfig:
    return InferenceConfig(
        device=selection.device,
        dtype=selection.dtype,
        allow_sensevoice_fallback=(
            selection.fallback_asr_profile == "sensevoice"
        ),
    )


def _runtime(
    engine: FakeEngine,
    vad: FakeVad,
    selection: RuntimeSelection | None = None,
) -> NanoRuntime:
    return NanoRuntime(
        engine=engine,
        vad=vad,
        selection=_selection("xpu") if selection is None else selection,
    )  # type: ignore[arg-type]


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
    assert runtime.health().lifecycle == "failed"

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


def test_nano_warmup_generates_synthetic_audio_without_vad() -> None:
    engine = FakeEngine(texts=[""])
    vad = FakeVad([(0, 100)])
    runtime = _runtime(engine, vad)

    elapsed_ms = runtime.warmup()

    assert elapsed_ms >= 0
    assert len(engine.inputs) == 1
    inputs, max_new_tokens = engine.inputs[0]
    assert len(inputs) == 1
    assert inputs[0].shape == (16_000,)
    assert inputs[0].dtype == np.float32
    assert max_new_tokens == 512
    assert vad.detect_calls == []


def test_native_nano_engine_adapts_repeated_in_memory_slices_without_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FakeNativeNanoModel()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=lambda samples: samples),
    )

    engine = NativeNanoEngine(native)
    samples = np.zeros(160, dtype=np.float32)
    result = engine.generate([samples], max_new_tokens=123)
    repeated = engine.generate([samples], max_new_tokens=123)

    assert result == [{"key": "sample_0", "text": "native"}]
    assert repeated == result
    assert native.calls == [
        {
            "input": [samples],
            "cache": {},
            "batch_size_s": 1,
            "max_length": 123,
            "llm_kwargs": {"do_sample": False},
        },
        {
            "input": [samples],
            "cache": {},
            "batch_size_s": 1,
            "max_length": 123,
            "llm_kwargs": {"do_sample": False},
        },
    ]
    check_engine_devices(engine, expected="xpu")
    assert engine.backend == "native_funasr_pytorch"
    assert engine.decoder_device_type == "xpu"


def test_native_nano_loader_uses_only_local_snapshot_and_xpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = FakeNativeNanoModel()
    captured: dict[str, Any] = {}

    def _auto_model(**kwargs: Any) -> FakeNativeNanoModel:
        captured.update(kwargs)
        return native

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=_auto_model))

    engine = nano_mod.load_native_nano_engine(
        "xpu:0", "bf16", model_dir="/local/nano"
    )

    assert isinstance(engine, NativeNanoEngine)
    assert captured == {
        "model": "/local/nano",
        "trust_remote_code": True,
        "device": "xpu:0",
        "dtype": "bf16",
        "bf16": True,
        "fp16": False,
        "disable_update": True,
    }
    assert engine.decoder_device_type == "xpu"


def test_runtime_live_fd_vad_and_window_preserve_source_offsets() -> None:
    runtime = _runtime(FakeEngine(texts=["window"]), FakeVad([(0, 100)]))
    with tempfile.TemporaryFile() as audio:
        fd = audio.fileno()
        audio.write(b"\x00\x00" * (16_000 * 300 // 1000))
        audio.flush()

        assert runtime.detect_vad_fd(fd, sample_rate=16000) == ((0, 100),)
        transcription = runtime.transcribe_window_fd(
            fd,
            sample_rate=16000,
            source_start_ms=1200,
            source_end_ms=1500,
        )

    assert transcription.text == "window"
    assert transcription.segments == (Segment(1200, 1300, "window"),)


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


def test_fsmn_vad_detect_passes_max_single_segment_time() -> None:
    model = FakeFsmnVadModel([{"key": "k", "value": [[100, 200]]}])
    FsmnVadSegmenter(model).detect(_silence(), 16000)
    assert model.generate_kwargs is not None
    assert (
        model.generate_kwargs["max_single_segment_time"]
        == VAD_MAX_SINGLE_SEGMENT_TIME_MS
    )
    assert VAD_MAX_SINGLE_SEGMENT_TIME_MS == 30000  # 30 s, in milliseconds


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
        engine=SlowEngine(),
        vad=FakeVad([(0, 100)]),
        selection=_selection("xpu"),
        default_timeout=0.05,
    )
    with pytest.raises(InferenceTimeoutError):
        runtime.transcribe_samples(_silence(), timeout=0.05)
    result = runtime.transcribe_samples(_silence(), timeout=2.0)
    assert result.text == "late"


# --- Device guard -----------------------------------------------------------


def _engine_on(
    device_type: str, dtype: str = "torch.bfloat16"
) -> SimpleNamespace:
    mod = SimpleNamespace(device=SimpleNamespace(type=device_type), dtype=dtype)
    return SimpleNamespace(audio_encoder=mod, audio_adaptor=mod, embed_tokens=mod)


def test_engine_device_check_uses_selected_cuda_type() -> None:
    engine = FakeEngine(texts=["ok"])
    engine.audio_encoder = FakeModule("cuda")  # type: ignore[attr-defined]
    engine.audio_adaptor = FakeModule("cuda")  # type: ignore[attr-defined]
    engine.embed_tokens = FakeModule("cuda")  # type: ignore[attr-defined]

    check_engine_devices(engine, expected="cuda")


def test_engine_device_check_rejects_cpu_when_cuda_selected() -> None:
    engine = FakeEngine(texts=["ok"])
    engine.audio_encoder = FakeModule("cpu")  # type: ignore[attr-defined]
    engine.audio_adaptor = FakeModule("cuda")  # type: ignore[attr-defined]
    engine.embed_tokens = FakeModule("cuda")  # type: ignore[attr-defined]

    with pytest.raises(DeviceMismatchError, match="expected 'cuda'"):
        check_engine_devices(engine, expected="cuda")


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


def test_live_fd_rejects_an_oversized_descriptor_before_vad() -> None:
    vad = FakeVad([(0, 100)])
    runtime = _runtime(FakeEngine(texts=[]), vad)
    with tempfile.TemporaryFile() as audio:
        audio.write(b"\x00" * (MAX_LIVE_PCM_BYTES + 2))
        audio.flush()
        with pytest.raises(LiveAudioProtocolError):
            runtime.detect_vad_fd(audio.fileno())

    assert vad.detect_calls == []


def test_vad_loader_rejects_a_non_xpu_model(monkeypatch: pytest.MonkeyPatch) -> None:
    parameter = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype="torch.bfloat16",
        is_floating_point=lambda: True,
    )
    model = SimpleNamespace(parameters=lambda: iter([parameter]))
    monkeypatch.setitem(
        sys.modules,
        "funasr",
        SimpleNamespace(AutoModel=lambda **_kwargs: model),
    )

    with pytest.raises(DeviceMismatchError):
        nano_mod._load_vad("xpu:0", "bf16")


def test_vad_loader_rejects_selected_dtype_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = SimpleNamespace(
        device=SimpleNamespace(type="xpu"),
        dtype="torch.float32",
        is_floating_point=lambda: True,
    )
    model = SimpleNamespace(parameters=lambda: iter([parameter]))
    captured: dict[str, object] = {}

    def auto_model(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return model

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=auto_model))

    with pytest.raises(DeviceMismatchError, match="dtype"):
        nano_mod._load_vad("xpu:0", "bf16")
    assert captured["dtype"] == "bf16"
    assert captured["bf16"] is True
    assert captured["fp16"] is False


def test_loaded_nano_runtime_health_rechecks_vad_selected_dtype() -> None:
    parameter = SimpleNamespace(
        device=SimpleNamespace(type="xpu"),
        dtype="torch.float32",
        is_floating_point=lambda: True,
    )
    vad_model = SimpleNamespace(parameters=lambda: iter([parameter]))
    runtime = NanoRuntime(
        engine=_engine_on("xpu"),
        vad=FsmnVadSegmenter(vad_model),
        selection=_selection("xpu"),
    )

    assert runtime.health().model_ready is False
    with pytest.raises(DeviceMismatchError, match="dtype"):
        runtime.device_evidence()


def test_nano_loader_rejects_a_non_xpu_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nano_mod,
        "load_native_nano_engine",
        lambda _device, _dtype: _engine_on("cpu"),
    )
    monkeypatch.setattr(
        nano_mod, "_load_vad", lambda _device, _dtype: FakeVad([(0, 100)])
    )

    with pytest.raises(DeviceMismatchError):
        nano_mod.load_nano_runtime(
            selection=_selection("xpu"), inference=_inference(_selection("xpu"))
        )


def test_nano_loader_rejects_engine_with_wrong_selected_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nano_mod,
        "load_native_nano_engine",
        lambda _device, _dtype: _engine_on("xpu", "torch.float32"),
    )
    monkeypatch.setattr(
        nano_mod, "_load_vad", lambda _device, _dtype: FakeVad([(0, 100)])
    )

    with pytest.raises(DeviceMismatchError, match="dtype"):
        nano_mod.load_nano_runtime(
            selection=_selection("xpu"), inference=_inference(_selection("xpu"))
        )


@pytest.mark.parametrize("backend", ["cuda", "xpu"], ids=["cuda", "xpu"])
def test_nano_loader_uses_accelerator_selection_not_toml(
    monkeypatch: pytest.MonkeyPatch, backend: Literal["cuda", "xpu"]
) -> None:
    selection = _selection(backend)
    captured: dict[str, str] = {}

    def load_engine(device: str, dtype: str) -> SimpleNamespace:
        captured["engine_device"] = device
        captured["engine_dtype"] = dtype
        return _engine_on(backend)

    def load_vad(device: str, dtype: str) -> FakeVad:
        captured["vad_device"] = device
        captured["vad_dtype"] = dtype
        return FakeVad([(0, 100)])

    monkeypatch.setattr(nano_mod, "load_native_nano_engine", load_engine)
    monkeypatch.setattr(nano_mod, "_load_vad", load_vad)

    runtime = nano_mod.load_nano_runtime(
        selection=selection, inference=_inference(selection)
    )

    assert runtime.device == f"{backend}:0"
    assert captured == {
        "engine_device": f"{backend}:0",
        "engine_dtype": "bf16",
        "vad_device": f"{backend}:0",
        "vad_dtype": "bf16",
    }


def test_cpu_sensevoice_loader_uses_selected_cpu_dtype_and_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("cpu")
    captured: dict[str, object] = {}
    parameter = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype="torch.float32",
        is_floating_point=lambda: True,
    )
    vad_parameter = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype="torch.float32",
        is_floating_point=lambda: True,
    )
    vad_model = SimpleNamespace(parameters=lambda: iter([vad_parameter]))
    model = SimpleNamespace(
        parameters=lambda: iter([parameter]), vad_model=vad_model
    )

    def auto_model(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return model

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=auto_model))

    runtime = nano_mod.load_sensevoice_runtime(
        selection=selection, inference=_inference(selection)
    )

    assert captured["device"] == "cpu"
    assert captured["dtype"] == "float32"
    assert captured["bf16"] is False
    assert captured["fp16"] is False
    assert runtime.device == "cpu"
    assert runtime.dtype == "float32"
    assert runtime.expected_device_type == "cpu"


def test_accelerator_sensevoice_passes_selected_dtype_to_nested_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("xpu")
    captured: dict[str, object] = {}
    parameter = SimpleNamespace(
        device=SimpleNamespace(type="xpu"),
        dtype="torch.bfloat16",
        is_floating_point=lambda: True,
    )
    model = SimpleNamespace(
        parameters=lambda: iter([parameter]),
        vad_model=SimpleNamespace(parameters=lambda: iter([parameter])),
    )

    def auto_model(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return model

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=auto_model))

    nano_mod.load_sensevoice_runtime(
        selection=selection, inference=_inference(selection)
    )

    assert captured["vad_kwargs"] == {
        "dtype": "bf16",
        "bf16": True,
        "fp16": False,
    }


def test_loaded_sensevoice_health_rechecks_vad_selected_dtype() -> None:
    main_parameter = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype="torch.float32",
        is_floating_point=lambda: True,
    )
    vad_parameter = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype="torch.bfloat16",
        is_floating_point=lambda: True,
    )
    model = SimpleNamespace(
        parameters=lambda: iter([main_parameter]),
        vad_model=SimpleNamespace(parameters=lambda: iter([vad_parameter])),
    )
    runtime = nano_mod.SenseVoiceRuntime(model, selection=_selection("cpu"))

    assert runtime.health().model_ready is False


def test_sensevoice_runtime_decodes_raw_pcm_before_model_generate(
    tmp_path: Path,
) -> None:
    """Captured s16le PCM must reach FunASR as 16 kHz float samples."""
    raw_pcm = tmp_path / "captured.pcm"
    raw_pcm.write_bytes(np.array([-32768, 0, 16384], dtype=np.int16).tobytes())
    parameter = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype="torch.float32",
        is_floating_point=lambda: True,
    )
    captured: list[object] = []

    def generate(*, input: object) -> list[dict[str, str]]:
        captured.append(input)
        return [{"key": "sample_0", "text": "ok"}]

    model = SimpleNamespace(
        parameters=lambda: iter([parameter]),
        vad_model=SimpleNamespace(parameters=lambda: iter([parameter])),
        generate=generate,
    )
    runtime = nano_mod.SenseVoiceRuntime(model, selection=_selection("cpu"))

    transcription = runtime.transcribe(str(raw_pcm), sample_rate=16000)

    assert transcription.text == "ok"
    assert len(captured) == 1
    assert isinstance(captured[0], np.ndarray)
    np.testing.assert_allclose(
        captured[0],
        np.array([-1.0, 0.0, 0.5], dtype=np.float32),
    )


def test_sensevoice_runtime_removes_control_tags_from_output(
    tmp_path: Path,
) -> None:
    """SenseVoice protocol metadata must never reach desktop text output."""
    raw_pcm = tmp_path / "captured.pcm"
    raw_pcm.write_bytes(np.array([0, 0], dtype=np.int16).tobytes())
    parameter = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype="torch.float32",
        is_floating_point=lambda: True,
    )

    def generate(*, input: object) -> list[dict[str, object]]:
        assert isinstance(input, np.ndarray)
        return [
            {
                "text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>请运行 pytest",
                "sentence_info": [
                    {
                        "start": 0,
                        "end": 500,
                        "text": "<|en|><|HAPPY|>use Git",
                    }
                ],
            }
        ]

    model = SimpleNamespace(
        parameters=lambda: iter([parameter]),
        vad_model=SimpleNamespace(parameters=lambda: iter([parameter])),
        generate=generate,
    )
    runtime = nano_mod.SenseVoiceRuntime(model, selection=_selection("cpu"))

    transcription = runtime.transcribe(str(raw_pcm), sample_rate=16000)

    assert transcription.text == "请运行 pytest"
    assert transcription.segments == (Segment(0, 500, "use Git"),)


def test_nano_loader_rejects_cpu_selection_before_model_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("cpu")
    monkeypatch.setattr(
        nano_mod,
        "load_native_nano_engine",
        lambda _device, _dtype: pytest.fail("Nano engine must not load for CPU"),
    )

    with pytest.raises(ModelLoadError, match="nano is not allowed"):
        nano_mod.load_nano_runtime(
            selection=selection, inference=_inference(selection)
        )


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
    assert response["engine"] == "nano"
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


def test_worker_error_response_never_echoes_an_audio_path() -> None:
    runtime = FakeRuntime(error=NanoRuntimeError("/private/audio.pcm"))
    response = _worker(runtime).handle(
        {"id": "u1", "op": "transcribe", "audio": "/private/audio.pcm"}
    )

    assert response["status"] == "error"
    assert response["error_code"] == "worker.internal"
    assert "/private/audio.pcm" not in repr(response)


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


def test_worker_live_requests_require_session_metadata_and_preserve_offsets(
) -> None:
    class LiveRuntime(FakeRuntime):
        def detect_vad_fd(
            self, fd: int, *, sample_rate: int
        ) -> tuple[tuple[int, int], ...]:
            assert fd >= 0
            assert sample_rate == 16000
            return ((0, 100), (120, 220))

        def transcribe_window_fd(
            self,
            fd: int,
            *,
            sample_rate: int,
            source_start_ms: int,
            source_end_ms: int,
        ) -> Transcription:
            assert fd >= 0
            assert sample_rate == 16000
            assert (source_start_ms, source_end_ms) == (1200, 1500)
            return Transcription(
                text="window",
                segments=(Segment(1200, 1500, "window"),),
            )

    worker = Worker(LazyTranscriber(LiveRuntime, device="xpu:0"))
    fd = os.open("/dev/null", os.O_RDONLY)
    try:
        missing = worker.handle({"id": "missing", "op": "detect_vad"})
        detected = worker.handle(
            {
                "id": "vad",
                "op": "detect_vad",
                "sample_rate": 16000,
                "session_id": "opaque-session",
                "generation": 2,
                "source_start_ms": 1000,
                "source_end_ms": 2000,
            },
            audio_fd=fd,
        )
        transcribed = worker.handle(
            {
                "id": "window",
                "op": "transcribe_window",
                "sample_rate": 16000,
                "session_id": "opaque-session",
                "generation": 2,
                "source_start_ms": 1200,
                "source_end_ms": 1500,
            },
            audio_fd=fd,
        )
    finally:
        os.close(fd)

    assert missing["status"] == "error"
    assert missing["error_code"] == "worker.protocol"
    assert detected == {
        "id": "vad",
        "status": "ok",
        "ranges": [
            {"start_ms": 0, "end_ms": 100},
            {"start_ms": 120, "end_ms": 220},
        ],
        "error_code": None,
    }
    assert transcribed["status"] == "ok"
    assert transcribed["segments"] == [
        {"start_ms": 1200, "end_ms": 1500, "text": "window"}
    ]


def test_worker_live_request_rejects_audio_path_and_never_echoes_it() -> None:
    response = _worker().handle(
        {
            "id": "live",
            "op": "detect_vad",
            "sample_rate": 16000,
            "session_id": "opaque-session",
            "generation": 1,
            "source_start_ms": 0,
            "source_end_ms": 100,
            "audio": "/private/audio.pcm",
        },
        audio_fd=3,
    )

    assert response["error_code"] == "worker.protocol"
    assert "/private/audio.pcm" not in repr(response)


def test_worker_live_request_rejects_an_unbounded_source_range() -> None:
    response = _worker().handle(
        {
            "id": "live",
            "op": "detect_vad",
            "sample_rate": 16000,
            "session_id": "opaque-session",
            "generation": 1,
            "source_start_ms": 0,
            "source_end_ms": MAX_LIVE_WINDOW_MS + 1,
        },
        audio_fd=3,
    )

    assert response["error_code"] == "worker.protocol"


def test_worker_unknown_op_returns_protocol_error() -> None:
    response = _worker().handle({"op": "bogus"})
    assert response["status"] == "error"
    assert response["error_code"] == "worker.protocol"


def test_worker_unknown_operation_never_echoes_caller_content() -> None:
    response = _worker().handle({"op": "/private/audio-or-text"})

    assert response["error_code"] == "worker.protocol"
    assert "/private/audio-or-text" not in repr(response)


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
