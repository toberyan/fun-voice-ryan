"""Unit tests for the XPU hard-gate preflight (fake torch / vLLM / Nano)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fun_voice.nano_runtime import VAD_OVERLAP_MS
from fun_voice.preflight import (
    CHECK_NAMES,
    DECODE_TOKENS_10S,
    DECODE_TOKENS_60S,
    RECOVERY_TOKENS,
    STATUS_FAIL,
    STATUS_PASS,
    CheckResult,
    PreflightReport,
    _transcribe_segmented,
    check_decode,
    check_vllm_xpu_decoder,
    check_worker_health,
    detect_cpu_fallback,
    load_nano_engine,
    probe_worker_health,
    run_preflight,
)

# --- Fakes -------------------------------------------------------------------


class _FakeTensor:
    pass


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class _FakeParameter:
    def __init__(self, device_type: str) -> None:
        self.device = _FakeDevice(device_type)


class _FakeModule:
    def __init__(self, device_type: str) -> None:
        self._device_type = device_type

    def parameters(self) -> Any:
        return iter([_FakeParameter(self._device_type)])


class _FakeProps:
    def __init__(self, total_memory: int) -> None:
        self.total_memory = total_memory


class _FakeXpu:
    def __init__(self, available: bool, total_memory: int) -> None:
        self._available = available
        self._total_memory = total_memory

    def is_available(self) -> bool:
        return self._available

    def memory_allocated(self, device: Any = None) -> int:
        return 1024

    def memory_reserved(self, device: Any = None) -> int:
        return 2048

    def get_device_properties(self, index: int) -> _FakeProps:
        return _FakeProps(self._total_memory)


class _FakeOOMError(RuntimeError):
    pass


class _FakeTorch:
    OutOfMemoryError = _FakeOOMError
    uint8 = "uint8"

    def __init__(self, available: bool = True, total_memory: int = 8 * 2**30) -> None:
        self.xpu = _FakeXpu(available, total_memory)

    def empty(self, size: int, *args: Any, **kwargs: Any) -> Any:
        if size > self.xpu._total_memory:
            raise _FakeOOMError("fake OOM")
        return _FakeTensor()


class _FakeVLLMEngine:
    def __init__(self, device_type: str = "xpu") -> None:
        self.device_type = device_type


class _FakeNanoEngine:
    def __init__(
        self,
        *,
        encoder_type: str = "xpu",
        adaptor_type: str = "xpu",
        embed_type: str = "xpu",
        device_type: str = "xpu",
        fail_tokens: set[int] | None = None,
    ) -> None:
        self.audio_encoder = _FakeModule(encoder_type)
        self.audio_adaptor = _FakeModule(adaptor_type)
        self.embed_tokens = _FakeModule(embed_type)
        self.vllm_engine = _FakeVLLMEngine(device_type)
        self._fail_tokens = fail_tokens or set()

    def generate(
        self, inputs: list[Any], max_new_tokens: int = 512, **kwargs: Any
    ) -> list[dict[str, str]]:
        if max_new_tokens in self._fail_tokens:
            raise RuntimeError("fake decode failure")
        return [
            {"key": f"sample_{index}", "text": "hello world"}
            for index, _input in enumerate(inputs)
        ]


class _FakeVad:
    """Fake VAD returning a caller-supplied list of ``(start_ms, end_ms)``."""

    def __init__(self, regions: list[tuple[int, int]] | None = None) -> None:
        self.regions = regions if regions is not None else [(0, 100), (200, 300)]
        self.detect_calls: list[tuple[int, int]] = []

    def detect(self, samples: Any, sample_rate: int) -> list[tuple[int, int]]:
        self.detect_calls.append((len(samples), sample_rate))
        return list(self.regions)


class _SegmentedEngine:
    """Fake Nano engine returning one text per input slice, in order."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.inputs: list[tuple[list[np.ndarray], int]] = []

    def generate(
        self, inputs: list[np.ndarray], max_new_tokens: int = 512, **kwargs: Any
    ) -> list[dict[str, str]]:
        self.inputs.append((list(inputs), max_new_tokens))
        return [
            {"key": f"sample_{i}", "text": text}
            for i, text in enumerate(self.texts)
        ]

TOTAL = 8 * 2**30


def _run(**kwargs: Any) -> PreflightReport:
    torch = _FakeTorch()
    engine = _FakeNanoEngine()
    vad = _FakeVad()
    if "torch" in kwargs:
        torch = kwargs.pop("torch")
    if "engine" in kwargs:
        engine = kwargs.pop("engine")
    if "vad" in kwargs:
        vad = kwargs.pop("vad")
    short = kwargs.pop("short", "short.wav")
    long = kwargs.pop("long", "long.wav")
    worker_health = kwargs.pop("worker_health", None)
    return run_preflight(
        torch=torch,
        engine=engine,
        vad=vad,
        short_sample=short,
        long_sample=long,
        worker_health=worker_health,
    )


@pytest.fixture(autouse=True)
def _stub_audio_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never hit disk: return a fixed 1 s silence buffer for every sample."""

    def _load(_path: str, sample_rate: int = 16000) -> np.ndarray:
        return np.zeros(sample_rate, dtype=np.float32)

    monkeypatch.setattr("fun_voice.preflight._load_audio_samples", _load)


# --- Tests -------------------------------------------------------------------


def test_check_names_are_the_nine_hard_gates() -> None:
    assert CHECK_NAMES == (
        "xpu_visible",
        "vllm_xpu_decoder",
        "nano_encoder_xpu",
        "nano_adaptor_xpu",
        "prompt_embeddings_xpu",
        "decode_10s",
        "decode_60s",
        "no_cpu_decoder_fallback",
        "oom_survives",
    )

@pytest.mark.parametrize(
    ("fail_name", "torch", "engine"),
    [
        ("xpu_visible", _FakeTorch(available=False), _FakeNanoEngine()),
        (
            "vllm_xpu_decoder",
            _FakeTorch(),
            _FakeNanoEngine(device_type="cuda"),
        ),
        ("nano_encoder_xpu", _FakeTorch(), _FakeNanoEngine(encoder_type="cpu")),
        ("nano_adaptor_xpu", _FakeTorch(), _FakeNanoEngine(adaptor_type="cpu")),
        (
            "prompt_embeddings_xpu",
            _FakeTorch(),
            _FakeNanoEngine(embed_type="cpu"),
        ),
        (
            "decode_10s",
            _FakeTorch(),
            _FakeNanoEngine(fail_tokens={DECODE_TOKENS_10S}),
        ),
        (
            "decode_60s",
            _FakeTorch(),
            _FakeNanoEngine(fail_tokens={DECODE_TOKENS_60S}),
        ),
        (
            "no_cpu_decoder_fallback",
            _FakeTorch(),
            _FakeNanoEngine(device_type="cpu"),
        ),
        (
            "oom_survives",
            _FakeTorch(),
            _FakeNanoEngine(fail_tokens={RECOVERY_TOKENS}),
        ),
    ],
    ids=[
        "xpu_visible",
        "vllm_xpu_decoder",
        "nano_encoder_xpu",
        "nano_adaptor_xpu",
        "prompt_embeddings_xpu",
        "decode_10s",
        "decode_60s",
        "no_cpu_decoder_fallback",
        "oom_survives",
    ],
)
def test_each_failing_check_blocks_ready(
    fail_name: str, torch: _FakeTorch, engine: _FakeNanoEngine
) -> None:
    report = _run(torch=torch, engine=engine)
    assert report.ready is False
    status_by_name = {c.name: c.status for c in report.checks}
    assert status_by_name[fail_name] == STATUS_FAIL


def test_report_json_omits_paths_and_transcript() -> None:
    report = _run(short="SECRET-SHORT.wav", long="SECRET-LONG.wav")
    payload = report.to_json()
    assert "SECRET-SHORT" not in payload
    assert "SECRET-LONG" not in payload
    assert "hello world" not in payload
    assert report.ready is True


def test_oom_check_records_recovery_length_not_text() -> None:
    report = _run()
    oom = next(c for c in report.checks if c.name == "oom_survives")
    assert oom.status == STATUS_PASS
    assert oom.detail["recovery_text_length"] == len("hello world")
    assert "hello world" not in report.to_json()


def test_vllm_xpu_decoder_alloc_probe_failure() -> None:
    """A failing allocation probe marks the decoder check as fail."""

    class _FailingAllocTorch(_FakeTorch):
        def empty(self, size: int, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("alloc boom")

    result = check_vllm_xpu_decoder(_FakeNanoEngine(), _FailingAllocTorch())
    assert result.status == STATUS_FAIL
    assert result.detail["alloc_probe"] == "failed:RuntimeError"
    assert result.detail["decoder_device_type"] == "xpu"


def test_detect_cpu_fallback_cpu_device() -> None:
    """A decoder resolved to ``cpu`` must be reported as a CPU fallback."""
    assert detect_cpu_fallback(_FakeNanoEngine(device_type="cpu")) == (
        "decoder device type is cpu"
    )
    assert detect_cpu_fallback(_FakeNanoEngine()) is None


def test_load_nano_engine_passes_triton_attn_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_nano_engine forwards attention_backend=TRITON_ATTN via vllm_kwargs."""
    import sys
    import types

    captured: dict[str, Any] = {}

    class _FakeFunASRNanoVLLM:
        @classmethod
        def from_pretrained(cls, **kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

    leaf = types.ModuleType("funasr.models.fun_asr_nano.inference_vllm")
    # types.ModuleType has no declared attribute; setattr avoids a mypy
    # attr-defined error on this dynamically-built fake module.
    setattr(leaf, "FunASRNanoVLLM", _FakeFunASRNanoVLLM)  # noqa: B010
    for pkg_name in ("funasr", "funasr.models", "funasr.models.fun_asr_nano"):
        monkeypatch.setitem(sys.modules, pkg_name, types.ModuleType(pkg_name))
    monkeypatch.setitem(
        sys.modules, "funasr.models.fun_asr_nano.inference_vllm", leaf
    )

    engine = load_nano_engine("/fake/model-dir")
    assert engine is not None
    assert captured["model"] == "/fake/model-dir"
    assert captured["gpu_memory_utilization"] == 0.15
    assert captured["max_model_len"] == 1536
    assert captured["device"] == "xpu:0"
    assert captured["dtype"] == "bf16"
    assert captured["vllm_kwargs"] == {"attention_backend": "TRITON_ATTN"}


def test_check_worker_health_pass() -> None:
    result = check_worker_health(
        {
            "status": "ok",
            "version": "1.0",
            "model_ready": True,
            "xpu_ready": True,
            "device": "xpu",
            "last_error": None,
        }
    )
    assert result.status == STATUS_PASS
    assert result.detail["device"] == "xpu"


def test_check_worker_health_fails_when_model_not_ready() -> None:
    result = check_worker_health(
        {
            "status": "ok",
            "version": "1.0",
            "model_ready": False,
            "xpu_ready": True,
            "device": "xpu",
            "last_error": "worker/internal",
        }
    )
    assert result.status == STATUS_FAIL


class _FakeConn:
    def __init__(self, response: bytes) -> None:
        self._response = response
        self.sent = bytearray()

    def settimeout(self, _timeout: float) -> None:
        pass

    def connect(self, _path: str) -> None:
        pass

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def recv(self, _n: int) -> bytes:
        if self._response:
            chunk = self._response
            self._response = b""
            return chunk
        return b""

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def test_probe_worker_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(
        b'{"status":"ok","model_ready":true,"xpu_ready":true,"device":"xpu"}\n'
    )
    monkeypatch.setattr("socket.socket", lambda *a, **k: conn)
    result = probe_worker_health("/tmp/fake.sock")
    assert result.status == STATUS_PASS
    assert result.detail["device"] == "xpu"
    assert conn.sent == b'{"op":"health"}\n'


def test_probe_worker_health_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingConn(_FakeConn):
        def connect(self, _path: str) -> None:
            raise ConnectionRefusedError("no worker")

    monkeypatch.setattr("socket.socket", lambda *a, **k: _FailingConn(b""))
    result = probe_worker_health("/tmp/fake.sock")
    assert result.status == STATUS_FAIL
    assert result.detail["error_class"] == "ConnectionRefusedError"


def test_probe_worker_health_requires_xdg_runtime_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    result = probe_worker_health()
    assert result.status == STATUS_FAIL
    assert result.detail["reason"] == "XDG_RUNTIME_DIR not set"


def test_run_preflight_ready_false_when_worker_unhealthy() -> None:
    worker_health = CheckResult("worker_health", STATUS_FAIL, {"reason": "down"})
    report = _run(worker_health=worker_health)
    assert report.ready is False
    assert report.worker_health is not None
    assert report.worker_health.status == STATUS_FAIL


def test_run_preflight_ready_true_when_worker_healthy() -> None:
    worker_health = CheckResult("worker_health", STATUS_PASS, {})
    report = _run(worker_health=worker_health)
    assert report.ready is True
    assert report.worker_health is not None
    assert report.worker_health.status == STATUS_PASS



def test_transcribe_segmented_sorts_and_concatenates_in_time_order() -> None:
    """VAD regions arrive out of order; text must follow sorted segment order."""
    engine = _SegmentedEngine(["a", "b", "c"])
    vad = _FakeVad([(300, 400), (0, 100), (600, 700)])
    samples = np.zeros(16000, dtype=np.float32)

    text, regions = _transcribe_segmented(engine, vad, samples, max_new_tokens=128)

    assert regions == [(0, 100), (300, 400), (600, 700)]
    assert text == "abc"
    (slices, tokens) = engine.inputs[0]
    assert tokens == 128
    assert len(slices) == 3


def test_transcribe_segmented_applies_fixed_overlap() -> None:
    """Each VAD region is sliced with VAD_OVERLAP_MS added to both boundaries."""
    engine = _SegmentedEngine(["ok"])
    vad = _FakeVad([(1000, 2000)])
    samples = np.arange(32000, dtype=np.float32)  # 2 s at 16 kHz

    _transcribe_segmented(engine, vad, samples, max_new_tokens=128)

    (slices, _tokens) = engine.inputs[0]
    overlap = int(VAD_OVERLAP_MS * 16000 / 1000)
    assert overlap == 4000
    # start = 1000 ms * 16 - overlap = 12000; end = 2000 ms * 16 + overlap,
    # clamped to total samples (32000).
    assert slices[0][0] == 12000.0
    assert slices[0][-1] == 31999.0


def test_check_decode_records_segment_count_not_text() -> None:
    """decode detail carries segment_count/text_length, never the transcript."""
    result = check_decode(
        "decode_60s",
        _SegmentedEngine(["MARKER-ONE", "MARKER-TWO"]),
        _FakeVad([(200, 300), (0, 100)]),
        "unused.wav",
        max_new_tokens=256,
        min_segments=2,
    )
    assert result.status == STATUS_PASS
    assert result.detail["segment_count"] == 2
    assert result.detail["text_length"] == len("MARKER-ONEMARKER-TWO")
    assert "MARKER" not in str(result.detail)


@pytest.mark.parametrize("texts", [["only-one"], ["one", "two", "three"]])
def test_check_decode_rejects_result_count_mismatch(texts: list[str]) -> None:
    result = check_decode(
        "decode_60s",
        _SegmentedEngine(texts),
        _FakeVad([(0, 100), (300, 400)]),
        "unused.wav",
        max_new_tokens=256,
        min_segments=2,
    )
    assert result.status == STATUS_FAIL
    assert result.detail["error_class"] == "ModelOutputError"


def test_check_decode_rejects_non_string_segment_text() -> None:
    class _MalformedEngine:
        def generate(self, *_args: Any, **_kwargs: Any) -> list[dict[str, object]]:
            return [{"text": "valid"}, {"text": None}]

    result = check_decode(
        "decode_60s",
        _MalformedEngine(),
        _FakeVad([(0, 100), (300, 400)]),
        "unused.wav",
        max_new_tokens=256,
        min_segments=2,
    )
    assert result.status == STATUS_FAIL
    assert result.detail["error_class"] == "ModelOutputError"
