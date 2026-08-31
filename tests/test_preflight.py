"""Unit tests for the XPU hard-gate preflight (fake torch / vLLM / Nano)."""

from __future__ import annotations

from typing import Any

import pytest

from fun_voice.preflight import (
    CHECK_NAMES,
    DECODE_TOKENS_10S,
    DECODE_TOKENS_60S,
    RECOVERY_TOKENS,
    STATUS_FAIL,
    STATUS_PASS,
    PreflightReport,
    check_vllm_xpu_decoder,
    detect_cpu_fallback,
    load_nano_engine,
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
        self, inputs: list[str], max_new_tokens: int = 512, **kwargs: Any
    ) -> list[dict[str, str]]:
        if max_new_tokens in self._fail_tokens:
            raise RuntimeError("fake decode failure")
        return [{"key": "sample", "text": "hello world"}]


TOTAL = 8 * 2**30


def _run(**kwargs: Any) -> PreflightReport:
    torch = _FakeTorch()
    engine = _FakeNanoEngine()
    if "torch" in kwargs:
        torch = kwargs.pop("torch")
    if "engine" in kwargs:
        engine = kwargs.pop("engine")
    short = kwargs.pop("short", "short.wav")
    long = kwargs.pop("long", "long.wav")
    return run_preflight(
        torch=torch, engine=engine, short_sample=short, long_sample=long
    )


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
    assert captured["device"] == "xpu:0"
    assert captured["dtype"] == "bf16"
    assert captured["vllm_kwargs"] == {"attention_backend": "TRITON_ATTN"}
