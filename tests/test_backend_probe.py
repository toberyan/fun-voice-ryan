from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fun_voice.backend_probe import (
    MODEL_IDS,
    ProbeRequest,
    ProbeResult,
    run_probe,
)


class FakeTensor:
    def __init__(self, *, dtype: str, failing: set[str]) -> None:
        self.dtype = dtype
        self.failing = failing

    def sum(self) -> FakeTensor:
        if self.dtype in self.failing:
            raise RuntimeError("tensor failure")
        return self

    def item(self) -> float:
        return 32.0


class FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "float32"

    def __init__(
        self,
        *,
        cuda: bool = True,
        xpu: bool = True,
        failing_dtypes: set[str] | None = None,
    ) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        self.xpu = SimpleNamespace(is_available=lambda: xpu)
        self.failing_dtypes = failing_dtypes or set()
        self.ones_calls: list[tuple[int, str, str]] = []

    def ones(self, size: int, *, device: str, dtype: str) -> FakeTensor:
        self.ones_calls.append((size, device, dtype))
        return FakeTensor(dtype=dtype, failing=self.failing_dtypes)


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float]] = []
        self.closed = False

    def transcribe(
        self, audio: str, *, sample_rate: int, timeout: float
    ) -> object:
        self.calls.append((audio, sample_rate, timeout))
        return object()

    def close(self) -> None:
        self.closed = True


class ProbeFakes:
    def __init__(self, tmp_path: Path, torch: FakeTorch) -> None:
        self.tmp_path = tmp_path
        self.torch = torch
        self.downloaded: list[tuple[str, str]] = []
        self.nano = FakeRuntime()
        self.sensevoice = FakeRuntime()
        self.loader_calls: list[tuple[str, Any, Any, float]] = []
        self.loader_models_roots: list[str | None] = []

    def snapshot_download(self, model_id: str, *, revision: str) -> str:
        self.downloaded.append((model_id, revision))
        snapshot = self.tmp_path / model_id.replace("/", "--")
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "configuration.json").write_text("{}", encoding="utf-8")
        return str(snapshot)

    def download_sample(self, url: str, destination: str) -> None:
        assert url.startswith("https://")
        Path(destination).write_bytes(b"RIFFfake")

    def load_nano(self, **kwargs: Any) -> FakeRuntime:
        self.loader_models_roots.append(os.environ.get("FUN_VOICE_MODELS_ROOT"))
        self.loader_calls.append(
            (
                "nano",
                kwargs["selection"],
                kwargs["inference"],
                kwargs["default_timeout"],
            )
        )
        return self.nano

    def load_sensevoice(self, **kwargs: Any) -> FakeRuntime:
        self.loader_models_roots.append(os.environ.get("FUN_VOICE_MODELS_ROOT"))
        self.loader_calls.append(
            (
                "sensevoice",
                kwargs["selection"],
                kwargs["inference"],
                kwargs["default_timeout"],
            )
        )
        return self.sensevoice

    def run(self, backend: str) -> ProbeResult:
        return run_probe(
            ProbeRequest(
                backend=backend,  # type: ignore[arg-type]
                runtime_root=self.tmp_path / "runtime",
                models_root=self.tmp_path / "models",
                revision="master",
            ),
            torch_module=self.torch,
            snapshot_downloader=self.snapshot_download,
            sample_downloader=self.download_sample,
            nano_loader=self.load_nano,
            sensevoice_loader=self.load_sensevoice,
        )


def test_cuda_falls_back_from_bf16_to_fp16(tmp_path: Path) -> None:
    fake = ProbeFakes(tmp_path, FakeTorch(failing_dtypes={"bf16"}))
    result = fake.run("cuda")
    assert result.status == "pass"
    assert result.dtype == "fp16"
    assert fake.torch.ones_calls == [
        (32, "cuda:0", "bf16"),
        (32, "cuda:0", "fp16"),
    ]
    assert [key for key, _ in fake.downloaded] == list(MODEL_IDS.values())
    assert len(fake.nano.calls) == 1
    assert fake.nano.closed


def test_xpu_only_tries_bf16(tmp_path: Path) -> None:
    fake = ProbeFakes(tmp_path, FakeTorch(failing_dtypes={"bf16"}))
    result = fake.run("xpu")
    assert result.status == "fail"
    assert result.error_category == "tensor"
    assert fake.torch.ones_calls == [(32, "xpu:0", "bf16")]
    assert fake.downloaded == []


def test_cpu_uses_float32_sensevoice_and_minimal_models(tmp_path: Path) -> None:
    fake = ProbeFakes(tmp_path, FakeTorch())
    result = fake.run("cpu")
    assert result.status == "pass"
    assert result.dtype == "float32"
    assert fake.torch.ones_calls == [(32, "cpu", "float32")]
    assert [key for key, _ in fake.downloaded] == [
        MODEL_IDS["sensevoice"],
        MODEL_IDS["vad"],
    ]
    assert [call[0] for call in fake.loader_calls] == ["sensevoice"]
    assert len(fake.sensevoice.calls) == 1
    assert fake.sensevoice.closed
    assert set(result.models) == {"sensevoice", "vad"}
    assert fake.loader_models_roots == [str(tmp_path / "models")]
    assert "FUN_VOICE_MODELS_ROOT" not in os.environ


@pytest.mark.parametrize(
    ("backend", "available"), [("cuda", {"cuda": False}), ("xpu", {"xpu": False})]
)
def test_unavailable_accelerator_fails_without_download(
    backend: str, available: dict[str, bool], tmp_path: Path
) -> None:
    fake = ProbeFakes(tmp_path, FakeTorch(**available))
    result = fake.run(backend)
    assert result.status == "fail"
    assert result.error_category == "availability"
    assert fake.downloaded == []


def test_probe_json_has_closed_privacy_preserving_schema(tmp_path: Path) -> None:
    fake = ProbeFakes(tmp_path, FakeTorch())
    payload = json.loads(fake.run("cpu").to_json())
    assert set(payload) == {
        "backend",
        "status",
        "error_category",
        "dtype",
        "models",
        "tensor_ms",
        "asr_ms",
    }
    serialized = json.dumps(payload)
    assert "RIFF" not in serialized
    assert "asr_example" not in serialized
    assert str(tmp_path) not in serialized


def test_missing_snapshot_metadata_fails_before_model_load(tmp_path: Path) -> None:
    fake = ProbeFakes(tmp_path, FakeTorch())

    def no_metadata(model_id: str, *, revision: str) -> str:
        snapshot = tmp_path / model_id.replace("/", "--")
        snapshot.mkdir(parents=True, exist_ok=True)
        return str(snapshot)

    result = run_probe(
        ProbeRequest("cpu", tmp_path / "runtime", tmp_path / "models", "master"),
        torch_module=fake.torch,
        snapshot_downloader=no_metadata,
        sample_downloader=fake.download_sample,
        nano_loader=fake.load_nano,
        sensevoice_loader=fake.load_sensevoice,
    )
    assert result.status == "fail"
    assert result.error_category == "model_download"
    assert fake.loader_calls == []


def test_transcribe_failure_still_closes_runtime(tmp_path: Path) -> None:
    fake = ProbeFakes(tmp_path, FakeTorch())

    def fail_transcribe(
        audio: str, *, sample_rate: int, timeout: float
    ) -> object:
        raise RuntimeError("private transcript")

    fake.sensevoice.transcribe = fail_transcribe  # type: ignore[method-assign]
    result = fake.run("cpu")
    assert result.status == "fail"
    assert result.error_category == "asr"
    assert fake.sensevoice.closed
