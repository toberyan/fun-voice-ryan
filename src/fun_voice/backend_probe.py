"""Privacy-preserving hardware and model probe for one isolated runtime.

This module is executed by the candidate runtime interpreter. Heavy optional
dependencies are imported only inside :func:`run_probe`, so importing the
schema remains safe in the repository development environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from fun_voice.runtime_selection import Backend, RuntimeSelection

MODEL_IDS = {
    "nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "sensevoice": "iic/SenseVoiceSmall",
    "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "qwen": "Qwen/Qwen3.5-0.8B",
    "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}
PUBLIC_SAMPLE_URL = (
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/"
    "test_audio/asr_example_zh.wav"
)
ACCELERATOR_MODEL_KEYS = ("nano", "sensevoice", "vad", "qwen", "campplus")
CPU_MODEL_KEYS = ("sensevoice", "vad")
ERROR_CATEGORIES = frozenset(
    {
        "environment",
        "import",
        "availability",
        "tensor",
        "dtype",
        "model_download",
        "asr",
        "internal",
    }
)

ProbeStatus = Literal["pass", "fail"]
ErrorCategory = Literal[
    "environment",
    "import",
    "availability",
    "tensor",
    "dtype",
    "model_download",
    "asr",
    "internal",
]


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    backend: Backend
    runtime_root: Path
    models_root: Path
    revision: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    backend: str
    status: ProbeStatus
    error_category: ErrorCategory | str | None
    dtype: str | None
    models: Mapping[str, str]
    tensor_ms: int
    asr_ms: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "backend": self.backend,
                "status": self.status,
                "error_category": self.error_category,
                "dtype": self.dtype,
                "models": dict(self.models),
                "tensor_ms": self.tensor_ms,
                "asr_ms": self.asr_ms,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


class ProbeRuntime(Protocol):
    def transcribe(
        self, audio: str, *, sample_rate: int, timeout: float
    ) -> object: ...

    def close(self) -> None: ...


RuntimeLoader = Callable[..., ProbeRuntime]
SnapshotDownloader = Callable[..., str]
SampleDownloader = Callable[[str, str], object]


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _failure(
    backend: Backend,
    category: ErrorCategory,
    *,
    dtype: str | None = None,
    models: Mapping[str, str] | None = None,
    tensor_ms: int = 0,
    asr_ms: int = 0,
) -> ProbeResult:
    return ProbeResult(
        backend=backend,
        status="fail",
        error_category=category,
        dtype=dtype,
        models={} if models is None else dict(models),
        tensor_ms=max(0, tensor_ms),
        asr_ms=max(0, asr_ms),
    )


def _tensor_gate(torch_module: Any, backend: Backend) -> tuple[str, int] | None:
    if backend == "cuda":
        if not torch_module.cuda.is_available():
            return None
        choices: tuple[tuple[str, Any], ...] = (
            ("bf16", torch_module.bfloat16),
            ("fp16", torch_module.float16),
        )
        device = "cuda:0"
    elif backend == "xpu":
        if not torch_module.xpu.is_available():
            return None
        choices = (("bf16", torch_module.bfloat16),)
        device = "xpu:0"
    else:
        choices = (("float32", torch_module.float32),)
        device = "cpu"

    started = time.perf_counter()
    for name, dtype in choices:
        try:
            torch_module.ones(32, device=device, dtype=dtype).sum().item()
        except Exception:  # noqa: BLE001 - candidate failures are classified only
            continue
        return name, _elapsed_ms(started)
    return "", _elapsed_ms(started)


def _snapshot_has_metadata(snapshot: Path) -> bool:
    return snapshot.is_dir() and any(
        (snapshot / name).is_file()
        for name in ("config.json", "config.yaml", "configuration.json")
    )


def _default_sample_downloader(url: str, destination: str) -> object:
    return urllib.request.urlretrieve(url, destination)


def run_probe(
    request: ProbeRequest,
    *,
    torch_module: Any | None = None,
    snapshot_downloader: SnapshotDownloader | None = None,
    sample_downloader: SampleDownloader | None = None,
    nano_loader: RuntimeLoader | None = None,
    sensevoice_loader: RuntimeLoader | None = None,
) -> ProbeResult:
    """Run tensor, snapshot and one public-sample ASR gate for a backend."""
    if request.backend not in {"cuda", "xpu", "cpu"}:
        return _failure(request.backend, "environment")
    backend = request.backend
    if request.revision != "master":
        return _failure(backend, "environment")

    try:
        if torch_module is None:
            import torch as imported_torch

            torch_module = imported_torch
        if snapshot_downloader is None:
            from modelscope.hub.snapshot_download import snapshot_download

            snapshot_downloader = snapshot_download
        if nano_loader is None or sensevoice_loader is None:
            from fun_voice.nano_runtime import (
                load_nano_runtime,
                load_sensevoice_runtime,
            )

            nano_loader = nano_loader or load_nano_runtime
            sensevoice_loader = sensevoice_loader or load_sensevoice_runtime
        from fun_voice.config import InferenceConfig
    except Exception:  # noqa: BLE001 - do not expose import exception details
        return _failure(backend, "import")

    try:
        available = (
            True
            if backend == "cpu"
            else bool(
                torch_module.cuda.is_available()
                if backend == "cuda"
                else torch_module.xpu.is_available()
            )
        )
    except Exception:  # noqa: BLE001
        return _failure(backend, "availability")
    if not available:
        return _failure(backend, "availability")

    try:
        tensor_result = _tensor_gate(torch_module, backend)
    except (AttributeError, TypeError):
        return _failure(backend, "dtype")
    if tensor_result is None:
        return _failure(backend, "availability")
    dtype, tensor_ms = tensor_result
    if not dtype:
        return _failure(backend, "tensor", tensor_ms=tensor_ms)

    model_keys = CPU_MODEL_KEYS if backend == "cpu" else ACCELERATOR_MODEL_KEYS
    revisions = {key: request.revision for key in model_keys}
    environment_keys = (
        "MODELSCOPE_CACHE",
        "FUN_VOICE_MODELS_ROOT",
        "MODELSCOPE_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    previous_environment = {key: os.environ.get(key) for key in environment_keys}
    os.environ["MODELSCOPE_CACHE"] = str(request.models_root)
    os.environ["FUN_VOICE_MODELS_ROOT"] = str(request.models_root)
    try:
        try:
            for key in model_keys:
                snapshot = Path(
                    snapshot_downloader(MODEL_IDS[key], revision=request.revision)
                )
                if not _snapshot_has_metadata(snapshot):
                    return _failure(
                        backend,
                        "model_download",
                        dtype=dtype,
                        tensor_ms=tensor_ms,
                    )
        except Exception:  # noqa: BLE001
            return _failure(
                backend, "model_download", dtype=dtype, tensor_ms=tensor_ms
            )

        for key in (
            "MODELSCOPE_OFFLINE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ):
            os.environ[key] = "1"
        sample_path: Path | None = None
        runtime: ProbeRuntime | None = None
        asr_started = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as sample:
                sample_path = Path(sample.name)
            (sample_downloader or _default_sample_downloader)(
                PUBLIC_SAMPLE_URL, str(sample_path)
            )
            device = "cpu" if backend == "cpu" else f"{backend}:0"
            selection = RuntimeSelection(
                schema_version=1,
                backend=backend,
                python=request.runtime_root / "bin/python",
                device=device,
                dtype=dtype,
                primary_asr_profile="sensevoice" if backend == "cpu" else "nano",
                fallback_asr_profile=None if backend == "cpu" else "sensevoice",
                enhanced_enabled=backend != "cpu",
                speaker_enabled=backend != "cpu",
                model_revisions=revisions,
                probe_status="pass",
                selected_at=max(1, int(time.time())),
            )
            inference = InferenceConfig(device=device, dtype=dtype)
            loader = sensevoice_loader if backend == "cpu" else nano_loader
            if loader is None:
                return _failure(
                    backend,
                    "import",
                    dtype=dtype,
                    models=revisions,
                    tensor_ms=tensor_ms,
                )
            runtime = loader(
                selection=selection, inference=inference, default_timeout=120.0
            )
            runtime.transcribe(str(sample_path), sample_rate=16000, timeout=120.0)
            asr_ms = _elapsed_ms(asr_started)
        except Exception:  # noqa: BLE001 - never serialize exception text
            return _failure(
                backend,
                "asr",
                dtype=dtype,
                models=revisions,
                tensor_ms=tensor_ms,
                asr_ms=_elapsed_ms(asr_started),
            )
        finally:
            if runtime is not None:
                with suppress(Exception):
                    runtime.close()
            if sample_path is not None:
                with suppress(OSError):
                    sample_path.unlink(missing_ok=True)

        return ProbeResult(
            backend=backend,
            status="pass",
            error_category=None,
            dtype=dtype,
            models=revisions,
            tensor_ms=tensor_ms,
            asr_ms=asr_ms,
        )
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="probe an isolated ASR runtime")
    parser.add_argument("--backend", required=True, choices=("cuda", "xpu", "cpu"))
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--json", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with (
        Path(os.devnull).open("w", encoding="utf-8") as sink,
        redirect_stdout(sink),
    ):
        result = run_probe(
            ProbeRequest(
                backend=cast(Backend, args.backend),
                runtime_root=Path(sys.prefix),
                models_root=args.models_root,
                revision=args.revision,
            )
        )
    print(result.to_json())
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
