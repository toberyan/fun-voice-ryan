"""Intel XPU hard-gate preflight for Fun-ASR-Nano.

The desktop service may only be installed or started after every hard gate in
this module passes. Each check is pure and takes its dependencies (``torch``,
the loaded Nano engine, sample paths) as parameters so the whole gate is
unit-testable with fakes and never imports ``torch`` / ``funasr`` at
module import time.

Privacy: check details and the JSON report carry only check names, statuses,
lengths, error classes, and device metrics. They never carry audio paths or
transcription text.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fun_voice.nano_runtime import (
    VAD_MAX_SINGLE_SEGMENT_TIME_MS,
    FsmnVadSegmenter,
    ModelOutputError,
    _load_audio_samples,
    _slice_windows,
    load_native_nano_engine,
)

STATUS_PASS = "pass"
STATUS_FAIL = "fail"

EXPECTED_DEVICE_TYPE = "xpu"
DEVICE = "xpu:0"
SAMPLE_RATE = 16000
NANO_BACKEND = "native_funasr_pytorch"


# The nine hard gates, in canonical order.
CHECK_NAMES: tuple[str, ...] = (
    "xpu_visible",
    "nano_decoder_xpu",
    "nano_encoder_xpu",
    "nano_adaptor_xpu",
    "prompt_embeddings_xpu",
    "decode_10s",
    "decode_60s",
    "no_cpu_decoder_fallback",
    "oom_survives",
)

# Decode token budgets (kept distinct so each check is independently failable
# and so the OOM recovery probe is a short, cheap request).
DECODE_TOKENS_10S = 128
DECODE_TOKENS_60S = 256
RECOVERY_TOKENS = 64
PROBE_BYTES = 1 << 20  # 1 MiB allocator probe

# Hard gate 5: a 60 s sample must VAD-segment into multiple speech segments
# (the POC sample inserts >=0.3 s silence between fragments to guarantee this).
MIN_SEGMENTS_60S = 2

# Worker health probe (only exercised with --require-live-worker).
WORKER_SOCKET_RELATIVE = "fun-voice-ryan/worker.sock"
WORKER_HEALTH_TIMEOUT_S = 5.0
WORKER_HEALTH_MAX_BYTES = 64 * 1024


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one hard-gate check (metrics only, no audio or text)."""

    name: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PreflightReport:
    """Aggregate of all hard-gate checks plus the overall ready verdict."""

    device: str
    checks: tuple[CheckResult, ...]
    ready: bool
    worker_health: CheckResult | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "device": self.device,
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
        }
        if self.worker_health is not None:
            data["worker_health"] = asdict(self.worker_health)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def _normalize_device_type(value: object) -> str | None:
    """Normalize a device specifier to its bare type (``"xpu"``, ``"cuda"`` …)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.split(":")[0].strip().lower() or None
    device_type = getattr(value, "type", None)
    if isinstance(device_type, str):
        return device_type.lower()
    return None


def get_decoder_device_type(engine: Any) -> str | None:
    """Best-effort extraction of the primary Nano decoder device type.

    Native FunASR adapters expose ``decoder_device_type`` directly. The active
    preflight deliberately has no vLLM inspection or fallback path.
    """
    return _normalize_device_type(getattr(engine, "decoder_device_type", None))


def detect_cpu_fallback(engine: Any) -> str | None:
    """Return a reason string when the decoder fell back to CPU, else ``None``.

    The native adapter proves all Nano parameters during loading, then exposes
    its decoder device type.  Only that resolved device is used here; process
    log keywords are deliberately not parsed.
    """
    if get_decoder_device_type(engine) == "cpu":
        return "decoder device type is cpu"
    return None


def _xpu_memory_stats(torch: Any) -> dict[str, int | None]:
    stats: dict[str, int | None] = {}
    for key in ("allocated", "reserved"):
        fn = getattr(torch.xpu, f"memory_{key}", None)
        if fn is None:
            stats[key] = None
            continue
        try:
            stats[key] = int(fn())
        except Exception:
            stats[key] = None
    return stats


def _total_memory(torch: Any) -> int | None:
    try:
        return int(torch.xpu.get_device_properties(0).total_memory)
    except Exception:
        return None


def _param_device_type(module: Any) -> str | None:
    try:
        device_type: object = next(module.parameters()).device.type
    except (StopIteration, AttributeError):
        return None
    return device_type if isinstance(device_type, str) else None


def check_xpu_visible(torch: Any) -> CheckResult:
    available = bool(torch.xpu.is_available())
    return CheckResult(
        "xpu_visible",
        STATUS_PASS if available else STATUS_FAIL,
        {"available": available},
    )


def check_nano_xpu_decoder(
    engine: Any, torch: Any, *, device: str = DEVICE
) -> CheckResult:
    """Verify the native Nano decoder runs on XPU and record XPU metrics.

    ``detail.backend`` makes a stale proof from any former backend fail closed.
    """
    device_type = get_decoder_device_type(engine)
    detail: dict[str, Any] = {
        "expected": EXPECTED_DEVICE_TYPE,
        "decoder_device_type": device_type,
        "configured_device": device,
        "backend": getattr(engine, "backend", None),
    }
    detail["memory_before"] = _xpu_memory_stats(torch)
    probe = "ok"
    tensor = None
    try:
        tensor = torch.empty(PROBE_BYTES, dtype=torch.uint8, device=device)
    except Exception as exc:
        probe = f"failed:{type(exc).__name__}"
    detail["memory_after"] = _xpu_memory_stats(torch)
    del tensor
    detail["alloc_probe"] = probe
    detail["total_memory"] = _total_memory(torch)
    ok = (
        device_type == EXPECTED_DEVICE_TYPE
        and probe == "ok"
        and detail["backend"] == NANO_BACKEND
    )
    return CheckResult(
        "nano_decoder_xpu", STATUS_PASS if ok else STATUS_FAIL, detail
    )


def check_module_on_device(
    name: str, module: Any, *, expected: str = EXPECTED_DEVICE_TYPE
) -> CheckResult:
    actual = _param_device_type(module)
    return CheckResult(
        name,
        STATUS_PASS if actual == expected else STATUS_FAIL,
        {"actual": actual, "expected": expected},
    )


def _transcribe_segmented(
    engine: Any,
    vad: Any,
    samples: np.ndarray,
    *,
    max_new_tokens: int,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[str, list[tuple[int, int]]]:
    """VAD-segment → time-sort → overlap-slice → per-segment ASR → concat.

    Mirrors ``nano_runtime.NanoRuntime._transcribe_impl``: segments are sorted
    by onset, each is sliced with the same fixed overlap (``VAD_OVERLAP_MS``),
    fed to the Nano engine as one batch, and the per-segment texts are joined
    verbatim in segment order. Returns ``(text, sorted_regions)``; the regions
    keep the raw VAD ``(start_ms, end_ms)`` boundaries (overlap is applied to
    audio only, never to the reported times).
    """
    regions: list[tuple[int, int]] = vad.detect(samples, sample_rate)
    regions = sorted(regions, key=lambda region: region[0])
    if not regions:
        return "", regions
    windows = _slice_windows(regions, len(samples), sample_rate)
    slices = [samples[start:end] for start, end in windows]
    results = engine.generate(slices, max_new_tokens=max_new_tokens)
    if not isinstance(results, list) or len(results) != len(slices):
        raise ModelOutputError("model result count does not match VAD segments")
    texts: list[str] = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ModelOutputError("malformed model result")
        texts.append(result["text"])
    return "".join(texts), regions


def check_decode(
    name: str,
    engine: Any,
    vad: Any,
    sample_path: str | Path,
    *,
    max_new_tokens: int,
    min_segments: int = 1,
    sample_rate: int = SAMPLE_RATE,
) -> CheckResult:
    """Decode one sample through the real VAD-segmented path (hard gates 6/7).

    Privacy: only ``segment_count`` and ``text_length`` are reported, never the
    transcription text or the audio path.
    """
    try:
        samples = _load_audio_samples(str(sample_path), sample_rate)
        text, regions = _transcribe_segmented(
            engine,
            vad,
            samples,
            max_new_tokens=max_new_tokens,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        return CheckResult(name, STATUS_FAIL, {"error_class": type(exc).__name__})
    detail: dict[str, Any] = {
        "segment_count": len(regions),
        "text_length": len(text),
    }
    ok = bool(text) and len(regions) >= min_segments
    if not ok and len(regions) < min_segments:
        detail["min_segments"] = min_segments
    return CheckResult(name, STATUS_PASS if ok else STATUS_FAIL, detail)


def check_no_cpu_fallback(engine: Any) -> CheckResult:
    reason = detect_cpu_fallback(engine)
    return CheckResult(
        "no_cpu_decoder_fallback",
        STATUS_FAIL if reason else STATUS_PASS,
        {
            "cpu_fallback_reason": reason,
            "decoder_device_type": get_decoder_device_type(engine),
        },
    )


def check_worker_health(response: Mapping[str, Any]) -> CheckResult:
    """Convert a worker health response into a pass/fail check (pure)."""
    detail: dict[str, Any] = {
        "status": response.get("status"),
        "version": response.get("version"),
        "model_ready": response.get("model_ready"),
        "xpu_ready": response.get("xpu_ready"),
        "device": response.get("device"),
        "last_error": response.get("last_error"),
    }
    ok = (
        response.get("status") == "ok"
        and response.get("model_ready") is True
        and response.get("xpu_ready") is True
    )
    return CheckResult("worker_health", STATUS_PASS if ok else STATUS_FAIL, detail)


def _read_socket_line(conn: socket.socket) -> bytes:
    """Read one newline-terminated line from ``conn`` (bounded)."""
    buf = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return bytes(buf)
        buf.extend(chunk)
        if len(buf) > WORKER_HEALTH_MAX_BYTES:
            raise RuntimeError("worker health response too large")
        if b"\n" in buf:
            line, _sep, _rest = buf.partition(b"\n")
            return bytes(line)


def probe_worker_health(
    socket_path: str | Path | None = None,
    *,
    timeout: float = WORKER_HEALTH_TIMEOUT_S,
) -> CheckResult:
    """Probe the worker Unix socket ``op=health`` endpoint; never raises.

    ``socket_path`` defaults to ``$XDG_RUNTIME_DIR/fun-voice-ryan/worker.sock``.
    Connection, framing, and protocol failures become a fail ``CheckResult``.
    """
    if socket_path is None:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if not xdg:
            return CheckResult(
                "worker_health", STATUS_FAIL, {"reason": "XDG_RUNTIME_DIR not set"}
            )
        socket_path = Path(xdg) / WORKER_SOCKET_RELATIVE
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(socket_path))
            conn.sendall(b'{"op":"health"}\n')
            line = _read_socket_line(conn)
        response = json.loads(line.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "worker_health", STATUS_FAIL, {"error_class": type(exc).__name__}
        )
    if not isinstance(response, dict):
        return CheckResult(
            "worker_health", STATUS_FAIL, {"error_class": "ProtocolError"}
        )
    return check_worker_health(response)


def _attempt_oom_allocations(
    torch: Any, device: str, total: int | None
) -> tuple[int | None, str | None]:
    """Allocate past total device memory until an OOM-class error is raised."""
    sizes: list[int] = []
    if total:
        sizes.extend([total, total * 2, total * 4])
    sizes.append(1 << 36)  # 64 GiB hard cap, always exceeds any local VRAM
    for size in sizes:
        if size <= 0:
            continue
        try:
            tensor = torch.empty(size, dtype=torch.uint8, device=device)
            del tensor
        except Exception as exc:
            return size, type(exc).__name__
    return None, None


def check_oom_survives(
    engine: Any,
    vad: Any,
    torch: Any,
    short_sample: str | Path,
    *,
    device: str = DEVICE,
) -> CheckResult:
    """Induce OOM, then prove the worker still serves a short decode.

    Never switches to CPU; a CPU fallback here is a failure.

    OOM is induced with a direct allocator probe (allocate past total device
    memory), rather than trying to manufacture a pathological decode request.
    This keeps the recovery test backend-independent and bounded.
    """
    detail: dict[str, Any] = {"total_memory": _total_memory(torch)}

    # Direct allocator OOM: allocate beyond total device memory.
    size, error = _attempt_oom_allocations(torch, device, _total_memory(torch))
    if error is not None:
        detail["allocator_oom"] = error
        detail["allocator_oom_bytes"] = size
    else:
        return CheckResult(
            "oom_survives", STATUS_FAIL, {**detail, "reason": "no OOM induced"}
        )

    # Prove the worker still serves a short decode through the same in-memory
    # VAD-segmented path as native Nano. Passing an audio pathname straight to
    # the engine was only valid for the retired vLLM adapter.
    try:
        samples = _load_audio_samples(str(short_sample), SAMPLE_RATE)
        text, regions = _transcribe_segmented(
            engine, vad, samples, max_new_tokens=RECOVERY_TOKENS
        )
    except Exception as exc:
        return CheckResult(
            "oom_survives",
            STATUS_FAIL,
            {**detail, "recovery_error": type(exc).__name__},
        )
    detail["recovery_text_length"] = len(text)
    detail["recovery_segment_count"] = len(regions)
    return CheckResult(
        "oom_survives", STATUS_PASS if text else STATUS_FAIL, detail
    )


def run_preflight(
    *,
    torch: Any,
    engine: Any,
    vad: Any,
    short_sample: str | Path,
    long_sample: str | Path,
    device: str = DEVICE,
    worker_health: CheckResult | None = None,
) -> PreflightReport:
    """Run all nine hard-gate checks and return the aggregate report.

    ``worker_health`` (only set under ``--require-live-worker``) is an
    independent extra check: it does not join the nine hard gates but does
    gate ``ready``.
    """
    checks: list[CheckResult] = [
        check_xpu_visible(torch),
        check_nano_xpu_decoder(engine, torch, device=device),
        check_module_on_device("nano_encoder_xpu", engine.audio_encoder),
        check_module_on_device("nano_adaptor_xpu", engine.audio_adaptor),
        check_module_on_device("prompt_embeddings_xpu", engine.embed_tokens),
        check_decode(
            "decode_10s", engine, vad, short_sample, max_new_tokens=DECODE_TOKENS_10S
        ),
        check_decode(
            "decode_60s",
            engine,
            vad,
            long_sample,
            max_new_tokens=DECODE_TOKENS_60S,
            min_segments=MIN_SEGMENTS_60S,
        ),
        check_no_cpu_fallback(engine),
        check_oom_survives(engine, vad, torch, short_sample, device=device),
    ]
    ready = all(check.status == STATUS_PASS for check in checks)
    if worker_health is not None:
        ready = ready and worker_health.status == STATUS_PASS
    return PreflightReport(
        device=device,
        checks=tuple(checks),
        ready=ready,
        worker_health=worker_health,
    )


def load_nano_engine(
    model_dir: str | Path,
    *,
    device: str = DEVICE,
) -> Any:
    """Load the worker's native FunASR/PyTorch Nano backend on XPU only."""
    return load_native_nano_engine(device, model_dir=str(model_dir))


def load_vad(device: str = DEVICE) -> FsmnVadSegmenter:
    """Load the local FSMN-VAD once on the requested XPU device.

    Points ModelScope at the same local model cache that
    ``scripts/run-nano-xpu-poc.sh`` populates, then wraps the model in the
    runtime's :class:`~fun_voice.nano_runtime.FsmnVadSegmenter` adapter.
    """
    from fun_voice.nano_runtime import models_root

    os.environ.setdefault("MODELSCOPE_CACHE", str(models_root()))
    from funasr import AutoModel

    vad_snapshot = (
        models_root()
        / "models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/master"
    )
    model = AutoModel(
        model=str(vad_snapshot),
        device=device,
        disable_update=True,
        max_single_segment_time=VAD_MAX_SINGLE_SEGMENT_TIME_MS,
    )
    return FsmnVadSegmenter(model)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fun-voice-preflight",
        description="Run the Fun-ASR-Nano XPU hard-gate preflight.",
    )
    parser.add_argument("--short", required=True, help="10s 16 kHz mono WAV sample")
    parser.add_argument("--long", required=True, help="60s 16 kHz mono WAV sample")
    parser.add_argument(
        "--model-dir", required=True, help="Fun-ASR-Nano local model directory"
    )
    parser.add_argument("--report", required=True, help="JSON report output path")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--require-live-worker",
        action="store_true",
        help="also probe the live worker socket and require a healthy response",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    worker_health: CheckResult | None = None
    if args.require_live_worker:
        worker_health = probe_worker_health()
    try:
        import torch

        engine = load_nano_engine(
            args.model_dir,
            device=args.device,
        )
        vad = load_vad(device=args.device)
        report = run_preflight(
            torch=torch,
            engine=engine,
            vad=vad,
            short_sample=args.short,
            long_sample=args.long,
            device=args.device,
            worker_health=worker_health,
        )
    except Exception as exc:  # noqa: BLE001
        # Turn any load/run failure into a structured fail report instead of a
        # bare traceback, so the POC harness always gets a parseable JSON.
        report = PreflightReport(
            device=args.device,
            checks=tuple(
                CheckResult(name, STATUS_FAIL, {"error": type(exc).__name__})
                for name in CHECK_NAMES
            ),
            ready=False,
            worker_health=worker_health,
        )
        print(f"preflight error: {type(exc).__name__}: {exc}", file=sys.stderr)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_json() + "\n", encoding="utf-8")

    for check in report.checks:
        print(f"{check.name}: {check.status}")
    if report.worker_health is not None:
        print(f"{report.worker_health.name}: {report.worker_health.status}")
    return 0 if report.ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
