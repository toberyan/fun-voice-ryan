"""Intel XPU hard-gate preflight for Fun-ASR-Nano.

The desktop service may only be installed or started after every hard gate in
this module passes. Each check is pure and takes its dependencies (``torch``,
the loaded Nano engine, sample paths) as parameters so the whole gate is
unit-testable with fakes and never imports ``torch`` / ``vllm`` / ``funasr`` at
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

STATUS_PASS = "pass"
STATUS_FAIL = "fail"

EXPECTED_DEVICE_TYPE = "xpu"
DEVICE = "xpu:0"

# The nine hard gates, in canonical order.
CHECK_NAMES: tuple[str, ...] = (
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

# Decode token budgets (kept distinct so each check is independently failable
# and so the OOM recovery probe is a short, cheap request).
DECODE_TOKENS_10S = 128
DECODE_TOKENS_60S = 256
RECOVERY_TOKENS = 64
PROBE_BYTES = 1 << 20  # 1 MiB allocator probe

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
    """Best-effort extraction of the vLLM decoder device type."""
    vllm_engine = getattr(engine, "vllm_engine", None)
    for obj in (vllm_engine, engine):
        if obj is None:
            continue
        for attr in ("decoder_device_type", "device_type"):
            value = getattr(obj, attr, None)
            if isinstance(value, str):
                return _normalize_device_type(value)
        for holder in (obj, getattr(obj, "llm_engine", None)):
            if holder is None:
                continue
            model_config = getattr(holder, "model_config", None)
            device = getattr(model_config, "device", None)
            if device is not None:
                return _normalize_device_type(device)
    try:
        from vllm.platforms import current_platform

        return _normalize_device_type(getattr(current_platform, "device_type", None))
    except Exception:
        return None


def detect_cpu_fallback(engine: Any) -> str | None:
    """Return a reason string when the decoder fell back to CPU, else ``None``.

    vLLM 0.28.0 exposes no dedicated "CPU fallback" engine attribute, so only the
    resolved decoder device type is consulted (``get_decoder_device_type`` reads
    the engine's device_type / model_config.device / current_platform.device_type).
    Engine log keywords are not inspected here: they are written to the process
    logger and are not programmatically reachable from this call site.
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


def check_vllm_xpu_decoder(
    engine: Any, torch: Any, *, device: str = DEVICE
) -> CheckResult:
    """Verify the vLLM decoder runs on XPU and record device + memory metrics."""
    device_type = get_decoder_device_type(engine)
    detail: dict[str, Any] = {
        "expected": EXPECTED_DEVICE_TYPE,
        "decoder_device_type": device_type,
        "configured_device": device,
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
    ok = device_type == EXPECTED_DEVICE_TYPE and probe == "ok"
    return CheckResult(
        "vllm_xpu_decoder", STATUS_PASS if ok else STATUS_FAIL, detail
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


def check_decode(
    name: str, engine: Any, sample_path: str | Path, *, max_new_tokens: int
) -> CheckResult:
    try:
        results = engine.generate([str(sample_path)], max_new_tokens=max_new_tokens)
    except Exception as exc:
        return CheckResult(name, STATUS_FAIL, {"error_class": type(exc).__name__})
    if not results:
        return CheckResult(name, STATUS_FAIL, {"reason": "no results returned"})
    first = results[0]
    text = first.get("text", "") if isinstance(first, dict) else ""
    return CheckResult(
        name,
        STATUS_PASS if text else STATUS_FAIL,
        {"result_count": len(results), "text_length": len(text)},
    )


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
    engine: Any, torch: Any, short_sample: str | Path, *, device: str = DEVICE
) -> CheckResult:
    """Induce OOM, then prove the worker still serves a short decode.

    Never switches to CPU; a CPU fallback here is a failure.

    OOM is induced with a direct allocator probe (allocate past total device
    memory). The former "oversized decode request" probe was removed: vLLM
    0.28.0 does not reject max_tokens > max_model_len (it clamps instead), and
    a large max_tokens makes the V1 scheduler hang reserving KV cache blocks
    when it follows a long decode.
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

    # Prove the worker still serves a short decode.
    try:
        results = engine.generate([str(short_sample)], max_new_tokens=RECOVERY_TOKENS)
    except Exception as exc:
        return CheckResult(
            "oom_survives",
            STATUS_FAIL,
            {**detail, "recovery_error": type(exc).__name__},
        )
    first = results[0] if results else None
    text = first.get("text", "") if isinstance(first, dict) else ""
    detail["recovery_text_length"] = len(text)
    return CheckResult(
        "oom_survives", STATUS_PASS if text else STATUS_FAIL, detail
    )


def run_preflight(
    *,
    torch: Any,
    engine: Any,
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
        check_vllm_xpu_decoder(engine, torch, device=device),
        check_module_on_device("nano_encoder_xpu", engine.audio_encoder),
        check_module_on_device("nano_adaptor_xpu", engine.audio_adaptor),
        check_module_on_device("prompt_embeddings_xpu", engine.embed_tokens),
        check_decode(
            "decode_10s", engine, short_sample, max_new_tokens=DECODE_TOKENS_10S
        ),
        check_decode(
            "decode_60s", engine, long_sample, max_new_tokens=DECODE_TOKENS_60S
        ),
        check_no_cpu_fallback(engine),
        check_oom_survives(engine, torch, short_sample, device=device),
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
    dtype: str = "bf16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.35,
    max_model_len: int = 4096,
    enforce_eager: bool = True,
    attention_backend: str = "TRITON_ATTN",
) -> Any:
    """Load Fun-ASR-Nano via FunASR's official vLLM calling convention on XPU.

    ``attention_backend`` defaults to ``TRITON_ATTN``: vllm-xpu-kernels 0.1.14.1
    ships CUTLASS FlashAttention kernels only for XE2/XE3 architectures, which
    raises ``Only XE2/XE3 cutlass kernel is supported currently`` on the Xe-LPG+
    iGPU (Arc 130T/140T, device_id 0x7D51). Triton attention runs on the Intel
    XPU triton backend (triton==3.7.2+xpu shim) and covers this device."""
    from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

    return FunASRNanoVLLM.from_pretrained(
        model=str(model_dir),
        hub="ms",
        device=device,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        vllm_kwargs={"attention_backend": attention_backend},
    )


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
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=4096)
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
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        report = run_preflight(
            torch=torch,
            engine=engine,
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
