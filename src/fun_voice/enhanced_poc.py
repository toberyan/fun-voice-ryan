"""Run the enhanced local-voice XPU proof of capability.

This module has a file-backed ``__main__`` entry point so the shell harness can
own model download and privacy-preserving runtime paths; this module only loads
local snapshots and writes the final, metrics-only success report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fun_voice.preflight import load_nano_engine

DEVICE = "xpu:0"


@dataclass(frozen=True, slots=True)
class Gate:
    """One metrics-only XPU gate outcome."""

    name: str
    status: str
    detail: dict[str, object]


@dataclass(frozen=True, slots=True)
class PocInputs:
    """Local snapshots and destination owned by the POC shell harness."""

    report_path: Path
    nano_dir: Path
    sensevoice_dir: Path
    vad_dir: Path
    camplus_dir: Path
    qwen_dir: Path
    revision: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_fingerprint(path: Path) -> str:
    metadata = sorted(path.glob("config.*"))
    if not metadata:
        raise RuntimeError("model snapshot lacks config metadata")
    return _sha256(metadata[0])


def _all_parameter_devices(module: Any, name: str) -> list[str]:
    candidates = [module]
    for attribute in ("model", "model_model", "network"):
        child = getattr(module, attribute, None)
        if child is not None:
            candidates.append(child)
    for candidate in candidates:
        parameters = getattr(candidate, "parameters", None)
        if not callable(parameters):
            continue
        devices = sorted({str(parameter.device.type) for parameter in parameters()})
        if devices:
            if devices != ["xpu"]:
                raise RuntimeError(f"{name} parameters are not all on XPU")
            return devices
    raise RuntimeError(f"{name} exposes no inspectable parameters")


def _module_gate(name: str, factory: Callable[[], Any]) -> Gate:
    module = factory()
    devices = _all_parameter_devices(module, name)
    return Gate(name, "pass", {"parameter_devices": devices})


def _ctc_primitives_gate(torch: Any) -> Gate:
    log_probs = torch.log_softmax(
        torch.randn((8, 4), device=DEVICE, dtype=torch.float32), dim=-1
    )
    targets = torch.tensor([1, 2], device=DEVICE, dtype=torch.long)
    table = torch.full((8, 5), float("-inf"), device=DEVICE)
    table[0, 0] = log_probs[0, 0]
    table[1:, 0] = torch.cumsum(log_probs[1:, 0], dim=0) + table[0, 0]
    _ = torch.max(table[:, :3], dim=1).values + targets.sum().to(dtype=table.dtype)
    if log_probs.device.type != "xpu" or targets.device.type != "xpu":
        raise RuntimeError("CTC tensors are not on XPU")
    return Gate("xpu_ctc_primitives", "pass", {"device": "xpu"})


def _qwen_gates(torch: Any, qwen_dir: Path) -> tuple[Gate, Gate, Gate]:
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # Qwen3.5's current vLLM XPU hybrid-attention path is numerically invalid
    # on Arc Pro 130T.  The corrector and this isolated gate use the native
    # Transformers XPU path, which creates only a request-local cache.
    loaded_model: Any = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(qwen_dir), torch_dtype=torch.bfloat16
    )
    model = loaded_model.to(DEVICE)
    model.eval()
    if _all_parameter_devices(model, "Qwen") != ["xpu"]:
        raise RuntimeError("Qwen parameters are not on XPU")
    processor: Any = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        str(qwen_dir)
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": "Reply: ."}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    model_inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    input_ids = model_inputs["input_ids"]
    memory_before = int(torch.xpu.memory_allocated())
    generated_ids = model.generate(**model_inputs, max_new_tokens=4, do_sample=False)
    output_length = len(
        processor.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
    )
    if output_length <= 0:
        raise RuntimeError("Qwen smoke generation returned no content")
    memory_after = int(torch.xpu.memory_allocated())
    return (
        Gate("qwen35_xpu", "pass", {"parameter_devices": ["xpu"]}),
        Gate(
            "qwen35_text_only",
            "pass",
            {"backend": "transformers_xpu", "request_local_kv": True},
        ),
        Gate(
            "qwen35_smoke",
            "pass",
            {
                "output_length": output_length,
                "memory_before": memory_before,
                "memory_after": memory_after,
            },
        ),
    )


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run(inputs: PocInputs) -> None:
    """Run every local XPU gate and atomically publish the success report."""
    import torch
    from funasr import AutoModel

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is unavailable")

    gates: list[Gate] = [
        Gate("xpu_visible", "pass", {"device_count": int(torch.xpu.device_count())}),
        _module_gate(
            "xpu_vad",
            lambda: AutoModel(
                model=str(inputs.vad_dir), device=DEVICE, disable_update=True
            ),
        ),
        _module_gate(
            "sensevoice_xpu",
            lambda: AutoModel(
                model=str(inputs.sensevoice_dir),
                vad_model=str(inputs.vad_dir),
                device=DEVICE,
                disable_update=True,
            ),
        ),
    ]
    nano = load_nano_engine(
        inputs.nano_dir,
        device=DEVICE,
    )
    if getattr(nano, "backend", None) != "native_funasr_pytorch":
        raise RuntimeError("Nano did not use the native FunASR/PyTorch backend")
    if getattr(nano, "decoder_device_type", None) != "xpu":
        raise RuntimeError("Nano decoder is not on XPU")
    gates.append(
        Gate(
            "nano_native_decoder_xpu",
            "pass",
            {"backend": nano.backend, "decoder_device_type": nano.decoder_device_type},
        )
    )
    for component_name in ("audio_encoder", "audio_adaptor", "embed_tokens"):
        component = getattr(nano, component_name, None)
        if component is None:
            raise RuntimeError("Nano component is unavailable")
        devices = _all_parameter_devices(component, f"Nano {component_name}")
        gates.append(
            Gate(
                f"nano_{component_name}_xpu",
                "pass",
                {"parameter_devices": devices},
            )
        )
    gates.append(
        _module_gate(
            "camplus_xpu",
            lambda: AutoModel(
                model=str(inputs.camplus_dir), device=DEVICE, disable_update=True
            ),
        )
    )
    # Drop native Nano references and release cached XPU blocks before creating
    # the independent Qwen engine, so this POC never intentionally holds both
    # large decoders.
    del nano
    gc.collect()
    torch.xpu.empty_cache()
    gates.append(_ctc_primitives_gate(torch))
    gates.extend(_qwen_gates(torch, inputs.qwen_dir))

    report = {
        "ready": True,
        "device": DEVICE,
        "model_revisions": {
            "nano": inputs.revision,
            "sensevoice": inputs.revision,
            "vad": inputs.revision,
            "camplus": inputs.revision,
            "qwen35": inputs.revision,
        },
        "snapshot_config_sha256": {
            "nano": _snapshot_fingerprint(inputs.nano_dir),
            "sensevoice": _snapshot_fingerprint(inputs.sensevoice_dir),
            "vad": _snapshot_fingerprint(inputs.vad_dir),
            "camplus": _snapshot_fingerprint(inputs.camplus_dir),
            "qwen35": _snapshot_fingerprint(inputs.qwen_dir),
        },
        "packages": {
            name: _version(name) for name in ("torch", "funasr", "modelscope")
        },
        "memory": {
            "allocated": int(torch.xpu.memory_allocated()),
            "reserved": int(torch.xpu.memory_reserved()),
            "total": int(torch.xpu.get_device_properties(0).total_memory),
        },
        "checks": [
            {"name": gate.name, "status": gate.status, "detail": gate.detail}
            for gate in gates
        ],
    }
    temporary = inputs.report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(inputs.report_path)
    print("[run-enhanced-xpu-poc] all enhanced XPU gates passed")


def _parse_args(argv: Sequence[str] | None = None) -> PocInputs:
    parser = argparse.ArgumentParser(prog="fun-voice-enhanced-poc")
    parser.add_argument("--report", required=True)
    parser.add_argument("--nano-dir", required=True)
    parser.add_argument("--sensevoice-dir", required=True)
    parser.add_argument("--vad-dir", required=True)
    parser.add_argument("--camplus-dir", required=True)
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    return PocInputs(
        report_path=Path(args.report),
        nano_dir=Path(args.nano_dir),
        sensevoice_dir=Path(args.sensevoice_dir),
        vad_dir=Path(args.vad_dir),
        camplus_dir=Path(args.camplus_dir),
        qwen_dir=Path(args.qwen_dir),
        revision=args.revision,
    )


def main(argv: Sequence[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the shell POC.
    raise SystemExit(main())
