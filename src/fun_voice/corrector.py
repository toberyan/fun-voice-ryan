"""One-request Qwen3.5-0.8B text correction on Intel XPU.

The daemon imports this module without importing a model runtime or loading a model.
``OnDemandQwenCorrector.correct`` starts this module as a child process for one
piece of text, reads its single JSON response, and lets the child exit.  This
keeps Qwen weights and its KV cache out of the login-resident daemon and out of
the ASR worker process.

Only the fixed local ``Qwen/Qwen3.5-0.8B`` snapshot is supported.  Any model
load, generation, protocol, or validation error is exposed to the caller so
the desktop pipeline can safely retain the ASR raw text.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from fun_voice import config
from fun_voice.contracts import CORRECTION_REJECTION_REASONS, CorrectionTiming

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEVICE = "xpu:0"
MAX_OUTPUT_CHARACTERS = 1536
DEFAULT_TIMEOUT_SECONDS = 30.0
_OPEN = "[[FINAL]]"
_CLOSE = "[[/FINAL]]"
_SYSTEM_PROMPT = """你是本地语音输入的文本校对器。保留原意和语言顺序，只修正明确的同音、
标点、空格与常见计算机术语错误。不要解释、不要总结、不要添加事实；英文、命令、
代码、路径和版本号在没有明确错误时必须原样保留。

只输出以下包裹格式，包裹外不能有任何字符：
[[FINAL]]修正后的完整文本[[/FINAL]]"""
_PROTECTED_PATTERNS = (
    re.compile(r"`[^`\n]+`"),
    re.compile(r"https?://[^\s<>()\[\]{}\"'，。！？；：]+"),
    re.compile(r"(?<!\w)(?:~?/|\.{1,2}/)[^\s<>|;&，。！？；：]+"),
    re.compile(r"(?<!\w)--?[A-Za-z][A-Za-z0-9-]*"),
    re.compile(r"\b(?:v?\d+(?:\.\d+){1,})\b"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*\b"),
    re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)+\b"),
    re.compile(
        r"\b(?:git|pytest|python|pip|uv|npm|pnpm|docker|kubectl|systemctl|"
        r"journalctl|grep|rg|bash|zsh)\b"
    ),
)

_DEFAULT_REJECTION_REASON = {
    "correction.input_too_large": "input_too_large",
    "correction.model_load": "model_load",
    "correction.oom": "oom",
    "correction.device": "device",
    "correction.protocol": "protocol",
    "correction.no_output": "no_output",
    "correction.generation": "generation",
    "correction.timeout": "timeout",
    "correction.unavailable": "unavailable",
    "correction.internal": "internal",
}


class CorrectionError(RuntimeError):
    """A correction error represented by a stable, non-text-bearing code."""

    def __init__(
        self,
        code: str,
        *,
        reason: str | None = None,
        timing: CorrectionTiming | None = None,
    ) -> None:
        self.code = code
        self.reason = (
            reason
            if reason in CORRECTION_REJECTION_REASONS
            else _DEFAULT_REJECTION_REASON.get(code, "internal")
        )
        self.timing = timing
        super().__init__(code)


@dataclass(frozen=True)
class GeneratedCorrection:
    """Verified child-model output plus its private stage durations."""

    text: str
    timing: CorrectionTiming


Runner = Callable[[Sequence[str], str, float], str]


def qwen_snapshot_dir() -> Path:
    """Return the required application-private Qwen snapshot directory."""
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return (
        root
        / "fun-voice-ryan"
        / "models"
        / "models"
        / "Qwen--Qwen3.5-0.8B"
        / "snapshots"
        / "master"
    )


def build_prompt(raw_text: str) -> str:
    """Build the bounded text-only correction request.

    The envelope is an output protocol, not user-visible content.  It forces
    the model to return only a candidate final text, which the parent verifies
    before committing it to Fcitx or the clipboard.
    """
    return "原始转写：\n" + raw_text


def parse_correction_output(output: str) -> str:
    """Accept a clean final envelope from the small local model.

    Qwen3.5-0.8B consistently produces the opening sentinel but may omit the
    closing sentinel before EOS.  A clean one-sided envelope is accepted only
    when it has no nested marker; later edit-density validation still decides
    whether it can replace the ASR text.
    """
    if not output.startswith(_OPEN):
        raise CorrectionError("correction.invalid_output", reason="envelope_missing")
    text = output[len(_OPEN) :]
    if text.endswith(_CLOSE):
        text = text[: -len(_CLOSE)]
    if _OPEN in text or _CLOSE in text:
        raise CorrectionError("correction.invalid_output", reason="envelope_malformed")
    text = text.strip()
    if not text:
        raise CorrectionError("correction.invalid_output", reason="output_empty")
    if len(text) > MAX_OUTPUT_CHARACTERS:
        raise CorrectionError("correction.invalid_output", reason="output_too_long")
    return text


def extract_protected_tokens(
    raw_text: str, configured_terms: Sequence[str] = ()
) -> tuple[str, ...]:
    """Return ordered, non-overlapping technical spans from the raw text."""
    matches: list[tuple[int, int, str]] = []
    for pattern in _PROTECTED_PATTERNS:
        matches.extend(
            (match.start(), match.end(), match.group(0))
            for match in pattern.finditer(raw_text)
        )
    for term in configured_terms:
        if not isinstance(term, str) or not term:
            continue
        start = raw_text.find(term)
        while start != -1:
            matches.append((start, start + len(term), term))
            start = raw_text.find(term, start + len(term))

    selected: list[str] = []
    last_end = -1
    for start, end, token in sorted(matches, key=lambda item: (item[0], -item[1])):
        if start < last_end or token in selected:
            continue
        selected.append(token)
        last_end = end
    return tuple(selected)


def validate_correction(
    raw_text: str,
    corrected_text: str,
    protected_terms: Sequence[str] = (),
) -> str:
    """Reject candidates that are too dissimilar to safely auto-commit."""
    if len(corrected_text) > max(32, len(raw_text) * 2):
        raise CorrectionError("correction.invalid_output", reason="output_too_long")
    if SequenceMatcher(None, raw_text, corrected_text).ratio() < 0.60:
        raise CorrectionError("correction.invalid_output", reason="similarity")
    candidate_offset = 0
    for token in extract_protected_tokens(raw_text, protected_terms):
        index = corrected_text.find(token, candidate_offset)
        if index < 0:
            raise CorrectionError("correction.invalid_output", reason="protected_token")
        candidate_offset = index + len(token)
    return corrected_text


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or "outofmemory" in (
        type(exc).__name__.lower()
    )


def generate_enveloped_correction(
    raw_text: str, *, inference: config.EnhancedInferenceConfig
) -> GeneratedCorrection:
    """Load Qwen on XPU, generate one envelope, then release the model.

    vLLM 0.28's Qwen3.5 hybrid-attention XPU path emits corrupt text on the
    Arc Pro 130T.  This dedicated Qwen process therefore uses Transformers'
    native XPU path.  It creates only the request-local generation cache (no
    vLLM KV pool), and process exit remains the release guarantee.
    """
    if not raw_text or len(raw_text) > inference.correction_max_source_characters:
        raise CorrectionError("correction.input_too_large")
    if inference.correction_model != MODEL_ID:
        raise CorrectionError("correction.model_load")
    model: Any | None = None
    model_load_started = time.perf_counter()
    try:
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        loaded_model: Any = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(qwen_snapshot_dir()), torch_dtype=torch.bfloat16
        )
        model = loaded_model.to(DEVICE)
        model.eval()
        processor: Any = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(qwen_snapshot_dir())
        )
    except Exception as exc:  # noqa: BLE001 - model details are not logged
        raise CorrectionError(
            "correction.oom" if _is_oom(exc) else "correction.model_load",
            timing=CorrectionTiming(model_load_ms=_elapsed_ms(model_load_started)),
        ) from exc

    model_load_ms = _elapsed_ms(model_load_started)
    generate_started = time.perf_counter()
    validate_started: float | None = None
    try:
        devices = {parameter.device.type for parameter in model.parameters()}
        if devices != {"xpu"}:
            raise CorrectionError("correction.device")
        messages: list[Any] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(raw_text)},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        model_inputs = {
            key: value.to(DEVICE) for key, value in inputs.items()
        }
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise CorrectionError("correction.protocol")
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=inference.correction_max_new_tokens,
            do_sample=False,
        )
        output_ids = generated_ids[:, input_ids.shape[1] :]
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        if not decoded or not isinstance(decoded[0], str):
            raise CorrectionError("correction.no_output")
        output = decoded[0]
        generate_ms = _elapsed_ms(generate_started)
        # Validate in the model-owning process as well.  The parent repeats the
        # validation because it treats the child output as an untrusted IPC.
        validate_started = time.perf_counter()
        validate_correction(
            raw_text,
            parse_correction_output(output),
            inference.correction_protected_terms,
        )
        return GeneratedCorrection(
            text=output,
            timing=CorrectionTiming(
                model_load_ms=model_load_ms,
                generate_ms=generate_ms,
                validate_ms=_elapsed_ms(validate_started),
            ),
        )
    except CorrectionError as exc:
        raise _with_timing(
            exc,
            model_load_ms=model_load_ms,
            generate_ms=_elapsed_ms(generate_started),
            validate_ms=_elapsed_ms(validate_started),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - normalise vLLM failures
        raise CorrectionError(
            "correction.oom" if _is_oom(exc) else "correction.generation",
            timing=CorrectionTiming(
                model_load_ms=model_load_ms,
                generate_ms=_elapsed_ms(generate_started),
            ),
        ) from exc
    finally:
        del model
        gc.collect()
        try:
            import torch

            torch.xpu.empty_cache()
        except Exception:
            pass


def _elapsed_ms(started: float | None) -> int | None:
    """Return a non-negative monotonic duration, or ``None`` for an unstarted stage."""
    if started is None:
        return None
    return max(0, round((time.perf_counter() - started) * 1000))


def _with_timing(
    exc: CorrectionError,
    *,
    model_load_ms: int | None,
    generate_ms: int | None,
    validate_ms: int | None = None,
) -> CorrectionError:
    """Attach only scalar stage durations while retaining the stable error code."""
    return CorrectionError(
        exc.code,
        reason=exc.reason,
        timing=CorrectionTiming(
            model_load_ms=model_load_ms,
            generate_ms=generate_ms,
            validate_ms=validate_ms,
        ),
    )


def _timing_from_payload(payload: object) -> CorrectionTiming | None:
    """Decode only known non-negative duration fields from child JSON."""
    if not isinstance(payload, dict):
        return None

    def duration(name: str) -> int | None:
        value = payload.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    timing = CorrectionTiming(
        model_load_ms=duration("model_load_ms"),
        generate_ms=duration("generate_ms"),
        validate_ms=duration("validate_ms"),
    )
    if all(
        value is None
        for value in (
            timing.model_load_ms,
            timing.generate_ms,
            timing.validate_ms,
        )
    ):
        return None
    return timing


def _timing_payload(timing: CorrectionTiming | None) -> dict[str, int | None]:
    """Serialize scalar child timings without exposing text or model internals."""
    if timing is None:
        return {}
    return {
        "model_load_ms": timing.model_load_ms,
        "generate_ms": timing.generate_ms,
        "validate_ms": timing.validate_ms,
    }


def _default_runner(command: Sequence[str], request: str, timeout: float) -> str:
    """Run the isolated corrector without exposing request text to a shell."""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise CorrectionError("correction.unavailable") from exc
    try:
        stdout, _stderr = process.communicate(request, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Model runtimes may create child processes.  A regular terminate could
        # leave one holding XPU memory, so stop the dedicated process group.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise CorrectionError("correction.timeout") from exc
    if not stdout:
        raise CorrectionError("correction.unavailable")
    return stdout


class OnDemandQwenCorrector:
    """Start exactly one Qwen process for each requested correction."""

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        inference: config.EnhancedInferenceConfig | None = None,
        timeout_seconds: float | None = None,
        runner: Runner = _default_runner,
    ) -> None:
        self._inference = config.validate_enhanced_inference_config(
            inference if inference is not None else config.EnhancedInferenceConfig()
        )
        self._command = tuple(command or (sys.executable, "-m", "fun_voice.corrector"))
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(self._inference.correction_timeout_seconds)
        )
        self._runner = runner
        self.last_timing: CorrectionTiming | None = None

    def correct(self, raw_text: str) -> str:
        self.last_timing = None
        if (
            not raw_text
            or len(raw_text) > self._inference.correction_max_source_characters
        ):
            raise CorrectionError("correction.input_too_large")
        request = json.dumps({"text": raw_text}, ensure_ascii=False)
        response = self._runner(self._command, request, self._timeout_seconds)
        try:
            # vLLM emits startup diagnostics before the corrector's final
            # response on some XPU releases.  The final non-empty line is our
            # own JSON frame; no generated text is accepted outside that frame.
            decoded: Any = json.loads(response.rstrip().rpartition("\n")[2])
        except (TypeError, json.JSONDecodeError) as exc:
            raise CorrectionError("correction.protocol") from exc
        if not isinstance(decoded, dict):
            raise CorrectionError("correction.protocol")
        timing = _timing_from_payload(decoded.get("timing_ms"))
        self.last_timing = timing
        if decoded.get("status") != "ok":
            code = decoded.get("error_code")
            raise CorrectionError(
                code if isinstance(code, str) else "correction.failed",
                reason=(
                    decoded.get("error_reason")
                    if isinstance(decoded.get("error_reason"), str)
                    else None
                ),
                timing=timing,
            )
        output = decoded.get("text")
        if not isinstance(output, str):
            raise CorrectionError("correction.protocol", timing=timing)
        try:
            return validate_correction(
                raw_text,
                parse_correction_output(output),
                self._inference.correction_protected_terms,
            )
        except CorrectionError as exc:
            raise CorrectionError(
                exc.code,
                reason=exc.reason,
                timing=timing,
            ) from exc

    def close(self) -> None:
        """No process is retained between calls."""


def _read_request() -> str:
    try:
        decoded: Any = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise CorrectionError("correction.protocol") from exc
    text = decoded.get("text") if isinstance(decoded, dict) else None
    if not isinstance(text, str):
        raise CorrectionError("correction.protocol")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    """Run a single stdin request and write one JSON result to stdout."""
    parser = argparse.ArgumentParser(prog="fun-voice-corrector")
    parser.parse_args(argv)
    try:
        inference = config.load_config().enhanced
        result = generate_enveloped_correction(_read_request(), inference=inference)
    except CorrectionError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": exc.code,
                    "error_reason": exc.reason,
                    "timing_ms": _timing_payload(exc.timing),
                }
            )
        )
        return 1
    except Exception:  # noqa: BLE001 - do not expose local details or text
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "correction.internal",
                    "error_reason": "internal",
                    "timing_ms": {},
                }
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "text": result.text,
                "timing_ms": _timing_payload(result.timing),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through child IPC.
    raise SystemExit(main())
