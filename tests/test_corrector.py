"""Tests for the one-request Qwen correction process boundary."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fun_voice import config
from fun_voice import corrector as corrector_module
from fun_voice.corrector import (
    DEFAULT_TIMEOUT_SECONDS,
    CorrectionError,
    OnDemandQwenCorrector,
    generate_enveloped_correction,
    parse_correction_output,
    validate_correction,
)
from fun_voice.runtime_selection import RuntimeSelection


def _selection(backend: str = "xpu", *, dtype: str | None = None) -> RuntimeSelection:
    is_cpu = backend == "cpu"
    return RuntimeSelection(
        schema_version=1,
        backend=backend,  # type: ignore[arg-type]
        python=Path(f"/runtime/{backend}/bin/python"),
        device="cpu" if is_cpu else f"{backend}:0",
        dtype=dtype if dtype is not None else "float32" if is_cpu else "bf16",
        primary_asr_profile="sensevoice" if is_cpu else "nano",
        fallback_asr_profile=None if is_cpu else "sensevoice",
        enhanced_enabled=not is_cpu,
        speaker_enabled=not is_cpu,
        model_revisions=(
            {"sensevoice": "master", "vad": "master"}
            if is_cpu
            else {
                "nano": "master",
                "sensevoice": "master",
                "vad": "master",
                "qwen": "master",
                "campplus": "master",
            }
        ),
        probe_status="pass",
        selected_at=1,
    )


def _cpu_effective() -> config.EffectiveRuntimeConfig:
    return config.effective_runtime_config(config.Config(), _selection("cpu"))


def test_qwen_on_demand_timeout_has_a_safe_upper_bound() -> None:
    assert DEFAULT_TIMEOUT_SECONDS == 30.0


def test_parse_correction_output_accepts_only_the_final_envelope() -> None:
    output = "[[FINAL]]git commit，然后运行 pytest。[[/FINAL]]"
    assert parse_correction_output(output) == (
        "git commit，然后运行 pytest。"
    )


def test_parse_correction_output_accepts_clean_opening_only_envelope() -> None:
    assert parse_correction_output("[[FINAL]]git commit\n") == "git commit"


def test_missing_envelope_keeps_generic_error_code_with_fixed_reason() -> None:
    with pytest.raises(CorrectionError) as caught:
        parse_correction_output("git commit")

    assert caught.value.code == "correction.invalid_output"
    assert caught.value.reason == "envelope_missing"


def test_candidate_must_remain_similar_to_the_raw_asr_text() -> None:
    assert validate_correction("get commit", "git commit") == "git commit"

    with pytest.raises(CorrectionError, match="invalid_output") as caught:
        validate_correction("get commit", "完全无关的长文本")
    assert caught.value.reason == "similarity"


def test_changed_protected_command_is_rejected() -> None:
    with pytest.raises(CorrectionError, match="invalid_output") as caught:
        validate_correction("运行 git commit --amend", "运行 get commit --amend")
    assert caught.value.reason == "protected_token"


@pytest.mark.parametrize(
    "output",
    [
        "git commit",
        "说明：[[FINAL]]git commit[[/FINAL]]",
        "[[FINAL]][[/FINAL]]",
        "[[FINAL]]git commit[[/FINAL]]\n额外说明",
    ],
)
def test_parse_correction_output_rejects_untrusted_text(output: str) -> None:
    with pytest.raises(CorrectionError, match="invalid_output"):
        parse_correction_output(output)


def test_qwen_is_not_started_until_a_correction_is_requested() -> None:
    calls: list[tuple[Sequence[str], str, float]] = []

    def runner(command: Sequence[str], request: str, timeout: float) -> str:
        calls.append((command, request, timeout))
        return '{"status":"ok","text":"[[FINAL]]git commit[[/FINAL]]"}'

    selection = _selection()
    corrector = OnDemandQwenCorrector(
        timeout_seconds=17.0, runner=runner, selection=selection
    )

    assert calls == []
    assert corrector.correct("get commit") == "git commit"
    assert len(calls) == 1
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            selection.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert calls[0][0] == (
        str(selection.python),
        "-m",
        "fun_voice.corrector",
        "--selection-fingerprint",
        expected_fingerprint,
    )
    assert calls[0][2] == 17.0
    assert json.loads(calls[0][1]) == {
        "text": "get commit",
        "selection_fingerprint": expected_fingerprint,
    }


def test_corrector_child_rejects_a_mismatched_cli_selection_before_reading_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(corrector_module, "load_runtime_selection", _selection)
    monkeypatch.setattr(
        corrector_module,
        "_read_request",
        lambda *_args: (_ for _ in ()).throw(AssertionError("text was accepted")),
    )

    assert corrector_module.main(["--selection-fingerprint", "0" * 64]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "correction.selection_mismatch"


def test_corrector_child_rejects_a_mismatched_request_selection_before_model_load(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    selection = _selection()
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            selection.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    generated: list[str] = []
    monkeypatch.setattr(corrector_module, "load_runtime_selection", lambda: selection)
    monkeypatch.setattr(
        corrector_module,
        "generate_enveloped_correction",
        lambda *_args, **_kwargs: generated.append("model") or None,
    )
    monkeypatch.setattr(
        corrector_module.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {"text": "get commit", "selection_fingerprint": "1" * 64}
            )
        ),
    )

    assert (
        corrector_module.main(["--selection-fingerprint", expected_fingerprint])
        == 1
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "correction.selection_mismatch"
    assert generated == []


def test_qwen_process_failure_is_exposed_for_raw_text_fallback() -> None:
    def runner(_command: Sequence[str], _request: str, _timeout: float) -> str:
        return '{"status":"error","error_code":"correction.oom"}'

    corrector = OnDemandQwenCorrector(
        command=("qwen-corrector",), runner=runner, selection=_selection()
    )

    with pytest.raises(CorrectionError, match="correction.oom"):
        corrector.correct("get commit")


def test_qwen_parent_preserves_child_rejection_reason_and_stage_timing() -> None:
    def runner(_command: Sequence[str], _request: str, _timeout: float) -> str:
        return json.dumps(
            {
                "status": "error",
                "error_code": "correction.invalid_output",
                "error_reason": "similarity",
                "timing_ms": {
                    "model_load_ms": 4,
                    "generate_ms": 8,
                    "validate_ms": 1,
                },
            }
        )

    corrector = OnDemandQwenCorrector(
        command=("qwen-corrector",), runner=runner, selection=_selection()
    )

    with pytest.raises(CorrectionError) as caught:
        corrector.correct("get commit")

    assert caught.value.code == "correction.invalid_output"
    assert caught.value.reason == "similarity"
    assert caught.value.timing is not None
    assert caught.value.timing.model_load_ms == 4
    assert caught.value.timing.generate_ms == 8
    assert caught.value.timing.validate_ms == 1


def test_qwen_client_accepts_only_the_final_json_frame_after_engine_logs() -> None:
    def runner(_command: Sequence[str], _request: str, _timeout: float) -> str:
        frame = '{"status":"ok","text":"[[FINAL]]git[[/FINAL]]"}'
        return f"INFO engine started\n{frame}\n"

    corrector = OnDemandQwenCorrector(
        command=("qwen-corrector",), runner=runner, selection=_selection()
    )

    assert corrector.correct("get") == "git"


def test_corrector_refuses_cpu_selection_before_subprocess() -> None:
    runner_calls: list[object] = []

    def runner(_command: Sequence[str], _request: str, _timeout: float) -> str:
        runner_calls.append(object())
        return ""

    corrector = OnDemandQwenCorrector(
        inference=_cpu_effective().enhanced,
        selection=_selection("cpu"),
        runner=runner,
    )

    with pytest.raises(CorrectionError, match="disabled_by_runtime_policy"):
        corrector.correct("get commit")
    assert runner_calls == []


@pytest.mark.parametrize(
    ("dtype", "torch_dtype_name"),
    [("float32", "float32"), ("bf16", "bfloat16"), ("fp16", "float16")],
)
def test_generate_maps_selected_dtype_before_model_load(
    monkeypatch: pytest.MonkeyPatch, dtype: str, torch_dtype_name: str
) -> None:
    observed_dtypes: list[object] = []
    fake_torch = SimpleNamespace(
        float32=object(), bfloat16=object(), float16=object()
    )

    class _FakeQwen:
        @staticmethod
        def from_pretrained(_path: str, *, torch_dtype: object) -> object:
            observed_dtypes.append(torch_dtype)
            raise RuntimeError("model unavailable")

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=object, Qwen3_5ForConditionalGeneration=_FakeQwen
        ),
    )

    with pytest.raises(CorrectionError, match="correction.model_load"):
        generate_enveloped_correction(
            "get commit",
            inference=config.EnhancedInferenceConfig(),
            selection=_selection("cuda", dtype=dtype),
        )

    assert observed_dtypes == [getattr(fake_torch, torch_dtype_name)]


def test_generate_rejects_unknown_selection_dtype_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loads: list[object] = []

    class _FakeQwen:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            model_loads.append(object())
            raise RuntimeError("model unavailable")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16=object()))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=object, Qwen3_5ForConditionalGeneration=_FakeQwen
        ),
    )

    with pytest.raises(CorrectionError, match="correction.device"):
        generate_enveloped_correction(
            "get commit",
            inference=config.EnhancedInferenceConfig(),
            selection=replace(_selection("cuda"), dtype="float64"),
        )

    assert model_loads == []
