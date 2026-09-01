"""Tests for the one-request Qwen correction process boundary."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from fun_voice.corrector import (
    DEFAULT_TIMEOUT_SECONDS,
    CorrectionError,
    OnDemandQwenCorrector,
    parse_correction_output,
    validate_correction,
)


def test_qwen_on_demand_timeout_has_a_safe_upper_bound() -> None:
    assert DEFAULT_TIMEOUT_SECONDS == 180.0


def test_parse_correction_output_accepts_only_the_final_envelope() -> None:
    output = "[[FINAL]]git commit，然后运行 pytest。[[/FINAL]]"
    assert parse_correction_output(output) == (
        "git commit，然后运行 pytest。"
    )


def test_parse_correction_output_accepts_clean_opening_only_envelope() -> None:
    assert parse_correction_output("[[FINAL]]git commit\n") == "git commit"


def test_candidate_must_remain_similar_to_the_raw_asr_text() -> None:
    assert validate_correction("get commit", "git commit") == "git commit"

    with pytest.raises(CorrectionError, match="invalid_output"):
        validate_correction("get commit", "完全无关的长文本")


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

    corrector = OnDemandQwenCorrector(
        command=("qwen-corrector",), timeout_seconds=17.0, runner=runner
    )

    assert calls == []
    assert corrector.correct("get commit") == "git commit"
    assert len(calls) == 1
    assert calls[0][0] == ("qwen-corrector",)
    assert calls[0][2] == 17.0
    assert json.loads(calls[0][1]) == {"text": "get commit"}


def test_qwen_process_failure_is_exposed_for_raw_text_fallback() -> None:
    def runner(_command: Sequence[str], _request: str, _timeout: float) -> str:
        return '{"status":"error","error_code":"correction.oom"}'

    corrector = OnDemandQwenCorrector(command=("qwen-corrector",), runner=runner)

    with pytest.raises(CorrectionError, match="correction.oom"):
        corrector.correct("get commit")


def test_qwen_client_accepts_only_the_final_json_frame_after_engine_logs() -> None:
    def runner(_command: Sequence[str], _request: str, _timeout: float) -> str:
        frame = '{"status":"ok","text":"[[FINAL]]git[[/FINAL]]"}'
        return f"INFO engine started\n{frame}\n"

    corrector = OnDemandQwenCorrector(command=("qwen-corrector",), runner=runner)

    assert corrector.correct("get") == "git"
