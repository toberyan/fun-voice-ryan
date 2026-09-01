"""Aggregate-only scoring tests for the explicit local benchmark command."""

from __future__ import annotations

import stat
from pathlib import Path

from fun_voice.benchmark import _write_report, aggregate_scores, score_text


def test_score_text_calculates_cer_terms_and_punctuation_without_text() -> None:
    score = score_text("运行 git commit。", "运行 git commit", ("git", "commit"))

    assert score["cer"] > 0
    assert score["term_exact"] == 1.0
    assert score["punctuation_f1"] == 0.0
    assert "运行" not in repr(score)


def test_aggregate_groups_category_without_audio_or_reference() -> None:
    report = aggregate_scores([("mixed", {"cer": 0.1, "term_exact": 1.0})])

    assert report["categories"]["mixed"]["count"] == 1
    assert "audio" not in repr(report)
    assert "reference" not in repr(report)


def test_explicit_report_is_owner_only_and_contains_aggregates(tmp_path: Path) -> None:
    report_path = tmp_path / "benchmark.json"
    _write_report(report_path, {"count": 1, "categories": {"mixed": {"count": 1}}})

    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert report_path.read_text(encoding="utf-8") == (
        '{"categories": {"mixed": {"count": 1}}, "count": 1}\n'
    )
