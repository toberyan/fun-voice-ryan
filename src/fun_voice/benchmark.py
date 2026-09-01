"""Explicit, aggregate-only local accuracy and latency benchmark.

The manifest is deliberately supplied by its owner on each invocation.  Audio,
reference text, terms, and model output remain in process only long enough to
score that invocation.  The optional report contains aggregates exclusively.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fun_voice import config
from fun_voice.contracts import CaptureArtifact, Transcription
from fun_voice.daemon import SocketWorkerClient, default_start_worker_service

_CATEGORY_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}\Z")
_SCORE_KEYS = (
    "cer",
    "term_exact",
    "punctuation_precision",
    "punctuation_recall",
    "punctuation_f1",
)
_MAX_REFERENCE_CHARACTERS = 16_384


@dataclass(frozen=True)
class BenchmarkCase:
    """One user-supplied benchmark case, retained only during the CLI run."""

    category: str
    audio: str
    reference: str
    terms: tuple[str, ...] = ()


def score_text(
    reference: str, candidate: str, terms: Sequence[str]
) -> dict[str, float]:
    """Score a candidate without returning either source string.

    CER is character-level edit distance normalized by reference length.
    Exact terms are searched with case-preserving substring matching so source
    code and product names remain meaningful.  Punctuation uses multiset
    precision/recall/F1 across both Chinese and ASCII punctuation.
    """
    if (
        len(reference) > _MAX_REFERENCE_CHARACTERS
        or len(candidate) > _MAX_REFERENCE_CHARACTERS
    ):
        raise ValueError("benchmark text exceeds the scoring limit")
    distance = _levenshtein_distance(reference, candidate)
    cer = distance / max(1, len(reference))
    normalized_terms = tuple(term for term in terms if term)
    term_exact = (
        sum(term in candidate for term in normalized_terms) / len(normalized_terms)
        if normalized_terms
        else 1.0
    )
    punctuation_precision, punctuation_recall, punctuation_f1 = _punctuation_score(
        reference, candidate
    )
    return {
        "cer": cer,
        "term_exact": term_exact,
        "punctuation_precision": punctuation_precision,
        "punctuation_recall": punctuation_recall,
        "punctuation_f1": punctuation_f1,
    }


def aggregate_scores(
    rows: Sequence[tuple[str, Mapping[str, float | int]]],
) -> dict[str, object]:
    """Group numeric scores by safe category labels, without source payloads."""
    grouped: dict[str, list[Mapping[str, float | int]]] = defaultdict(list)
    for category, score in rows:
        _validate_category(category)
        grouped[category].append(score)

    categories: dict[str, object] = {}
    for category, scores in sorted(grouped.items()):
        summary: dict[str, object] = {"count": len(scores)}
        for key in _SCORE_KEYS:
            values = [float(score[key]) for score in scores if key in score]
            if values:
                summary[key] = _numeric_summary(values)
        categories[category] = summary
    return {"count": len(rows), "categories": categories}


def _run_cases(
    cases: Sequence[BenchmarkCase],
    transcribe: Callable[[CaptureArtifact], Transcription],
) -> dict[str, object]:
    """Transcribe and score cases; returned data is aggregate-only."""
    scores: list[tuple[str, Mapping[str, float | int]]] = []
    request_durations: dict[str, list[int]] = {"cold": [], "warm": []}
    for index, case in enumerate(cases):
        started = time.monotonic()
        transcription = transcribe(CaptureArtifact(audio=case.audio))
        elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
        phase = "cold" if index == 0 else "warm"
        request_durations[phase].append(elapsed_ms)
        scores.append(
            (
                case.category,
                score_text(case.reference, transcription.text, case.terms),
            )
        )

    report = aggregate_scores(scores)
    report["request_latency_ms"] = {
        phase: _numeric_summary(values)
        for phase, values in request_durations.items()
        if values
    }
    return report


def _load_manifest(path: Path) -> tuple[BenchmarkCase, ...]:
    """Parse a user-owned JSONL manifest without ever echoing its contents."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("cannot read benchmark manifest") from exc
    cases: list[BenchmarkCase] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("manifest contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("manifest rows must be objects")
        category = value.get("category")
        audio = value.get("audio")
        reference = value.get("reference")
        terms = value.get("terms", [])
        if (
            not isinstance(category, str)
            or not isinstance(audio, str)
            or not audio
            or not isinstance(reference, str)
            or not isinstance(terms, list)
            or not all(isinstance(term, str) for term in terms)
        ):
            raise ValueError("manifest row has invalid fields")
        _validate_category(category)
        if len(reference) > _MAX_REFERENCE_CHARACTERS:
            raise ValueError("manifest reference exceeds the scoring limit")
        cases.append(
            BenchmarkCase(
                category=category,
                audio=audio,
                reference=reference,
                terms=tuple(terms),
            )
        )
    if not cases:
        raise ValueError("manifest has no benchmark cases")
    return tuple(cases)


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    """Write an explicit aggregate report with private owner-only mode."""
    encoded = (json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            config.FILE_MODE,
        )
        try:
            os.fchmod(descriptor, config.FILE_MODE)
        except BaseException:
            os.close(descriptor)
            raise
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
    except OSError as exc:
        raise ValueError("cannot write benchmark report") from exc


def _levenshtein_distance(reference: str, candidate: str) -> int:
    """Character edit distance using one short dynamic-programming row."""
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    row = list(range(len(candidate) + 1))
    for index, reference_char in enumerate(reference, start=1):
        previous_diagonal = row[0]
        row[0] = index
        for column, candidate_char in enumerate(candidate, start=1):
            previous_above = row[column]
            row[column] = min(
                row[column] + 1,
                row[column - 1] + 1,
                previous_diagonal + (reference_char != candidate_char),
            )
            previous_diagonal = previous_above
    return row[-1]


def _punctuation_score(reference: str, candidate: str) -> tuple[float, float, float]:
    expected = Counter(char for char in reference if _is_punctuation(char))
    actual = Counter(char for char in candidate if _is_punctuation(char))
    matches = sum((expected & actual).values())
    actual_count = sum(actual.values())
    expected_count = sum(expected.values())
    precision = matches / actual_count if actual_count else 0.0
    recall = matches / expected_count if expected_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def _validate_category(category: str) -> None:
    if not _CATEGORY_RE.fullmatch(category):
        raise ValueError("category must use only lowercase letters, digits, _ or -")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fun-voice-benchmark",
        description="Run aggregate-only local ASR accuracy and latency checks.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        cases = _load_manifest(args.manifest)
        paths = config.build_runtime_paths(config.resolve_runtime_dir())
        worker = SocketWorkerClient(
            paths.worker_socket,
            start_service=lambda: default_start_worker_service("nano"),
        )
        report = _run_cases(cases, worker.transcribe)
        if args.output is not None:
            _write_report(args.output, report)
    except Exception:  # noqa: BLE001 - never echo private manifest/model data
        print("benchmark failed", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
