"""Bounded, aggregate-only timing metrics for the local voice pipeline.

The ledger deliberately has no fields for text, audio, desktop focus, file
paths, or identifiers from external input systems.  It remains memory-only and
the daemon exposes only its aggregate summary over the owner-only socket.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Final, cast

from fun_voice.contracts import CORRECTION_REJECTION_REASONS

_TIMING_FIELDS: Final = frozenset(
    {
        "capture_duration_ms",
        "preload_ms",
        "preload_worker_ms",
        "preload_runtime_load_ms",
        "preload_warmup_ms",
        "active_idle_ms",
        "asr_ms",
        "asr_worker_ms",
        "asr_queue_transport_ms",
        "asr_audio_load_ms",
        "asr_vad_ms",
        "asr_generate_ms",
        "asr_release_ms",
        "correction_ms",
        "correction_model_load_ms",
        "correction_generate_ms",
        "correction_validate_ms",
        "commit_ms",
        "end_to_end_ms",
    }
)
_ENUM_FIELDS: Final = frozenset(
    {
        "asr_profile",
        "asr_preload",
        "asr_warmup",
        "asr_preload_profile",
        "nano_preload",
        "nano_warmup",
        "correction",
        "correction_rejection",
        "error_code",
        "nano_was_stopped_for_qwen",
        "session_policy",
        "session_transition",
        "risk_gate",
        "nano_rehydration",
        "background_enrichment",
    }
)
_ALLOWED_ASR_PROFILES: Final = frozenset({"nano", "sensevoice"})
_ALLOWED_ASR_PRELOAD: Final = frozenset(
    {"not_requested", "scheduled", "ready", "failed"}
)
_ALLOWED_ASR_WARMUP: Final = frozenset({"not_requested", "ready", "failed"})
_ALLOWED_NANO_PRELOAD: Final = frozenset(
    {"not_requested", "scheduled", "ready", "failed"}
)
_ALLOWED_NANO_WARMUP: Final = frozenset({"not_requested", "ready", "failed"})
_ALLOWED_CORRECTION: Final = frozenset(
    {"disabled", "corrected", "raw_fallback", "skipped_lease", "failed"}
)
_ALLOWED_SESSION_POLICIES: Final = frozenset(
    {"memory_saver", "balanced", "sustained"}
)
_ALLOWED_SESSION_TRANSITIONS: Final = frozenset(
    {
        "preparing",
        "recording",
        "finalizing",
        "correcting",
        "committing",
        "rehydrating",
        "enriching",
        "active_idle",
        "idle",
    }
)
_ALLOWED_RISK_GATE: Final = frozenset(
    {"none", "punctuation", "term", "mixed_technical", "explicit_polish"}
)
_ALLOWED_NANO_REHYDRATION: Final = frozenset(
    {"not_requested", "scheduled", "ready", "failed", "skipped"}
)
_ALLOWED_BACKGROUND_ENRICHMENT: Final = frozenset(
    {"not_requested", "scheduled", "completed", "cancelled", "failed"}
)
_ALLOWED_ERROR_CODES: Final = frozenset(
    {
        "capture",
        "empty_speech",
        "internal",
        "worker.device",
        "worker.empty_speech",
        "worker.format",
        "worker.internal",
        "worker.model_load",
        "worker.no_output",
        "worker.oom",
        "worker.protocol",
        "worker.timeout",
        "worker.unavailable",
        "worker.vllm",
    }
)
_ALLOWED_FIELDS: Final = _TIMING_FIELDS | _ENUM_FIELDS


@dataclass(frozen=True)
class SessionMetric:
    """One non-sensitive pipeline measurement retained only in memory."""

    sequence: int
    capture_duration_ms: int | None = None
    preload_ms: int | None = None
    preload_worker_ms: int | None = None
    preload_runtime_load_ms: int | None = None
    preload_warmup_ms: int | None = None
    active_idle_ms: int | None = None
    asr_ms: int | None = None
    asr_worker_ms: int | None = None
    asr_queue_transport_ms: int | None = None
    asr_audio_load_ms: int | None = None
    asr_vad_ms: int | None = None
    asr_generate_ms: int | None = None
    asr_release_ms: int | None = None
    correction_ms: int | None = None
    correction_model_load_ms: int | None = None
    correction_generate_ms: int | None = None
    correction_validate_ms: int | None = None
    commit_ms: int | None = None
    end_to_end_ms: int | None = None
    asr_profile: str | None = None
    asr_preload: str | None = None
    asr_warmup: str | None = None
    asr_preload_profile: str | None = None
    nano_preload: str | None = None
    nano_warmup: str | None = None
    correction: str = "disabled"
    correction_rejection: str | None = None
    error_code: str | None = None
    nano_was_stopped_for_qwen: bool = False
    session_policy: str | None = None
    session_transition: str | None = None
    risk_gate: str | None = None
    nano_rehydration: str | None = None
    background_enrichment: str | None = None


class MetricsLedger:
    """Thread-safe fixed-size ledger returning aggregates, never row data."""

    def __init__(self, *, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._entries: deque[SessionMetric] = deque(maxlen=max_entries)
        self._next_sequence = 1
        self._lock = Lock()

    def begin(self) -> int:
        """Create a metric row and return its opaque process-local sequence."""
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._entries.append(SessionMetric(sequence=sequence))
            return sequence

    def record(self, sequence: int, **updates: object) -> None:
        """Apply validated non-sensitive values to a retained metric row.

        A row which has been evicted is intentionally ignored.  This prevents
        a completed asynchronous preload from resurrecting historical data.
        """
        self._validate_updates(updates)
        with self._lock:
            for index in range(len(self._entries) - 1, -1, -1):
                row = self._entries[index]
                if row.sequence == sequence:
                    # `_validate_updates` has checked every key and value before
                    # this point; the cast only bridges dataclasses' dynamic
                    # `replace(**kwargs)` signature for the type checker.
                    self._entries[index] = replace(row, **cast(Any, updates))
                    return

    def summary(self) -> dict[str, object]:
        """Return only counts, enum histograms, and integer percentiles."""
        with self._lock:
            rows = tuple(self._entries)

        report: dict[str, object] = {"count": len(rows)}
        for field in sorted(_TIMING_FIELDS):
            values = [getattr(row, field) for row in rows]
            timings = [value for value in values if isinstance(value, int)]
            if timings:
                report[field] = {
                    "p50": _percentile(timings, 0.50),
                    "p95": _percentile(timings, 0.95),
                }

        for field in (
            "asr_profile",
            "asr_preload",
            "asr_warmup",
            "asr_preload_profile",
            "nano_preload",
            "nano_warmup",
            "correction",
            "correction_rejection",
            "error_code",
            "session_policy",
            "session_transition",
            "risk_gate",
            "nano_rehydration",
            "background_enrichment",
        ):
            values = [getattr(row, field) for row in rows]
            counts = Counter(value for value in values if value is not None)
            if counts:
                report[field] = dict(sorted(counts.items()))

        stopped = sum(row.nano_was_stopped_for_qwen for row in rows)
        if stopped:
            report["nano_was_stopped_for_qwen"] = stopped
        return report

    @staticmethod
    def _validate_updates(updates: dict[str, object]) -> None:
        unknown = set(updates) - _ALLOWED_FIELDS
        if unknown:
            raise ValueError("unsupported metric fields")
        for field in _TIMING_FIELDS:
            if field in updates:
                value = updates[field]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"{field} must be a non-negative integer")
        if (
            "asr_profile" in updates
            and updates["asr_profile"] not in _ALLOWED_ASR_PROFILES
        ):
            raise ValueError("invalid asr_profile")
        if (
            "asr_preload" in updates
            and updates["asr_preload"] not in _ALLOWED_ASR_PRELOAD
        ):
            raise ValueError("invalid asr_preload")
        if (
            "asr_warmup" in updates
            and updates["asr_warmup"] not in _ALLOWED_ASR_WARMUP
        ):
            raise ValueError("invalid asr_warmup")
        if (
            "asr_preload_profile" in updates
            and updates["asr_preload_profile"] not in _ALLOWED_ASR_PROFILES
        ):
            raise ValueError("invalid asr_preload_profile")
        if (
            "nano_preload" in updates
            and updates["nano_preload"] not in _ALLOWED_NANO_PRELOAD
        ):
            raise ValueError("invalid nano_preload")
        if (
            "nano_warmup" in updates
            and updates["nano_warmup"] not in _ALLOWED_NANO_WARMUP
        ):
            raise ValueError("invalid nano_warmup")
        if (
            "correction" in updates
            and updates["correction"] not in _ALLOWED_CORRECTION
        ):
            raise ValueError("invalid correction")
        if (
            "correction_rejection" in updates
            and updates["correction_rejection"] not in CORRECTION_REJECTION_REASONS
        ):
            raise ValueError("invalid correction_rejection")
        if (
            "error_code" in updates
            and updates["error_code"] not in _ALLOWED_ERROR_CODES
        ):
            raise ValueError("invalid error_code")
        if (
            "session_policy" in updates
            and updates["session_policy"] not in _ALLOWED_SESSION_POLICIES
        ):
            raise ValueError("invalid session_policy")
        if (
            "session_transition" in updates
            and updates["session_transition"] not in _ALLOWED_SESSION_TRANSITIONS
        ):
            raise ValueError("invalid session_transition")
        if "risk_gate" in updates and updates["risk_gate"] not in _ALLOWED_RISK_GATE:
            raise ValueError("invalid risk_gate")
        if (
            "nano_rehydration" in updates
            and updates["nano_rehydration"] not in _ALLOWED_NANO_REHYDRATION
        ):
            raise ValueError("invalid nano_rehydration")
        if (
            "background_enrichment" in updates
            and updates["background_enrichment"]
            not in _ALLOWED_BACKGROUND_ENRICHMENT
        ):
            raise ValueError("invalid background_enrichment")
        if "nano_was_stopped_for_qwen" in updates and not isinstance(
            updates["nano_was_stopped_for_qwen"], bool
        ):
            raise ValueError("nano_was_stopped_for_qwen must be boolean")


def _percentile(values: list[int], quantile: float) -> int:
    """Return a rounded linear percentile without exposing individual values."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)
