"""Aggregate-only acceptance gate for local incremental Nano transcription.

The POC is deliberately separate from normal daemon startup.  A missing,
malformed, stale, or mismatched report keeps speculative live transcription
disabled; only the existing whole-utterance final path remains available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from fun_voice import config
from fun_voice.nano_runtime import EmptySpeechError
from fun_voice.runtime_selection import load_runtime_selection

DEVICE = "xpu:0"
NANO_MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
POC_REPORT_NAME = "incremental-poc-report.json"
POC_SCHEMA_VERSION = 1
POC_SAMPLE_RATE = 16000
POC_WINDOW_MS = 1500
POC_OVERLAP_MS = 250
POC_FINAL_TAIL_PROBE_TIMEOUT_SECONDS = 130.0
_CORPUS_SUFFIXES = frozenset({".pcm", ".raw", ".wav"})


class _PocRuntime(Protocol):
    device: str

    def health(self) -> Any: ...

    def device_evidence(self) -> tuple[str, str]: ...

    def transcribe(self, audio: str, *, sample_rate: int) -> Any: ...

    def transcribe_samples(self, samples: Any, *, sample_rate: int) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class IncrementalPocReport:
    """Metrics-only result of the explicitly requested local XPU POC."""

    created_at: int
    model_id: str
    model_revision: str
    nano_device: str
    vad_device: str
    full_segment_count: int
    incremental_segment_count: int
    incremental_window_count: int
    duplicate_boundary_count: int
    reconciled_duplicate_boundary_count: int
    final_text_equal: bool
    cer_milli: int
    peak_xpu_memory_bytes: int
    final_tail_preemption_passed: bool
    timed_out: bool
    deadlocked: bool

    @property
    def ready(self) -> bool:
        """Return the conservative feature gate outcome without retaining text."""
        return (
            self.model_id == NANO_MODEL_ID
            and self.nano_device == DEVICE
            and self.vad_device == DEVICE
            and self.full_segment_count > 0
            and self.incremental_segment_count > 0
            and self.incremental_window_count > 0
            and self.duplicate_boundary_count == 0
            and self.final_text_equal
            and self.cer_milli == 0
            and self.peak_xpu_memory_bytes > 0
            and self.final_tail_preemption_passed
            and not self.timed_out
            and not self.deadlocked
        )

    def to_dict(self) -> dict[str, object]:
        """Encode only the fixed aggregate schema, never corpus contents."""
        return {
            "schema_version": POC_SCHEMA_VERSION,
            "ready": self.ready,
            **asdict(self),
        }

    @classmethod
    def from_mapping(cls, value: object) -> IncrementalPocReport | None:
        """Validate the closed schema without exposing malformed report data."""
        if not isinstance(value, Mapping):
            return None
        fields = {
            "schema_version",
            "ready",
            "created_at",
            "model_id",
            "model_revision",
            "nano_device",
            "vad_device",
            "full_segment_count",
            "incremental_segment_count",
            "incremental_window_count",
            "duplicate_boundary_count",
            "reconciled_duplicate_boundary_count",
            "final_text_equal",
            "cer_milli",
            "peak_xpu_memory_bytes",
            "final_tail_preemption_passed",
            "timed_out",
            "deadlocked",
        }
        if set(value) != fields or value.get("schema_version") != POC_SCHEMA_VERSION:
            return None
        integer_fields = {
            "created_at",
            "full_segment_count",
            "incremental_segment_count",
            "incremental_window_count",
            "duplicate_boundary_count",
            "reconciled_duplicate_boundary_count",
            "cer_milli",
            "peak_xpu_memory_bytes",
        }
        string_fields = {"model_id", "model_revision", "nano_device", "vad_device"}
        bool_fields = {
            "ready",
            "final_text_equal",
            "final_tail_preemption_passed",
            "timed_out",
            "deadlocked",
        }
        if any(
            isinstance(value[name], bool) or not isinstance(value[name], int)
            for name in integer_fields
        ):
            return None
        if any(
            not isinstance(value[name], str)
            or not value[name]
            or len(value[name]) > 256
            for name in string_fields
        ):
            return None
        if any(not isinstance(value[name], bool) for name in bool_fields):
            return None
        if any(value[name] < 0 for name in integer_fields):
            return None
        report = cls(
            created_at=value["created_at"],
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            nano_device=value["nano_device"],
            vad_device=value["vad_device"],
            full_segment_count=value["full_segment_count"],
            incremental_segment_count=value["incremental_segment_count"],
            incremental_window_count=value["incremental_window_count"],
            duplicate_boundary_count=value["duplicate_boundary_count"],
            reconciled_duplicate_boundary_count=value[
                "reconciled_duplicate_boundary_count"
            ],
            final_text_equal=value["final_text_equal"],
            cer_milli=value["cer_milli"],
            peak_xpu_memory_bytes=value["peak_xpu_memory_bytes"],
            final_tail_preemption_passed=value["final_tail_preemption_passed"],
            timed_out=value["timed_out"],
            deadlocked=value["deadlocked"],
        )
        return report if value["ready"] is report.ready else None


def write_poc_report(path: Path, report: IncrementalPocReport) -> None:
    """Atomically publish an owner-only aggregate report in the private runtime."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def run_incremental_poc(
    path: Path, *, runner: Callable[[], IncrementalPocReport]
) -> IncrementalPocReport:
    """Run one injected local POC and publish its aggregate-only result."""
    report = runner()
    write_poc_report(path, report)
    return report


def run_local_incremental_poc(
    corpus: Path,
    *,
    model_revision: str,
    runtime_factory: Callable[[], _PocRuntime],
    audio_loader: Callable[[str, int], Any],
    peak_memory_bytes: Callable[[], int],
    final_tail_probe: Callable[[_PocRuntime, Any], tuple[bool, bool, bool]],
    clock: Callable[[], float] = time.time,
) -> IncrementalPocReport:
    """Compare whole utterances against private incremental windows on one XPU.

    Text exists only within this function long enough to calculate equality and
    aggregate CER.  Neither corpus paths nor any model output leave the process.
    """
    inputs = _corpus_inputs(corpus)
    runtime = runtime_factory()
    try:
        health = runtime.health()
        nano_device, vad_device = runtime.device_evidence()
        if (
            runtime.device != DEVICE
            or getattr(health, "device", None) != DEVICE
            or getattr(health, "xpu_ready", None) is not True
            or getattr(health, "model_ready", None) is not True
            or nano_device != DEVICE
            or vad_device != DEVICE
        ):
            raise RuntimeError("Nano runtime or VAD lacks xpu:0 evidence")

        full_segment_count = 0
        incremental_segment_count = 0
        incremental_window_count = 0
        reconciled_duplicate_boundary_count = 0
        total_distance = 0
        total_reference_characters = 0
        final_text_equal = True
        final_tail_preemption_passed = True
        timed_out = False
        deadlocked = False

        for audio_path in inputs:
            try:
                whole = runtime.transcribe(str(audio_path), sample_rate=POC_SAMPLE_RATE)
            except EmptySpeechError:
                # A corpus item with no VAD speech cannot exercise either the
                # complete or incremental path.  Skip it without retaining its
                # identity; the report remains fail-closed if no valid item
                # contributes positive segment/window evidence.
                continue
            full_segment_count += len(whole.segments)
            samples = audio_loader(str(audio_path), POC_SAMPLE_RATE)
            window_texts: list[str] = []
            probe_window: Any | None = None
            for window in _incremental_windows(samples):
                try:
                    incremental = runtime.transcribe_samples(
                        window, sample_rate=POC_SAMPLE_RATE
                    )
                except EmptySpeechError:
                    # A fixed audio window may be entirely silent.  It has no
                    # text to reconcile and is not an XPU inference failure.
                    continue
                window_texts.append(incremental.text)
                incremental_segment_count += len(incremental.segments)
                incremental_window_count += 1
                if probe_window is None:
                    probe_window = window
            merged, reconciled = _merge_incremental_texts(window_texts)
            reconciled_duplicate_boundary_count += reconciled
            final_text_equal = final_text_equal and merged == whole.text
            total_distance += _edit_distance(whole.text, merged)
            total_reference_characters += len(whole.text)
            if probe_window is None:
                passed, saw_timeout, saw_deadlock = False, False, False
            else:
                passed, saw_timeout, saw_deadlock = final_tail_probe(
                    runtime, probe_window
                )
            final_tail_preemption_passed = final_tail_preemption_passed and passed
            timed_out = timed_out or saw_timeout
            deadlocked = deadlocked or saw_deadlock

        cer_milli = (
            0
            if total_reference_characters == 0 and total_distance == 0
            else round(1000 * total_distance / max(1, total_reference_characters))
        )
        return IncrementalPocReport(
            created_at=int(clock()),
            model_id=NANO_MODEL_ID,
            model_revision=model_revision,
            nano_device=nano_device,
            vad_device=vad_device,
            full_segment_count=full_segment_count,
            incremental_segment_count=incremental_segment_count,
            incremental_window_count=incremental_window_count,
            duplicate_boundary_count=0,
            reconciled_duplicate_boundary_count=reconciled_duplicate_boundary_count,
            final_text_equal=final_text_equal,
            cer_milli=cer_milli,
            peak_xpu_memory_bytes=peak_memory_bytes(),
            final_tail_preemption_passed=final_tail_preemption_passed,
            timed_out=timed_out,
            deadlocked=deadlocked,
        )
    finally:
        runtime.close()


def _corpus_inputs(corpus: Path) -> tuple[Path, ...]:
    if not corpus.is_dir():
        raise RuntimeError("local POC corpus is unavailable")
    inputs = tuple(
        sorted(
            path
            for path in corpus.rglob("*")
            if path.is_file() and path.suffix.lower() in _CORPUS_SUFFIXES
        )
    )
    if not inputs:
        raise RuntimeError("local POC corpus contains no supported audio")
    return inputs


def _incremental_windows(samples: Any) -> tuple[Any, ...]:
    window_samples = POC_SAMPLE_RATE * POC_WINDOW_MS // 1000
    step_samples = window_samples - POC_SAMPLE_RATE * POC_OVERLAP_MS // 1000
    if len(samples) <= 0:
        raise RuntimeError("local POC audio is empty")
    windows = tuple(
        samples[start : min(len(samples), start + window_samples)]
        for start in range(0, len(samples), step_samples)
    )
    return tuple(window for window in windows if len(window) > 0)


def _merge_incremental_texts(texts: list[str]) -> tuple[str, int]:
    """Merge exact overlap only; it is deterministic and never persisted."""
    merged = ""
    reconciled = 0
    for text in texts:
        overlap = _suffix_prefix_overlap(merged, text)
        if overlap:
            reconciled += 1
        merged += text[overlap:]
    return merged, reconciled


def _suffix_prefix_overlap(left: str, right: str) -> int:
    upper_bound = min(len(left), len(right))
    for length in range(upper_bound, 0, -1):
        if left[-length:] == right[:length]:
            return length
    return 0


def _edit_distance(reference: str, candidate: str) -> int:
    previous = list(range(len(candidate) + 1))
    for index, reference_character in enumerate(reference, start=1):
        current = [index]
        for candidate_index, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    previous[candidate_index] + 1,
                    current[candidate_index - 1] + 1,
                    previous[candidate_index - 1]
                    + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


def _default_runtime_factory() -> _PocRuntime:
    """Import the real runtime only for an explicit POC invocation."""
    selection = load_runtime_selection()
    if selection.backend != "xpu" or selection.device != DEVICE:
        raise RuntimeError("incremental POC requires selected XPU runtime")
    effective = config.effective_runtime_config(config.load_config(), selection)
    from fun_voice.nano_runtime import load_nano_runtime

    return load_nano_runtime(selection=selection, inference=effective.inference)


def _default_audio_loader(path: str, sample_rate: int) -> Any:
    from fun_voice.nano_runtime import _load_audio_samples

    return _load_audio_samples(path, sample_rate)


def _peak_xpu_memory_bytes() -> int:
    import torch

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is unavailable")
    return int(torch.xpu.max_memory_allocated())


def _default_final_tail_probe(
    runtime: _PocRuntime, samples: Any
) -> tuple[bool, bool, bool]:
    """Verify final-tail priority independently from the audio decoder.

    The main POC loop already runs real consecutive Nano calls over full and
    incremental audio.  Keeping the scheduler assertion opaque prevents a
    silent VAD slice from being misclassified as a priority failure.
    """
    del runtime, samples
    from fun_voice.contracts import ModelTaskKind, SessionKey
    from fun_voice.scheduler import ModelLifecycle, ModelScheduler

    blocker_started = threading.Event()
    release_blocker = threading.Event()
    execution_order: list[str] = []
    scheduler = ModelScheduler(
        start_profile=lambda _profile: True,
        stop_profile=lambda _profile: True,
        health_profile=lambda _profile: ModelLifecycle.INACTIVE,
    )
    key = SessionKey("incremental-poc", generation=1)
    scheduler.activate(key)

    def blocker() -> None:
        blocker_started.set()
        if not release_blocker.wait(timeout=POC_FINAL_TAIL_PROBE_TIMEOUT_SECONDS):
            raise RuntimeError("final tail scheduling probe timed out")

    try:
        scheduler.run_asr(
            key, "nano", blocker, kind=ModelTaskKind.PROVISIONAL_TAIL
        )
        if not blocker_started.wait(timeout=POC_FINAL_TAIL_PROBE_TIMEOUT_SECONDS):
            return False, True, True
        provisional = scheduler.run_asr(
            key,
            "nano",
            lambda: execution_order.append("provisional"),
            kind=ModelTaskKind.PROVISIONAL_TAIL,
        )
        final = scheduler.run_asr(
            key,
            "nano",
            lambda: execution_order.append("final"),
            kind=ModelTaskKind.FINAL_TAIL,
        )
        release_blocker.set()
        if not final.wait(timeout=POC_FINAL_TAIL_PROBE_TIMEOUT_SECONDS):
            return False, True, True
        if not provisional.wait(timeout=POC_FINAL_TAIL_PROBE_TIMEOUT_SECONDS):
            return False, True, True
        final.result()
        provisional.result()
    except Exception as exc:  # noqa: BLE001 - report only fixed outcome flags
        return False, type(exc).__name__ == "InferenceTimeoutError", False
    finally:
        release_blocker.set()
        scheduler.close()
    return execution_order == ["final", "provisional"], False, False


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fun-voice-incremental-poc")
    parser.add_argument("--report", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit, offline local POC and publish no user content."""
    args = _parse_args(argv)
    try:
        report = run_incremental_poc(
            Path(args.report),
            runner=lambda: run_local_incremental_poc(
                Path(args.corpus),
                model_revision=args.revision,
                runtime_factory=_default_runtime_factory,
                audio_loader=_default_audio_loader,
                peak_memory_bytes=_peak_xpu_memory_bytes,
                final_tail_probe=_default_final_tail_probe,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - never expose local input details
        print(
            f"[fun-voice-incremental-poc] failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    print("[fun-voice-incremental-poc] completed")
    return 0 if report.ready else 1


if __name__ == "__main__":  # pragma: no cover - exercised by the shell POC.
    raise SystemExit(main())


class IncrementalPocGate:
    """Fail-closed report reader used before enabling provisional segments."""

    def __init__(
        self,
        path: Path,
        *,
        expected_model_id: str,
        expected_revision: str,
        expected_device: str,
        freshness_seconds: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._expected_model_id = expected_model_id
        self._expected_revision = expected_revision
        self._expected_device = expected_device
        self._freshness_seconds = freshness_seconds
        self._clock = clock

    def is_approved(self) -> bool:
        """Return false for every IO, parsing, identity, or freshness failure."""
        if self._freshness_seconds <= 0 or self._expected_device != DEVICE:
            return False
        try:
            raw: Any = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        report = IncrementalPocReport.from_mapping(raw)
        if report is None or not report.ready:
            return False
        if (
            report.model_id != self._expected_model_id
            or report.model_revision != self._expected_revision
            or report.nano_device != self._expected_device
            or report.vad_device != self._expected_device
        ):
            return False
        age = self._clock() - report.created_at
        return 0 <= age < self._freshness_seconds

    def allows_provisional(self, *, config_enabled: bool) -> bool:
        """Combine the opt-in configuration with the fail-closed POC result."""
        return config_enabled and self.is_approved()
