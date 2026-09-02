"""Single-owner, priority-ordered scheduler for local model work.

The scheduler intentionally knows no model API, desktop state or text.  It
serializes opaque callbacks and makes a completion callback conditional on the
still-current :class:`SessionKey`.  A cancellation cannot interrupt a decoder
already executing in a model runtime; it makes that decoder's eventual result
unpublishable instead.
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from fun_voice.contracts import ModelTaskKind, SessionKey

AsrProfile = Literal["nano", "sensevoice"]


class ModelLifecycle(StrEnum):
    """Non-sensitive lifecycle states returned by a model profile supervisor."""

    LOADING = "loading"
    READY = "ready"
    INACTIVE = "inactive"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CorrectionOutcome:
    """Result of a Qwen lease attempt; value is absent when not permitted."""

    permitted: bool
    value: object | None = None


class ModelProfileError(RuntimeError):
    """The scheduler could not establish exclusive ownership of an ASR profile."""


class TaskHandle:
    """Thread-safe handle for one queued or running opaque model task."""

    def __init__(self, key: SessionKey, kind: ModelTaskKind) -> None:
        self.key = key
        self.kind = kind
        self._cancelled = False
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._value: object | None = None
        self._error: Exception | None = None

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for physical execution/skipping; returns ``False`` on timeout."""
        return self._done.wait(timeout)

    def result(self) -> object | None:
        """Return the task value after completion, preserving its owner error."""
        if not self._done.is_set():
            raise RuntimeError("task result requested before completion")
        with self._lock:
            if self._cancelled:
                raise RuntimeError("task was cancelled")
            if self._error is not None:
                raise self._error
            return self._value

    def _finish(
        self, *, value: object | None = None, error: Exception | None = None
    ) -> None:
        with self._lock:
            self._value = value
            self._error = error
        self._done.set()


@dataclass(order=True, slots=True)
class _PendingTask:
    priority: int
    sequence: int
    handle: TaskHandle = field(compare=False)
    fn: Callable[[], object] = field(compare=False)
    on_complete: Callable[[object], None] | None = field(compare=False)


_PRIORITY = {
    ModelTaskKind.FINAL_TAIL: 0,
    ModelTaskKind.STABLE_SEGMENT: 1,
    ModelTaskKind.PROVISIONAL_TAIL: 2,
    ModelTaskKind.CORRECTION: 3,
    ModelTaskKind.ENRICHMENT: 4,
}


class ModelScheduler:
    """Run exactly one model callback at a time in approved priority order."""

    def __init__(
        self,
        *,
        start_profile: Callable[[AsrProfile], bool] | None = None,
        stop_profile: Callable[[AsrProfile], bool] | None = None,
        health_profile: Callable[[AsrProfile], ModelLifecycle] | None = None,
    ) -> None:
        self._start_profile = (
            start_profile if start_profile is not None else self._deny_profile_start
        )
        self._stop_profile = (
            stop_profile if stop_profile is not None else self._deny_profile_stop
        )
        self._health_profile = (
            health_profile
            if health_profile is not None
            else lambda _p: ModelLifecycle.FAILED
        )
        self._condition = threading.Condition()
        self._pending: list[_PendingTask] = []
        self._sequence = 0
        self._current_key: SessionKey | None = None
        self._profile_states: dict[AsrProfile, ModelLifecycle] = {
            "nano": ModelLifecycle.INACTIVE,
            "sensevoice": ModelLifecycle.INACTIVE,
        }
        self._running: TaskHandle | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._dispatch, name="fun-voice-model-scheduler", daemon=True
        )
        self._thread.start()

    def activate(self, key: SessionKey) -> None:
        """Make ``key`` current and invalidate all earlier model results."""
        with self._condition:
            self._current_key = key
            for task in self._pending:
                if task.handle.key != key:
                    task.handle.cancel()
            running = self._running
            if running is not None and running.key != key:
                running.cancel()
            self._condition.notify_all()

    def submit(
        self,
        key: SessionKey,
        kind: ModelTaskKind,
        fn: Callable[[], object],
        *,
        on_complete: Callable[[object], None] | None = None,
    ) -> TaskHandle:
        """Queue model work; the supplied callback never runs on this caller."""
        handle = TaskHandle(key, kind)
        with self._condition:
            if self._closed or key != self._current_key:
                handle.cancel()
                handle._finish()
                return handle
            self._sequence += 1
            heapq.heappush(
                self._pending,
                _PendingTask(_PRIORITY[kind], self._sequence, handle, fn, on_complete),
            )
            self._condition.notify_all()
        return handle

    def run_correction(
        self,
        key: SessionKey,
        profile: AsrProfile,
        fn: Callable[[], object],
        *,
        on_complete: Callable[[CorrectionOutcome], None] | None = None,
    ) -> TaskHandle:
        """Queue Qwen only after every ASR profile is confirmed gone."""

        def guarded() -> CorrectionOutcome:
            other: AsrProfile = "sensevoice" if profile == "nano" else "nano"
            for candidate in (profile, other):
                # Always ask the producing profile to release: its process can
                # have outlived our local state. A second profile is released
                # whenever the scheduler has observed it load or become ready.
                needs_release = (
                    candidate == profile
                    or self._profile_states[candidate]
                    in {ModelLifecycle.LOADING, ModelLifecycle.READY}
                )
                try:
                    if needs_release:
                        if not self._stop_profile(candidate):
                            return CorrectionOutcome(permitted=False)
                        state = self._health_profile(candidate)
                    else:
                        state = self._profile_states[candidate]
                except Exception:  # noqa: BLE001 - deny lease on uncertainty
                    return CorrectionOutcome(permitted=False)
                if state not in {ModelLifecycle.INACTIVE, ModelLifecycle.FAILED}:
                    return CorrectionOutcome(permitted=False)
                self._profile_states[candidate] = state
            return CorrectionOutcome(permitted=True, value=fn())

        callback: Callable[[object], None] | None = None
        if on_complete is not None:
            def callback(value: object) -> None:
                assert on_complete is not None
                if isinstance(value, CorrectionOutcome):
                    on_complete(value)
        return self.submit(
            key, ModelTaskKind.CORRECTION, guarded, on_complete=callback
        )

    def run_asr(
        self,
        key: SessionKey,
        profile: AsrProfile,
        fn: Callable[[], object],
        *,
        kind: ModelTaskKind = ModelTaskKind.FINAL_TAIL,
    ) -> TaskHandle:
        """Queue ASR work and switch profiles only after a confirmed release."""
        if kind not in {
            ModelTaskKind.FINAL_TAIL,
            ModelTaskKind.STABLE_SEGMENT,
            ModelTaskKind.PROVISIONAL_TAIL,
        }:
            raise ValueError("ASR work must use an ASR task kind")

        def guarded() -> object:
            other: AsrProfile = "sensevoice" if profile == "nano" else "nano"
            if self._profile_states[other] is not ModelLifecycle.INACTIVE:
                if not self._stop_profile(other):
                    raise ModelProfileError("ASR profile release was unconfirmed")
                observed = self._health_profile(other)
                if observed not in {ModelLifecycle.INACTIVE, ModelLifecycle.FAILED}:
                    raise ModelProfileError("ASR profile remained active")
                self._profile_states[other] = observed
            if self._profile_states[profile] is not ModelLifecycle.READY:
                self._profile_states[profile] = ModelLifecycle.LOADING
                if not self._start_profile(profile):
                    self._profile_states[profile] = ModelLifecycle.FAILED
                    raise ModelProfileError("ASR profile did not start")
            try:
                result = fn()
            except Exception:
                self._profile_states[profile] = ModelLifecycle.FAILED
                raise
            self._profile_states[profile] = ModelLifecycle.READY
            return result

        return self.submit(key, kind, guarded)

    def profile_state(self, profile: AsrProfile) -> ModelLifecycle:
        """Return scheduler-owned profile state without a VRAM/process probe."""
        with self._condition:
            return self._profile_states[profile]

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until no queued or physically executing task remains."""
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._pending and self._running is None, timeout=timeout
            )

    def close(self) -> None:
        """Cancel queued work and stop the dispatcher after current work ends."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for task in self._pending:
                task.handle.cancel()
            if self._running is not None:
                self._running.cancel()
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _dispatch(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._pending))
                if self._closed and not self._pending:
                    return
                task = heapq.heappop(self._pending)
                if task.handle.cancelled or task.handle.key != self._current_key:
                    task.handle.cancel()
                    task.handle._finish()
                    self._condition.notify_all()
                    continue
                self._running = task.handle
            try:
                value = task.fn()
            except Exception as exc:  # noqa: BLE001 - owner records operation-specific error
                value = None
                error: Exception | None = exc
            else:
                error = None
            with self._condition:
                publish = (
                    not task.handle.cancelled
                    and task.handle.key == self._current_key
                    and not self._closed
                )
                self._running = None
                task.handle._finish(value=value, error=error)
                self._condition.notify_all()
            if publish and task.on_complete is not None:
                # A UI/daemon callback cannot kill the sole dispatcher.
                with suppress(Exception):
                    task.on_complete(value)

    @staticmethod
    def _deny_profile_stop(_profile: AsrProfile) -> bool:
        return False

    @staticmethod
    def _deny_profile_start(_profile: AsrProfile) -> bool:
        return False
