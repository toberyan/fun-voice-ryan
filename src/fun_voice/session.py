"""Pure active-session policy state machine with no desktop or model imports."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from fun_voice.config import ActiveSessionConfig, ResourcePolicy
from fun_voice.contracts import DaemonState, SessionKey


class SessionActionKind(StrEnum):
    """Side effects requested by the pure session controller."""

    START_NANO = "start_nano"
    SHOW_OVERLAY = "show_overlay"
    FINALIZE = "finalize"
    BEGIN_CORRECTION = "begin_correction"
    REHYDRATE = "rehydrate"
    ENQUEUE_ENRICHMENT = "enqueue_enrichment"
    STOP_MODELS = "stop_models"


@dataclass(frozen=True, slots=True)
class SessionAction:
    """A non-sensitive, key-bound action emitted to the daemon integration."""

    kind: SessionActionKind
    key: SessionKey


class ActiveSessionController:
    """Own active Nano residency without knowing how models or UI are run."""

    def __init__(
        self,
        config: ActiveSessionConfig,
        *,
        clock: Callable[[], float],
        on_ac_power: Callable[[], bool],
        incremental_poc_approved: bool,
        emit: Callable[[SessionAction], None],
    ) -> None:
        self._config = config
        self._clock = clock
        self._emit = emit
        self._effective_policy = config.policy
        self._policy_reason = "configured"
        if config.policy is ResourcePolicy.SUSTAINED and not on_ac_power():
            self._effective_policy = ResourcePolicy.BALANCED
            self._policy_reason = "ac_required"
        self._active_window = ActiveSessionConfig.for_policy(
            self._effective_policy,
            provisional_enabled=config.provisional_enabled,
            worker_failsafe_idle_seconds=config.worker_failsafe_idle_seconds,
        ).active_idle_seconds
        self._provisional_enabled = (
            config.provisional_enabled and incremental_poc_approved
        )
        self._state = DaemonState.IDLE
        self._generation = 0
        self._key: SessionKey | None = None
        self._active_deadline: float | None = None
        self._model_failures = 0

    @property
    def state(self) -> DaemonState:
        return self._state

    @property
    def active_deadline(self) -> float | None:
        return self._active_deadline

    @property
    def effective_policy(self) -> ResourcePolicy:
        return self._effective_policy

    @property
    def policy_reason(self) -> str:
        return self._policy_reason

    @property
    def provisional_enabled(self) -> bool:
        return self._provisional_enabled

    def on_press(self) -> SessionKey:
        """Begin a recording generation from cold or hot active-idle state."""
        if self._state not in {DaemonState.IDLE, DaemonState.ACTIVE_IDLE}:
            raise RuntimeError("session is already busy")
        hot = self._state is DaemonState.ACTIVE_IDLE
        self._generation += 1
        self._key = SessionKey(
            session_id=secrets.token_urlsafe(16), generation=self._generation
        )
        self._active_deadline = None
        self._model_failures = 0
        self._state = DaemonState.RECORDING if hot else DaemonState.PREPARING
        if not hot:
            self._emit_action(SessionActionKind.START_NANO)
        self._emit_action(SessionActionKind.SHOW_OVERLAY)
        return self._key

    def on_nano_ready(self, key: SessionKey) -> bool:
        """Accept a Nano-ready completion only for the current preparation."""
        if not self._is_current(key) or self._state is not DaemonState.PREPARING:
            return False
        self._state = DaemonState.RECORDING
        self._emit_action(SessionActionKind.SHOW_OVERLAY)
        return True

    def on_finalized(self, key: SessionKey, *, has_text: bool) -> bool:
        """Finish one recording and establish an active window on success."""
        if not self._is_current(key) or self._state is not DaemonState.RECORDING:
            return False
        if not has_text:
            self._stop_models()
            return True
        self._state = DaemonState.ACTIVE_IDLE
        self._active_deadline = self._clock() + self._active_window
        self._emit_action(SessionActionKind.ENQUEUE_ENRICHMENT)
        return True

    def on_enrichment_ready(self, key: SessionKey) -> bool:
        """Only report whether a background completion still belongs to this key."""
        return self._is_current(key)

    def on_model_failure(self, key: SessionKey) -> bool:
        """Release Nano after two consecutive current-generation model failures."""
        if not self._is_current(key):
            return False
        self._model_failures += 1
        if self._model_failures < 2:
            return False
        self._stop_models()
        return True

    def on_lock(self) -> None:
        self._stop_models()

    def on_resource_pressure(self) -> None:
        self._stop_models()

    def on_memory_saver(self) -> None:
        self._stop_models()

    def tick(self) -> bool:
        """Expire the active session exactly at its policy-owned deadline."""
        deadline = self._active_deadline
        if self._state is not DaemonState.ACTIVE_IDLE or deadline is None:
            return False
        if self._clock() < deadline:
            return False
        self._stop_models()
        return True

    def _is_current(self, key: SessionKey) -> bool:
        return self._key == key

    def _emit_action(self, kind: SessionActionKind) -> None:
        key = self._key
        assert key is not None
        self._emit(SessionAction(kind=kind, key=key))

    def _stop_models(self) -> None:
        if self._key is not None:
            self._emit_action(SessionActionKind.STOP_MODELS)
        self._state = DaemonState.IDLE
        self._active_deadline = None
