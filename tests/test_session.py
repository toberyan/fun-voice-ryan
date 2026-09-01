"""Pure active-session policy and lifecycle tests without model imports."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fun_voice.config import ActiveSessionConfig, ResourcePolicy
from fun_voice.contracts import DaemonState
from fun_voice.session import (
    ActiveSessionController,
    SessionActionKind,
)


@dataclass
class FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _controller(
    *,
    policy: ResourcePolicy = ResourcePolicy.BALANCED,
    on_ac_power: bool = True,
    incremental_poc_approved: bool = False,
) -> tuple[ActiveSessionController, FakeClock, list[SessionActionKind]]:
    clock = FakeClock()
    actions: list[SessionActionKind] = []
    controller = ActiveSessionController(
        ActiveSessionConfig.for_policy(policy, provisional_enabled=True),
        clock=clock,
        on_ac_power=lambda: on_ac_power,
        incremental_poc_approved=incremental_poc_approved,
        emit=lambda action: actions.append(action.kind),
    )
    return controller, clock, actions


def test_cold_press_prepares_nano_then_nonempty_result_enters_active_idle() -> None:
    controller, clock, actions = _controller()

    key = controller.on_press()

    assert controller.state is DaemonState.PREPARING
    assert actions == [SessionActionKind.START_NANO, SessionActionKind.SHOW_OVERLAY]
    assert controller.on_nano_ready(key) is True
    assert controller.state is DaemonState.RECORDING

    assert controller.on_finalized(key, has_text=True) is True

    assert controller.state is DaemonState.ACTIVE_IDLE
    assert controller.active_deadline == pytest.approx(clock.now + 480)
    assert actions[-1] is SessionActionKind.ENQUEUE_ENRICHMENT


@pytest.mark.parametrize(
    ("policy", "window"),
    [
        (ResourcePolicy.MEMORY_SAVER, 120),
        (ResourcePolicy.BALANCED, 480),
        (ResourcePolicy.SUSTAINED, 1800),
    ],
)
def test_policy_window_stops_nano_exactly_at_deadline(
    policy: ResourcePolicy, window: int
) -> None:
    controller, clock, actions = _controller(policy=policy)
    key = controller.on_press()
    assert controller.on_nano_ready(key)
    assert controller.on_finalized(key, has_text=True)

    clock.advance(window - 1)
    assert controller.tick() is False
    assert controller.state is DaemonState.ACTIVE_IDLE

    clock.advance(1)
    assert controller.tick() is True
    assert controller.state is DaemonState.IDLE
    assert actions[-1] is SessionActionKind.STOP_MODELS


@pytest.mark.parametrize(
    "stop",
    ["on_lock", "on_resource_pressure", "on_memory_saver"],
)
def test_urgent_resource_events_stop_active_nano_immediately(stop: str) -> None:
    controller, _clock, actions = _controller()
    key = controller.on_press()
    assert controller.on_nano_ready(key)
    assert controller.on_finalized(key, has_text=True)

    getattr(controller, stop)()

    assert controller.state is DaemonState.IDLE
    assert actions[-1] is SessionActionKind.STOP_MODELS


def test_second_model_failure_stops_the_active_session() -> None:
    controller, _clock, actions = _controller()
    key = controller.on_press()
    assert controller.on_nano_ready(key)
    assert controller.on_finalized(key, has_text=True)

    assert controller.on_model_failure(key) is False
    assert controller.state is DaemonState.ACTIVE_IDLE
    assert controller.on_model_failure(key) is True

    assert controller.state is DaemonState.IDLE
    assert actions[-1] is SessionActionKind.STOP_MODELS


def test_stale_generation_callbacks_have_no_side_effect() -> None:
    controller, _clock, actions = _controller()
    first = controller.on_press()
    assert controller.on_nano_ready(first)
    assert controller.on_finalized(first, has_text=True)
    second = controller.on_press()
    action_count = len(actions)

    assert second.generation == first.generation + 1
    assert controller.on_nano_ready(first) is False
    assert controller.on_enrichment_ready(first) is False
    assert len(actions) == action_count


def test_sustained_policy_on_battery_degrades_to_balanced_with_fixed_reason() -> None:
    controller, _clock, _actions = _controller(
        policy=ResourcePolicy.SUSTAINED, on_ac_power=False
    )

    assert controller.effective_policy is ResourcePolicy.BALANCED
    assert controller.policy_reason == "ac_required"


def test_provisional_text_stays_disabled_without_an_approved_incremental_poc() -> None:
    controller, _clock, _actions = _controller(incremental_poc_approved=False)

    assert controller.provisional_enabled is False
