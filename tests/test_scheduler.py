"""Concurrency contracts for the single-owner XPU task scheduler."""

from __future__ import annotations

import threading

import pytest

from fun_voice.contracts import ModelTaskKind, SessionKey
from fun_voice.scheduler import ModelLifecycle, ModelProfileError, ModelScheduler


def _key(generation: int = 1) -> SessionKey:
    return SessionKey(session_id=f"session-{generation}", generation=generation)


def test_final_tail_precedes_queued_provisional_tail() -> None:
    scheduler = ModelScheduler()
    scheduler.activate(_key())
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def blocker() -> None:
        order.append("blocker")
        started.set()
        assert release.wait(timeout=1.0)

    scheduler.submit(_key(), ModelTaskKind.STABLE_SEGMENT, blocker)
    assert started.wait(timeout=1.0)
    scheduler.submit(
        _key(), ModelTaskKind.PROVISIONAL_TAIL, lambda: order.append("provisional")
    )
    final = scheduler.submit(
        _key(), ModelTaskKind.FINAL_TAIL, lambda: order.append("final")
    )
    release.set()

    assert final.wait(timeout=1.0)
    assert scheduler.wait_idle(timeout=1.0)
    assert order == ["blocker", "final", "provisional"]
    scheduler.close()


def test_all_model_tasks_run_on_one_dispatcher_without_overlap() -> None:
    scheduler = ModelScheduler()
    scheduler.activate(_key())
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    active = 0
    maximum = 0
    lock = threading.Lock()

    def first() -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        first_entered.set()
        assert release_first.wait(timeout=1.0)
        with lock:
            active -= 1

    def second() -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            active -= 1
        second_entered.set()

    scheduler.submit(_key(), ModelTaskKind.STABLE_SEGMENT, first)
    assert first_entered.wait(timeout=1.0)
    scheduler.submit(_key(), ModelTaskKind.FINAL_TAIL, second)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    assert second_entered.wait(timeout=1.0)
    assert maximum == 1
    scheduler.close()


def test_new_generation_cancels_queued_enrichment() -> None:
    scheduler = ModelScheduler()
    first = _key(1)
    second = _key(2)
    scheduler.activate(first)
    started = threading.Event()
    release = threading.Event()
    completed: list[str] = []

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=1.0)

    scheduler.submit(first, ModelTaskKind.STABLE_SEGMENT, blocker)
    assert started.wait(timeout=1.0)
    enrichment = scheduler.submit(
        first, ModelTaskKind.ENRICHMENT, lambda: completed.append("enrichment")
    )
    scheduler.activate(second)
    release.set()

    assert scheduler.wait_idle(timeout=1.0)
    assert enrichment.cancelled is True
    assert completed == []
    scheduler.close()


def test_cancelled_running_task_completes_physically_but_has_no_callback() -> None:
    scheduler = ModelScheduler()
    key = _key()
    scheduler.activate(key)
    started = threading.Event()
    release = threading.Event()
    physical: list[str] = []
    published: list[str] = []

    def decode() -> str:
        started.set()
        assert release.wait(timeout=1.0)
        physical.append("finished")
        return "result"

    handle = scheduler.submit(
        key,
        ModelTaskKind.FINAL_TAIL,
        decode,
        on_complete=lambda value: published.append(str(value)),
    )
    assert started.wait(timeout=1.0)
    handle.cancel()
    release.set()

    assert scheduler.wait_idle(timeout=1.0)
    assert physical == ["finished"]
    assert published == []
    scheduler.close()


def test_correction_runs_only_after_producing_asr_is_inactive() -> None:
    stopped: list[str] = []
    ran: list[str] = []
    scheduler = ModelScheduler(
        stop_profile=lambda profile: stopped.append(profile) or True,
        health_profile=lambda _profile: ModelLifecycle.INACTIVE,
    )
    key = _key()
    scheduler.activate(key)
    outcomes: list[tuple[bool, object | None]] = []

    handle = scheduler.run_correction(
        key,
        "nano",
        lambda: ran.append("qwen") or "corrected",
        on_complete=lambda outcome: outcomes.append((outcome.permitted, outcome.value)),
    )

    assert handle.wait(timeout=1.0)
    assert stopped == ["nano", "sensevoice"]
    assert ran == ["qwen"]
    assert outcomes == [(True, "corrected")]
    scheduler.close()


def test_correction_is_skipped_when_profile_release_is_uncertain() -> None:
    scheduler = ModelScheduler(
        stop_profile=lambda _profile: True,
        health_profile=lambda _profile: ModelLifecycle.READY,
    )
    key = _key()
    scheduler.activate(key)
    ran: list[str] = []
    outcomes: list[tuple[bool, object | None]] = []

    handle = scheduler.run_correction(
        key,
        "nano",
        lambda: ran.append("qwen") or "corrected",
        on_complete=lambda outcome: outcomes.append((outcome.permitted, outcome.value)),
    )

    assert handle.wait(timeout=1.0)
    assert ran == []
    assert outcomes == [(False, None)]
    scheduler.close()


def test_correction_reconciles_an_unobserved_sibling_before_qwen() -> None:
    stopped: list[str] = []
    checked: list[str] = []
    ran: list[str] = []

    def health(profile: str) -> ModelLifecycle:
        checked.append(profile)
        return (
            ModelLifecycle.READY
            if profile == "sensevoice"
            else ModelLifecycle.INACTIVE
        )

    scheduler = ModelScheduler(
        stop_profile=lambda profile: stopped.append(profile) or True,
        health_profile=health,
    )
    key = _key()
    scheduler.activate(key)

    handle = scheduler.run_correction(
        key, "nano", lambda: ran.append("qwen") or "corrected"
    )

    assert handle.wait(timeout=1.0)
    assert stopped == ["nano", "sensevoice"]
    assert checked == ["nano", "sensevoice"]
    assert ran == []
    assert getattr(handle.result(), "permitted", None) is False
    scheduler.close()


def test_correction_is_denied_when_another_asr_profile_remains_active() -> None:
    stopped: list[str] = []
    ran: list[str] = []

    def stop(profile: str) -> bool:
        stopped.append(profile)
        return profile != "sensevoice"

    scheduler = ModelScheduler(
        start_profile=lambda _profile: True,
        stop_profile=stop,
        health_profile=lambda _profile: ModelLifecycle.INACTIVE,
    )
    key = _key()
    scheduler.activate(key)
    sensevoice = scheduler.run_asr(key, "sensevoice", lambda: "asr")
    assert sensevoice.wait(timeout=1.0)

    handle = scheduler.run_correction(
        key,
        "nano",
        lambda: ran.append("qwen") or "corrected",
    )

    assert handle.wait(timeout=1.0)
    assert stopped == ["nano", "sensevoice"]
    assert ran == []
    scheduler.close()


def test_correction_rechecks_an_observed_failed_asr_profile() -> None:
    stopped: list[str] = []
    ran: list[str] = []
    state = {
        "nano": ModelLifecycle.INACTIVE,
        "sensevoice": ModelLifecycle.INACTIVE,
    }

    def start(profile: str) -> bool:
        state[profile] = ModelLifecycle.READY
        return True

    def stop(profile: str) -> bool:
        stopped.append(profile)
        state[profile] = ModelLifecycle.INACTIVE
        return True

    scheduler = ModelScheduler(
        start_profile=start,
        stop_profile=stop,
        health_profile=lambda profile: state[profile],
    )
    key = _key()
    scheduler.activate(key)

    failed_asr = scheduler.run_asr(
        key, "nano", lambda: (_ for _ in ()).throw(RuntimeError("ASR failed"))
    )
    assert failed_asr.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="ASR failed"):
        failed_asr.result()

    correction = scheduler.run_correction(
        key, "sensevoice", lambda: ran.append("qwen")
    )

    assert correction.wait(timeout=1.0)
    assert stopped == ["sensevoice", "nano"]
    assert ran == ["qwen"]
    scheduler.close()


def test_task_handle_returns_value_or_reraises_owner_error() -> None:
    scheduler = ModelScheduler()
    key = _key()
    scheduler.activate(key)

    success = scheduler.submit(key, ModelTaskKind.FINAL_TAIL, lambda: "value")
    failure = scheduler.submit(
        key,
        ModelTaskKind.STABLE_SEGMENT,
        lambda: (_ for _ in ()).throw(RuntimeError("model failed")),
    )

    assert success.wait(timeout=1.0)
    assert success.result() == "value"
    assert failure.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="model failed"):
        failure.result()
    scheduler.close()


def test_asr_profile_is_started_by_scheduler_and_switches_only_after_release() -> None:
    calls: list[tuple[str, str]] = []
    state = {
        "nano": ModelLifecycle.INACTIVE,
        "sensevoice": ModelLifecycle.INACTIVE,
    }

    def start(profile: str) -> bool:
        calls.append(("start", profile))
        state[profile] = ModelLifecycle.READY
        return True

    def stop(profile: str) -> bool:
        calls.append(("stop", profile))
        state[profile] = ModelLifecycle.INACTIVE
        return True

    scheduler = ModelScheduler(
        start_profile=start,
        stop_profile=stop,
        health_profile=lambda profile: state[profile],
    )
    key = _key()
    scheduler.activate(key)

    nano = scheduler.run_asr(key, "nano", lambda: "nano-result")
    assert nano.wait(timeout=1.0)
    assert nano.result() == "nano-result"
    sensevoice = scheduler.run_asr(key, "sensevoice", lambda: "fallback-result")
    assert sensevoice.wait(timeout=1.0)
    assert sensevoice.result() == "fallback-result"

    assert calls == [
        ("start", "nano"),
        ("stop", "nano"),
        ("start", "sensevoice"),
    ]
    scheduler.close()


def test_asr_restarts_after_a_cached_ready_profile_is_observed_inactive() -> None:
    health_states = iter(
        (ModelLifecycle.READY, ModelLifecycle.INACTIVE, ModelLifecycle.READY)
    )
    checked: list[str] = []
    starts: list[str] = []
    ran: list[str] = []
    scheduler = ModelScheduler(
        allowed_profiles=("nano",),
        start_profile=lambda profile: starts.append(profile) or True,
        stop_profile=lambda _profile: True,
        health_profile=lambda profile: checked.append(profile) or next(health_states),
    )
    key = _key()
    scheduler.activate(key)

    first = scheduler.run_asr(key, "nano", lambda: ran.append("first"))
    assert first.wait(timeout=1.0)
    second = scheduler.run_asr(key, "nano", lambda: ran.append("second"))
    assert second.wait(timeout=1.0)

    assert checked == ["nano", "nano", "nano"]
    assert starts == ["nano"]
    assert ran == ["first", "second"]
    scheduler.close()


def test_asr_unknown_health_denies_callback_when_restart_is_unconfirmed() -> None:
    health_states = iter((ModelLifecycle.READY, ModelLifecycle.UNKNOWN))
    checked: list[str] = []
    starts: list[str] = []
    ran: list[str] = []
    scheduler = ModelScheduler(
        allowed_profiles=("nano",),
        start_profile=lambda profile: starts.append(profile) or True,
        stop_profile=lambda _profile: True,
        health_profile=lambda profile: checked.append(profile) or next(health_states),
    )
    key = _key()
    scheduler.activate(key)

    initial = scheduler.run_asr(key, "nano", lambda: ran.append("initial"))
    assert initial.wait(timeout=1.0)
    handle = scheduler.run_asr(key, "nano", lambda: ran.append("asr"))

    assert handle.wait(timeout=1.0)
    with pytest.raises(ModelProfileError, match="health was unconfirmed"):
        handle.result()
    assert checked == ["nano", "nano"]
    assert starts == ["nano"]
    assert ran == ["initial"]
    scheduler.close()


def test_asr_waits_for_transport_ready_after_service_start() -> None:
    now = 0.0
    sleeps: list[float] = []
    state = {"nano": ModelLifecycle.INACTIVE}
    starts: list[str] = []
    ran: list[str] = []

    def monotonic() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds
        state["nano"] = ModelLifecycle.READY

    scheduler = ModelScheduler(
        allowed_profiles=("nano",),
        start_profile=lambda profile: (
            starts.append(profile)
            or state.__setitem__(profile, ModelLifecycle.LOADING)
            or True
        ),
        health_profile=lambda profile: state[profile],
        startup_timeout=1.0,
        monotonic=monotonic,
        sleep=sleep,
    )
    key = _key()
    scheduler.activate(key)

    handle = scheduler.run_asr(key, "nano", lambda: ran.append("asr"))

    assert handle.wait(timeout=1.0)
    assert handle.result() is None
    assert starts == ["nano"]
    assert sleeps == [0.05]
    assert ran == ["asr"]
    scheduler.close()


def test_asr_denies_callback_when_started_service_never_becomes_transport_ready() -> (
    None
):
    now = 0.0
    sleeps: list[float] = []
    state = {"nano": ModelLifecycle.INACTIVE}
    starts: list[str] = []
    ran: list[str] = []

    def monotonic() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    def start(profile: str) -> bool:
        starts.append(profile)
        state[profile] = ModelLifecycle.UNKNOWN
        return True

    scheduler = ModelScheduler(
        allowed_profiles=("nano",),
        start_profile=start,
        health_profile=lambda profile: state[profile],
        startup_timeout=0.1,
        monotonic=monotonic,
        sleep=sleep,
    )
    key = _key()
    scheduler.activate(key)

    handle = scheduler.run_asr(key, "nano", lambda: ran.append("asr"))

    assert handle.wait(timeout=1.0)
    with pytest.raises(ModelProfileError, match="transport was not ready"):
        handle.result()
    assert starts == ["nano"]
    assert sleeps == [0.05, 0.05]
    assert ran == []
    assert scheduler.profile_state("nano") is ModelLifecycle.UNKNOWN
    scheduler.close()


def test_asr_releases_freshly_observed_active_sibling_before_target_runs() -> None:
    state = {
        "nano": ModelLifecycle.INACTIVE,
        "sensevoice": ModelLifecycle.READY,
    }
    events: list[tuple[str, str] | str] = []

    def health(profile: str) -> ModelLifecycle:
        events.append(("health", profile))
        return state[profile]

    def stop(profile: str) -> bool:
        events.append(("stop", profile))
        state[profile] = ModelLifecycle.INACTIVE
        return True

    def start(profile: str) -> bool:
        events.append(("start", profile))
        state[profile] = ModelLifecycle.READY
        return True

    scheduler = ModelScheduler(
        allowed_profiles=("nano", "sensevoice"),
        start_profile=start,
        stop_profile=stop,
        health_profile=health,
    )
    key = _key()
    scheduler.activate(key)

    handle = scheduler.run_asr(key, "nano", lambda: events.append("asr"))

    assert handle.wait(timeout=1.0)
    assert events == [
        ("health", "sensevoice"),
        ("stop", "sensevoice"),
        ("health", "sensevoice"),
        ("health", "nano"),
        ("start", "nano"),
        ("health", "nano"),
        "asr",
    ]
    scheduler.close()


def test_asr_denies_unknown_allowed_sibling_before_starting_target() -> None:
    checked: list[str] = []
    starts: list[str] = []
    ran: list[str] = []
    scheduler = ModelScheduler(
        allowed_profiles=("nano", "sensevoice"),
        start_profile=lambda profile: starts.append(profile) or True,
        health_profile=lambda profile: (
            checked.append(profile) or ModelLifecycle.UNKNOWN
        ),
    )
    key = _key()
    scheduler.activate(key)

    handle = scheduler.run_asr(key, "nano", lambda: ran.append("asr"))

    assert handle.wait(timeout=1.0)
    with pytest.raises(ModelProfileError, match="sibling health was unconfirmed"):
        handle.result()
    assert checked == ["sensevoice"]
    assert starts == []
    assert ran == []
    scheduler.close()


def test_cpu_scheduler_never_probes_disallowed_nano_profile() -> None:
    checked: list[str] = []
    scheduler = ModelScheduler(
        allowed_profiles=("sensevoice",),
        health_profile=lambda profile: checked.append(profile) or ModelLifecycle.READY,
    )
    key = _key()
    scheduler.activate(key)

    handle = scheduler.run_asr(key, "sensevoice", lambda: "asr")

    assert handle.wait(timeout=1.0)
    assert handle.result() == "asr"
    assert checked == ["sensevoice"]
    scheduler.close()
