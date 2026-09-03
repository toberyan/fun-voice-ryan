"""Concurrency contracts for the single-owner XPU task scheduler."""

from __future__ import annotations

import threading

import pytest

from fun_voice.contracts import ModelTaskKind, SessionKey
from fun_voice.scheduler import ModelLifecycle, ModelScheduler


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
    scheduler = ModelScheduler(
        start_profile=lambda _profile: True,
        stop_profile=lambda profile: stopped.append(profile) or True,
        health_profile=lambda _profile: ModelLifecycle.INACTIVE,
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

    def start(profile: str) -> bool:
        calls.append(("start", profile))
        return True

    def stop(profile: str) -> bool:
        calls.append(("stop", profile))
        return True

    scheduler = ModelScheduler(
        start_profile=start,
        stop_profile=stop,
        health_profile=lambda _profile: ModelLifecycle.INACTIVE,
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
