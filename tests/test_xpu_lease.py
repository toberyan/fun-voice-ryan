"""Process-level mutual exclusion tests for ASR and Qwen XPU residency."""

from __future__ import annotations

from fun_voice.xpu_lease import XpuLeaseCoordinator


def test_qwen_lease_stops_the_producing_asr_profile() -> None:
    calls: list[str] = []
    lease = XpuLeaseCoordinator(
        stop_service=lambda profile: calls.append(profile) or True
    )

    assert lease.release_asr_for_qwen("nano") is True
    assert calls == ["nano"]
    assert lease.last_release_ms is not None
    assert lease.last_release_ms >= 0


def test_qwen_lease_converts_stop_errors_to_a_safe_rejection() -> None:
    def fail(_profile: str) -> bool:
        raise RuntimeError("service manager unavailable")

    assert XpuLeaseCoordinator(stop_service=fail).release_asr_for_qwen("nano") is False
