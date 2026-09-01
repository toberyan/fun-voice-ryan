"""Conservative process-level lease for serial ASR and Qwen XPU use."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

AsrProfile = Literal["nano", "sensevoice"]


class XpuLeaseCoordinator:
    """Permit Qwen only after the profile that produced ASR has stopped.

    The supplied callback must return ``True`` only after systemd has observed
    that the exact template instance is inactive.  There is intentionally no
    optimistic VRAM probe or fallback model path.
    """

    def __init__(self, *, stop_service: Callable[[AsrProfile], bool]) -> None:
        self._stop_service = stop_service
        self.last_release_ms: int | None = None

    def release_asr_for_qwen(self, profile: AsrProfile) -> bool:
        started = time.perf_counter()
        try:
            return self._stop_service(profile)
        except Exception:  # noqa: BLE001 - deny lease when confirmation fails
            return False
        finally:
            self.last_release_ms = max(
                0, round((time.perf_counter() - started) * 1000)
            )
