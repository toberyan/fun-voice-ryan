"""Unit tests for the non-interactive transient X11 overlay."""

from __future__ import annotations

from dataclasses import dataclass

from fun_voice.contracts import DaemonState
from fun_voice.overlay import OverlayModel, X11TransientOverlay


@dataclass
class FakePointer:
    root_x: int = 320
    root_y: int = 240


class FakeGc:
    def __init__(self) -> None:
        self.foregrounds: list[int] = []

    def change(self, *, foreground: int) -> None:
        self.foregrounds.append(foreground)


class FakeWindow:
    def __init__(self) -> None:
        self.gc = FakeGc()
        self.maps = 0
        self.unmaps = 0
        self.clears = 0
        self.draws: list[str] = []

    def create_gc(self, **_kwargs: object) -> FakeGc:
        return self.gc

    def map(self) -> None:
        self.maps += 1

    def unmap(self) -> None:
        self.unmaps += 1

    def clear_area(self) -> None:
        self.clears += 1

    def draw_text(self, _gc: FakeGc, _x: int, _y: int, text: str) -> None:
        self.draws.append(text)


class FakeRoot:
    def __init__(self) -> None:
        self.window = FakeWindow()
        self.create_calls: list[dict[str, object]] = []
        self.focus_calls = 0

    def query_pointer(self) -> FakePointer:
        return FakePointer()

    def create_window(self, **kwargs: object) -> FakeWindow:
        self.create_calls.append(kwargs)
        return self.window

    def set_input_focus(self) -> None:
        self.focus_calls += 1


class FakeScreen:
    def __init__(self) -> None:
        self.root = FakeRoot()
        self.root_depth = 24
        self.white_pixel = 0xFFFFFF
        self.black_pixel = 0
        self.width_in_pixels = 1920
        self.height_in_pixels = 1080


class FakeDisplay:
    def __init__(self) -> None:
        self._screen = FakeScreen()
        self.sync_calls = 0
        self.close_calls = 0

    def screen(self) -> FakeScreen:
        return self._screen

    def sync(self) -> None:
        self.sync_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_overlay_maps_an_override_redirect_window_without_focusing() -> None:
    display = FakeDisplay()
    overlay = X11TransientOverlay(make_display=lambda: display)
    try:
        overlay.show(OverlayModel(phase=DaemonState.PREPARING))
        assert overlay.wait_idle(timeout=1.0)

        root = display.screen().root
        assert root.create_calls[0]["override_redirect"] == 1
        assert root.create_calls[0]["event_mask"] == 0
        assert root.window.maps == 1
        assert root.focus_calls == 0
    finally:
        overlay.close()


def test_overlay_marks_stable_and_provisional_content_with_distinct_tones() -> None:
    display = FakeDisplay()
    overlay = X11TransientOverlay(make_display=lambda: display)
    try:
        overlay.show(
            OverlayModel(
                phase=DaemonState.RECORDING,
                stable_text="stable",
                provisional_text="tail",
                level=42,
            )
        )
        assert overlay.wait_idle(timeout=1.0)

        frame = overlay.last_frame
        assert frame is not None
        assert frame.stable_tone == "dark"
        assert frame.provisional_tone == "light"
        assert "stable" in display.screen().root.window.draws
        assert "tail" in display.screen().root.window.draws
    finally:
        overlay.close()


def test_overlay_clear_overwrites_unmaps_and_discards_text_references() -> None:
    display = FakeDisplay()
    overlay = X11TransientOverlay(make_display=lambda: display)
    try:
        overlay.show(
            OverlayModel(
                phase=DaemonState.RECORDING,
                stable_text="private stable",
                provisional_text="private tail",
            )
        )
        assert overlay.wait_idle(timeout=1.0)

        overlay.clear()
        assert overlay.wait_idle(timeout=1.0)

        window = display.screen().root.window
        assert window.clears >= 2
        assert window.unmaps == 1
        assert overlay.last_frame is None
        assert overlay.current_model is None
    finally:
        overlay.close()
        overlay.close()
    assert display.close_calls == 1


def test_overlay_becomes_a_noop_after_x11_creation_failure() -> None:
    attempts: list[None] = []

    def fail_display() -> FakeDisplay:
        attempts.append(None)
        raise RuntimeError("no display")

    overlay = X11TransientOverlay(make_display=fail_display)
    try:
        overlay.show(OverlayModel(phase=DaemonState.PREPARING))
        assert overlay.wait_idle(timeout=1.0)
        overlay.show(OverlayModel(phase=DaemonState.RECORDING))
        assert overlay.wait_idle(timeout=1.0)

        assert overlay.unavailable is True
        assert overlay.current_model is None
        assert attempts == [None]
    finally:
        overlay.close()
