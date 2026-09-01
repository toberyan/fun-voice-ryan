"""Private, non-interactive transient status overlay for X11.

The daemon submits immutable models to a queue; the dedicated UI thread owns
the Xlib display, window and graphics context.  The overlay never requests
input focus, writes a selection, sends an XTEST event or logs supplied text.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from fun_voice.contracts import DaemonState
from fun_voice.desktop import default_make_display

STABLE_DARK = 0x202020
PROVISIONAL_LIGHT = 0x7A7A7A
WINDOW_WIDTH = 420
WINDOW_HEIGHT = 112
WINDOW_OFFSET = 20


@dataclass(frozen=True, slots=True)
class OverlayModel:
    """An in-memory overlay snapshot; it is never a desktop input payload."""

    phase: DaemonState
    stable_text: str = ""
    provisional_text: str = ""
    level: int | None = None


@dataclass(frozen=True, slots=True)
class OverlayFrame:
    """Renderer-ready form with fixed visual treatment labels for tests."""

    phase: DaemonState
    stable_text: str
    provisional_text: str
    level: int | None
    stable_tone: Literal["dark"] = "dark"
    provisional_tone: Literal["light"] = "light"


class OverlayController(Protocol):
    """Non-blocking UI seam used by the daemon state machine."""

    def show(self, model: OverlayModel) -> None: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...


class _OverlayWindow(Protocol):
    def create_gc(self, **kwargs: object) -> object: ...

    def map(self) -> None: ...

    def unmap(self) -> None: ...

    def clear_area(self) -> None: ...

    def draw_text(self, gc: object, x: int, y: int, text: str) -> None: ...


class _OverlayRoot(Protocol):
    def query_pointer(self) -> object: ...

    def create_window(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        border_width: int,
        depth: int,
        window_class: int = 0,
        visual: int = 0,
        **kwargs: object,
    ) -> _OverlayWindow: ...


class _OverlayScreen(Protocol):
    root: _OverlayRoot
    root_depth: int
    white_pixel: int
    black_pixel: int
    width_in_pixels: int
    height_in_pixels: int


class _OverlayDisplay(Protocol):
    def screen(self) -> _OverlayScreen: ...

    def sync(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _Barrier:
    ready: threading.Event


class _Clear:
    pass


class _Close:
    pass


class NullOverlay:
    """No-op fallback when an X11 overlay cannot be constructed."""

    def show(self, model: OverlayModel) -> None:
        del model

    def clear(self) -> None:
        pass

    def close(self) -> None:
        pass


class X11TransientOverlay:
    """Draw a small override-redirect X11 status window on one UI thread."""

    def __init__(
        self,
        *,
        make_display: Callable[[], _OverlayDisplay] | None = None,
    ) -> None:
        self._make_display = (
            make_display
            if make_display is not None
            else cast(Callable[[], _OverlayDisplay], default_make_display)
        )
        self._commands: queue.Queue[OverlayModel | _Clear | _Close | _Barrier] = (
            queue.Queue()
        )
        self._closed = False
        self._unavailable = False
        self._state_lock = threading.Lock()
        self._last_frame: OverlayFrame | None = None
        self._current_model: OverlayModel | None = None
        self._thread = threading.Thread(
            target=self._run, name="x11-transient-overlay", daemon=True
        )
        self._thread.start()

    @property
    def unavailable(self) -> bool:
        with self._state_lock:
            return self._unavailable

    @property
    def last_frame(self) -> OverlayFrame | None:
        with self._state_lock:
            return self._last_frame

    @property
    def current_model(self) -> OverlayModel | None:
        with self._state_lock:
            return self._current_model

    def show(self, model: OverlayModel) -> None:
        with self._state_lock:
            if self._closed:
                return
        self._commands.put(model)

    def clear(self) -> None:
        with self._state_lock:
            if self._closed:
                return
        self._commands.put(_Clear())

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._commands.put(_Close())
        self._thread.join(timeout=1.0)

    def wait_idle(self, timeout: float = 1.0) -> bool:
        """Test-only barrier that never exposes the Xlib objects to callers."""
        with self._state_lock:
            if self._closed:
                return True
        ready = threading.Event()
        self._commands.put(_Barrier(ready))
        return ready.wait(timeout)

    def _run(self) -> None:
        display: _OverlayDisplay | None = None
        window: _OverlayWindow | None = None
        gc: object | None = None
        while True:
            command = self._commands.get()
            try:
                if isinstance(command, _Barrier):
                    command.ready.set()
                    continue
                if isinstance(command, _Close):
                    self._clear_window(display, window)
                    self._clear_model()
                    return
                if self.unavailable:
                    self._clear_model()
                    continue
                if display is None:
                    try:
                        display, window, gc = self._open_window()
                    except Exception:  # no text/model error ever leaves this thread
                        with self._state_lock:
                            self._unavailable = True
                        self._clear_model()
                        continue
                if isinstance(command, _Clear):
                    self._clear_window(display, window)
                    self._clear_model()
                    continue
                frame = OverlayFrame(
                    phase=command.phase,
                    stable_text=command.stable_text,
                    provisional_text=command.provisional_text,
                    level=command.level,
                )
                assert window is not None and gc is not None
                self._draw_frame(display, window, gc, frame)
                with self._state_lock:
                    self._current_model = command
                    self._last_frame = frame
            finally:
                self._commands.task_done()
                if isinstance(command, _Close) and display is not None:
                    with suppress(Exception):
                        display.close()

    def _open_window(self) -> tuple[_OverlayDisplay, _OverlayWindow, object]:
        """Create the only overlay window; input event selection is explicitly 0."""
        from Xlib import X

        display = self._make_display()
        screen = display.screen()
        root = screen.root
        pointer = root.query_pointer()
        pointer_x = getattr(pointer, "root_x", WINDOW_OFFSET)
        pointer_y = getattr(pointer, "root_y", WINDOW_OFFSET)
        x = max(
            0,
            min(
                int(pointer_x) + WINDOW_OFFSET,
                screen.width_in_pixels - WINDOW_WIDTH,
            ),
        )
        y = max(
            0,
            min(
                int(pointer_y) + WINDOW_OFFSET,
                screen.height_in_pixels - WINDOW_HEIGHT,
            ),
        )
        window = root.create_window(
            x=x,
            y=y,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            border_width=0,
            depth=screen.root_depth,
            window_class=X.InputOutput,
            visual=X.CopyFromParent,
            background_pixel=screen.white_pixel,
            override_redirect=1,
            event_mask=0,
        )
        gc = window.create_gc(foreground=screen.black_pixel)
        return display, window, gc

    def _draw_frame(
        self,
        display: _OverlayDisplay,
        window: _OverlayWindow,
        gc: object,
        frame: OverlayFrame,
    ) -> None:
        window.clear_area()
        self._set_foreground(gc, STABLE_DARK)
        window.draw_text(gc, 12, 22, _phase_label(frame.phase))
        y = 46
        if frame.level is not None:
            safe_level = max(0, min(100, frame.level))
            window.draw_text(gc, 12, y, f"音量 {safe_level}%")
            y += 20
        if frame.stable_text:
            self._set_foreground(gc, STABLE_DARK)
            window.draw_text(gc, 12, y, frame.stable_text)
            y += 20
        if frame.provisional_text:
            self._set_foreground(gc, PROVISIONAL_LIGHT)
            window.draw_text(gc, 12, y, frame.provisional_text)
        window.map()
        display.sync()

    @staticmethod
    def _set_foreground(gc: object, color: int) -> None:
        change = getattr(gc, "change", None)
        if callable(change):
            change(foreground=color)

    @staticmethod
    def _clear_window(
        display: _OverlayDisplay | None, window: _OverlayWindow | None
    ) -> None:
        if window is None:
            return
        try:
            window.clear_area()
            window.unmap()
            if display is not None:
                display.sync()
        except Exception:
            pass

    def _clear_model(self) -> None:
        with self._state_lock:
            self._current_model = None
            self._last_frame = None


def _phase_label(phase: DaemonState) -> str:
    """Return a fixed status label; it never contains user-provided text."""
    return {
        DaemonState.PREPARING: "正在准备本地模型",
        DaemonState.RECORDING: "录音中",
        DaemonState.FINALIZING: "正在整理",
        DaemonState.CORRECTING: "正在精修",
        DaemonState.COMMITTING: "正在输入",
        DaemonState.REHYDRATING: "正在恢复本地模型",
        DaemonState.ENRICHING: "正在整理结果",
        DaemonState.ACTIVE_IDLE: "本地模型就绪",
    }.get(phase, "语音输入")
