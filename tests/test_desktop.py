"""Tests for X11 hotkey/focus, clipboard and XTEST desktop adapters.

Every external boundary is a replaceable adapter, so these tests use fakes for
the command runner, the X11 display and the monotonic clock. No real X server
is required.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from fun_voice.contracts import FocusSnapshot
from fun_voice.desktop import (
    X_KEY_PRESS,
    X_KEY_RELEASE,
    XK_C,
    XK_CONTROL_L,
    XK_V,
    ClipboardError,
    ClipboardMirror,
    X11Error,
    X11FocusGuard,
    X11HotkeyListener,
    X11HotkeyUnavailable,
    XTestError,
    XTestInjector,
    default_runner,
)

# --- Fakes -------------------------------------------------------------------

class FakeRunner:
    """Records invocations and returns canned (returncode, stdout, stderr)."""

    def __init__(self, *results: tuple[int, str, str]) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []
        self._results = list(results)

    def __call__(
        self,
        argv: list[str] | tuple[str, ...],
        input_text: str | None = None,
        *,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        self.calls.append((tuple(argv), input_text))
        if self._results:
            return self._results.pop(0)
        return (0, "", "")


class _FakeProperty:
    def __init__(self, value: list[int]) -> None:
        self.value = value


class _FakeRootWindow:
    def __init__(self, active_window: int | None) -> None:
        self._active = active_window

    def get_full_property(self, atom: str, prop_type: int) -> _FakeProperty | None:
        if atom == "_NET_ACTIVE_WINDOW" and self._active is not None:
            return _FakeProperty([self._active])
        return None


class _FakeWindow:
    def __init__(self, wid: int, pid: int | None) -> None:
        self.id = wid
        self._pid = pid

    def get_full_property(self, atom: str, prop_type: int) -> _FakeProperty | None:
        if atom == "_NET_WM_PID" and self._pid is not None:
            return _FakeProperty([self._pid])
        return None


class _FakeScreen:
    def __init__(self, root: _FakeRootWindow) -> None:
        self.root = root


class _InputFocusReply:
    """Matches python-xlib's attribute-based GetInputFocus reply."""

    def __init__(self, focus: int | None, revert_to: int = 0) -> None:
        self.focus = focus
        self.revert_to = revert_to


class FakeDisplay:
    """Minimal X display double shared by the focus guard and XTEST injector."""

    def __init__(
        self,
        *,
        active_window: int | None = None,
        input_focus: int | None = None,
        window_pid: int | None = None,
        keymap: list[int] | None = None,
        c_keycode: int = 54,
        explode: bool = False,
        input_focus_reply: bool = False,
    ) -> None:
        self.root = _FakeRootWindow(active_window)
        self._focus = input_focus
        self._pid = window_pid
        self._keymap = keymap if keymap is not None else [0] * 32
        self._c_keycode = c_keycode
        self._explode = explode
        self._input_focus_reply = input_focus_reply
        self.keysym_calls: list[int] = []
        self.sync_calls = 0
        self.fake_input_events: list[tuple[int, int]] = []

    def screen(self) -> _FakeScreen:
        return _FakeScreen(self.root)

    def intern_atom(self, name: str) -> str:
        return name

    def get_input_focus(self) -> tuple[int | None, int] | _InputFocusReply:
        if self._explode:
            raise RuntimeError("X server connection lost")
        if self._input_focus_reply:
            return _InputFocusReply(self._focus)
        return (self._focus, 0)

    def create_resource_object(self, kind: str, wid: int) -> _FakeWindow:
        return _FakeWindow(wid, self._pid)

    def query_keymap(self) -> list[int]:
        return self._keymap

    def keysym_to_keycode(self, keysym: int) -> int:
        self.keysym_calls.append(keysym)
        if keysym == XK_C:
            return self._c_keycode
        if keysym == XK_CONTROL_L:
            return 37
        if keysym == XK_V:
            return 55
        return 0

    def sync(self) -> None:
        self.sync_calls += 1


class _FakeHotkeyRoot(_FakeRootWindow):
    """X11 root double that records passive keyboard grabs."""

    def __init__(self, display: FakeHotkeyDisplay) -> None:
        super().__init__(active_window=None)
        self._display = display

    def grab_key(
        self,
        key: int,
        modifiers: int,
        owner_events: bool,
        pointer_mode: int,
        keyboard_mode: int,
        onerror: object | None = None,
    ) -> None:
        self._display.grabs.append((key, modifiers))
        if modifiers == self._display.fail_modifier and callable(onerror):
            self._display.grabs_before_failure = list(self._display.grabs[:-1])
            onerror(RuntimeError("BadAccess"), object())

    def ungrab_key(self, key: int, modifiers: int) -> None:
        self._display.ungrabs.append((key, modifiers))


class FakeHotkeyDisplay:
    """Minimal X11 display double for global grab registration and events."""

    c_keycode = 54
    super_keycode = 133
    num_lock_keycode = 77
    scroll_lock_keycode = 78

    def __init__(
        self,
        *,
        super_index: int = 6,
        num_index: int = 4,
        scroll_index: int = 7,
        fail_modifier: int | None = None,
        c_keycode: int | None = None,
    ) -> None:
        self.c_keycode = self.c_keycode if c_keycode is None else c_keycode
        self._root = _FakeHotkeyRoot(self)
        self._mapping = [[] for _ in range(8)]
        self._mapping[super_index].append(self.super_keycode)
        self._mapping[num_index].append(self.num_lock_keycode)
        self._mapping[scroll_index].append(self.scroll_lock_keycode)
        self.fail_modifier = fail_modifier
        self.grabs: list[tuple[int, int]] = []
        self.grabs_before_failure: list[tuple[int, int]] = []
        self.ungrabs: list[tuple[int, int]] = []
        self.sync_calls = 0
        self.closed = False

    def screen(self) -> _FakeScreen:
        return _FakeScreen(self._root)

    def keysym_to_keycode(self, keysym: int) -> int:
        mapping = {
            XK_C: self.c_keycode,
            0xFFEB: self.super_keycode,  # XK_Super_L
            0xFFEC: 134,  # XK_Super_R
            0xFF7F: self.num_lock_keycode,  # XK_Num_Lock
            0xFF14: self.scroll_lock_keycode,  # XK_Scroll_Lock
        }
        return mapping.get(keysym, 0)

    def get_modifier_mapping(self) -> list[list[int]]:
        return self._mapping

    def fileno(self) -> int:
        return 0

    def next_event(self) -> object:
        raise AssertionError("select_ready=False must avoid next_event")

    def pending_events(self) -> int:
        return 0

    def sync(self) -> None:
        self.sync_calls += 1

    def close(self) -> None:
        self.closed = True


class FakeKeyEvent:
    def __init__(self, event_type: int, detail: int, state: int) -> None:
        self.type = event_type
        self.detail = detail
        self.state = state


def _make_hotkey_listener(
    display: FakeHotkeyDisplay,
    on_press: Callable[[], None],
    on_release: Callable[[], None],
) -> X11HotkeyListener:
    return X11HotkeyListener(
        on_press,
        on_release,
        make_display=lambda: display,
        select_ready=lambda _fd, _timeout: False,
    )


# --- X11 global hotkey -------------------------------------------------------


def test_x11_hotkey_grabs_super_c_with_every_lock_combination() -> None:
    display = FakeHotkeyDisplay(super_index=6, num_index=4, scroll_index=7)
    listener = _make_hotkey_listener(display, lambda: None, lambda: None)

    listener.start()

    super_mask = 1 << 6
    lock_mask = 1 << 1
    num_mask = 1 << 4
    scroll_mask = 1 << 7
    assert {modifiers for _key, modifiers in display.grabs} == {
        super_mask,
        super_mask | lock_mask,
        super_mask | num_mask,
        super_mask | scroll_mask,
        super_mask | lock_mask | num_mask,
        super_mask | lock_mask | scroll_mask,
        super_mask | num_mask | scroll_mask,
        super_mask | lock_mask | num_mask | scroll_mask,
    }
    listener.close()


def test_x11_hotkey_rolls_back_every_successful_grab_on_bad_access() -> None:
    super_mask = 1 << 6
    lock_mask = 1 << 1
    display = FakeHotkeyDisplay(fail_modifier=super_mask | lock_mask)
    listener = _make_hotkey_listener(display, lambda: None, lambda: None)
    with pytest.raises(X11HotkeyUnavailable, match="already grabbed"):
        listener.start()

    assert display.ungrabs == display.grabs_before_failure
    assert display.closed is True


def test_x11_hotkey_press_repeat_and_release_call_each_callback_once() -> None:
    calls: list[str] = []
    display = FakeHotkeyDisplay()
    listener = _make_hotkey_listener(
        display,
        lambda: calls.append("start"),
        lambda: calls.append("stop"),
    )
    listener.start()

    super_mask = 1 << 6
    listener.handle_event(
        FakeKeyEvent(X_KEY_PRESS, display.c_keycode, super_mask)
    )
    listener.handle_event(
        FakeKeyEvent(X_KEY_PRESS, display.c_keycode, super_mask)
    )
    # Super may be released before C. The matching C release still stops once.
    listener.handle_event(
        FakeKeyEvent(X_KEY_RELEASE, display.c_keycode, 0)
    )
    listener.flush_pending_release()
    listener.handle_event(
        FakeKeyEvent(X_KEY_RELEASE, display.c_keycode, 0)
    )
    listener.flush_pending_release()

    assert calls == ["start", "stop"]
    listener.close()


def test_x11_hotkey_ignores_core_auto_repeat_release_press_pairs() -> None:
    calls: list[str] = []
    display = FakeHotkeyDisplay()
    listener = _make_hotkey_listener(
        display,
        lambda: calls.append("start"),
        lambda: calls.append("stop"),
    )
    listener.start()

    super_mask = 1 << 6
    listener.handle_event(FakeKeyEvent(X_KEY_PRESS, display.c_keycode, super_mask))
    listener.handle_event(FakeKeyEvent(X_KEY_RELEASE, display.c_keycode, super_mask))
    listener.handle_event(FakeKeyEvent(X_KEY_PRESS, display.c_keycode, super_mask))
    listener.flush_pending_release()

    assert calls == ["start"]

    listener.handle_event(FakeKeyEvent(X_KEY_RELEASE, display.c_keycode, 0))
    listener.flush_pending_release()
    assert calls == ["start", "stop"]
    listener.close()


def test_x11_hotkey_rejects_missing_super_mapping_and_releases_nothing() -> None:
    display = FakeHotkeyDisplay()
    display.get_modifier_mapping()[6].clear()
    listener = _make_hotkey_listener(display, lambda: None, lambda: None)
    with pytest.raises(X11HotkeyUnavailable, match="Super"):
        listener.start()

    assert display.grabs == []
    assert display.ungrabs == []
    assert display.closed is True


def test_x11_hotkey_close_is_idempotent_and_releases_every_grab() -> None:
    display = FakeHotkeyDisplay()
    listener = _make_hotkey_listener(display, lambda: None, lambda: None)
    listener.start()

    listener.close()
    listener.close()

    assert display.ungrabs == display.grabs
    assert display.closed is True


# --- X11 focus guard ---------------------------------------------------------


def test_focus_guard_captures_snapshot(tmp_path: Path) -> None:
    proc = tmp_path / "1234" / "comm"
    proc.parent.mkdir()
    proc.write_text("firefox\n", encoding="utf-8")
    display = FakeDisplay(active_window=0x123, input_focus=0x456, window_pid=1234)
    guard = X11FocusGuard(display=display, monotonic=lambda: 111, proc_root=tmp_path)

    snap = guard.capture()
    assert snap == FocusSnapshot(
        active_window=0x123,
        process_name="firefox",
        input_focus=0x456,
        monotonic_ns=111,
        window_pid=1234,
    )


def test_focus_guard_missing_properties_are_none() -> None:
    display = FakeDisplay(active_window=None, input_focus=None, window_pid=None)
    guard = X11FocusGuard(display=display, monotonic=lambda: 0)
    snap = guard.capture()
    assert snap.active_window is None
    assert snap.input_focus is None
    assert snap.window_pid is None
    assert snap.process_name is None


def test_focus_guard_accepts_python_xlib_input_focus_reply() -> None:
    display = FakeDisplay(
        active_window=0x123,
        input_focus=0x456,
        input_focus_reply=True,
    )
    guard = X11FocusGuard(display=display, monotonic=lambda: 0)

    assert guard.capture().input_focus == 0x456


def test_focus_guard_x_server_error_raises() -> None:
    display = FakeDisplay(explode=True)
    guard = X11FocusGuard(display=display, monotonic=lambda: 0)
    with pytest.raises(X11Error):
        guard.capture()


def test_focus_guard_display_open_failure_raises_x11_error() -> None:
    def boom() -> FakeDisplay:
        raise RuntimeError("cannot open display")

    guard = X11FocusGuard(display=None, make_display=boom, monotonic=lambda: 0)
    with pytest.raises(X11Error):
        guard.capture()


def test_focus_guard_is_same_compares_focus_identity_only() -> None:
    guard = X11FocusGuard(display=FakeDisplay(), monotonic=lambda: 0)
    a = FocusSnapshot(
        active_window=1, process_name="app", input_focus=2, monotonic_ns=100,
        window_pid=9,
    )
    # The timestamp differs, but the focus identity is unchanged.
    b = FocusSnapshot(
        active_window=1, process_name="app", input_focus=2, monotonic_ns=999,
        window_pid=9,
    )
    assert guard.is_same(a, b)


def test_focus_guard_is_same_detects_every_focus_change() -> None:
    guard = X11FocusGuard(display=FakeDisplay(), monotonic=lambda: 0)
    base = FocusSnapshot(
        active_window=1, process_name="app", input_focus=2, monotonic_ns=0,
        window_pid=9,
    )
    changes = [
        FocusSnapshot(active_window=3, process_name="app", input_focus=2,
                      monotonic_ns=0, window_pid=9),  # active window switch
        FocusSnapshot(active_window=1, process_name="other", input_focus=2,
                      monotonic_ns=0, window_pid=9),  # process change
        FocusSnapshot(active_window=1, process_name="app", input_focus=4,
                      monotonic_ns=0, window_pid=9),  # focus lost to other window
        FocusSnapshot(active_window=1, process_name="app", input_focus=2,
                      monotonic_ns=0, window_pid=10),  # pid change
    ]
    for changed in changes:
        assert not guard.is_same(base, changed)


def test_focus_guard_c_is_down() -> None:
    keymap = [0] * 32
    display = FakeDisplay(keymap=keymap, c_keycode=54)
    guard = X11FocusGuard(display=display, monotonic=lambda: 0)

    assert guard.c_is_down() is False
    keymap[6] |= 1 << 6  # byte 6, bit 6
    assert guard.c_is_down() is True
    keymap[6] = 0
    assert guard.c_is_down() is False


def test_c_is_down_invalid_keycode_returns_false() -> None:
    display = FakeDisplay(c_keycode=0)
    guard = X11FocusGuard(display=display, monotonic=lambda: 0)
    assert guard.c_is_down() is False


# --- Clipboard mirror --------------------------------------------------------


def test_default_runner_does_not_capture_xclip_background_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """xclip forks its selection owner, which inherits captured stderr."""

    seen: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = None
        stderr = None

    def fake_run(*args: object, **kwargs: object) -> Result:
        seen.update(kwargs)
        return Result()

    monkeypatch.setattr("fun_voice.desktop.subprocess.run", fake_run)

    assert default_runner(["xclip", "-selection", "clipboard", "-in"], "text") == (
        0,
        "",
        "",
    )
    assert seen["stdout"] is subprocess.DEVNULL
    assert seen["stderr"] is subprocess.DEVNULL
    assert "capture_output" not in seen


def test_clipboard_writes_via_xclip() -> None:
    runner = FakeRunner((0, "", ""))
    mirror = ClipboardMirror(runner, binary="xclip")
    mirror.write_utf8("你好 world")
    assert runner.calls == [
        (("xclip", "-selection", "clipboard", "-in"), "你好 world")
    ]


def test_clipboard_writes_via_xsel() -> None:
    runner = FakeRunner((0, "", ""))
    mirror = ClipboardMirror(runner, binary="xsel")
    mirror.write_utf8("你好 world")
    assert runner.calls == [
        (("xsel", "--clipboard", "--input"), "你好 world")
    ]


def test_clipboard_failure_raises() -> None:
    runner = FakeRunner((1, "", "xclip: error"))
    mirror = ClipboardMirror(runner, binary="xclip")
    with pytest.raises(ClipboardError, match="xclip"):
        mirror.write_utf8("text")


def test_clipboard_timeout_raises() -> None:
    def timeout_runner(
        argv: list[str] | tuple[str, ...],
        input_text: str | None = None,
        *,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        raise subprocess.TimeoutExpired(argv, timeout or 5)

    mirror = ClipboardMirror(timeout_runner, binary="xclip")
    with pytest.raises(ClipboardError, match="timed out"):
        mirror.write_utf8("text")


def test_clipboard_without_tool_raises_clear_error() -> None:
    mirror = ClipboardMirror(FakeRunner(), which=lambda name: None)
    with pytest.raises(ClipboardError, match="xclip or xsel"):
        mirror.write_utf8("text")


# --- XTEST injector ----------------------------------------------------------


def test_xtest_injects_ctrl_v_sequence() -> None:
    display = FakeDisplay()
    events: list[tuple[object, int, int]] = []

    def record(display_obj: object, event_type: int, keycode: int) -> None:
        events.append((display_obj, event_type, keycode))

    injector = XTestInjector(display=display, fake_input=record)
    injector.paste_ctrl_v()
    assert [event[1:] for event in events] == [
        (2, 37),  # Ctrl press
        (2, 55),  # V press
        (3, 55),  # V release
        (3, 37),  # Ctrl release
    ]
    assert display.sync_calls == 1


def test_xtest_requires_display() -> None:
    injector = XTestInjector(display=None)
    with pytest.raises(XTestError):
        injector.paste_ctrl_v()
