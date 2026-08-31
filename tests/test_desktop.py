"""Tests for the DDE, X11 focus, clipboard and XTEST desktop adapters.

Every external boundary is a replaceable adapter, so these tests use fakes for
the D-Bus command runner, the X11 display and the monotonic clock. No real X
server or DDE session is required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fun_voice.contracts import FocusSnapshot, encode_message
from fun_voice.desktop import (
    XK_C,
    XK_CONTROL_L,
    XK_V,
    ClipboardError,
    ClipboardMirror,
    DdeKeybindingClient,
    DdeKeybindingError,
    DdeShortcutConflict,
    HotkeyBridge,
    X11Error,
    X11FocusGuard,
    XTestError,
    XTestInjector,
    parse_busctl_string,
    parse_shortcut_struct,
    register_shortcut,
    unregister_shortcut,
)

# --- Fakes -------------------------------------------------------------------

_SERVICE = "org.deepin.dde.Keybinding1"
_PATH = "/org/deepin/dde/Keybinding1"
_INTERFACE = "org.deepin.dde.Keybinding1"


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
    ) -> None:
        self.root = _FakeRootWindow(active_window)
        self._focus = input_focus
        self._pid = window_pid
        self._keymap = keymap if keymap is not None else [0] * 32
        self._c_keycode = c_keycode
        self._explode = explode
        self.keysym_calls: list[int] = []
        self.sync_calls = 0
        self.fake_input_events: list[tuple[int, int]] = []

    def screen(self) -> _FakeScreen:
        return _FakeScreen(self.root)

    def intern_atom(self, name: str) -> str:
        return name

    def get_input_focus(self) -> tuple[int | None, int]:
        if self._explode:
            raise RuntimeError("X server connection lost")
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


def _conflict_output(
    *,
    shortcut_id: str = "org.deepin.dde.keybinding.shortcut.app.uos-ai-talk",
    name: str = "UOS AI Talk",
) -> str:
    return (
        f'(sssasssbb) "{shortcut_id}" "{name}" "UOS AI" 1 '
        '"<Control><Super>space" "" "" false true'
    )


# --- DdeKeybindingClient -----------------------------------------------------


def test_lookup_conflict_returns_none_when_free() -> None:
    runner = FakeRunner((0, '(sssasssbb) "" "" "" 0 "" "" false false', ""))
    client = DdeKeybindingClient(runner)
    assert client.lookup_conflict("<Super>C") is None
    assert runner.calls[0][0] == (
        "busctl",
        "--user",
        "call",
        _SERVICE,
        _PATH,
        _INTERFACE,
        "LookupConflictShortcut",
        "s",
        "<Super>C",
    )


def test_lookup_conflict_reports_owner() -> None:
    runner = FakeRunner((0, _conflict_output(), ""))
    client = DdeKeybindingClient(runner)
    assert client.lookup_conflict("<Super>C") == "UOS AI Talk"


def test_lookup_conflict_decodes_octal_escaped_owner() -> None:
    # "语音对话" escaped by busctl as octal UTF-8 byte sequences.
    runner = FakeRunner(
        (
            0,
            _conflict_output(name="\\350\\257\\255\\351\\237\\263\\345\\257\\271\\350\\257\\235"),
            "",
        )
    )
    client = DdeKeybindingClient(runner)
    assert client.lookup_conflict("<Super>C") == "语音对话"


def test_lookup_conflict_falls_back_to_id_when_name_empty() -> None:
    runner = FakeRunner((0, _conflict_output(name=""), ""))
    client = DdeKeybindingClient(runner)
    assert (
        client.lookup_conflict("<Super>C")
        == "org.deepin.dde.keybinding.shortcut.app.uos-ai-talk"
    )


def test_add_custom_shortcut_returns_id() -> None:
    runner = FakeRunner((0, 's "org.deepin.dde.keybinding.shortcut.custom.42"', ""))
    client = DdeKeybindingClient(runner)
    got = client.add_custom_shortcut(
        "Fun Voice Ryan", "/usr/bin/fun-voice-bridge", "<Super>C"
    )
    assert got == "org.deepin.dde.keybinding.shortcut.custom.42"
    assert runner.calls[0][0] == (
        "busctl",
        "--user",
        "call",
        _SERVICE,
        _PATH,
        _INTERFACE,
        "AddCustomShortcut",
        "sss",
        "Fun Voice Ryan",
        "/usr/bin/fun-voice-bridge",
        "<Super>C",
    )


def test_delete_custom_shortcut_is_a_noop_return() -> None:
    runner = FakeRunner((0, "b true", ""))
    client = DdeKeybindingClient(runner)
    assert client.delete_custom_shortcut("custom.42") is None
    assert runner.calls[0][0] == (
        "busctl",
        "--user",
        "call",
        _SERVICE,
        _PATH,
        _INTERFACE,
        "DeleteCustomShortcut",
        "s",
        "custom.42",
    )


def test_dde_call_failure_raises() -> None:
    runner = FakeRunner((1, "", "No such method"))
    client = DdeKeybindingClient(runner)
    with pytest.raises(DdeKeybindingError, match="No such method"):
        client.lookup_conflict("<Super>C>")


def test_parse_busctl_string() -> None:
    assert parse_busctl_string('s "hello"') == "hello"


def test_parse_shortcut_struct_empty() -> None:
    info = parse_shortcut_struct('(sssasssbb) "" "" "" 0 "" "" false false')
    assert info is not None
    assert info.id == ""
    assert info.name == ""
    assert info.accels == ()


# --- Registration ------------------------------------------------------------


def test_register_shortcut_persists_id(tmp_path: Path) -> None:
    state_file = tmp_path / "shortcut-id"
    runner = FakeRunner(
        (0, '(sssasssbb) "" "" "" 0 "" "" false false', ""),  # lookup: free
        (0, 's "custom.7"', ""),  # add
    )
    client = DdeKeybindingClient(runner)
    got = register_shortcut(
        client,
        name="Fun Voice Ryan",
        action="/usr/bin/fun-voice-bridge",
        hotkey="<Super>C",
        state_file=state_file,
    )
    assert got == "custom.7"
    assert state_file.read_text(encoding="utf-8") == "custom.7\n"
    methods = [call[0][6] for call in runner.calls]
    assert methods == ["LookupConflictShortcut", "AddCustomShortcut"]


def test_register_shortcut_conflict_does_not_modify_dde(tmp_path: Path) -> None:
    state_file = tmp_path / "shortcut-id"
    runner = FakeRunner((0, _conflict_output(), ""))
    client = DdeKeybindingClient(runner)
    with pytest.raises(DdeShortcutConflict, match="UOS AI Talk"):
        register_shortcut(
            client,
            name="Fun Voice Ryan",
            action="/usr/bin/fun-voice-bridge",
            hotkey="<Super>C",
            state_file=state_file,
        )
    assert len(runner.calls) == 1
    assert runner.calls[0][0][6] == "LookupConflictShortcut"
    assert not state_file.exists()


def test_unregister_shortcut_deletes_and_clears(tmp_path: Path) -> None:
    state_file = tmp_path / "shortcut-id"
    state_file.write_text("custom.9\n", encoding="utf-8")
    runner = FakeRunner((0, "b true", ""))
    client = DdeKeybindingClient(runner)
    assert unregister_shortcut(client, state_file=state_file) == "custom.9"
    assert runner.calls[0][0][6] == "DeleteCustomShortcut"
    assert runner.calls[0][0][-1] == "custom.9"
    assert not state_file.exists()


def test_unregister_shortcut_without_state_is_noop(tmp_path: Path) -> None:
    runner = FakeRunner()
    client = DdeKeybindingClient(runner)
    assert unregister_shortcut(client, state_file=tmp_path / "missing") is None
    assert runner.calls == []


def test_unregister_corrupted_state_file_is_noop(tmp_path: Path) -> None:
    state_file = tmp_path / "shortcut-id"
    state_file.write_bytes(b"\xff\xfe\x00garbage")
    runner = FakeRunner()
    client = DdeKeybindingClient(runner)
    assert unregister_shortcut(client, state_file=state_file) is None
    assert runner.calls == []


# --- Hotkey bridge -----------------------------------------------------------


def _make_bridge(
    keymap: list[int] | None = None,
    *,
    c_keycode: int = 54,
    c_down: bool = False,
) -> tuple[HotkeyBridge, FakeDisplay, list[bytes]]:
    sent: list[bytes] = []
    display = FakeDisplay(keymap=keymap, c_keycode=c_keycode)
    if c_down:
        byte = c_keycode // 8
        bit = c_keycode % 8
        display._keymap[byte] |= 1 << bit
    guard = X11FocusGuard(display=display, monotonic=lambda: 0)
    bridge = HotkeyBridge(guard, sent.append)
    return bridge, display, sent


def test_bridge_c_down_sends_start_if_idle() -> None:
    bridge, _, sent = _make_bridge(c_down=True)
    assert bridge.handle() == "start"
    assert sent == [encode_message({"op": "start_if_idle"})]


def test_bridge_c_up_sends_stop() -> None:
    bridge, _, sent = _make_bridge(c_down=False)
    assert bridge.handle() == "stop"
    assert sent == [encode_message({"op": "stop"})]


def test_bridge_repeated_actions_are_idempotent_requests() -> None:
    bridge, _, sent = _make_bridge(c_down=True)
    bridge.handle()
    bridge.handle()
    bridge.handle()
    # Every hold trigger is a start_if_idle; the daemon makes it a no-op while
    # already recording, so repeats never toggle or stack.
    assert sent == [encode_message({"op": "start_if_idle"})] * 3


def test_bridge_uses_live_c_key_state() -> None:
    # Same keycode, but the keymap flips between down and up across calls.
    keymap = [0] * 32
    display = FakeDisplay(keymap=keymap, c_keycode=54)
    guard = X11FocusGuard(display=display, monotonic=lambda: 0)
    sent: list[bytes] = []
    bridge = HotkeyBridge(guard, sent.append)

    keymap[6] = 1 << 6  # C down
    assert bridge.handle() == "start"
    keymap[6] = 0  # C up
    assert bridge.handle() == "stop"
    assert sent == [
        encode_message({"op": "start_if_idle"}),
        encode_message({"op": "stop"}),
    ]


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
    # Timestamp and Fcitx token differ, but focus identity is unchanged.
    b = FocusSnapshot(
        active_window=1, process_name="app", input_focus=2, monotonic_ns=999,
        window_pid=9, focus_token="tok",
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
