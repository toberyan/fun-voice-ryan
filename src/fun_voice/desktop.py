"""Desktop adapters for DDE Keybinding1, X11 focus/key state, clipboard and XTEST.

Every external boundary is a replaceable adapter so the daemon state machine can
depend on small interfaces and be tested with fakes:

- :class:`DdeKeybindingClient` wraps ``busctl --user call`` against the DDE
  ``org.deepin.dde.Keybinding1`` service (LookupConflictShortcut,
  AddCustomShortcut, DeleteCustomShortcut). DDE is only ever told to run the
  bridge command; the model is never invoked by a shortcut action.
- :class:`X11FocusGuard` captures :class:`~fun_voice.contracts.FocusSnapshot`
  (active window, window process, input focus, monotonic timestamp) and reads
  the live ``C`` key state.
- :class:`HotkeyBridge` turns a DDE action into a ``start_if_idle``/``stop``
  daemon request based on the live ``C`` key state.
- :class:`ClipboardMirror` writes UTF-8 text to the CLIPBOARD selection.
- :class:`XTestInjector` sends a Ctrl+V XTEST sequence, only as a post-Fcitx
  fallback.

No ``/dev/input`` is read anywhere; the push-to-talk hold semantics come from
DDE triggering plus X11 key state, never from raw keyboard devices.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from fun_voice.contracts import FocusSnapshot, StartRequest, StopRequest, encode_message

# --- X11 constants (stable protocol values, no Xlib import needed) -----------

X_ANY_PROPERTY_TYPE = 0  # X.AnyPropertyType
XK_C = 0x63  # XK_c
XK_CONTROL_L = 0xFFE3
XK_V = 0x76  # XK_v
X_KEY_PRESS = 2
X_KEY_RELEASE = 3

# --- Errors ------------------------------------------------------------------


class DesktopError(RuntimeError):
    """Base class for desktop adapter errors."""


class DdeShortcutConflict(DesktopError):
    """Raised when the configured hotkey is already owned by another shortcut."""

    def __init__(self, hotkey: str, owner: str) -> None:
        self.hotkey = hotkey
        self.owner = owner
        super().__init__(f"hotkey {hotkey} is already owned by {owner!r}")


class DdeKeybindingError(DesktopError):
    """Raised when a DDE Keybinding1 D-Bus call fails."""


class ClipboardError(DesktopError):
    """Raised when the CLIPBOARD selection cannot be written."""


class X11Error(DesktopError):
    """Raised when the X11 server cannot answer a focus or key-state query."""


class XTestError(DesktopError):
    """Raised when XTEST injection cannot be performed."""


# --- Command runner ----------------------------------------------------------

RunResult = tuple[int, str, str]


class Runner(Protocol):
    """Runs a command and returns ``(returncode, stdout, stderr)``."""

    def __call__(
        self,
        argv: Sequence[str],
        input_text: str | None = None,
        *,
        timeout: float | None = None,
    ) -> RunResult: ...


def default_runner(
    argv: Sequence[str],
    input_text: str | None = None,
    *,
    timeout: float | None = None,
) -> RunResult:
    """Run a command via subprocess, capturing UTF-8 text output."""
    proc = subprocess.run(
        argv, input=input_text, capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# --- DDE Keybinding1 client --------------------------------------------------

DEFAULT_HOTKEY = "<Super>C"
DEFAULT_SHORTCUT_NAME = "Fun Voice Ryan — 按住说话"


@dataclass(frozen=True)
class ShortcutInfo:
    """The subset of a DDE shortcut struct that the client needs."""

    id: str
    name: str
    accels: tuple[str, ...]


class DdeKeybindingClient:
    """Client for ``org.deepin.dde.Keybinding1`` over ``busctl --user call``."""

    SERVICE = "org.deepin.dde.Keybinding1"
    PATH = "/org/deepin/dde/Keybinding1"
    INTERFACE = "org.deepin.dde.Keybinding1"

    def __init__(self, runner: Runner | None = None) -> None:
        self._runner: Runner = runner if runner is not None else default_runner

    def _call(self, method: str, signature: str, *args: str) -> str:
        argv: list[str] = [
            "busctl",
            "--user",
            "call",
            self.SERVICE,
            self.PATH,
            self.INTERFACE,
            method,
            signature,
            *args,
        ]
        code, stdout, stderr = self._runner(argv)
        if code != 0:
            raise DdeKeybindingError(
                f"{method} failed (exit {code}): {stderr.strip()}"
            )
        return stdout.strip()

    def lookup_conflict(self, hotkey: str) -> str | None:
        """Return the display name owning ``hotkey``, or ``None`` when free."""
        out = self._call("LookupConflictShortcut", "s", hotkey)
        info = parse_shortcut_struct(out)
        if info is None or not info.accels:
            return None
        return info.name or info.id

    def add_custom_shortcut(self, name: str, action: str, hotkey: str) -> str:
        """Register a custom shortcut and return its DDE shortcut id."""
        return parse_busctl_string(
            self._call("AddCustomShortcut", "sss", name, action, hotkey)
        )

    def delete_custom_shortcut(self, shortcut_id: str) -> None:
        """Remove a custom shortcut by id."""
        self._call("DeleteCustomShortcut", "s", shortcut_id)


# --- busctl output parsing ---------------------------------------------------


def parse_busctl_string(out: str) -> str:
    """Parse a busctl single-string reply such as ``s "value"``."""
    body = out.strip()
    if body.startswith("s "):
        body = body[2:].lstrip()
    tokens = _split_busctl_args(body)
    return tokens[0] if tokens else ""


def parse_shortcut_struct(out: str) -> ShortcutInfo | None:
    """Parse a busctl ``(sssasssbb)`` shortcut struct reply."""
    body = out.strip()
    if body.startswith("("):
        end = body.find(") ")
        if end == -1:
            return None
        body = body[end + 2 :]
    tokens = _split_busctl_args(body)
    if len(tokens) < 4:
        return None
    try:
        count = int(tokens[3])
    except ValueError:
        count = 0
    accels = tuple(tokens[4 : 4 + count])
    return ShortcutInfo(id=tokens[0], name=tokens[1], accels=accels)


def _split_busctl_args(text: str) -> list[str]:
    """Tokenize busctl-style output, decoding quoted strings and octal escapes."""
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        if text[i] == '"':
            i += 1
            data = bytearray()
            while i < n:
                ch = text[i]
                if ch == '"':
                    i += 1
                    break
                if ch == "\\":
                    i += 1
                    if i >= n:
                        break
                    esc = text[i]
                    if esc == "n":
                        data.append(0x0A)
                    elif esc == "r":
                        data.append(0x0D)
                    elif esc == "t":
                        data.append(0x09)
                    elif esc in "01234567":
                        j = i
                        while j < n and j < i + 3 and text[j] in "01234567":
                            j += 1
                        data.append(int(text[i:j], 8))
                        i = j - 1
                    else:
                        data.extend(esc.encode("utf-8"))
                else:
                    data.extend(ch.encode("utf-8"))
                i += 1
            tokens.append(data.decode("utf-8"))
        else:
            j = i
            while j < n and not text[j].isspace():
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


# --- Shortcut registration lifecycle -----------------------------------------


def shortcut_state_path(env: Mapping[str, str] | None = None) -> Path:
    """Return the file that persists the registered custom shortcut id.

    This must survive logout (unlike XDG_RUNTIME_DIR), so it lives under
    ``$XDG_CONFIG_HOME``.
    """
    if env is None:
        env = os.environ
    base = env.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "fun-voice-ryan" / "dde-shortcut-id"


def register_shortcut(
    client: DdeKeybindingClient,
    *,
    name: str,
    action: str,
    hotkey: str = DEFAULT_HOTKEY,
    state_file: Path,
) -> str:
    """Register ``hotkey`` via DDE and persist the returned shortcut id.

    Re-checks for a conflict immediately before registering; on conflict the
    DDE configuration is left untouched and the owner is reported. ``action``
    must be the bridge command — DDE runs it on trigger and never runs the
    model directly.
    """
    owner = client.lookup_conflict(hotkey)
    if owner is not None:
        raise DdeShortcutConflict(hotkey, owner)
    shortcut_id = client.add_custom_shortcut(name, action, hotkey)
    _write_shortcut_id(state_file, shortcut_id)
    return shortcut_id


def unregister_shortcut(
    client: DdeKeybindingClient, *, state_file: Path
) -> str | None:
    """Delete the previously registered shortcut and clear its persisted id."""
    shortcut_id = _read_shortcut_id(state_file)
    if shortcut_id is None:
        return None
    client.delete_custom_shortcut(shortcut_id)
    _clear_shortcut_id(state_file)
    return shortcut_id


def _write_shortcut_id(path: Path, shortcut_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(shortcut_id + "\n", encoding="utf-8")


def _read_shortcut_id(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


def _clear_shortcut_id(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


# --- X11 display interface ---------------------------------------------------


class XProperty(Protocol):
    value: object


class XWindow(Protocol):
    id: int

    def get_full_property(self, atom: int, prop_type: int) -> XProperty | None: ...


class XScreen(Protocol):
    root: XWindow


class XDisplay(Protocol):
    """The minimal Xlib ``Display`` surface used by the adapters."""

    def screen(self) -> XScreen: ...

    def intern_atom(self, name: str) -> int: ...

    def get_input_focus(self) -> tuple[object, int]: ...

    def create_resource_object(self, kind: str, window_id: int) -> XWindow: ...

    def query_keymap(self) -> list[int]: ...

    def keysym_to_keycode(self, keysym: int) -> int: ...

    def sync(self) -> None: ...

    def close(self) -> None: ...


def default_make_display() -> XDisplay:
    """Open the default X11 display (imports python-xlib lazily)."""
    from Xlib import display as xdisplay

    return cast(XDisplay, xdisplay.Display())


def _first_int(value: object) -> int | None:
    """Extract an integer from an X11 property value (list, int or bytes)."""
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, int):
        return value
    if isinstance(value, (bytes, bytearray)):
        if not value:
            return None
        return int.from_bytes(value, "little")
    return None

def _window_id(window: object) -> int | None:
    """Resolve an X11 window handle to its id, mapping None/PointerRoot to ``None``."""
    if window is None:
        return None
    wid = getattr(window, "id", window)
    if isinstance(wid, int):
        return wid if wid not in (0, 1) else None
    return None


# --- X11 focus guard ---------------------------------------------------------


class X11FocusGuard:
    """Captures focus snapshots and queries the live ``C`` key state."""

    def __init__(
        self,
        display: XDisplay | None = None,
        *,
        make_display: Callable[[], XDisplay] | None = None,
        monotonic: Callable[[], int] = time.monotonic_ns,
        proc_root: Path | None = None,
    ) -> None:
        self._display = display
        self._make_display = make_display
        self._monotonic = monotonic
        self._proc_root = Path("/proc") if proc_root is None else proc_root

    @property
    def display(self) -> XDisplay:
        display = self._display
        if display is None:
            factory = (
                self._make_display
                if self._make_display is not None
                else default_make_display
            )
            display = factory()
            self._display = display
        return display

    def capture(self) -> FocusSnapshot:
        """Capture the current X11 focus state, raising X11Error on server error."""
        try:
            display = self.display
            root = display.screen().root
            active_atom = display.intern_atom("_NET_ACTIVE_WINDOW")
            active = self._read_active_window(root, active_atom)
            input_focus = self._read_input_focus(display)
            pid = (
                self._read_window_pid(display, active)
                if active is not None
                else None
            )
            name = self._process_name(pid) if pid is not None else None
        except X11Error:
            raise
        except Exception as exc:
            raise X11Error(f"X11 focus capture failed: {exc}") from exc
        return FocusSnapshot(
            active_window=active,
            process_name=name,
            input_focus=input_focus,
            monotonic_ns=self._monotonic(),
            window_pid=pid,
        )

    def is_same(self, a: FocusSnapshot, b: FocusSnapshot) -> bool:
        """Full equality of focus identity (timestamp and Fcitx token excluded)."""
        return (
            a.active_window == b.active_window
            and a.process_name == b.process_name
            and a.input_focus == b.input_focus
            and a.window_pid == b.window_pid
        )

    def c_is_down(self) -> bool:
        """Return whether the physical ``C`` key is currently held down."""
        try:
            keycode = self.display.keysym_to_keycode(XK_C)
            keymap = self.display.query_keymap()
        except Exception as exc:
            raise X11Error(f"X11 key state query failed: {exc}") from exc
        if keycode == 0:
            return False
        if not (0 <= keycode < len(keymap) * 8):
            return False
        return bool(keymap[keycode // 8] & (1 << (keycode % 8)))

    def _read_active_window(self, root: XWindow, active_atom: int) -> int | None:
        prop = root.get_full_property(active_atom, X_ANY_PROPERTY_TYPE)
        if prop is None:
            return None
        return _window_id(_first_int(prop.value))

    def _read_input_focus(self, display: XDisplay) -> int | None:
        focus, _revert = display.get_input_focus()
        return _window_id(focus)

    def _read_window_pid(self, display: XDisplay, active: int) -> int | None:
        window = display.create_resource_object("window", active)
        pid_atom = display.intern_atom("_NET_WM_PID")
        prop = window.get_full_property(pid_atom, X_ANY_PROPERTY_TYPE)
        if prop is None:
            return None
        return _first_int(prop.value)

    def _process_name(self, pid: int) -> str | None:
        try:
            return (self._proc_root / str(pid) / "comm").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            return None


# --- Hotkey bridge -----------------------------------------------------------


class HotkeyBridge:
    """Translates a DDE custom-shortcut action into a daemon request.

    Stateless by design: DDE invokes the action once per trigger, and this
    bridge reads the live X11 ``C`` key state to choose ``start_if_idle`` (key
    held) or ``stop`` (key released). Repeated triggers are idempotent because
    ``start_if_idle`` is a no-op while the daemon is already recording.
    """

    def __init__(self, guard: X11FocusGuard, send: Callable[[bytes], None]) -> None:
        self._guard = guard
        self._send = send

    def handle(self) -> str:
        """Forward the daemon request implied by the current ``C`` key state."""
        if self._guard.c_is_down():
            self._send(encode_message(asdict(StartRequest())))
            return "start"
        self._send(encode_message(asdict(StopRequest())))
        return "stop"


# --- Clipboard mirror --------------------------------------------------------

CLIPBOARD_TIMEOUT_SECONDS = 5.0


class ClipboardMirror:
    """Writes UTF-8 text to the CLIPBOARD selection via xclip or xsel."""

    def __init__(
        self,
        runner: Runner | None = None,
        *,
        binary: str | None = None,
        which: Callable[[str], str | None] = shutil.which,
        timeout: float | None = CLIPBOARD_TIMEOUT_SECONDS,
    ) -> None:
        self._runner: Runner = runner if runner is not None else default_runner
        self._binary = binary
        self._which = which
        self._timeout = timeout

    def write_utf8(self, text: str) -> None:
        """Mirror ``text`` to the CLIPBOARD selection, raising on failure."""
        binary = self._binary if self._binary is not None else self._resolve_binary()
        if binary == "xclip":
            argv = ["xclip", "-selection", "clipboard", "-in"]
        elif binary == "xsel":
            argv = ["xsel", "--clipboard", "--input"]
        else:
            raise ClipboardError(f"unsupported clipboard tool: {binary!r}")
        try:
            code, _stdout, stderr = self._runner(argv, text, timeout=self._timeout)
        except subprocess.TimeoutExpired as exc:
            raise ClipboardError(
                f"clipboard write via {binary} timed out after {self._timeout}s"
            ) from exc
        if code != 0:
            raise ClipboardError(
                f"failed to write clipboard via {binary}: {stderr.strip()}"
            )

    def _resolve_binary(self) -> str:
        for name in ("xclip", "xsel"):
            if self._which(name) is not None:
                return name
        raise ClipboardError("no clipboard tool available (need xclip or xsel)")


# --- XTEST injector ----------------------------------------------------------


class XTestInjector:
    """Sends a Ctrl+V XTEST sequence; only a fallback after Fcitx fails."""

    def __init__(
        self,
        display: XDisplay | None = None,
        fake_input: Callable[[object, int, int], None] | None = None,
    ) -> None:
        self._display = display
        self._fake_input = fake_input

    @property
    def display(self) -> XDisplay | None:
        """The X11 display used for injection, or ``None`` when not provided."""
        return self._display

    def paste_ctrl_v(self, display: XDisplay | None = None) -> None:
        """Inject Ctrl+V via the XTEST extension.

        Callers must re-check the focus snapshot before falling back here; this
        method only performs the injection.
        """
        d = display if display is not None else self._display
        if d is None:
            raise XTestError("XTEST injection requires an X11 display")
        fake_input = self._fake_input
        if fake_input is None:
            from Xlib.ext import xtest

            fake_input = xtest.fake_input
        try:
            ctrl = d.keysym_to_keycode(XK_CONTROL_L)
            v = d.keysym_to_keycode(XK_V)
            if ctrl == 0 or v == 0:
                raise XTestError("Ctrl or V keycode unavailable on this display")
            fake_input(d, X_KEY_PRESS, ctrl)
            fake_input(d, X_KEY_PRESS, v)
            fake_input(d, X_KEY_RELEASE, v)
            fake_input(d, X_KEY_RELEASE, ctrl)
            d.sync()
        except XTestError:
            raise
        except Exception as exc:
            raise XTestError(f"XTEST injection failed: {exc}") from exc


# --- CLI (used by the register/unregister scripts) ----------------------------


def _state_file_or_default(state_file: Path | None) -> Path:
    return state_file if state_file is not None else shortcut_state_path()


def main(argv: Sequence[str] | None = None) -> int:
    """Register or unregister the DDE push-to-talk shortcut."""
    parser = argparse.ArgumentParser(
        prog="fun-voice-desktop",
        description="Register/unregister the Fun Voice Ryan DDE shortcut.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="register the Super+C shortcut")
    register.add_argument("--action", required=True, help="bridge command DDE runs")
    register.add_argument("--name", default=DEFAULT_SHORTCUT_NAME)
    register.add_argument("--hotkey", default=DEFAULT_HOTKEY)
    register.add_argument("--state-file", type=Path, default=None)

    unregister = subparsers.add_parser("unregister", help="remove the shortcut")
    unregister.add_argument("--state-file", type=Path, default=None)

    ns = parser.parse_args(argv)
    client = DdeKeybindingClient()
    try:
        if ns.command == "register":
            state_file = _state_file_or_default(ns.state_file)
            shortcut_id = register_shortcut(
                client,
                name=ns.name,
                action=ns.action,
                hotkey=ns.hotkey,
                state_file=state_file,
            )
            print(shortcut_id)
            return 0
        state_file = _state_file_or_default(ns.state_file)
        unregister_shortcut(client, state_file=state_file)
        return 0
    except DesktopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
