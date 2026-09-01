"""Desktop adapters for X11 hotkeys/focus, clipboard and XTEST.

Every external boundary is a replaceable adapter so the daemon state machine can
depend on small interfaces and be tested with fakes:

- :class:`X11HotkeyListener` exclusively grabs ``Super+C`` and emits an ordered
  press/release lifecycle to the daemon.
- :class:`X11FocusGuard` captures :class:`~fun_voice.contracts.FocusSnapshot`
  (active window, window process, input focus, monotonic timestamp) and reads
  the live ``C`` key state.
- :class:`ClipboardMirror` writes UTF-8 text to the CLIPBOARD selection.
- :class:`XTestInjector` sends a Ctrl+V XTEST sequence, only as a post-Fcitx
  fallback.

No ``/dev/input`` is read anywhere; the push-to-talk hold semantics come from
X11 keyboard events, never from raw keyboard devices.
"""

from __future__ import annotations

import logging
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from fun_voice.contracts import FocusSnapshot

logger = logging.getLogger(__name__)

# --- X11 constants (stable protocol values, no Xlib import needed) -----------

X_ANY_PROPERTY_TYPE = 0  # X.AnyPropertyType
XK_C = 0x63  # XK_c
XK_CONTROL_L = 0xFFE3
XK_V = 0x76  # XK_v
XK_NUM_LOCK = 0xFF7F
XK_SCROLL_LOCK = 0xFF14
XK_SUPER_L = 0xFFEB
XK_SUPER_R = 0xFFEC
X_KEY_PRESS = 2
X_KEY_RELEASE = 3
X_LOCK_MASK = 1 << 1
HOTKEY_EVENT_WAIT_SECONDS = 0.1

# --- Errors ------------------------------------------------------------------


class DesktopError(RuntimeError):
    """Base class for desktop adapter errors."""


class ClipboardError(DesktopError):
    """Raised when the CLIPBOARD selection cannot be written."""


class X11Error(DesktopError):
    """Raised when the X11 server cannot answer a focus or key-state query."""


class X11HotkeyUnavailable(X11Error):
    """Raised when the X server cannot exclusively grab ``Super+C``."""


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
    if argv and argv[0] == "xclip":
        # xclip forks a background selection owner.  Capturing stderr would
        # keep ``communicate()`` waiting on that child's inherited pipe until
        # the clipboard changes, despite the parent having succeeded.
        proc = subprocess.run(
            argv,
            input=input_text,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    else:
        proc = subprocess.run(
            argv, input=input_text, capture_output=True, text=True, timeout=timeout
        )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# --- X11 display interface ---------------------------------------------------


class XProperty(Protocol):
    value: object


class XWindow(Protocol):
    id: int

    def get_full_property(self, atom: int, prop_type: int) -> XProperty | None: ...

    def grab_key(
        self,
        key: int,
        modifiers: int,
        owner_events: bool,
        pointer_mode: int,
        keyboard_mode: int,
        onerror: Callable[[object, object], None] | None = None,
    ) -> None: ...

    def ungrab_key(self, key: int, modifiers: int) -> None: ...


class XScreen(Protocol):
    root: XWindow


class XDisplay(Protocol):
    """The minimal Xlib ``Display`` surface used by the adapters."""

    def screen(self) -> XScreen: ...

    def intern_atom(self, name: str) -> int: ...

    def get_input_focus(self) -> object: ...

    def create_resource_object(self, kind: str, window_id: int) -> XWindow: ...

    def query_keymap(self) -> list[int]: ...

    def keysym_to_keycode(self, keysym: int) -> int: ...

    def get_modifier_mapping(self) -> list[list[int]]: ...

    def fileno(self) -> int: ...

    def next_event(self) -> object: ...

    def pending_events(self) -> int: ...

    def sync(self) -> None: ...

    def close(self) -> None: ...


def default_make_display() -> XDisplay:
    """Open the default X11 display (imports python-xlib lazily)."""
    from Xlib import display as xdisplay

    return cast(XDisplay, xdisplay.Display())


def _select_ready(fd: int, timeout: float) -> bool:
    """Return whether an X11 Display file descriptor is readable."""
    readable, _writable, _exceptional = select.select([fd], [], [], timeout)
    return bool(readable)


class X11HotkeyListener:
    """Exclusively grab ``Super+C`` and expose its press/release lifecycle.

    The listener owns a dedicated X11 Display because its event-loop thread must
    not share a connection with focus queries or XTEST injection. It never
    persists input data: only the current in-memory held flag is tracked.
    """

    def __init__(
        self,
        on_press: Callable[[], object],
        on_release: Callable[[], None],
        *,
        make_display: Callable[[], XDisplay] = default_make_display,
        select_ready: Callable[[int, float], bool] = _select_ready,
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._make_display = make_display
        self._select_ready = select_ready
        self._display: XDisplay | None = None
        self._root: XWindow | None = None
        self._keycode = 0
        self._super_mask = 0
        self._ignored_mask = X_LOCK_MASK
        self._ignored_masks: tuple[int, ...] = (X_LOCK_MASK,)
        self._grabs: list[tuple[int, int]] = []
        self._held = False
        self._release_pending = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Atomically register all ``Super+C`` lock variants and start polling."""
        with self._lock:
            if self._thread is not None:
                return
            if self._closed:
                raise X11HotkeyUnavailable("X11 hotkey listener is closed")
            try:
                display = self._make_display()
                root = display.screen().root
                self._display = display
                self._root = root
                keycode = display.keysym_to_keycode(XK_C)
                if keycode == 0:
                    raise X11HotkeyUnavailable("C keycode is unavailable")
                super_mask = self._modifier_mask(
                    display, (XK_SUPER_L, XK_SUPER_R)
                )
                if super_mask == 0:
                    raise X11HotkeyUnavailable("Super modifier is unavailable")
                num_mask = self._modifier_mask(display, (XK_NUM_LOCK,))
                scroll_mask = self._modifier_mask(display, (XK_SCROLL_LOCK,))
                self._keycode = keycode
                self._super_mask = super_mask
                self._ignored_mask = X_LOCK_MASK | num_mask | scroll_mask
                self._ignored_masks = tuple(
                    mask for mask in (X_LOCK_MASK, num_mask, scroll_mask) if mask
                )
                self._grab_all()
            except X11HotkeyUnavailable:
                self._abort_start()
                raise
            except Exception as exc:
                self._abort_start()
                raise X11HotkeyUnavailable(
                    f"cannot register Super+C: {type(exc).__name__}"
                ) from exc
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="x11-hotkey-listener",
            )
            self._thread.start()

    def close(self) -> None:
        """Stop polling, release every grab and close the dedicated Display."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=HOTKEY_EVENT_WAIT_SECONDS * 3)
        with self._lock:
            self._release_grabs()
            display = self._display
            self._display = None
            self._root = None
        if display is not None:
            try:
                display.close()
            except Exception:
                logger.warning("X11 hotkey display close failed")

    def handle_event(self, event: object) -> None:
        """Route a single matching key event; exposed for deterministic tests."""
        if getattr(event, "detail", None) != self._keycode:
            return
        event_type = getattr(event, "type", None)
        if event_type == X_KEY_PRESS and self._press_matches_super(event):
            # Core X11 keyboard auto-repeat emits KeyRelease + KeyPress pairs.
            # Keep the session held when this press immediately follows such a
            # release; only an idle event wait confirms a real key release.
            self._release_pending = False
            if not self._held:
                self._held = True
                self._on_press()
        elif event_type == X_KEY_RELEASE and self._held:
            # Do not inspect modifier state here: Super may already be released.
            self._release_pending = True

    def flush_pending_release(self) -> None:
        """Stop after an idle event wait confirms a real C key release."""
        if not self._release_pending or not self._held:
            return
        self._release_pending = False
        self._held = False
        self._on_release()

    def _modifier_mask(self, display: XDisplay, keysyms: tuple[int, ...]) -> int:
        keycodes = {display.keysym_to_keycode(keysym) for keysym in keysyms}
        keycodes.discard(0)
        if not keycodes:
            return 0
        for index, mapping in enumerate(display.get_modifier_mapping()):
            if keycodes.intersection(mapping):
                return 1 << index
        return 0

    def _grab_all(self) -> None:
        display = self._require_display()
        root = self._require_root()
        errors: list[object] = []

        def record_error(error: object, _request: object) -> None:
            errors.append(error)

        from Xlib import X

        masks = self._grab_masks()
        for modifiers in masks:
            root.grab_key(
                self._keycode,
                modifiers,
                False,
                X.GrabModeAsync,
                X.GrabModeAsync,
                onerror=record_error,
            )
            if errors:
                break
            self._grabs.append((self._keycode, modifiers))
        try:
            display.sync()
        except Exception as exc:
            raise X11HotkeyUnavailable(
                f"cannot register Super+C: {type(exc).__name__}"
            ) from exc
        if errors:
            raise X11HotkeyUnavailable(
                "Super+C is already grabbed by another X11 client"
            )
        if len(self._grabs) != len(masks):
            raise X11HotkeyUnavailable("Super+C grab did not complete")

    def _grab_masks(self) -> tuple[int, ...]:
        variants = {0}
        for ignored in self._ignored_masks:
            variants |= {variant | ignored for variant in tuple(variants)}
        return tuple(sorted(self._super_mask | variant for variant in variants))

    def _press_matches_super(self, event: object) -> bool:
        state = getattr(event, "state", None)
        if not isinstance(state, int):
            return False
        return (state & ~self._ignored_mask) == self._super_mask

    def _run(self) -> None:
        display = self._require_display()
        while not self._stop.is_set():
            try:
                if not self._select_ready(display.fileno(), HOTKEY_EVENT_WAIT_SECONDS):
                    self.flush_pending_release()
                    continue
                self.handle_event(display.next_event())
                while display.pending_events():
                    self.handle_event(display.next_event())
            except Exception as exc:
                if not self._stop.is_set():
                    logger.warning(
                        "X11 hotkey event loop stopped: %s", type(exc).__name__
                    )
                return

    def _abort_start(self) -> None:
        self._release_grabs()
        display = self._display
        self._display = None
        self._root = None
        self._closed = True
        if display is not None:
            try:
                display.close()
            except Exception:
                logger.warning("X11 hotkey rollback display close failed")

    def _release_grabs(self) -> None:
        root = self._root
        display = self._display
        if root is None:
            return
        for keycode, modifiers in self._grabs:
            try:
                root.ungrab_key(keycode, modifiers)
            except Exception:
                logger.warning("X11 hotkey ungrab failed")
        self._grabs.clear()
        if display is not None:
            try:
                display.sync()
            except Exception:
                logger.warning("X11 hotkey ungrab sync failed")

    def _require_display(self) -> XDisplay:
        if self._display is None:
            raise X11HotkeyUnavailable("X11 display is unavailable")
        return self._display

    def _require_root(self) -> XWindow:
        if self._root is None:
            raise X11HotkeyUnavailable("X11 root window is unavailable")
        return self._root


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
        reply = display.get_input_focus()
        if isinstance(reply, tuple):
            if not reply:
                return None
            focus = reply[0]
        else:
            # python-xlib returns a GetInputFocus reply object with a Window
            # resource in ``focus``; lightweight test adapters often return
            # the historical ``(focus, revert_to)`` tuple instead.
            focus = getattr(reply, "focus", None)
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
