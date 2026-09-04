"""Private, non-interactive native DTK transient overlay controller.

The daemon sends immutable, in-memory models through a private pipe to a
short-lived native process.  The controller never logs or persists supplied
text and every process or protocol failure degrades to no UI.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, Protocol

from fun_voice.config import OverlayConfig
from fun_voice.contracts import DaemonState

OVERLAY_MAX_FRAME_BYTES = 64 * 1024
OVERLAY_REPLY_MAX_BYTES = 1024
OVERLAY_CLOSE_TIMEOUT_SECONDS = 0.2


@dataclass(frozen=True, slots=True)
class OverlayModel:
    """An in-memory overlay snapshot; it is never a desktop input payload."""

    phase: DaemonState
    stable_text: str = ""
    provisional_text: str = ""
    level: int | None = None


@dataclass(frozen=True, slots=True)
class OverlayFrame:
    """Renderer-ready snapshot retained as a compatibility-only value object."""

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


class _OverlayProcess(Protocol):
    stdin: IO[Any] | None
    stdout: IO[Any] | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


OverlayPopen = Callable[[list[str]], _OverlayProcess]


class NullOverlay:
    """No-op fallback when no desktop overlay can be constructed."""

    def show(self, model: OverlayModel) -> None:
        del model

    def clear(self) -> None:
        pass

    def close(self) -> None:
        pass


def default_overlay_executable() -> Path:
    """Return the user-scoped overlay executable installed by install-user.sh."""
    return Path.home() / ".local/lib/fun-voice-ryan/fun-voice-overlay"


def _default_popen(argv: list[str]) -> _OverlayProcess:
    return subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def _encode_payload(payload: dict[str, object]) -> bytes:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if not encoded or len(encoded) > OVERLAY_MAX_FRAME_BYTES:
        raise ValueError("overlay frame exceeds bound")
    return len(encoded).to_bytes(4, "big") + encoded


def _show_payload(model: OverlayModel) -> bytes:
    payload: dict[str, object] = {
        "command": "show",
        "phase": model.phase.value,
        "stable_text": model.stable_text,
        "provisional_text": model.provisional_text,
    }
    if model.level is not None:
        payload["level"] = max(0, min(100, model.level))
    return _encode_payload(payload)


class DtkOverlayController:
    """Lazy, best-effort controller for the private native DTK child process."""

    def __init__(
        self,
        *,
        executable: Path | None = None,
        layout: OverlayConfig | None = None,
        popen: OverlayPopen = _default_popen,
    ) -> None:
        self._executable = (
            default_overlay_executable() if executable is None else executable
        )
        self._layout = OverlayConfig() if layout is None else layout
        self._popen = popen
        self._process: _OverlayProcess | None = None
        self._closed = False
        self._lock = threading.RLock()

    def show(self, model: OverlayModel) -> None:
        try:
            frame = _show_payload(model)
        except (TypeError, ValueError):
            return
        with self._lock:
            if self._closed:
                return
            process = self._ensure_process_locked()
            if process is not None:
                self._write_or_discard_locked(process, frame)

    def clear(self) -> None:
        try:
            frame = _encode_payload({"command": "clear"})
        except ValueError:  # pragma: no cover - fixed, bounded command
            return
        with self._lock:
            if self._closed or self._process is None:
                return
            self._write_or_discard_locked(self._process, frame)

    def close(self) -> None:
        try:
            frame = _encode_payload({"command": "shutdown"})
        except ValueError:  # pragma: no cover - fixed, bounded command
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            self._process = None
        if process is None:
            return
        self._write_and_close(process, frame)

    def _ensure_process_locked(self) -> _OverlayProcess | None:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        self._process = None
        try:
            process = self._popen(self._argv())
        except (OSError, RuntimeError):
            return None
        self._process = process
        stdout = process.stdout
        if stdout is not None:
            threading.Thread(
                target=self._drain_replies,
                args=(stdout,),
                name="dtk-overlay-replies",
                daemon=True,
            ).start()
        threading.Thread(
            target=self._reap_child,
            args=(process,),
            name="dtk-overlay-reaper",
            daemon=True,
        ).start()
        return process

    def _argv(self) -> list[str]:
        return [
            str(self._executable),
            "--vertical-center-ratio",
            str(self._layout.vertical_center_ratio),
            "--width-px",
            str(self._layout.width_px),
            "--font-scale",
            str(self._layout.font_scale),
        ]

    def _write_or_discard_locked(self, process: _OverlayProcess, frame: bytes) -> None:
        if not self._write_frame(process, frame):
            if self._process is process:
                self._process = None
            self._terminate(process)

    @staticmethod
    def _write_frame(process: _OverlayProcess, frame: bytes) -> bool:
        stream = process.stdin
        if stream is None:
            return False
        try:
            stream.write(frame)
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            return False
        return True

    @classmethod
    def _write_and_close(cls, process: _OverlayProcess, frame: bytes) -> None:
        cls._write_frame(process, frame)
        stream = process.stdin
        if stream is not None:
            with suppress(OSError, ValueError):
                stream.close()
        try:
            process.wait(timeout=OVERLAY_CLOSE_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            cls._terminate(process)

    @staticmethod
    def _terminate(process: _OverlayProcess) -> None:
        with suppress(OSError):
            process.terminate()

    def _reap_child(self, process: _OverlayProcess) -> None:
        """Wait for one owned child so its normal idle exit cannot become a zombie."""
        try:
            process.wait()
        except (OSError, subprocess.TimeoutExpired):
            return
        with self._lock:
            if self._process is process:
                self._process = None

    @staticmethod
    def _drain_replies(stream: IO[Any]) -> None:
        """Consume fixed-size ACKs without retaining or reporting their payloads."""
        while True:
            try:
                header = stream.read(4)
            except (OSError, ValueError):
                return
            if len(header) != 4:
                return
            length = int.from_bytes(header, "big")
            if length == 0 or length > OVERLAY_REPLY_MAX_BYTES:
                return
            try:
                payload = stream.read(length)
            except (OSError, ValueError):
                return
            if len(payload) != length:
                return
            try:
                reply = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if not isinstance(reply, dict) or reply.get("reply") not in {
                "ready",
                "error_frame",
                "error_command",
            }:
                return
