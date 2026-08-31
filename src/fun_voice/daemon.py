"""Voice daemon: state machine, capture→worker→output pipeline, and socket server.

The daemon owns the whole push-to-talk lifecycle. It listens on
``$XDG_RUNTIME_DIR/fun-voice-ryan/daemon.sock`` (mode ``0600``, same-uid
``SO_PEERCRED`` gated) for single-line JSON requests from the bridge:

    {"op": "start_if_idle"}   -> start recording if idle (ignore otherwise)
    {"op": "stop"}            -> stop recording and run the pipeline

The pipeline is strictly linear: stop capture, transcribe via the worker, then
route the text to the desktop (clipboard mirror first, then Fcitx commit with a
strictly focus-guarded XTEST Ctrl+V fallback). Every path — success, error,
empty speech, focus change — runs the same idempotent cleanup before returning
to ``IDLE``.

Privacy red lines: no log or notification ever carries transcription text or
audio; audio is never persisted; a failed path never injects partial text.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from fun_voice import config
from fun_voice.capture import CaptureError, PipeWireRecorder
from fun_voice.contracts import (
    MAX_MESSAGE_BYTES,
    CaptureArtifact,
    CommitResult,
    DaemonState,
    ErrorCode,
    FocusSnapshot,
    MessageTooLarge,
    ProtocolError,
    Segment,
    Transcription,
    decode_message,
    encode_message,
)
from fun_voice.desktop import (
    ClipboardError,
    ClipboardMirror,
    Runner,
    X11Error,
    X11FocusGuard,
    XTestError,
    XTestInjector,
    default_runner,
)
from fun_voice.fcitx import FcitxClient, FcitxCommitError

logger = logging.getLogger(__name__)

# --- Notification copy (fixed; never carries transcription text) -------------

NOTIFICATION_APP_NAME = "Fun Voice Ryan"
NOTIFY_RECORDING = "录音中"
NOTIFY_TRANSCRIBING = "识别中"
NOTIFY_EMPTY_SPEECH = "未检测到语音"
NOTIFY_FOCUS_CHANGED = "焦点已变化，结果已复制到剪贴板"
NOTIFY_CLIPBOARD_FAILED = "文本已输入，但剪贴板备份失败"
NOTIFY_RECOGNITION_FAILED = "本地识别失败：{category}"
NOTIFY_LIMIT_REACHED = "已达到 30 分钟录音上限，开始识别"

# The worker may need up to its 120s inference timeout; leave margin.
WORKER_RESPONSE_TIMEOUT_SECONDS = 130.0
FUN_VOICE_WORKER_SERVICE = "fun-voice-worker.service"
SOCKET_BACKLOG = 4

# Hold-to-talk confirmation: after the bridge reports C down, the daemon must
# confirm C is still physically held within this window before recording.
C_CONFIRM_TIMEOUT_SECONDS = 0.5
C_CONFIRM_POLL_SECONDS = 0.05


# --- Structured worker errors ------------------------------------------------


class WorkerError(RuntimeError):
    """The worker returned a structured error, or could not be reached."""

    def __init__(self, code: ErrorCode, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail or str(code))


class EmptySpeechError(WorkerError):
    """The worker's VAD found no speech."""

    def __init__(self) -> None:
        super().__init__(ErrorCode("worker", "empty_speech"))


class _WorkerConnectFailure(Exception):
    """Internal marker: the worker socket could not be reached/parsed."""


def _parse_error_code(value: object) -> ErrorCode:
    """Parse a worker ``error_code`` string back into an :class:`ErrorCode`."""
    if not isinstance(value, str) or not value:
        return ErrorCode("worker", "internal")
    category, sep, code = value.partition(".")
    if sep and code:
        return ErrorCode(category, code)
    return ErrorCode("worker", value)


# --- Adapter seams (the fakes implement these) -------------------------------


class FocusGuard(Protocol):
    def capture(self) -> FocusSnapshot: ...

    def is_same(self, a: FocusSnapshot, b: FocusSnapshot) -> bool: ...

    def c_is_down(self) -> bool: ...


class Recorder(Protocol):
    def start(self) -> None: ...

    def stop(self) -> CaptureArtifact: ...

    def cancel(self) -> None: ...

    def cleanup(self) -> None: ...


class FcitxClientLike(Protocol):
    def start_focus(self) -> str | None: ...

    def commit(self, focus_token: str, text: str) -> CommitResult: ...

    def close(self) -> None: ...


class ClipboardLike(Protocol):
    def write_utf8(self, text: str) -> None: ...


class InjectorLike(Protocol):
    def paste_ctrl_v(self) -> None: ...


class Notifier(Protocol):
    def notify(self, message: str) -> None: ...


class WorkerClient(Protocol):
    def transcribe(self, artifact: CaptureArtifact) -> Transcription: ...

    def close(self) -> None: ...


class _NullFcitx:
    """No-op Fcitx stand-in used when the client cannot be constructed."""

    def start_focus(self) -> None:
        return None

    def commit(self, focus_token: str, text: str) -> CommitResult:
        return CommitResult(
            committed=False,
            method="fcitx",
            error=ErrorCode("fcitx", "unavailable"),
        )

    def close(self) -> None:
        pass


# --- Notifier (notify-send) --------------------------------------------------


class NotifySendNotifier:
    """Best-effort DDE notification via ``notify-send`` (org.freedesktop.Notifications).

    Notification failure must never break the pipeline, so every call swallows
    its own errors.
    """

    def __init__(
        self,
        runner: Runner | None = None,
        *,
        app_name: str = NOTIFICATION_APP_NAME,
    ) -> None:
        self._runner: Runner = runner if runner is not None else default_runner
        self._app_name = app_name

    def notify(self, message: str) -> None:
        try:
            self._runner(["notify-send", self._app_name, message], timeout=5.0)
        except Exception:
            logger.debug("notify-send failed (ignored)")


# --- Worker client (socket) --------------------------------------------------


def default_start_worker_service() -> None:
    """Best-effort, idempotent start of the worker user service."""
    try:
        subprocess.run(
            ["systemctl", "--user", "start", FUN_VOICE_WORKER_SERVICE],
            check=False,
            capture_output=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("could not start %s", FUN_VOICE_WORKER_SERVICE)


class SocketWorkerClient:
    """Speaks the worker's single-line JSON protocol over a Unix socket.

    Opens a fresh connection per request and starts the worker user service
    once (best-effort, idempotent) before a single retry when the socket is
    unreachable.
    """

    def __init__(
        self,
        socket_path: Path | str,
        *,
        timeout: float = WORKER_RESPONSE_TIMEOUT_SECONDS,
        start_service: Callable[[], None] | None = None,
    ) -> None:
        self._socket_path = Path(socket_path)
        self._timeout = timeout
        self._start_service = (
            start_service if start_service is not None else default_start_worker_service
        )

    def transcribe(self, artifact: CaptureArtifact) -> Transcription:
        request = {
            "id": uuid.uuid4().hex,
            "op": "transcribe",
            "audio": artifact.audio,
            "sample_rate": artifact.sample_rate,
        }
        last: _WorkerConnectFailure | None = None
        for attempt in (0, 1):
            try:
                response = self._round_trip(request)
                return self._parse_response(response)
            except _WorkerConnectFailure as exc:
                last = exc
                if attempt == 0:
                    self._start_service()
        raise WorkerError(ErrorCode("worker", "unavailable"), str(last))

    def close(self) -> None:
        # Per-request connections; nothing to release between requests.
        pass

    def _round_trip(self, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self._timeout)
                sock.connect(str(self._socket_path))
                sock.sendall(encode_message(request) + b"\n")
                line = _read_line(sock)
        except (OSError, ProtocolError) as exc:
            raise _WorkerConnectFailure(str(exc)) from exc
        if line is None:
            raise _WorkerConnectFailure("worker closed the connection without a reply")
        try:
            return decode_message(line)
        except ProtocolError as exc:
            raise _WorkerConnectFailure(f"invalid worker reply: {exc}") from exc

    @staticmethod
    def _parse_response(response: Mapping[str, Any]) -> Transcription:
        if response.get("status") == "ok":
            text = response.get("text", "")
            segments = tuple(
                Segment(
                    start_ms=int(seg["start_ms"]),
                    end_ms=int(seg["end_ms"]),
                    text=str(seg["text"]),
                )
                for seg in response.get("segments", [])
            )
            request_id = (
                response.get("id") if isinstance(response.get("id"), str) else None
            )
            return Transcription(
                text=text if isinstance(text, str) else "",
                segments=segments,
                request_id=request_id,
            )
        code = _parse_error_code(response.get("error_code"))
        detail = str(response.get("error_message") or "")
        if code.code == "empty_speech":
            raise EmptySpeechError()
        raise WorkerError(code, detail)


# --- Per-session state -------------------------------------------------------


@dataclass
class _Session:
    """State held for one recording → commit session."""

    snapshot: FocusSnapshot
    token: str | None
    fcitx: FcitxClientLike


# --- State machine -----------------------------------------------------------


class VoiceDaemon:
    """Drives the capture → transcribe → commit pipeline on small interfaces."""

    def __init__(
        self,
        *,
        guard: FocusGuard,
        recorder: Recorder,
        fcitx_factory: Callable[[], FcitxClientLike],
        clipboard: ClipboardLike,
        injector: InjectorLike,
        notifier: Notifier,
        worker: WorkerClient,
        auto_stop_event: threading.Event | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._guard = guard
        self._recorder = recorder
        self._fcitx_factory = fcitx_factory
        self._clipboard = clipboard
        self._injector = injector
        self._notifier = notifier
        self._worker = worker
        self._auto_stop_event = auto_stop_event
        self._monotonic = monotonic
        self._sleep = sleep
        self._state = DaemonState.IDLE
        self._session: _Session | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> DaemonState:
        return self._state

    # --- Requests ------------------------------------------------------------

    def dispatch(self, message: Mapping[str, Any]) -> str:
        op = message.get("op")
        if op == "start_if_idle":
            return self.start_if_idle()
        if op == "stop":
            self.stop()
            return "ok"
        logger.warning("ignoring unknown op: %r", op)
        return "error"

    def start_if_idle(self) -> str:
        """Start recording if idle; ``"started"`` / ``"busy"`` / ``"error"`` /
        ``"cancelled"``."""
        if not self._lock.acquire(blocking=False):
            logger.info("start_if_idle rejected: session already in progress")
            return "busy"
        try:
            if self._state is not DaemonState.IDLE:
                logger.info("start_if_idle ignored: state=%s", self._state.value)
                return "busy"

            try:
                snapshot = self._guard.capture()
            except X11Error:
                logger.warning("start failed: X11 focus capture unavailable")
                self._notify(NOTIFY_RECOGNITION_FAILED.format(category="x11"))
                return "error"

            # Hold-to-talk gate: confirm C is still held before recording.
            if not self._confirm_c_pressed():
                logger.info("start_if_idle cancelled: C not held in window")
                self._recorder.cancel()
                self._cleanup()
                return "cancelled"

            try:
                fcitx = self._fcitx_factory()
            except Exception:
                logger.warning("fcitx client unavailable; XTEST-only commit")
                fcitx = _NullFcitx()
            token: str | None = None
            try:
                token = fcitx.start_focus()
            except FcitxCommitError:
                token = None  # decision: still record; clipboard + XTEST only

            self._session = _Session(snapshot=snapshot, token=token, fcitx=fcitx)
            try:
                self._recorder.start()
            except CaptureError:
                logger.warning("start failed: capture could not start")
                self._cleanup()
                self._notify(NOTIFY_RECOGNITION_FAILED.format(category="capture"))
                return "error"

            self._state = DaemonState.RECORDING
            logger.info("state -> recording")
            self._notify(NOTIFY_RECORDING)
            return "started"
        finally:
            self._lock.release()

    def _confirm_c_pressed(self) -> bool:
        """Return ``True`` when C is still held within the confirmation window."""
        deadline = self._monotonic() + C_CONFIRM_TIMEOUT_SECONDS
        while True:
            try:
                if self._guard.c_is_down():
                    return True
            except X11Error:
                return False
            if self._monotonic() >= deadline:
                return False
            self._sleep(C_CONFIRM_POLL_SECONDS)

    def stop(self) -> None:
        """Handle a bridge ``stop`` (key released); only acts while RECORDING."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._state is not DaemonState.RECORDING:
                logger.info("stop ignored: state=%s", self._state.value)
                return
            self._notify(NOTIFY_TRANSCRIBING)
            self._transcribe_and_commit()
        finally:
            self._lock.release()

    def handle_auto_stop(self) -> None:
        """Handle the recorder's 30-minute auto-stop (limit reached)."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._state is not DaemonState.RECORDING:
                return
            self._notify(NOTIFY_LIMIT_REACHED)
            self._transcribe_and_commit()
        finally:
            self._lock.release()

    def shutdown(self) -> None:
        """Final teardown: cleanup any session and close long-lived resources."""
        with self._lock:
            self._cleanup()
            with contextlib.suppress(Exception):
                self._worker.close()

    # --- Pipeline ------------------------------------------------------------

    def _transcribe_and_commit(self) -> None:
        self._state = DaemonState.TRANSCRIBING
        logger.info("state -> transcribing")
        try:
            try:
                artifact = self._recorder.stop()
            except CaptureError as exc:
                logger.warning(
                    "transcribe aborted: capture failed (%s)", type(exc).__name__
                )
                self._notify(NOTIFY_RECOGNITION_FAILED.format(category="capture"))
                return

            try:
                transcription = self._worker.transcribe(artifact)
            except EmptySpeechError:
                self._notify(NOTIFY_EMPTY_SPEECH)
                return
            except WorkerError as exc:
                logger.warning("transcribe failed: %s", exc.code)
                self._notify(NOTIFY_RECOGNITION_FAILED.format(category=str(exc.code)))
                return

            text = transcription.text
            if not text:
                self._notify(NOTIFY_EMPTY_SPEECH)
                return

            self._state = DaemonState.COMMITTING
            logger.info("state -> committing")
            self._commit(text)
        finally:
            self._cleanup()

    def _commit(self, text: str) -> None:
        session = self._session
        if session is None:
            return
        snapshot = session.snapshot

        # 1. Mirror to the clipboard immediately and independently of injection.
        clipboard_ok = True
        try:
            self._clipboard.write_utf8(text)
        except ClipboardError:
            clipboard_ok = False
            logger.warning("clipboard mirror failed")

        # 2. Re-check X11 focus before any injection.
        try:
            current = self._guard.capture()
        except X11Error:
            self._notify(NOTIFY_RECOGNITION_FAILED.format(category="x11"))
            return
        if not self._guard.is_same(snapshot, current):
            logger.info("focus changed; skipping injection")
            self._notify_focus_or_clipboard(clipboard_ok)
            return

        # 3. Prefer Fcitx commit (reusing the focus token), else XTEST fallback.
        if session.token is not None:
            outcome = self._fcitx_commit(session, text)
            if outcome is True:
                if not clipboard_ok:
                    self._notify(NOTIFY_CLIPBOARD_FAILED)
                return
            if outcome is False:
                return  # explicit stale-focus reject: never fall back

        # 4. XTEST fallback: only when focus is still unchanged and XTEST works.
        self._xtest_fallback(snapshot, clipboard_ok)

    def _fcitx_commit(self, session: _Session, text: str) -> bool | None:
        """Return ``True`` (committed), ``False`` (explicit stale-focus reject),
        or ``None`` (channel failure → XTEST fallback allowed)."""
        assert session.token is not None
        try:
            result = session.fcitx.commit(session.token, text)
        except FcitxCommitError:
            logger.info("fcitx channel failure; XTEST fallback allowed")
            return None
        return self._interpret_commit(result)

    def _interpret_commit(self, result: CommitResult) -> bool | None:
        if result.committed:
            return True
        if result.error is not None and result.error.code == "stale-focus":
            logger.info("fcitx rejected with stale focus; no fallback")
            self._notify(NOTIFY_RECOGNITION_FAILED.format(category=str(result.error)))
            return False
        logger.info("fcitx returned error; XTEST fallback allowed")
        return None

    def _xtest_fallback(self, snapshot: FocusSnapshot, clipboard_ok: bool) -> None:
        try:
            current = self._guard.capture()
        except X11Error:
            self._notify(NOTIFY_RECOGNITION_FAILED.format(category="x11"))
            return
        if not self._guard.is_same(snapshot, current):
            logger.info("focus changed before XTEST; skipping injection")
            self._notify_focus_or_clipboard(clipboard_ok)
            return
        try:
            self._injector.paste_ctrl_v()
        except XTestError:
            logger.info("XTEST injection failed")
            self._notify(NOTIFY_RECOGNITION_FAILED.format(category="injection"))
            return
        if not clipboard_ok:
            self._notify(NOTIFY_CLIPBOARD_FAILED)

    def _notify_focus_or_clipboard(self, clipboard_ok: bool) -> None:
        if clipboard_ok:
            self._notify(NOTIFY_FOCUS_CHANGED)
        else:
            self._notify(NOTIFY_RECOGNITION_FAILED.format(category="clipboard"))

    # --- Cleanup -------------------------------------------------------------

    def _cleanup(self) -> None:
        """Idempotent teardown; every path leaving a non-IDLE state calls it."""
        self._recorder.cleanup()
        session = self._session
        self._session = None
        if session is not None:
            with contextlib.suppress(Exception):
                session.fcitx.close()
        self._state = DaemonState.IDLE

    def _notify(self, message: str) -> None:
        try:
            self._notifier.notify(message)
        except Exception:
            logger.debug("notification failed (ignored)")


# --- Socket transport --------------------------------------------------------


class _ShutdownRequested(Exception):
    pass


def _read_line(conn: socket.socket, max_bytes: int = MAX_MESSAGE_BYTES) -> bytes | None:
    """Read one newline-terminated line, bounded to ``max_bytes``."""
    buf = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return bytes(buf) if buf else None
        buf.extend(chunk)
        if len(buf) > max_bytes + 1:
            raise MessageTooLarge(f"message exceeds {max_bytes} bytes")
        if b"\n" in buf:
            break
    line, _sep, _rest = buf.partition(b"\n")
    return bytes(line)


def peer_uid(conn: socket.socket) -> int | None:
    """Return the uid of the peer, or ``None`` when credentials are unavailable."""
    try:
        creds = conn.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        )
    except OSError:
        return None
    _pid, uid, _gid = struct.unpack("3i", creds)
    return int(uid)


class DaemonRequestHandler(socketserver.StreamRequestHandler):
    """Handle one JSON-line request and reply with a one-line status."""

    def handle(self) -> None:
        server = cast("DaemonServer", self.server)
        conn = self.connection
        if not server.client_allowed(conn):
            logger.warning("rejecting connection from non-owner uid")
            return
        try:
            line = _read_line(conn)
        except MessageTooLarge:
            return
        except OSError:
            return
        if not line:
            return
        try:
            message = decode_message(line)
        except ProtocolError:
            return
        status = server.daemon.dispatch(message)
        with contextlib.suppress(OSError):
            conn.sendall(encode_message({"status": status}) + b"\n")


class DaemonServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Concurrent Unix socket server; rejects non-owner clients.

    Each connection is handled in its own thread so a long-running
    transcription never blocks ``accept``: a ``start_if_idle`` arriving while a
    session is in flight is rejected immediately (the state machine lock is
    non-blocking) instead of queueing in the backlog.
    """

    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        socket_path: Path,
        daemon: VoiceDaemon,
        *,
        uid: int | None = None,
        auto_stop_event: threading.Event | None = None,
    ) -> None:
        self.daemon = daemon
        self.uid = os.getuid() if uid is None else uid
        self.socket_path = socket_path
        self._auto_stop_event = auto_stop_event
        self.allow_reuse_address = True
        self.request_queue_size = SOCKET_BACKLOG
        super().__init__(str(socket_path), DaemonRequestHandler)
        os.chmod(socket_path, config.SOCKET_MODE)

    def client_allowed(self, conn: socket.socket) -> bool:
        return peer_uid(conn) == self.uid

    def service_actions(self) -> None:
        """Poll the recorder's 30-minute auto-stop signal every loop iteration.

        The pipeline runs on a short-lived worker thread so the accept loop
        stays responsive; the state machine lock serializes it against any
        concurrent request.
        """
        event = self._auto_stop_event
        if event is not None and event.is_set():
            event.clear()
            thread = threading.Thread(
                target=self.daemon.handle_auto_stop, daemon=True
            )
            thread.start()


def _unlink_socket(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("could not remove socket %s: %s", path, exc)


def prepare_runtime_dir(path: Path) -> None:
    """Create the private runtime dir and force its mode to ``0700``."""
    path.mkdir(mode=config.DIRECTORY_MODE, parents=True, exist_ok=True)
    os.chmod(path, config.DIRECTORY_MODE)


def serve(
    socket_path: Path,
    daemon: VoiceDaemon,
    *,
    uid: int | None = None,
    auto_stop_event: threading.Event | None = None,
) -> int:
    """Run the daemon socket server until SIGTERM/SIGINT, cleaning up on exit."""
    _unlink_socket(socket_path)
    server = DaemonServer(
        socket_path, daemon, uid=uid, auto_stop_event=auto_stop_event
    )

    def _stop(signum: int, frame: object) -> None:
        raise _ShutdownRequested(signum)

    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            previous[signum] = signal.signal(signum, _stop)
    try:
        logger.info("daemon listening on %s", socket_path)
        server.serve_forever(poll_interval=0.1)
    except _ShutdownRequested:
        logger.info("shutdown requested")
    finally:
        server.server_close()
        daemon.shutdown()
        _unlink_socket(socket_path)
        for saved, handler in previous.items():
            signal.signal(saved, handler)
    return 0


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fun-voice-daemon",
        description="Fun Voice Ryan push-to-talk daemon (capture -> ASR -> commit).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        paths = config.build_runtime_paths(config.resolve_runtime_dir())
    except config.ConfigError as exc:
        logger.error("cannot resolve runtime dir: %s", exc)
        return 1

    prepare_runtime_dir(paths.runtime_dir)

    notifier = NotifySendNotifier()
    auto_stop_event = threading.Event()
    recorder = PipeWireRecorder(
        notifier=notifier.notify,
        on_auto_stop=auto_stop_event.set,
        runtime_dir=paths.runtime_dir,
    )
    daemon = VoiceDaemon(
        guard=X11FocusGuard(),
        recorder=recorder,
        fcitx_factory=FcitxClient,
        clipboard=ClipboardMirror(),
        injector=XTestInjector(),
        notifier=notifier,
        worker=SocketWorkerClient(paths.worker_socket),
        auto_stop_event=auto_stop_event,
    )
    return serve(paths.daemon_socket, daemon, auto_stop_event=auto_stop_event)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
