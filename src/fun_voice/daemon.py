"""Voice daemon: state machine, capture→worker→output pipeline, and socket server.

The daemon owns the whole push-to-talk lifecycle. An in-process X11 listener
delivers the ``Super+C`` press/release lifecycle. It also listens on
``$XDG_RUNTIME_DIR/fun-voice-ryan/daemon.sock`` (mode ``0600``, same-uid
``SO_PEERCRED`` gated) for diagnostics and local integration requests:

    {"op": "diagnostics"}     -> return non-sensitive hotkey readiness booleans

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
import functools
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
from fun_voice.capture import CaptureConfig, CaptureError, PipeWireRecorder
from fun_voice.contracts import (
    MAX_MESSAGE_BYTES,
    WORKER_RESPONSE_MAX_BYTES,
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
from fun_voice.corrector import CorrectionError, OnDemandQwenCorrector
from fun_voice.desktop import (
    ClipboardError,
    ClipboardMirror,
    Runner,
    X11Error,
    X11FocusGuard,
    X11HotkeyListener,
    X11HotkeyUnavailable,
    XTestError,
    XTestInjector,
    default_runner,
)
from fun_voice.fcitx import FcitxClient, FcitxCommitError
from fun_voice.metrics import MetricsLedger

logger = logging.getLogger(__name__)

# --- Notification copy (fixed; never carries transcription text) -------------

NOTIFICATION_APP_NAME = "Fun Voice Ryan"
NOTIFY_RECORDING = "录音中"
NOTIFY_TRANSCRIBING = "识别中"
NOTIFY_EMPTY_SPEECH = "未检测到语音"
NOTIFY_FOCUS_CHANGED = "焦点已变化，结果已复制到剪贴板"
NOTIFY_COMMIT_CANCELLED = "输入已取消，结果已复制到剪贴板"
NOTIFY_RESULT_LOST = "结果已丢失：焦点已变化且剪贴板不可用"
NOTIFY_CLIPBOARD_FAILED = "文本已输入，但剪贴板备份失败"
NOTIFY_RECOGNITION_FAILED = "本地识别失败：{category}"
NOTIFY_LIMIT_REACHED = "已达到 30 分钟录音上限，开始识别"

# The worker may need up to its 120s inference timeout; leave margin.
WORKER_RESPONSE_TIMEOUT_SECONDS = 130.0
WORKER_STARTUP_TIMEOUT_SECONDS = 15.0
WORKER_STARTUP_POLL_SECONDS = 0.05
WORKER_TEMPLATE = "fun-voice-worker@{}.service"
SOCKET_BACKLOG = 4
REQUEST_READ_TIMEOUT_SECONDS = 10.0
STOP_LOCK_TIMEOUT_SECONDS = 2.0
HOTKEY_UNAVAILABLE_EXIT = 2

# Hold-to-talk confirmation: after an X11 press event, the daemon confirms C
# is still physically held within this window before recording.
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


_ALLOWED_WORKER_CODES = frozenset(
    {
        "empty_speech",
        "oom",
        "vllm",
        "no_output",
        "format",
        "timeout",
        "device",
        "internal",
        "protocol",
        "unavailable",
        "model_load",
    }
)


def _parse_error_code(value: object) -> ErrorCode:
    """Parse a worker ``error_code`` string into an :class:`ErrorCode`.

    Only the known worker categories/codes are accepted; anything else falls
    back to ``worker.internal`` so a malformed or hostile code never flows into
    a notification or log line.
    """
    if not isinstance(value, str) or not value:
        return ErrorCode("worker", "internal")
    category, sep, code = value.partition(".")
    if sep and category == "worker" and code in _ALLOWED_WORKER_CODES:
        return ErrorCode(category, code)
    return ErrorCode("worker", "internal")


# --- Adapter seams (the fakes implement these) -------------------------------


class FocusGuard(Protocol):
    def capture(self) -> FocusSnapshot: ...

    def is_same(self, a: FocusSnapshot, b: FocusSnapshot) -> bool: ...

    def c_is_down(self) -> bool: ...


class Recorder(Protocol):
    def start(self, config: CaptureConfig | None = None) -> None: ...

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


class TextCorrector(Protocol):
    """A text-only candidate corrector; it never owns ASR state."""

    def correct(self, raw_text: str) -> str: ...

    def close(self) -> None: ...


class HotkeyLifecycle(Protocol):
    """Owns the X11 hotkey registration and event-loop lifetime."""

    def start(self) -> None: ...

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


class _DisabledInjector:
    """XTEST fallback turned off by configuration; never injects."""

    def paste_ctrl_v(self) -> None:
        raise XTestError("XTEST fallback disabled by configuration")


# --- Notifier (notify-send) --------------------------------------------------


class NotifySendNotifier:
    """Best-effort desktop notification via ``notify-send``.

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


def worker_service_name(profile: str) -> str:
    """Return the non-enabled systemd template instance for an ASR profile."""
    if profile not in {"nano", "sensevoice"}:
        raise ValueError(f"unsupported worker profile: {profile}")
    return WORKER_TEMPLATE.format(profile)


def default_start_worker_service(profile: str = "nano") -> None:
    """Best-effort start of one on-demand model-worker template instance."""
    service = worker_service_name(profile)
    try:
        subprocess.run(
            ["systemctl", "--user", "start", service],
            check=False,
            capture_output=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("could not start %s", service)


def default_stop_worker_service(profile: str = "nano") -> None:
    """Best-effort stop used before switching an OOM Nano request to fallback."""
    service = worker_service_name(profile)
    try:
        subprocess.run(
            ["systemctl", "--user", "stop", service],
            check=False,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("could not stop %s", service)


class SocketWorkerClient:
    """Speaks the worker's single-line JSON protocol over a Unix socket.

    Opens a fresh connection per request. When its socket is unreachable it
    starts the assigned user service and polls for a successful request, so a
    model-worker process has time to bind its private socket.
    """

    def __init__(
        self,
        socket_path: Path | str,
        *,
        timeout: float = WORKER_RESPONSE_TIMEOUT_SECONDS,
        start_service: Callable[[], None] | None = None,
        startup_timeout: float = WORKER_STARTUP_TIMEOUT_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._socket_path = Path(socket_path)
        self._timeout = timeout
        self._start_service = (
            start_service if start_service is not None else default_start_worker_service
        )
        self._startup_timeout = startup_timeout
        self._monotonic = monotonic
        self._sleep = sleep

    def transcribe(self, artifact: CaptureArtifact) -> Transcription:
        request = {
            "id": uuid.uuid4().hex,
            "op": "transcribe",
            "audio": artifact.audio,
            "sample_rate": artifact.sample_rate,
        }
        try:
            return self._parse_response(self._round_trip(request))
        except _WorkerConnectFailure as first_failure:
            self._start_service()
            return self._wait_for_worker(request, first_failure)

    def preload(self) -> None:
        """Request model materialization without sending any audio."""
        request = {"id": uuid.uuid4().hex, "op": "preload"}
        try:
            self._parse_preload_response(self._round_trip(request))
        except _WorkerConnectFailure as first_failure:
            self._start_service()
            self._wait_for_preload(request, first_failure)

    def _wait_for_worker(
        self, request: Mapping[str, Any], first_failure: _WorkerConnectFailure
    ) -> Transcription:
        deadline = self._monotonic() + self._startup_timeout
        last = first_failure
        while self._monotonic() < deadline:
            try:
                return self._parse_response(self._round_trip(request))
            except _WorkerConnectFailure as exc:
                last = exc
                self._sleep(WORKER_STARTUP_POLL_SECONDS)
        raise WorkerError(ErrorCode("worker", "unavailable"), str(last))

    def _wait_for_preload(
        self, request: Mapping[str, Any], first_failure: _WorkerConnectFailure
    ) -> None:
        deadline = self._monotonic() + self._startup_timeout
        last = first_failure
        while self._monotonic() < deadline:
            try:
                self._parse_preload_response(self._round_trip(request))
                return
            except _WorkerConnectFailure as exc:
                last = exc
                self._sleep(WORKER_STARTUP_POLL_SECONDS)
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
                line = _read_line(sock, WORKER_RESPONSE_MAX_BYTES)
        except (OSError, ProtocolError) as exc:
            raise _WorkerConnectFailure(str(exc)) from exc
        if line is None:
            raise _WorkerConnectFailure("worker closed the connection without a reply")
        try:
            return decode_message(line, WORKER_RESPONSE_MAX_BYTES)
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

    @staticmethod
    def _parse_preload_response(response: Mapping[str, Any]) -> None:
        if response.get("status") == "ok" and response.get("model_ready") is True:
            return
        code = _parse_error_code(response.get("error_code"))
        detail = str(response.get("error_message") or "")
        raise WorkerError(code, detail)


class FallbackWorkerClient:
    """Use SenseVoiceSmall only when Nano could not load or exhausted XPU."""

    def __init__(
        self,
        nano: WorkerClient,
        sensevoice: WorkerClient,
        *,
        stop_primary: Callable[[], None],
    ) -> None:
        self._nano = nano
        self._sensevoice = sensevoice
        self._stop_primary = stop_primary

    def transcribe(self, artifact: CaptureArtifact) -> Transcription:
        try:
            return self._nano.transcribe(artifact)
        except WorkerError as exc:
            if exc.code.code not in {"model_load", "oom"}:
                raise
            self._stop_primary()
            return self._sensevoice.transcribe(artifact)

    def close(self) -> None:
        self._nano.close()
        self._sensevoice.close()


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
        corrector: TextCorrector | None = None,
        metrics: MetricsLedger | None = None,
        nano_preloader: Callable[[], None] | None = None,
        auto_stop_event: threading.Event | None = None,
        capture_config: CaptureConfig | None = None,
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
        self._corrector = corrector
        self._metrics = metrics if metrics is not None else MetricsLedger()
        self._metric_sequence: int | None = None
        self._nano_preloader = nano_preloader
        self._auto_stop_event = auto_stop_event
        self._capture_config = (
            capture_config if capture_config is not None else CaptureConfig()
        )
        self._monotonic = monotonic
        self._sleep = sleep
        self._state = DaemonState.IDLE
        self._session: _Session | None = None
        self._lock = threading.Lock()
        # Ephemeral X11 hotkey evidence. It is deliberately only booleans:
        # no timestamp, audio, text, focus snapshot, or Fcitx token is retained.
        self._hotkey_registered = False
        self._hotkey_press_seen = False

    @property
    def state(self) -> DaemonState:
        return self._state

    # --- Requests ------------------------------------------------------------

    def dispatch(self, message: Mapping[str, Any]) -> str | Mapping[str, object]:
        try:
            op = message.get("op")
            if op == "start_if_idle":
                return self.start_if_idle()
            if op == "stop":
                self.stop()
                return "ok"
            if op == "diagnostics":
                return self.diagnostics()
            if op == "metrics":
                return self._metrics.summary()
            logger.warning("ignoring unknown op: %r", op)
            return "error"
        except Exception as exc:
            # Never leak a message body or transcription text into logs.
            logger.error("dispatch failed: %s", type(exc).__name__)
            self._notify(NOTIFY_RECOGNITION_FAILED.format(category="internal"))
            return "error"

    def diagnostics(self) -> dict[str, bool]:
        """Return non-sensitive, process-lifetime X11 hotkey evidence."""
        return {
            "hotkey_registered": self._hotkey_registered,
            "hotkey_press_seen": self._hotkey_press_seen,
        }

    def mark_hotkey_registered(self) -> None:
        """Record a successfully installed X11 grab without retaining event data."""
        self._hotkey_registered = True

    def handle_hotkey_press(self) -> str:
        """Handle a real matching X11 KeyPress and start recording if idle."""
        self._hotkey_press_seen = True
        return self.start_if_idle()

    def handle_hotkey_release(self) -> None:
        """Handle a matching KeyRelease without blocking the X11 event loop."""
        threading.Thread(target=self.stop, daemon=True, name="hotkey-stop").start()

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
                self._recorder.start(self._capture_config)
            except CaptureError:
                logger.warning("start failed: capture could not start")
                self._cleanup()
                self._notify(NOTIFY_RECOGNITION_FAILED.format(category="capture"))
                return "error"

            # Re-check C after starting: a release during the start window must
            # cancel instead of leaving an unintended recording running.
            try:
                if not self._guard.c_is_down():
                    logger.info("start_if_idle cancelled: C released during start")
                    self._cleanup()
                    return "cancelled"
            except X11Error:
                logger.warning("start_if_idle cancelled: X11 unavailable after start")
                self._cleanup()
                return "cancelled"

            self._state = DaemonState.RECORDING
            self._metric_sequence = self._metrics.begin()
            self._schedule_nano_preload(self._metric_sequence)
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

    def _schedule_nano_preload(self, sequence: int) -> None:
        """Start one non-blocking Nano preload after capture has begun."""
        preloader = self._nano_preloader
        if preloader is None:
            return
        self._metrics.record(sequence, nano_preload="scheduled")
        threading.Thread(
            target=self._preload_nano,
            args=(sequence, preloader),
            daemon=True,
            name="nano-preload",
        ).start()

    def _preload_nano(self, sequence: int, preloader: Callable[[], None]) -> None:
        started = self._monotonic()
        try:
            preloader()
        except Exception as exc:  # noqa: BLE001 - recording stays unaffected
            logger.info("nano preload unavailable: %s", type(exc).__name__)
            self._metrics.record(
                sequence,
                preload_ms=_elapsed_ms(started, self._monotonic()),
                nano_preload="failed",
            )
            return
        self._metrics.record(
            sequence,
            preload_ms=_elapsed_ms(started, self._monotonic()),
            nano_preload="ready",
        )

    def stop(self) -> None:
        """Handle a hotkey release; only acts while ``RECORDING``.

        Uses a bounded blocking acquire so a ``stop`` arriving while ``start``
        still holds the lock (e.g. during ``fcitx.start_focus``) waits for it
        and then stops, instead of being silently dropped. The bound keeps it
        from ever blocking on a long transcription.
        """
        if not self._lock.acquire(timeout=STOP_LOCK_TIMEOUT_SECONDS):
            return
        try:
            if self._state is not DaemonState.RECORDING:
                logger.info("stop ignored: state=%s", self._state.value)
                return
            self._stop_locked()
        finally:
            self._lock.release()

    def _stop_locked(self) -> None:
        """Notify and transcribe; caller holds the lock and observed RECORDING."""
        self._notify(NOTIFY_TRANSCRIBING)
        self._transcribe_and_commit()

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
        """Final teardown; never blocks on an in-flight transcription.

        An in-flight session already finalized its capture shards via
        ``recorder.stop()`` and only holds a memfd that the OS reclaims on
        process exit, so skipping cleanup here is safe and keeps shutdown
        within systemd's ``TimeoutStopSec``.
        """
        if not self._lock.acquire(blocking=False):
            logger.info("shutdown: in-flight session; skipping cleanup")
            return
        try:
            self._cleanup()
            with contextlib.suppress(Exception):
                self._worker.close()
            if self._corrector is not None:
                with contextlib.suppress(Exception):
                    self._corrector.close()
        finally:
            self._lock.release()

    # --- Pipeline ------------------------------------------------------------

    def _transcribe_and_commit(self) -> None:
        self._state = DaemonState.TRANSCRIBING
        logger.info("state -> transcribing")
        pipeline_started = self._monotonic()
        try:
            try:
                artifact = self._recorder.stop()
            except CaptureError as exc:
                logger.warning(
                    "transcribe aborted: capture failed (%s)", type(exc).__name__
                )
                self._record_metric(error_code="capture")
                self._notify(NOTIFY_RECOGNITION_FAILED.format(category="capture"))
                return
            if artifact.duration_ms is not None:
                self._record_metric(capture_duration_ms=artifact.duration_ms)

            asr_started = self._monotonic()
            try:
                transcription = self._worker.transcribe(artifact)
            except EmptySpeechError:
                self._record_metric(
                    asr_ms=_elapsed_ms(asr_started, self._monotonic()),
                    error_code="empty_speech",
                )
                self._notify(NOTIFY_EMPTY_SPEECH)
                return
            except WorkerError as exc:
                self._record_metric(
                    asr_ms=_elapsed_ms(asr_started, self._monotonic()),
                    error_code=str(exc.code),
                )
                logger.warning("transcribe failed: %s", exc.code)
                self._notify(NOTIFY_RECOGNITION_FAILED.format(category=str(exc.code)))
                return
            self._record_metric(asr_ms=_elapsed_ms(asr_started, self._monotonic()))

            text = transcription.text
            if not text:
                self._record_metric(error_code="empty_speech")
                self._notify(NOTIFY_EMPTY_SPEECH)
                return

            corrector = self._corrector
            if corrector is not None:
                correction_started = self._monotonic()
                try:
                    candidate = corrector.correct(text)
                    self._record_metric(
                        correction_ms=_elapsed_ms(
                            correction_started, self._monotonic()
                        ),
                        correction="corrected" if candidate else "raw_fallback",
                    )
                    if candidate:
                        text = candidate
                except CorrectionError as exc:
                    # The raw ASR result remains the safe final output.  Keep
                    # the code-only log and avoid an extra desktop notification
                    # for every transient model failure.
                    logger.warning("correction unavailable: %s", exc.code)
                    self._record_metric(
                        correction_ms=_elapsed_ms(
                            correction_started, self._monotonic()
                        ),
                        correction="failed",
                    )
                except Exception as exc:  # noqa: BLE001 - raw text is resilient
                    logger.warning("correction unavailable: %s", type(exc).__name__)
                    self._record_metric(
                        correction_ms=_elapsed_ms(
                            correction_started, self._monotonic()
                        ),
                        correction="failed",
                    )

            self._state = DaemonState.COMMITTING
            logger.info("state -> committing")
            commit_started = self._monotonic()
            self._commit(text)
            self._record_metric(
                commit_ms=_elapsed_ms(commit_started, self._monotonic())
            )
        finally:
            self._record_metric(
                end_to_end_ms=_elapsed_ms(pipeline_started, self._monotonic())
            )
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
            self._notify_focus_or_clipboard(clipboard_ok, NOTIFY_FOCUS_CHANGED)
            return

        # 3. Prefer Fcitx commit (reusing the focus token), else XTEST fallback.
        if session.token is not None:
            outcome = self._fcitx_commit(session, text, clipboard_ok)
            if outcome is True:
                if not clipboard_ok:
                    self._notify(NOTIFY_CLIPBOARD_FAILED)
                return
            if outcome is False:
                return  # explicit stale-focus reject: never fall back

        # 4. XTEST fallback: only when focus is still unchanged and XTEST works.
        self._xtest_fallback(snapshot, clipboard_ok)

    def _fcitx_commit(
        self, session: _Session, text: str, clipboard_ok: bool
    ) -> bool | None:
        """Return ``True`` (committed), ``False`` (explicit stale-focus reject),
        or ``None`` (channel failure → XTEST fallback allowed)."""
        assert session.token is not None
        try:
            result = session.fcitx.commit(session.token, text)
        except FcitxCommitError:
            logger.info("fcitx channel failure; XTEST fallback allowed")
            return None
        return self._interpret_commit(result, clipboard_ok)

    def _interpret_commit(
        self, result: CommitResult, clipboard_ok: bool
    ) -> bool | None:
        if result.committed:
            return True
        if result.error is not None and result.error.code == "stale-focus":
            logger.info("fcitx rejected with stale focus; no fallback")
            self._notify_focus_or_clipboard(clipboard_ok, NOTIFY_COMMIT_CANCELLED)
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
            self._notify_focus_or_clipboard(clipboard_ok, NOTIFY_FOCUS_CHANGED)
            return
        try:
            self._injector.paste_ctrl_v()
        except XTestError:
            logger.info("XTEST injection failed")
            self._notify(NOTIFY_RECOGNITION_FAILED.format(category="injection"))
            return
        if not clipboard_ok:
            self._notify(NOTIFY_CLIPBOARD_FAILED)

    def _notify_focus_or_clipboard(self, clipboard_ok: bool, message: str) -> None:
        if clipboard_ok:
            self._notify(message)
        else:
            self._notify(NOTIFY_RESULT_LOST)

    # --- Cleanup -------------------------------------------------------------

    def _cleanup(self) -> None:
        """Idempotent teardown; every path leaving a non-IDLE state calls it."""
        self._recorder.cleanup()
        session = self._session
        self._session = None
        self._metric_sequence = None
        if session is not None:
            with contextlib.suppress(Exception):
                session.fcitx.close()
        self._state = DaemonState.IDLE

    def _notify(self, message: str) -> None:
        try:
            self._notifier.notify(message)
        except Exception:
            logger.debug("notification failed (ignored)")

    def _record_metric(self, **updates: object) -> None:
        """Best-effort telemetry that can never affect the input pipeline."""
        sequence = self._metric_sequence
        if sequence is None:
            return
        with contextlib.suppress(ValueError):
            self._metrics.record(sequence, **updates)


def _elapsed_ms(started: float, finished: float) -> int:
    """Convert a monotonic interval to a non-negative integer milliseconds."""
    return max(0, round((finished - started) * 1000))


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
        conn.settimeout(REQUEST_READ_TIMEOUT_SECONDS)
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
        result = server.daemon.dispatch(message)
        response: dict[str, Any]
        if isinstance(result, Mapping):
            response = {"status": "ok", **result}
        else:
            response = {"status": result}
        with contextlib.suppress(OSError):
            conn.sendall(encode_message(response) + b"\n")


class DaemonServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Concurrent Unix socket server; rejects non-owner clients.

    Each connection is handled in its own thread so a long-running
    transcription never blocks ``accept``: a ``start_if_idle`` arriving while a
    session is in flight is rejected immediately (the state machine lock is
    non-blocking) instead of queueing in the backlog.

    ``daemon_threads`` keeps request threads daemonic so ``server_close`` never
    waits on an in-flight transcription; shutdown must not exceed systemd's
    default ``TimeoutStopSec``.
    """

    daemon_threads = True

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
        """Run the main-loop side effects every ``poll_interval`` tick.

        Polls the recorder's 30-minute auto-stop signal. The pipeline action
        runs off the accept loop so it never blocks on transcription; the state
        machine lock serializes it against concurrent local requests.
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
    hotkey_listener: HotkeyLifecycle | None = None,
) -> int:
    """Run the daemon socket server until SIGTERM/SIGINT, cleaning up on exit.

    The X11 hotkey is acquired before binding the control socket. A failed
    exclusive grab means no push-to-talk input exists, so the daemon exits
    rather than silently falling back to a competing registration mechanism.
    """
    _unlink_socket(socket_path)
    server: DaemonServer | None = None
    listener_started = False

    def _stop(signum: int, frame: object) -> None:
        raise _ShutdownRequested(signum)

    previous: dict[int, Any] = {}
    try:
        if hotkey_listener is not None:
            try:
                hotkey_listener.start()
            except X11HotkeyUnavailable as exc:
                logger.error("X11 hotkey unavailable: %s", exc)
                return HOTKEY_UNAVAILABLE_EXIT
            listener_started = True
            daemon.mark_hotkey_registered()

        server = DaemonServer(
            socket_path, daemon, uid=uid, auto_stop_event=auto_stop_event
        )
        for signum in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):
                previous[signum] = signal.signal(signum, _stop)
        logger.info("daemon listening on %s", socket_path)
        server.serve_forever(poll_interval=0.1)
    except _ShutdownRequested:
        logger.info("shutdown requested")
    finally:
        if listener_started and hotkey_listener is not None:
            hotkey_listener.close()
        if server is not None:
            server.server_close()
        daemon.shutdown()
        _unlink_socket(socket_path)
        for saved, handler in previous.items():
            signal.signal(saved, handler)
    return 0


def build_fcitx_factory(cfg: config.Config) -> Callable[[], FcitxClient]:
    """Return the Fcitx client factory implied by the configuration."""
    return functools.partial(FcitxClient, timeout=cfg.fcitx_commit_timeout_ms / 1000)


def build_injector(cfg: config.Config, guard: X11FocusGuard) -> InjectorLike:
    """Return the XTEST injector implied by the configuration.

    The injector shares the focus guard's live X11 display so injection never
    fails with "requires an X11 display". When the fallback is disabled by
    configuration it returns a no-op injector that only ever reports failure.
    """
    if cfg.allow_x11_paste_fallback:
        return XTestInjector(display=guard.display)
    return _DisabledInjector()


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
        cfg = config.load_config()
    except config.ConfigError as exc:
        logger.error("cannot load config: %s", exc)
        return 1
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
    guard = X11FocusGuard()
    nano_worker = SocketWorkerClient(
        paths.worker_socket,
        start_service=functools.partial(default_start_worker_service, "nano"),
    )
    worker: WorkerClient = nano_worker
    if cfg.inference.allow_sensevoice_fallback:
        sensevoice_worker = SocketWorkerClient(
            paths.runtime_dir / "worker-sensevoice.sock",
            start_service=functools.partial(
                default_start_worker_service, "sensevoice"
            ),
        )
        worker = FallbackWorkerClient(
            nano_worker,
            sensevoice_worker,
            stop_primary=functools.partial(default_stop_worker_service, "nano"),
        )
    corrector: TextCorrector | None = None
    if cfg.enhanced.enabled:
        corrector = OnDemandQwenCorrector()
    daemon = VoiceDaemon(
        guard=guard,
        recorder=recorder,
        fcitx_factory=build_fcitx_factory(cfg),
        clipboard=ClipboardMirror(),
        injector=build_injector(cfg, guard),
        notifier=notifier,
        worker=worker,
        corrector=corrector,
        nano_preloader=nano_worker.preload,
        auto_stop_event=auto_stop_event,
        capture_config=CaptureConfig(source=cfg.audio_source),
    )
    hotkey_listener = X11HotkeyListener(
        daemon.handle_hotkey_press,
        daemon.handle_hotkey_release,
    )
    return serve(
        paths.daemon_socket,
        daemon,
        auto_stop_event=auto_stop_event,
        hotkey_listener=hotkey_listener,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
