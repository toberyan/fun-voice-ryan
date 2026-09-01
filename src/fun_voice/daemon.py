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
from typing import Any, Literal, Protocol, cast

from fun_voice import config
from fun_voice.capture import CaptureConfig, CaptureError, PipeWireRecorder
from fun_voice.contracts import (
    MAX_MESSAGE_BYTES,
    WORKER_RESPONSE_MAX_BYTES,
    AsrStageTiming,
    CaptureArtifact,
    CommitResult,
    CorrectionTiming,
    DaemonState,
    ErrorCode,
    FocusSnapshot,
    MessageTooLarge,
    ModelTaskKind,
    PreloadTiming,
    ProtocolError,
    Segment,
    SessionKey,
    Transcription,
    WorkerHealth,
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
from fun_voice.scheduler import (
    CorrectionOutcome,
    ModelLifecycle,
    TaskHandle,
    XpuScheduler,
)

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
WORKER_STOP_TIMEOUT_SECONDS = 30.0
WORKER_STOP_POLL_SECONDS = 0.05
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


def _optional_duration(payload: Mapping[str, Any], field: str) -> int | None:
    """Accept one bounded, scalar duration from an untrusted local peer."""
    value = payload.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


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


AsrProfile = Literal["nano", "sensevoice"]


class HealthWorkerClient(Protocol):
    """Private-socket worker endpoint used only for a bounded health probe."""

    def health(self) -> WorkerHealth: ...


class ModelProfileSupervisor(Protocol):
    """The scheduler's sole authority for ASR worker process transitions."""

    def start_profile(self, profile: AsrProfile) -> bool: ...

    def stop_profile(self, profile: AsrProfile) -> bool: ...

    def health_profile(self, profile: AsrProfile) -> ModelLifecycle: ...


class SystemdModelProfileSupervisor:
    """Pair systemd lifecycle confirmation with a same-UID worker health probe.

    A socket outage is considered inactive only after this instance has
    successfully stopped that profile through systemd.  It is never treated as
    proof that an unconfirmed model release freed the XPU.
    """

    def __init__(
        self,
        *,
        workers: Mapping[AsrProfile, HealthWorkerClient],
        start_service: Callable[[AsrProfile], bool] | None = None,
        stop_service: Callable[[AsrProfile], bool] | None = None,
    ) -> None:
        self._workers = dict(workers)
        self._start_service = (
            start_service if start_service is not None else default_start_worker_service
        )
        self._stop_service = (
            stop_service if stop_service is not None else default_stop_worker_service
        )
        self._confirmed_inactive: dict[AsrProfile, bool] = {
            "nano": False,
            "sensevoice": False,
        }

    def start_profile(self, profile: AsrProfile) -> bool:
        try:
            started = self._start_service(profile)
        except Exception:  # noqa: BLE001 - deny start on supervisor uncertainty
            return False
        if started:
            self._confirmed_inactive[profile] = False
        return started

    def stop_profile(self, profile: AsrProfile) -> bool:
        try:
            stopped = self._stop_service(profile)
        except Exception:  # noqa: BLE001 - deny release on supervisor uncertainty
            return False
        if stopped:
            self._confirmed_inactive[profile] = True
        return stopped

    def health_profile(self, profile: AsrProfile) -> ModelLifecycle:
        worker = self._workers.get(profile)
        if worker is None:
            return (
                ModelLifecycle.INACTIVE
                if self._confirmed_inactive[profile]
                else ModelLifecycle.FAILED
            )
        try:
            lifecycle = ModelLifecycle(worker.health().lifecycle)
        except Exception:  # noqa: BLE001 - no untrusted worker error propagation
            return (
                ModelLifecycle.INACTIVE
                if self._confirmed_inactive[profile]
                else ModelLifecycle.FAILED
            )
        if lifecycle in {ModelLifecycle.INACTIVE, ModelLifecycle.FAILED}:
            self._confirmed_inactive[profile] = True
        return lifecycle


class TextCorrector(Protocol):
    """A text-only candidate corrector; it never owns ASR state."""

    def correct(self, raw_text: str) -> str: ...

    def close(self) -> None: ...


class XpuLease(Protocol):
    """Confirms the producing ASR profile has yielded the XPU to Qwen."""

    def release_asr_for_qwen(
        self, profile: Literal["nano", "sensevoice"]
    ) -> bool: ...


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


def default_start_worker_service(profile: str = "nano") -> bool:
    """Start one on-demand worker and return whether systemd accepted it."""
    service = worker_service_name(profile)
    try:
        result = subprocess.run(
            ["systemctl", "--user", "start", service],
            check=False,
            capture_output=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("could not start %s", service)
        return False
    if result.returncode != 0:
        logger.warning("could not start %s", service)
        return False
    return True


def default_stop_worker_service(profile: str = "nano") -> bool:
    """Stop and confirm the exact ASR service is no longer active.

    A false result is a hard denial for the Qwen lease. This conservative check
    avoids concurrent model residency when no cross-process XPU telemetry is
    available.
    """
    service = worker_service_name(profile)
    try:
        result = subprocess.run(
            ["systemctl", "--user", "stop", service],
            check=False,
            capture_output=True,
            timeout=WORKER_STOP_TIMEOUT_SECONDS,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("could not stop %s", service)
            return False
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("could not stop %s", service)
        return False

    deadline = time.monotonic() + WORKER_STOP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            state = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    "--property=ActiveState",
                    "--value",
                    service,
                ],
                check=False,
                capture_output=True,
                timeout=5.0,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("could not confirm %s state", service)
            return False
        if state.returncode != 0:
            logger.warning("could not confirm %s state", service)
            return False
        if state.stdout.strip() in {"inactive", "failed"}:
            return True
        time.sleep(WORKER_STOP_POLL_SECONDS)
    logger.warning("timed out waiting for %s to stop", service)
    return False


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
        start_service: Callable[[], object] | None = None,
        auto_start_service: bool = True,
        startup_timeout: float = WORKER_STARTUP_TIMEOUT_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._socket_path = Path(socket_path)
        self._timeout = timeout
        self._start_service = (
            start_service if start_service is not None else default_start_worker_service
        )
        self._auto_start_service = auto_start_service
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
            if self._auto_start_service:
                self._start_service()
            return self._wait_for_worker(request, first_failure)

    def preload(self) -> PreloadTiming:
        """Request model materialization without sending any audio."""
        request = {"id": uuid.uuid4().hex, "op": "preload"}
        try:
            return self._parse_preload_response(self._round_trip(request))
        except _WorkerConnectFailure as first_failure:
            if self._auto_start_service:
                self._start_service()
            return self._wait_for_preload(request, first_failure)

    def health(self) -> WorkerHealth:
        """Read the worker's fixed health state without starting a model."""
        request = {"id": uuid.uuid4().hex, "op": "health"}
        try:
            response = self._round_trip(request)
        except _WorkerConnectFailure as exc:
            raise WorkerError(ErrorCode("worker", "unavailable")) from exc
        if response.get("status") != "ok":
            raise WorkerError(_parse_error_code(response.get("error_code")))
        lifecycle_value = response.get("lifecycle")
        lifecycle: Literal["loading", "ready", "inactive", "failed"] = (
            lifecycle_value
            if lifecycle_value in {"loading", "ready", "inactive", "failed"}
            else "failed"
        )
        device = response.get("device")
        error_value = response.get("last_error")
        return WorkerHealth(
            version=str(response.get("version") or "unknown"),
            xpu_ready=response.get("xpu_ready") is True,
            model_ready=response.get("model_ready") is True,
            device=device if isinstance(device, str) else None,
            last_error=(
                _parse_error_code(error_value) if isinstance(error_value, str) else None
            ),
            lifecycle=lifecycle,
        )

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
    ) -> PreloadTiming:
        deadline = self._monotonic() + self._startup_timeout
        last = first_failure
        while self._monotonic() < deadline:
            try:
                return self._parse_preload_response(self._round_trip(request))
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
            engine: Literal["nano", "sensevoice"] = (
                "sensevoice" if response.get("engine") == "sensevoice" else "nano"
            )
            timing_value = response.get("timing_ms")
            timing_payload = (
                timing_value if isinstance(timing_value, Mapping) else {}
            )
            return Transcription(
                text=text if isinstance(text, str) else "",
                segments=segments,
                request_id=request_id,
                engine=engine,
                timing=AsrStageTiming(
                    audio_load_ms=_optional_duration(timing_payload, "audio_load_ms"),
                    vad_ms=_optional_duration(timing_payload, "vad_ms"),
                    generate_ms=_optional_duration(timing_payload, "generate_ms"),
                ),
                worker_elapsed_ms=_optional_duration(response, "elapsed_ms"),
            )
        code = _parse_error_code(response.get("error_code"))
        detail = str(response.get("error_message") or "")
        if code.code == "empty_speech":
            raise EmptySpeechError()
        raise WorkerError(code, detail)

    @staticmethod
    def _parse_preload_response(response: Mapping[str, Any]) -> PreloadTiming:
        if response.get("status") == "ok" and response.get("model_ready") is True:
            status = response.get("warmup_status")
            warmup_status: Literal["not_requested", "ready", "failed"] = (
                status
                if status in {"not_requested", "ready", "failed"}
                else "not_requested"
            )
            return PreloadTiming(
                worker_elapsed_ms=_optional_duration(response, "elapsed_ms"),
                runtime_load_ms=_optional_duration(response, "runtime_load_ms"),
                warmup_ms=_optional_duration(response, "warmup_ms"),
                warmup_status=warmup_status,
            )
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
        stop_primary: Callable[[], object],
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

    key: SessionKey
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
        fallback_worker: WorkerClient | None = None,
        corrector: TextCorrector | None = None,
        xpu_lease: XpuLease | None = None,
        metrics: MetricsLedger | None = None,
        nano_preloader: Callable[[], PreloadTiming | None] | None = None,
        scheduler: XpuScheduler | None = None,
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
        self._fallback_worker = fallback_worker
        self._corrector = corrector
        self._xpu_lease = xpu_lease
        self._scheduler = scheduler if scheduler is not None else XpuScheduler(
            start_profile=self._start_asr_for_scheduler,
            stop_profile=self._release_asr_for_scheduler,
            health_profile=self._asr_lifecycle_after_release,
        )
        self._metrics = metrics if metrics is not None else MetricsLedger()
        self._metric_sequence: int | None = None
        self._nano_preloader = nano_preloader
        self._preload_lock = threading.Lock()
        self._preload_cancel: threading.Event | None = None
        self._preload_handle: TaskHandle | None = None
        self._auto_stop_event = auto_stop_event
        self._capture_config = (
            capture_config if capture_config is not None else CaptureConfig()
        )
        self._monotonic = monotonic
        self._sleep = sleep
        self._state = DaemonState.IDLE
        self._session: _Session | None = None
        self._session_generation = 0
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

            self._session_generation += 1
            key = SessionKey(
                session_id=uuid.uuid4().hex, generation=self._session_generation
            )
            self._session = _Session(
                key=key, snapshot=snapshot, token=token, fcitx=fcitx
            )
            self._scheduler.activate(key)
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
        session = self._session
        if preloader is None or session is None:
            return
        cancelled = threading.Event()
        self._preload_cancel = cancelled
        self._metrics.record(sequence, nano_preload="scheduled")
        self._preload_handle = self._scheduler.run_asr(
            session.key,
            "nano",
            lambda: self._preload_nano(sequence, preloader, cancelled),
            kind=ModelTaskKind.STABLE_SEGMENT,
        )

    def _preload_nano(
        self,
        sequence: int,
        preloader: Callable[[], PreloadTiming | None],
        cancelled: threading.Event,
    ) -> None:
        with self._preload_lock:
            if cancelled.is_set():
                return
            started = self._monotonic()
            try:
                timing = preloader()
            except Exception as exc:  # noqa: BLE001 - recording stays unaffected
                logger.info("nano preload unavailable: %s", type(exc).__name__)
                self._metrics.record(
                    sequence,
                    preload_ms=_elapsed_ms(started, self._monotonic()),
                    nano_preload="failed",
                )
                return
            updates: dict[str, object] = {
                "preload_ms": _elapsed_ms(started, self._monotonic()),
                "nano_preload": "ready",
            }
            if timing is not None:
                if timing.worker_elapsed_ms is not None:
                    updates["preload_worker_ms"] = timing.worker_elapsed_ms
                if timing.runtime_load_ms is not None:
                    updates["preload_runtime_load_ms"] = timing.runtime_load_ms
                if timing.warmup_ms is not None:
                    updates["preload_warmup_ms"] = timing.warmup_ms
                updates["nano_warmup"] = timing.warmup_status
            self._metrics.record(sequence, **updates)

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
            if self._fallback_worker is not None:
                with contextlib.suppress(Exception):
                    self._fallback_worker.close()
            if self._corrector is not None:
                with contextlib.suppress(Exception):
                    self._corrector.close()
            self._scheduler.close()
        finally:
            self._lock.release()

    # --- Pipeline ------------------------------------------------------------

    def _transcribe_and_commit(self) -> None:
        self._state = DaemonState.TRANSCRIBING
        logger.info("state -> transcribing")
        if self._preload_cancel is not None:
            self._preload_cancel.set()
        if self._preload_handle is not None:
            self._preload_handle.cancel()
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
                transcription = self._transcribe_on_scheduler(artifact)
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
            asr_elapsed_ms = _elapsed_ms(asr_started, self._monotonic())
            asr_updates: dict[str, object] = {
                "asr_ms": asr_elapsed_ms,
                "asr_profile": transcription.engine,
            }
            if transcription.worker_elapsed_ms is not None:
                asr_updates["asr_worker_ms"] = transcription.worker_elapsed_ms
                asr_updates["asr_queue_transport_ms"] = max(
                    0, asr_elapsed_ms - transcription.worker_elapsed_ms
                )
            timing = transcription.timing
            if timing.audio_load_ms is not None:
                asr_updates["asr_audio_load_ms"] = timing.audio_load_ms
            if timing.vad_ms is not None:
                asr_updates["asr_vad_ms"] = timing.vad_ms
            if timing.generate_ms is not None:
                asr_updates["asr_generate_ms"] = timing.generate_ms
            self._record_metric(**asr_updates)

            text = transcription.text
            if not text:
                self._record_metric(error_code="empty_speech")
                self._notify(NOTIFY_EMPTY_SPEECH)
                return

            corrector = self._corrector
            if corrector is not None:
                text = self._correct_after_asr(corrector, text, transcription.engine)

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

    def _correct_after_asr(
        self,
        corrector: TextCorrector,
        raw_text: str,
        profile: Literal["nano", "sensevoice"],
    ) -> str:
        """Run Qwen on the single scheduler after an ASR-release confirmation."""
        with self._preload_lock:
            session = self._session
            if session is None:
                self._record_metric(correction="skipped_lease")
                return raw_text
            correction_started = self._monotonic()
            try:
                task = self._scheduler.run_correction(
                    session.key, profile, lambda: corrector.correct(raw_text)
                )
                if not task.wait(timeout=WORKER_RESPONSE_TIMEOUT_SECONDS):
                    task.cancel()
                    self._record_metric(correction="skipped_lease")
                    return raw_text
                outcome = task.result()
                if not isinstance(outcome, CorrectionOutcome) or not outcome.permitted:
                    self._record_metric(correction="skipped_lease")
                    return raw_text
                candidate = outcome.value if isinstance(outcome.value, str) else ""
                release_ms = getattr(self._xpu_lease, "last_release_ms", None)
                if (
                    isinstance(release_ms, int)
                    and not isinstance(release_ms, bool)
                    and release_ms >= 0
                ):
                    self._record_metric(asr_release_ms=release_ms)
                if profile == "nano":
                    self._record_metric(nano_was_stopped_for_qwen=True)
                self._record_correction_timing(getattr(corrector, "last_timing", None))
                self._record_metric(
                    correction_ms=_elapsed_ms(correction_started, self._monotonic()),
                    correction="corrected" if candidate else "raw_fallback",
                )
                return candidate or raw_text
            except CorrectionError as exc:
                # Raw ASR remains usable; do not show a noisy secondary alert.
                logger.warning("correction unavailable: %s", exc.code)
                self._record_correction_timing(exc.timing)
                self._record_metric(correction_rejection=exc.reason)
            except Exception as exc:  # noqa: BLE001 - raw text is resilient
                logger.warning("correction unavailable: %s", type(exc).__name__)
                self._record_metric(correction_rejection="internal")
            self._record_metric(
                correction_ms=_elapsed_ms(correction_started, self._monotonic()),
                correction="failed",
            )
            return raw_text

    def _transcribe_on_scheduler(self, artifact: CaptureArtifact) -> Transcription:
        """Execute the final ASR request on the single model dispatcher."""
        session = self._session
        if session is None:
            raise WorkerError(ErrorCode("worker", "unavailable"))
        try:
            return self._run_asr_profile(
                session.key, "nano", lambda: self._worker.transcribe(artifact)
            )
        except WorkerError as exc:
            fallback = self._fallback_worker
            if fallback is None or exc.code.code not in {"model_load", "oom"}:
                raise
            return self._run_asr_profile(
                session.key,
                "sensevoice",
                lambda: fallback.transcribe(artifact),
            )

    def _run_asr_profile(
        self,
        key: SessionKey,
        profile: Literal["nano", "sensevoice"],
        fn: Callable[[], Transcription],
    ) -> Transcription:
        """Execute one profile through the scheduler and preserve worker errors."""
        task = self._scheduler.run_asr(key, profile, fn)
        if not task.wait(timeout=WORKER_RESPONSE_TIMEOUT_SECONDS):
            task.cancel()
            raise WorkerError(ErrorCode("worker", "timeout"))
        try:
            result = task.result()
        except (EmptySpeechError, WorkerError):
            raise
        except Exception as exc:  # scheduler/profile errors stay a worker failure
            raise WorkerError(ErrorCode("worker", "unavailable")) from exc
        if not isinstance(result, Transcription):
            raise WorkerError(ErrorCode("worker", "internal"))
        return result

    def _release_asr_for_scheduler(
        self, profile: Literal["nano", "sensevoice"]
    ) -> bool:
        lease = self._xpu_lease
        if lease is None:
            return False
        try:
            return lease.release_asr_for_qwen(profile)
        except Exception:  # noqa: BLE001 - deny XPU release uncertainty
            return False

    @staticmethod
    def _start_asr_for_scheduler(profile: Literal["nano", "sensevoice"]) -> bool:
        """Permit a scheduled socket client to start its selected profile.

        ``main()`` always injects :class:`SystemdModelProfileSupervisor`, so
        production process control remains explicit.  This compatibility path
        deliberately performs no lifecycle side effect during construction or
        tests; a legacy auto-starting socket client can only start from the
        scheduler callback that follows it.
        """
        del profile
        return True

    def _asr_lifecycle_after_release(
        self, _profile: Literal["nano", "sensevoice"]
    ) -> ModelLifecycle:
        """The existing lease is affirmative only after systemd saw inactive."""
        return ModelLifecycle.INACTIVE

    def _record_correction_timing(self, timing: object) -> None:
        """Retain only bounded Qwen duration fields from a local child result."""
        if not isinstance(timing, CorrectionTiming):
            return
        updates: dict[str, object] = {}
        if timing.model_load_ms is not None:
            updates["correction_model_load_ms"] = timing.model_load_ms
        if timing.generate_ms is not None:
            updates["correction_generate_ms"] = timing.generate_ms
        if timing.validate_ms is not None:
            updates["correction_validate_ms"] = timing.validate_ms
        if updates:
            self._record_metric(**updates)

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
        auto_start_service=False,
    )
    profile_workers: dict[AsrProfile, HealthWorkerClient] = {"nano": nano_worker}
    fallback_worker: WorkerClient | None = None
    if cfg.inference.allow_sensevoice_fallback:
        sensevoice_worker = SocketWorkerClient(
            paths.runtime_dir / "worker-sensevoice.sock",
            start_service=functools.partial(
                default_start_worker_service, "sensevoice"
            ),
            auto_start_service=False,
        )
        fallback_worker = sensevoice_worker
        profile_workers["sensevoice"] = sensevoice_worker
    supervisor = SystemdModelProfileSupervisor(workers=profile_workers)
    scheduler = XpuScheduler(
        start_profile=supervisor.start_profile,
        stop_profile=supervisor.stop_profile,
        health_profile=supervisor.health_profile,
    )
    corrector: TextCorrector | None = None
    if cfg.enhanced.enabled:
        corrector = OnDemandQwenCorrector(inference=cfg.enhanced)
    daemon = VoiceDaemon(
        guard=guard,
        recorder=recorder,
        fcitx_factory=build_fcitx_factory(cfg),
        clipboard=ClipboardMirror(),
        injector=build_injector(cfg, guard),
        notifier=notifier,
        worker=nano_worker,
        fallback_worker=fallback_worker,
        corrector=corrector,
        nano_preloader=nano_worker.preload,
        scheduler=scheduler,
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
