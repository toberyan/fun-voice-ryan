"""State-machine and pipeline unit tests for the voice daemon.

Every desktop/capture/fcitx/clipboard/injector/worker adapter is a fake, so no
X server, PipeWire, Fcitx or GPU is required. The tests drive the daemon's
public surface (``start_if_idle`` / ``stop`` / ``handle_auto_stop``) and assert
state transitions, notification copy, and that the single cleanup function runs
on every path.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from fun_voice import daemon as daemon_mod
from fun_voice.capture import CaptureConfig, CaptureError
from fun_voice.config import Config
from fun_voice.contracts import (
    CaptureArtifact,
    CommitResult,
    DaemonState,
    ErrorCode,
    FocusSnapshot,
    Transcription,
)
from fun_voice.corrector import CorrectionError
from fun_voice.daemon import (
    HOTKEY_UNAVAILABLE_EXIT,
    NOTIFY_EMPTY_SPEECH,
    NOTIFY_LIMIT_REACHED,
    NOTIFY_RECOGNITION_FAILED,
    NOTIFY_RECORDING,
    NOTIFY_TRANSCRIBING,
    EmptySpeechError,
    VoiceDaemon,
    WorkerError,
    _DisabledInjector,
    build_fcitx_factory,
    build_injector,
    serve,
)
from fun_voice.desktop import (
    ClipboardError,
    X11Error,
    X11FocusGuard,
    X11HotkeyUnavailable,
    XTestError,
    XTestInjector,
)
from fun_voice.fcitx import FcitxClient, FcitxCommitError

ARTIFACT = CaptureArtifact(
    audio="/proc/self/fd/3", sample_rate=16000, channels=1, format="s16le",
    duration_ms=1000,
)

SNAPSHOT = FocusSnapshot(
    active_window=1, process_name="app", input_focus=2, monotonic_ns=0, window_pid=9
)


# --- Fakes -------------------------------------------------------------------


class FakeGuard:
    def __init__(
        self,
        snapshots: list[FocusSnapshot] | None = None,
        *,
        error: X11Error | None = None,
        c_down: bool = True,
        c_error: X11Error | None = None,
    ) -> None:
        self._snapshots = list(snapshots) if snapshots else [SNAPSHOT]
        self._error = error
        self.c_down = c_down
        self.c_error = c_error
        self.captures = 0

    def capture(self) -> FocusSnapshot:
        self.captures += 1
        if self._error is not None:
            raise self._error
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]

    def is_same(self, a: FocusSnapshot, b: FocusSnapshot) -> bool:
        return (
            a.active_window == b.active_window
            and a.process_name == b.process_name
            and a.input_focus == b.input_focus
            and a.window_pid == b.window_pid
        )

    def c_is_down(self) -> bool:
        if self.c_error is not None:
            raise self.c_error
        return self.c_down


class FakeRecorder:
    def __init__(
        self,
        artifact: CaptureArtifact = ARTIFACT,
        *,
        start_error: CaptureError | None = None,
        stop_error: CaptureError | None = None,
    ) -> None:
        self.artifact = artifact
        self.start_error = start_error
        self.stop_error = stop_error
        self.start_calls = 0
        self.stop_calls = 0
        self.cleanup_calls = 0
        self.cancel_calls = 0
        self.start_configs: list[CaptureConfig] = []

    def start(self, config: CaptureConfig | None = None) -> None:
        self.start_calls += 1
        self.start_configs.append(config if config is not None else CaptureConfig())
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> CaptureArtifact:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        return self.artifact

    def cancel(self) -> None:
        self.cancel_calls += 1

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class FakeFcitx:
    def __init__(
        self,
        *,
        token: str | None = "tok-123",
        start_error: FcitxCommitError | None = None,
        commit_result: CommitResult | None = None,
        commit_error: FcitxCommitError | None = None,
    ) -> None:
        self.token = token
        self.start_error = start_error
        self.commit_result = commit_result
        self.commit_error = commit_error
        self.commits: list[tuple[str, str]] = []
        self.closed = False

    def start_focus(self) -> str | None:
        if self.start_error is not None:
            raise self.start_error
        return self.token

    def commit(self, focus_token: str, text: str) -> CommitResult:
        self.commits.append((focus_token, text))
        if self.commit_error is not None:
            raise self.commit_error
        if self.commit_result is not None:
            return self.commit_result
        return CommitResult(committed=True, method="fcitx")

    def close(self) -> None:
        self.closed = True


class FakeClipboard:
    def __init__(self, error: ClipboardError | None = None) -> None:
        self.error = error
        self.writes: list[str] = []

    def write_utf8(self, text: str) -> None:
        if self.error is not None:
            raise self.error
        self.writes.append(text)


class FakeInjector:
    def __init__(self, error: XTestError | None = None) -> None:
        self.error = error
        self.pastes = 0

    def paste_ctrl_v(self) -> None:
        self.pastes += 1
        if self.error is not None:
            raise self.error


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def notify(self, message: str) -> None:
        self.messages.append(message)


class FakeWorker:
    def __init__(
        self,
        text: str = "你好",
        *,
        error: BaseException | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.transcriptions: list[CaptureArtifact] = []
        self.closed = False

    def transcribe(self, artifact: CaptureArtifact) -> Transcription:
        self.transcriptions.append(artifact)
        if self.error is not None:
            raise self.error
        return Transcription(text=self.text, segments=())

    def close(self) -> None:
        self.closed = True


class FakeCorrector:
    def __init__(
        self, *, text: str | None = None, error: Exception | None = None
    ) -> None:
        self.text = text
        self.error = error
        self.calls: list[str] = []

    def correct(self, raw_text: str) -> str:
        self.calls.append(raw_text)
        if self.error is not None:
            raise self.error
        assert self.text is not None
        return self.text

    def close(self) -> None:
        pass


class Harness:
    def __init__(
        self,
        *,
        guard: FakeGuard | None = None,
        recorder: FakeRecorder | None = None,
        fcitx: FakeFcitx | None = None,
        clipboard: FakeClipboard | None = None,
        injector: FakeInjector | None = None,
        worker: FakeWorker | None = None,
        corrector: FakeCorrector | None = None,
        nano_preloader: Callable[[], None] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        capture_config: CaptureConfig | None = None,
    ) -> None:
        self.guard = guard if guard is not None else FakeGuard()
        self.recorder = recorder if recorder is not None else FakeRecorder()
        self.clipboard = clipboard if clipboard is not None else FakeClipboard()
        self.injector = injector if injector is not None else FakeInjector()
        self.notifier = FakeNotifier()
        self.worker = worker if worker is not None else FakeWorker()
        self.corrector = corrector
        self.fcitx_instances: list[FakeFcitx] = []
        self._fcitx_template = fcitx if fcitx is not None else FakeFcitx()

        def factory() -> FakeFcitx:
            instance = FakeFcitx(
                token=self._fcitx_template.token,
                start_error=self._fcitx_template.start_error,
                commit_result=self._fcitx_template.commit_result,
                commit_error=self._fcitx_template.commit_error,
            )
            self.fcitx_instances.append(instance)
            return instance

        self.daemon = VoiceDaemon(
            guard=self.guard,
            recorder=self.recorder,
            fcitx_factory=factory,
            clipboard=self.clipboard,
            injector=self.injector,
            notifier=self.notifier,
            worker=self.worker,
            corrector=self.corrector,
            nano_preloader=nano_preloader,
            monotonic=monotonic if monotonic is not None else time.monotonic,
            sleep=sleep if sleep is not None else time.sleep,
            capture_config=(
                capture_config if capture_config is not None else CaptureConfig()
            ),
        )

    @property
    def fcitx(self) -> FakeFcitx:
        assert self.fcitx_instances, "no Fcitx instance was created"
        return self.fcitx_instances[-1]


def _started(harness: Harness) -> None:
    assert harness.daemon.start_if_idle() == "started"


# --- State machine -----------------------------------------------------------


def test_initial_state_is_idle() -> None:
    assert Harness().daemon.state is DaemonState.IDLE


def test_stop_in_idle_is_a_noop() -> None:
    h = Harness()
    h.daemon.stop()
    assert h.daemon.state is DaemonState.IDLE
    assert h.recorder.stop_calls == 0
    assert h.worker.transcriptions == []


def test_start_in_idle_transitions_to_recording() -> None:
    h = Harness()
    assert h.daemon.start_if_idle() == "started"
    assert h.daemon.state is DaemonState.RECORDING
    assert h.recorder.start_calls == 1


def test_start_records_focus_snapshot_and_requests_token() -> None:
    h = Harness()
    h.daemon.start_if_idle()
    assert h.guard.captures == 1
    assert len(h.fcitx_instances) == 1


def test_start_notifies_recording() -> None:
    h = Harness()
    h.daemon.start_if_idle()
    assert NOTIFY_RECORDING in h.notifier.messages


def test_start_confirms_c_is_still_pressed() -> None:
    h = Harness(guard=FakeGuard(c_down=True))
    assert h.daemon.start_if_idle() == "started"
    assert h.daemon.state is DaemonState.RECORDING


def test_hotkey_press_starts_recording_and_keeps_private_boolean() -> None:
    h = Harness(guard=FakeGuard(c_down=True))
    assert h.daemon.diagnostics() == {
        "hotkey_registered": False,
        "hotkey_press_seen": False,
    }

    assert h.daemon.handle_hotkey_press() == "started"

    assert h.daemon.diagnostics() == {
        "hotkey_registered": False,
        "hotkey_press_seen": True,
    }


def test_mark_hotkey_registered_changes_only_registration_boolean() -> None:
    h = Harness()
    h.daemon.mark_hotkey_registered()
    assert h.daemon.diagnostics() == {
        "hotkey_registered": True,
        "hotkey_press_seen": False,
    }


def test_dispatch_metrics_returns_empty_aggregate_without_session_data() -> None:
    assert Harness().daemon.dispatch({"op": "metrics"}) == {"count": 0}


def test_completed_session_metrics_contain_only_aggregate_stage_data() -> None:
    h = Harness()
    _started(h)
    h.daemon.stop()

    report = h.daemon.dispatch({"op": "metrics"})

    assert report["count"] == 1
    assert report["capture_duration_ms"] == {"p50": 1000, "p95": 1000}
    assert "asr_ms" in report
    assert "commit_ms" in report
    assert "你好" not in repr(report)
    assert ARTIFACT.audio not in repr(report)


def test_daemon_requests_nano_preload_only_after_recording_starts() -> None:
    called = threading.Event()
    states: list[DaemonState] = []
    holder: dict[str, VoiceDaemon] = {}

    def preload() -> None:
        states.append(holder["daemon"].state)
        called.set()

    h = Harness(nano_preloader=preload)
    holder["daemon"] = h.daemon

    assert h.daemon.start_if_idle() == "started"
    assert called.wait(timeout=2.0)
    assert states == [DaemonState.RECORDING]


def test_start_cancels_when_c_not_pressed() -> None:
    values = iter([0.0, 1.0])
    h = Harness(
        guard=FakeGuard(c_down=False),
        monotonic=lambda: next(values),
        sleep=lambda _secs: None,
    )
    assert h.daemon.start_if_idle() == "cancelled"
    assert h.daemon.state is DaemonState.IDLE
    assert h.recorder.cancel_calls == 0  # recorder never started; cleanup suffices
    assert h.recorder.start_calls == 0
    assert h.notifier.messages == []  # cancel is silent


@pytest.mark.parametrize(
    "bad_state", [DaemonState.TRANSCRIBING, DaemonState.COMMITTING]
)
def test_stop_while_transcribing_or_committing_is_noop(bad_state: DaemonState) -> None:
    h = Harness()
    h.daemon._state = bad_state
    h.daemon.stop()
    assert h.recorder.stop_calls == 0
    assert h.worker.transcriptions == []


class BlockingStartFcitx:
    """Blocks inside ``start_focus`` so a test can hold the daemon lock."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def start_focus(self) -> str:
        self.entered.set()
        self.release.wait(timeout=5.0)
        return "tok-123"

    def commit(self, focus_token: str, text: str) -> CommitResult:
        return CommitResult(committed=True, method="fcitx")

    def close(self) -> None:
        pass


def test_stop_during_start_is_not_dropped() -> None:
    fcitx = BlockingStartFcitx()
    recorder = FakeRecorder()
    daemon = VoiceDaemon(
        guard=FakeGuard(),
        recorder=recorder,
        fcitx_factory=lambda: fcitx,
        clipboard=FakeClipboard(),
        injector=FakeInjector(),
        notifier=FakeNotifier(),
        worker=FakeWorker(text="你好"),
    )
    result: dict[str, str] = {}

    def do_start() -> None:
        result["status"] = daemon.start_if_idle()

    start_thread = threading.Thread(target=do_start)
    start_thread.start()
    assert fcitx.entered.wait(timeout=2.0)  # start is now holding the lock

    stop_thread = threading.Thread(target=daemon.stop)
    stop_thread.start()
    time.sleep(0.05)  # let stop block on the lock
    fcitx.release.set()  # release start; stop must then proceed

    start_thread.join(timeout=2.0)
    stop_thread.join(timeout=2.0)

    assert result["status"] == "started"
    assert recorder.stop_calls == 1  # the queued stop was not dropped
    assert daemon.state is DaemonState.IDLE


@pytest.mark.parametrize(
    "bad_state",
    [DaemonState.RECORDING, DaemonState.TRANSCRIBING, DaemonState.COMMITTING],
)
def test_start_if_idle_while_busy_is_rejected(bad_state: DaemonState) -> None:
    h = Harness()
    h.daemon._state = bad_state
    assert h.daemon.start_if_idle() == "busy"
    assert h.guard.captures == 0
    assert h.recorder.start_calls == 0


def test_repeated_start_while_recording_is_noop() -> None:
    h = Harness()
    _started(h)
    assert h.daemon.start_if_idle() == "busy"
    assert h.daemon.start_if_idle() == "busy"
    assert h.guard.captures == 1
    assert h.recorder.start_calls == 1


def test_start_x11_error_returns_error_and_stays_idle() -> None:
    h = Harness(guard=FakeGuard(error=X11Error("no X")))
    assert h.daemon.start_if_idle() == "error"
    assert h.daemon.state is DaemonState.IDLE
    assert h.recorder.start_calls == 0


def test_start_capture_error_cleans_up_and_returns_to_idle() -> None:
    h = Harness(recorder=FakeRecorder(start_error=CaptureError("boom")))
    assert h.daemon.start_if_idle() == "error"
    assert h.daemon.state is DaemonState.IDLE
    assert h.recorder.cleanup_calls == 1


@pytest.mark.parametrize("token_or_error", ["none", "error"])
def test_fcitx_token_unavailable_still_records(token_or_error: str) -> None:
    fcitx = FakeFcitx(token=None) if token_or_error == "none" else FakeFcitx(
        start_error=FcitxCommitError("down")
    )
    h = Harness(fcitx=fcitx)
    assert h.daemon.start_if_idle() == "started"
    assert h.daemon.state is DaemonState.RECORDING


def test_fcitx_factory_failure_still_records_and_uses_xtest() -> None:
    def boom() -> FakeFcitx:
        raise FcitxCommitError("cannot construct client")

    injector = FakeInjector()
    notifier = FakeNotifier()
    daemon = VoiceDaemon(
        guard=FakeGuard(),
        recorder=FakeRecorder(),
        fcitx_factory=boom,
        clipboard=FakeClipboard(),
        injector=injector,
        notifier=notifier,
        worker=FakeWorker(text="你好"),
    )
    assert daemon.start_if_idle() == "started"
    assert daemon.state is DaemonState.RECORDING
    daemon.stop()
    assert daemon.state is DaemonState.IDLE
    assert injector.pastes == 1  # no Fcitx → XTEST fallback
    assert notifier.messages


def test_stop_transitions_through_pipeline_to_idle() -> None:
    h = Harness()
    _started(h)
    h.daemon.stop()
    assert h.daemon.state is DaemonState.IDLE
    assert h.recorder.stop_calls == 1
    assert h.worker.transcriptions == [ARTIFACT]
    assert NOTIFY_TRANSCRIBING in h.notifier.messages


def test_corrected_text_is_committed_and_copied_instead_of_raw_text() -> None:
    corrector = FakeCorrector(text="今天执行 git commit，然后运行 pytest。")
    h = Harness(
        worker=FakeWorker(text="今天执行 get commit，然后运行 py test。"),
        corrector=corrector,
    )

    _started(h)
    h.daemon.stop()

    assert corrector.calls == ["今天执行 get commit，然后运行 py test。"]
    assert h.clipboard.writes == ["今天执行 git commit，然后运行 pytest。"]
    assert h.fcitx.commits == [("tok-123", "今天执行 git commit，然后运行 pytest。")]


def test_correction_error_keeps_raw_text_usable() -> None:
    h = Harness(
        worker=FakeWorker(text="get commit"),
        corrector=FakeCorrector(error=CorrectionError("correction.oom")),
    )

    _started(h)
    h.daemon.stop()

    assert h.clipboard.writes == ["get commit"]
    assert h.fcitx.commits == [("tok-123", "get commit")]


def test_unexpected_correction_error_also_keeps_raw_text_usable() -> None:
    h = Harness(
        worker=FakeWorker(text="get commit"),
        corrector=FakeCorrector(error=RuntimeError("unavailable")),
    )

    _started(h)
    h.daemon.stop()

    assert h.clipboard.writes == ["get commit"]
    assert h.fcitx.commits == [("tok-123", "get commit")]


# --- Every path returns to IDLE and cleans up --------------------------------


def _run_failing_stop(
    recorder: FakeRecorder | None = None,
    worker: FakeWorker | None = None,
) -> Harness:
    h = Harness(recorder=recorder, worker=worker)
    _started(h)
    h.daemon.stop()
    return h


def test_capture_error_returns_to_idle_and_notifies() -> None:
    h = _run_failing_stop(recorder=FakeRecorder(stop_error=CaptureError("no audio")))
    assert h.daemon.state is DaemonState.IDLE
    assert h.worker.transcriptions == []
    assert NOTIFY_RECOGNITION_FAILED.format(category="capture") in h.notifier.messages


def test_empty_speech_returns_to_idle_and_notifies() -> None:
    h = _run_failing_stop(worker=FakeWorker(error=EmptySpeechError()))
    assert h.daemon.state is DaemonState.IDLE
    assert NOTIFY_EMPTY_SPEECH in h.notifier.messages
    assert h.clipboard.writes == []


def test_worker_error_returns_to_idle_and_notifies_category() -> None:
    h = _run_failing_stop(
        worker=FakeWorker(
            error=WorkerError(ErrorCode("worker", "oom"), "out of memory")
        )
    )
    assert h.daemon.state is DaemonState.IDLE
    assert (
        NOTIFY_RECOGNITION_FAILED.format(category="worker.oom")
        in h.notifier.messages
    )
    assert h.clipboard.writes == []


def test_empty_text_treated_as_empty_speech() -> None:
    h = _run_failing_stop(worker=FakeWorker(text=""))
    assert h.daemon.state is DaemonState.IDLE
    assert NOTIFY_EMPTY_SPEECH in h.notifier.messages


@pytest.mark.parametrize(
    ("recorder", "worker"),
    [
        (FakeRecorder(stop_error=CaptureError("no audio")), None),
        (None, FakeWorker(error=EmptySpeechError())),
        (None, FakeWorker(error=WorkerError(ErrorCode("worker", "oom")))),
        (None, FakeWorker(text="")),
    ],
)
def test_cleanup_runs_on_every_failure_path(
    recorder: FakeRecorder | None, worker: FakeWorker | None
) -> None:
    h = _run_failing_stop(recorder=recorder, worker=worker)
    assert h.recorder.cleanup_calls == 1
    assert h.daemon.state is DaemonState.IDLE


def test_cleanup_runs_on_success_path() -> None:
    h = Harness()
    _started(h)
    h.daemon.stop()
    assert h.recorder.cleanup_calls == 1
    assert h.daemon.state is DaemonState.IDLE


def test_cleanup_closes_session_fcitx_client() -> None:
    h = Harness()
    _started(h)
    h.daemon.stop()
    assert h.fcitx.closed is True


# --- Auto-stop ---------------------------------------------------------------


def test_auto_stop_notifies_limit_and_transcribes() -> None:
    h = Harness()
    _started(h)
    h.daemon.handle_auto_stop()
    assert h.daemon.state is DaemonState.IDLE
    assert NOTIFY_LIMIT_REACHED in h.notifier.messages
    assert h.worker.transcriptions == [ARTIFACT]


def test_auto_stop_outside_recording_is_noop() -> None:
    h = Harness()
    h.daemon.handle_auto_stop()
    assert h.recorder.stop_calls == 0
    assert h.notifier.messages == []


# --- X11 hotkey release ------------------------------------------------------


def test_hotkey_release_stops_in_a_background_thread() -> None:
    h = Harness(guard=FakeGuard(c_down=True))
    assert h.daemon.handle_hotkey_press() == "started"

    h.daemon.handle_hotkey_release()

    deadline = time.monotonic() + 1.0
    while h.recorder.stop_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert h.recorder.stop_calls == 1
    assert h.daemon.state is DaemonState.IDLE
    assert NOTIFY_TRANSCRIBING in h.notifier.messages


class _FakeHotkeyListener:
    def __init__(self, *, unavailable: bool = False) -> None:
        self.unavailable = unavailable
        self.started = False
        self.closed = False

    def start(self) -> None:
        if self.unavailable:
            raise X11HotkeyUnavailable("Super+C is already grabbed")
        self.started = True

    def close(self) -> None:
        self.closed = True


class _StoppingDaemonServer:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.closed = False

    def serve_forever(self, *, poll_interval: float) -> None:
        raise daemon_mod._ShutdownRequested(signal.SIGTERM)  # noqa: SLF001

    def server_close(self) -> None:
        self.closed = True


def test_serve_starts_x11_hotkey_before_server_and_closes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = Harness()
    listener = _FakeHotkeyListener()
    monkeypatch.setattr(daemon_mod, "DaemonServer", _StoppingDaemonServer)

    assert serve(tmp_path / "daemon.sock", h.daemon, hotkey_listener=listener) == 0
    assert listener.started is True
    assert listener.closed is True
    assert h.daemon.diagnostics() == {
        "hotkey_registered": True,
        "hotkey_press_seen": False,
    }


def test_serve_exits_without_binding_when_x11_hotkey_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = Harness()
    listener = _FakeHotkeyListener(unavailable=True)

    def unexpected_server(*args: object, **kwargs: object) -> None:
        pytest.fail("server must not bind when the X11 hotkey grab fails")

    monkeypatch.setattr(daemon_mod, "DaemonServer", unexpected_server)

    assert (
        serve(tmp_path / "daemon.sock", h.daemon, hotkey_listener=listener)
        == HOTKEY_UNAVAILABLE_EXIT
    )
    assert listener.closed is False
    assert h.worker.closed is True


# --- Configuration wiring (fix 4: config.toml lands) -------------------------


def test_start_passes_configured_source_to_recorder() -> None:
    h = Harness(capture_config=CaptureConfig(source="alsa_input.custom"))
    _started(h)
    assert h.recorder.start_configs == [CaptureConfig(source="alsa_input.custom")]


def test_build_fcitx_factory_wires_timeout() -> None:
    factory = build_fcitx_factory(Config(fcitx_commit_timeout_ms=2500))
    assert factory.func is FcitxClient  # type: ignore[attr-defined]
    assert factory.keywords == {"timeout": 2.5}  # type: ignore[attr-defined]


def test_build_injector_enabled_uses_guard_display() -> None:
    class _StubDisplay:
        pass

    display = _StubDisplay()
    guard = X11FocusGuard(display=display, monotonic=lambda: 0)
    injector = build_injector(Config(allow_x11_paste_fallback=True), guard)
    assert isinstance(injector, XTestInjector)
    assert injector.display is display


def test_build_injector_disabled_uses_noop() -> None:
    guard = X11FocusGuard(display=object(), monotonic=lambda: 0)
    injector = build_injector(Config(allow_x11_paste_fallback=False), guard)
    assert isinstance(injector, _DisabledInjector)
    with pytest.raises(XTestError):
        injector.paste_ctrl_v()


def test_disabled_xtest_fallback_notifies_without_injecting() -> None:
    h = Harness(fcitx=FakeFcitx(token=None), injector=_DisabledInjector())
    _started(h)
    h.daemon.stop()
    assert h.clipboard.writes == ["你好"]  # still mirrored to clipboard
    assert NOTIFY_RECOGNITION_FAILED.format(category="injection") in h.notifier.messages
