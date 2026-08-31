"""End-to-end fake tests: daemon pipeline, bridge, socket server, worker client.

Every external boundary is a fake, but the tests exercise the real wiring: the
``VoiceDaemon`` state machine driven through its public surface, the real
``DaemonServer`` over an actual AF_UNIX socket, the real ``SocketWorkerClient``
against an in-process worker socket, and the bridge's ``send_request`` against
an in-process daemon socket.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fun_voice import daemon as daemon_mod
from fun_voice.bridge import send_request
from fun_voice.capture import CaptureError
from fun_voice.contracts import (
    FCITX_CHUNK_MAX_BYTES,
    CaptureArtifact,
    CommitResult,
    DaemonState,
    ErrorCode,
    FocusSnapshot,
    Transcription,
    split_utf8,
)
from fun_voice.daemon import (
    NOTIFY_CLIPBOARD_FAILED,
    NOTIFY_FOCUS_CHANGED,
    NOTIFY_RECOGNITION_FAILED,
    DaemonServer,
    EmptySpeechError,
    SocketWorkerClient,
    VoiceDaemon,
    WorkerError,
    peer_uid,
)
from fun_voice.desktop import ClipboardError, X11Error, XTestError
from fun_voice.fcitx import FcitxCommitError

ARTIFACT = CaptureArtifact(
    audio="/proc/self/fd/3", sample_rate=16000, channels=1, format="s16le",
    duration_ms=1000,
)

SNAPSHOT = FocusSnapshot(
    active_window=1, process_name="app", input_focus=2, monotonic_ns=0, window_pid=9
)
CHANGED = FocusSnapshot(
    active_window=99, process_name="other", input_focus=100, monotonic_ns=0,
    window_pid=88,
)


# --- Fakes -------------------------------------------------------------------


class FakeGuard:
    def __init__(
        self,
        snapshots: list[FocusSnapshot] | None = None,
        *,
        error: X11Error | None = None,
        c_down: bool = True,
    ) -> None:
        self._snapshots = list(snapshots) if snapshots else [SNAPSHOT]
        self._error = error
        self.c_down = c_down
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

    def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> CaptureArtifact:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        return self.artifact

    def cancel(self) -> None:
        pass

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class FakeFcitx:
    def __init__(
        self,
        *,
        token: str | None = "tok-123",
        commit_result: CommitResult | None = None,
        commit_error: FcitxCommitError | None = None,
    ) -> None:
        self.token = token
        self.commit_result = commit_result
        self.commit_error = commit_error
        self.commits: list[tuple[str, str]] = []
        self.closed = False

    def start_focus(self) -> str | None:
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


class _TextWorker:
    def __init__(self, text: str) -> None:
        self.text = text
        self.transcriptions: list[CaptureArtifact] = []

    def transcribe(self, artifact: CaptureArtifact) -> Transcription:
        self.transcriptions.append(artifact)
        return Transcription(text=self.text, segments=())

    def close(self) -> None:
        pass


class FlakyWorker:
    """Raises once (OOM), then succeeds — the next session must still work."""

    def __init__(self, text: str = "你好") -> None:
        self.text = text
        self.calls = 0
        self.closed = False

    def transcribe(self, artifact: CaptureArtifact) -> Transcription:
        self.calls += 1
        if self.calls == 1:
            raise WorkerError(ErrorCode("worker", "oom"), "out of memory")
        return Transcription(text=self.text, segments=())

    def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(
        self,
        *,
        guard: FakeGuard | None = None,
        recorder: FakeRecorder | None = None,
        fcitx: FakeFcitx | None = None,
        clipboard: FakeClipboard | None = None,
        injector: FakeInjector | None = None,
        worker: Any = None,
    ) -> None:
        self.guard = guard if guard is not None else FakeGuard()
        self.recorder = recorder if recorder is not None else FakeRecorder()
        self.clipboard = clipboard if clipboard is not None else FakeClipboard()
        self.injector = injector if injector is not None else FakeInjector()
        self.notifier = FakeNotifier()
        self.worker = worker if worker is not None else _TextWorker("你好")
        self._fcitx_template = fcitx if fcitx is not None else FakeFcitx()
        self.fcitx_instances: list[FakeFcitx] = []

        def factory() -> FakeFcitx:
            instance = FakeFcitx(
                token=self._fcitx_template.token,
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
        )

    @property
    def fcitx(self) -> FakeFcitx:
        assert self.fcitx_instances, "no Fcitx instance was created"
        return self.fcitx_instances[-1]


def _record(h: Harness, text: str) -> None:
    h.daemon._worker = _TextWorker(text)  # noqa: SLF001 - test wiring
    assert h.daemon.start_if_idle() == "started"
    h.daemon.stop()


# --- End-to-end scenarios ----------------------------------------------------


def test_normal_fcitx_commit() -> None:
    h = Harness()
    _record(h, "你好世界")
    assert h.daemon.state is DaemonState.IDLE
    assert h.clipboard.writes == ["你好世界"]
    assert h.injector.pastes == 0
    assert h.fcitx.commits == [("tok-123", "你好世界")]
    assert h.recorder.cleanup_calls == 1


def test_focus_change_writes_only_clipboard() -> None:
    h = Harness(guard=FakeGuard(snapshots=[SNAPSHOT, CHANGED]))
    _record(h, "你好")
    assert h.clipboard.writes == ["你好"]
    assert h.fcitx.commits == []
    assert h.injector.pastes == 0
    assert NOTIFY_FOCUS_CHANGED in h.notifier.messages


def test_fcitx_channel_failure_falls_back_to_xtest() -> None:
    h = Harness(fcitx=FakeFcitx(commit_error=FcitxCommitError("socket gone")))
    _record(h, "你好")
    assert h.injector.pastes == 1
    assert h.clipboard.writes == ["你好"]


def test_fcitx_stale_focus_reject_does_not_fall_back() -> None:
    h = Harness(
        fcitx=FakeFcitx(
            commit_result=CommitResult(
                committed=False, method="fcitx", error=ErrorCode("fcitx", "stale-focus")
            )
        )
    )
    _record(h, "你好")
    assert h.injector.pastes == 0
    assert h.fcitx.commits == [("tok-123", "你好")]
    assert (
        NOTIFY_RECOGNITION_FAILED.format(category="fcitx.stale-focus")
        in h.notifier.messages
    )


def test_clipboard_failure_does_not_undo_fcitx_success() -> None:
    h = Harness(clipboard=FakeClipboard(error=ClipboardError("xclip missing")))
    _record(h, "你好")
    assert h.fcitx.commits == [("tok-123", "你好")]
    assert h.injector.pastes == 0
    assert NOTIFY_CLIPBOARD_FAILED in h.notifier.messages


def test_worker_oom_then_next_session_succeeds() -> None:
    h = Harness(worker=FlakyWorker())
    assert h.daemon.start_if_idle() == "started"
    h.daemon.stop()
    assert h.daemon.state is DaemonState.IDLE
    assert (
        NOTIFY_RECOGNITION_FAILED.format(category="worker.oom")
        in h.notifier.messages
    )
    # A fresh session succeeds and injects.
    assert h.daemon.start_if_idle() == "started"
    h.daemon.stop()
    assert h.daemon.state is DaemonState.IDLE
    assert h.clipboard.writes == ["你好"]
    assert h.injector.pastes == 0
    assert h.fcitx.commits == [("tok-123", "你好")]


def test_long_text_reject_never_injects_partial_text() -> None:
    long_text = "你" * 30000
    h = Harness(
        fcitx=FakeFcitx(
            commit_result=CommitResult(
                committed=False, method="fcitx", error=ErrorCode("fcitx", "stale-focus")
            )
        )
    )
    _record(h, long_text)
    assert h.fcitx.commits == [("tok-123", long_text)]
    assert h.injector.pastes == 0
    assert h.daemon.state is DaemonState.IDLE


class ChunkingFakeFcitx:
    """Models the real FcitxClient chunking: split on UTF-8 boundaries, reject
    at a caller-chosen chunk, and record only the chunks sent before it."""

    def __init__(self, reject_at: int = 3) -> None:
        self.reject_at = reject_at
        self.sent_chunks: list[str] = []
        self.commits: list[tuple[str, str]] = []

    def start_focus(self) -> str:
        return "tok-123"

    def commit(self, focus_token: str, text: str) -> CommitResult:
        self.commits.append((focus_token, text))
        chunks = split_utf8(text, FCITX_CHUNK_MAX_BYTES)
        for index, chunk in enumerate(chunks, start=1):
            if index >= self.reject_at:
                return CommitResult(
                    committed=False,
                    method="fcitx",
                    error=ErrorCode("fcitx", "stale-focus"),
                )
            self.sent_chunks.append(chunk)
        return CommitResult(committed=True, method="fcitx")

    def close(self) -> None:
        pass


def test_long_text_reject_sends_only_prefix_chunks() -> None:
    long_text = "你" * 30000  # > 64 KiB → multiple 8 KiB chunks
    fcitx = ChunkingFakeFcitx(reject_at=3)
    guard = FakeGuard()
    recorder = FakeRecorder()
    notifier = FakeNotifier()
    injector = FakeInjector()
    daemon = VoiceDaemon(
        guard=guard,
        recorder=recorder,
        fcitx_factory=lambda: fcitx,
        clipboard=FakeClipboard(),
        injector=injector,
        notifier=notifier,
        worker=_TextWorker(long_text),
    )
    assert daemon.start_if_idle() == "started"
    daemon.stop()
    assert daemon.state is DaemonState.IDLE
    # Rejected at chunk 3 → exactly the first two chunks were sent.
    assert len(fcitx.sent_chunks) == 2
    assert "".join(fcitx.sent_chunks) != long_text
    # The daemon never injects partial text via XTEST or a retry.
    assert injector.pastes == 0
    assert fcitx.commits == [("tok-123", long_text)]
    assert (
        NOTIFY_RECOGNITION_FAILED.format(category="fcitx.stale-focus")
        in notifier.messages
    )

def test_no_token_records_and_uses_xtest_only() -> None:
    h = Harness(fcitx=FakeFcitx(token=None))
    _record(h, "你好")
    assert h.fcitx.commits == []
    assert h.injector.pastes == 1
    assert h.clipboard.writes == ["你好"]


# --- Bridge ------------------------------------------------------------------


class _BridgeGuard:
    def __init__(self, c_down: bool, error: X11Error | None = None) -> None:
        self.c_down = c_down
        self.error = error

    def c_is_down(self) -> bool:
        if self.error is not None:
            raise self.error
        return self.c_down


class _LineServer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[bytes] = []
        self._received = threading.Event()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(2.0)
                data = bytearray()
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data.extend(chunk)
                if data:
                    self.lines.append(bytes(data).rstrip(b"\n"))
                    self._received.set()

    def wait_for_line(self, timeout: float = 2.0) -> None:
        self._received.wait(timeout)

    def close(self) -> None:
        self._server.close()
        self._thread.join(timeout=2.0)


def test_bridge_c_down_sends_start_if_idle(tmp_path: Path) -> None:
    server = _LineServer(tmp_path / "daemon.sock")
    try:
        assert send_request(_BridgeGuard(c_down=True), server.path) == 0
        server.wait_for_line()
        assert server.lines == [b'{"op":"start_if_idle"}']
    finally:
        server.close()


def test_bridge_c_up_sends_stop(tmp_path: Path) -> None:
    server = _LineServer(tmp_path / "daemon.sock")
    try:
        assert send_request(_BridgeGuard(c_down=False), server.path) == 0
        server.wait_for_line()
        assert server.lines == [b'{"op":"stop"}']
    finally:
        server.close()


def test_bridge_connection_failure_exits_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.sock"
    assert send_request(_BridgeGuard(c_down=True), missing) == 1


def test_bridge_x11_failure_exits_nonzero(tmp_path: Path) -> None:
    server = _LineServer(tmp_path / "daemon.sock")
    try:
        assert (
            send_request(_BridgeGuard(c_down=True, error=X11Error("no X")), server.path)
            == 1
        )
        assert server.lines == []
    finally:
        server.close()


# --- Daemon socket server ----------------------------------------------------


@pytest.fixture
def running_server(tmp_path: Path):
    h = Harness()
    path = tmp_path / "daemon.sock"
    server = DaemonServer(path, h.daemon, uid=os.getuid())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, h, path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _send(path: Path, message: dict[str, str]) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        client.sendall(json.dumps(message).encode("utf-8") + b"\n")


def _wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.time() + 2.0
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)


def test_daemon_socket_start_then_stop(running_server) -> None:
    _server, h, path = running_server
    _send(path, {"op": "start_if_idle"})
    _wait_for(lambda: h.daemon.state is DaemonState.RECORDING)
    assert h.daemon.state is DaemonState.RECORDING

    _send(path, {"op": "stop"})
    _wait_for(lambda: h.daemon.state is DaemonState.IDLE)
    assert h.daemon.state is DaemonState.IDLE


def test_daemon_socket_mode_0600(running_server) -> None:
    _server, _h, path = running_server
    assert (os.stat(path).st_mode & 0o777) == 0o600


def test_daemon_rejects_non_owner(monkeypatch, tmp_path: Path) -> None:
    h = Harness()
    path = tmp_path / "daemon.sock"
    server = DaemonServer(path, h.daemon, uid=os.getuid())
    monkeypatch.setattr(daemon_mod, "peer_uid", lambda conn: os.getuid() + 1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _send(path, {"op": "start_if_idle"})
        assert h.daemon.state is DaemonState.IDLE  # request was ignored
    finally:
        server.shutdown()
        server.server_close()


def _request(path: Path, message: dict[str, str], timeout: float = 2.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(json.dumps(message).encode("utf-8") + b"\n")
        data = bytearray()
        while b"\n" not in data:
            chunk = client.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
    return json.loads(bytes(data).decode("utf-8"))


class BlockingWorker:
    """Blocks inside ``transcribe`` until released, to hold the daemon lock."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.transcriptions: list[CaptureArtifact] = []

    def transcribe(self, artifact: CaptureArtifact) -> Transcription:
        self.transcriptions.append(artifact)
        self.started.set()
        self.release.wait(timeout=10.0)
        return Transcription(text="你好", segments=())

    def close(self) -> None:
        pass


def test_start_if_idle_rejected_promptly_during_transcription(tmp_path: Path) -> None:
    worker = BlockingWorker()
    recorder = FakeRecorder()
    guard = FakeGuard()
    notifier = FakeNotifier()
    daemon = VoiceDaemon(
        guard=guard,
        recorder=recorder,
        fcitx_factory=lambda: FakeFcitx(),
        clipboard=FakeClipboard(),
        injector=FakeInjector(),
        notifier=notifier,
        worker=worker,
    )
    path = tmp_path / "daemon.sock"
    server = DaemonServer(path, daemon, uid=os.getuid())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _request(path, {"op": "start_if_idle"})["status"] == "started"
        # Stop blocks inside transcribe; fire-and-forget so we do not wait on it.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(path))
            client.sendall(b'{"op":"stop"}\n')
        worker.started.wait(timeout=2.0)
        # While transcription is blocked, a new start_if_idle is rejected now.
        started = time.monotonic()
        response = _request(path, {"op": "start_if_idle"}, timeout=2.0)
        assert response["status"] == "busy"
        assert time.monotonic() - started < 1.0
    finally:
        worker.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- Socket worker client ----------------------------------------------------


class _WorkerSocket:
    def __init__(self, path: Path, responder: Callable[[dict], dict]) -> None:
        self.requests: list[dict] = []
        self._responder = responder
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(5.0)
                data = bytearray()
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data.extend(chunk)
                if not data:
                    continue
                request = json.loads(bytes(data).decode("utf-8"))
                self.requests.append(request)
                response = self._responder(request)
                conn.sendall(json.dumps(response).encode("utf-8") + b"\n")

    def close(self) -> None:
        self._server.close()
        self._thread.join(timeout=2.0)


def _ok_responder(request: dict) -> dict:
    return {
        "id": request["id"],
        "status": "ok",
        "text": "你好",
        "segments": [{"start_ms": 0, "end_ms": 100, "text": "你好"}],
        "elapsed_ms": 12,
        "error_code": None,
    }


def _error_responder(error_code: str) -> Callable[[dict], dict]:
    def respond(request: dict) -> dict:
        return {
            "id": request["id"],
            "status": "error",
            "text": "",
            "segments": [],
            "elapsed_ms": 1,
            "error_code": error_code,
            "error_message": "boom",
        }

    return respond


def test_worker_client_round_trip(tmp_path: Path) -> None:
    server = _WorkerSocket(tmp_path / "worker.sock", _ok_responder)
    try:
        client = SocketWorkerClient(tmp_path / "worker.sock", timeout=2.0)
        result = client.transcribe(ARTIFACT)
        assert result.text == "你好"
        assert server.requests[0]["op"] == "transcribe"
        assert server.requests[0]["audio"] == ARTIFACT.audio
    finally:
        server.close()


def test_worker_client_maps_empty_speech(tmp_path: Path) -> None:
    server = _WorkerSocket(
        tmp_path / "worker.sock", _error_responder("worker.empty_speech")
    )
    try:
        client = SocketWorkerClient(tmp_path / "worker.sock", timeout=2.0)
        with pytest.raises(EmptySpeechError):
            client.transcribe(ARTIFACT)
    finally:
        server.close()


def test_worker_client_maps_oom_code(tmp_path: Path) -> None:
    server = _WorkerSocket(tmp_path / "worker.sock", _error_responder("worker.oom"))
    try:
        client = SocketWorkerClient(tmp_path / "worker.sock", timeout=2.0)
        with pytest.raises(WorkerError) as excinfo:
            client.transcribe(ARTIFACT)
        assert excinfo.value.code == ErrorCode("worker", "oom")
    finally:
        server.close()


def test_worker_client_starts_service_then_retries(tmp_path: Path) -> None:
    path = tmp_path / "worker.sock"
    starts: list[None] = []
    server_holder: dict[str, _WorkerSocket] = {}

    def start_service() -> None:
        starts.append(None)
        server_holder["server"] = _WorkerSocket(path, _ok_responder)

    client = SocketWorkerClient(path, timeout=2.0, start_service=start_service)
    try:
        result = client.transcribe(ARTIFACT)
        assert result.text == "你好"
        assert starts == [None]
    finally:
        if "server" in server_holder:
            server_holder["server"].close()


def test_worker_client_unavailable_after_retry(tmp_path: Path) -> None:
    path = tmp_path / "worker.sock"
    starts: list[None] = []
    client = SocketWorkerClient(
        path, timeout=0.2, start_service=lambda: starts.append(None)
    )
    with pytest.raises(WorkerError) as excinfo:
        client.transcribe(ARTIFACT)
    assert excinfo.value.code == ErrorCode("worker", "unavailable")
    assert starts == [None]


def test_peer_uid_matches_current_user() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert peer_uid(right) == os.getuid()
    finally:
        left.close()
        right.close()
