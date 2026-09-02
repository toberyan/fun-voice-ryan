"""Protocol/transport tests for the worker Unix-socket server.

These exercise the real ``WorkerServer`` (socketserver) over an actual AF_UNIX
socket with a fake runtime, plus SO_PEERCRED same-uid gating and the 64 KiB
single-line JSON framing.
"""

from __future__ import annotations

import array
import contextlib
import os
import socket
import threading
from collections.abc import Iterator

import pytest

from fun_voice import worker as worker_mod
from fun_voice.config import Config, InferenceConfig, RuntimePaths
from fun_voice.contracts import (
    MAX_MESSAGE_BYTES,
    WORKER_RESPONSE_MAX_BYTES,
    ProtocolError,
    Segment,
    Transcription,
    WorkerHealth,
    decode_message,
    encode_message,
)
from fun_voice.worker import LazyTranscriber, Worker, WorkerServer, peer_uid


class FakeRuntime:
    def __init__(self) -> None:
        self.transcription = Transcription(
            text="你好", segments=(Segment(0, 100, "你好"),)
        )
        self.health_result = WorkerHealth(
            version="test", xpu_ready=True, model_ready=True, device="xpu:0"
        )
        self.closed = False

    def transcribe(
        self, audio: str, *, sample_rate: int = 16000, timeout: float | None = None
    ) -> Transcription:
        return self.transcription

    def health(self) -> WorkerHealth:
        return self.health_result

    def close(self) -> None:
        self.closed = True

    def detect_vad_fd(
        self, fd: int, *, sample_rate: int
    ) -> tuple[tuple[int, int], ...]:
        assert fd >= 0
        assert sample_rate == 16000
        return ((0, 100),)


class FlakyRuntime(FakeRuntime):
    """Raises on the first transcribe, then succeeds."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def transcribe(
        self, audio: str, *, sample_rate: int = 16000, timeout: float | None = None
    ) -> Transcription:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("out of memory")
        return self.transcription


class WarmableRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.warmup_calls = 0

    def warmup(self) -> int:
        self.warmup_calls += 1
        return 7


@pytest.fixture
def server(tmp_path) -> Iterator[tuple[WorkerServer, FakeRuntime, str]]:
    runtime = FakeRuntime()
    socket_path = tmp_path / "worker.sock"
    server = WorkerServer(socket_path, Worker(runtime), uid=os.getuid())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, runtime, str(socket_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(socket_path: str, message: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(socket_path)
        client.sendall(encode_message(message) + b"\n")
        data = _read_response(client)
    return decode_message(data.rstrip(b"\n"))


def _read_response(client: socket.socket) -> bytes:
    data = bytearray()
    while b"\n" not in data:
        chunk = client.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


# --- Peer credential gating -------------------------------------------------


def test_peer_uid_matches_current_user() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert peer_uid(right) == os.getuid()
    finally:
        left.close()
        right.close()


def test_server_rejects_non_owner_uid(monkeypatch, tmp_path) -> None:
    socket_path = tmp_path / "reject.sock"
    server = WorkerServer(socket_path, Worker(FakeRuntime()), uid=os.getuid())
    monkeypatch.setattr(worker_mod, "peer_uid", lambda conn: os.getuid() + 1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(socket_path))
            with contextlib.suppress(OSError):  # server already closed
                client.sendall(b'{"op":"health"}\n')
            try:
                response = client.recv(4096)
            except OSError:
                response = b""
            assert response == b""  # closed without a response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- Socket permissions -----------------------------------------------------


def test_socket_is_created_mode_0600(server) -> None:
    _, _, socket_path = server
    mode = os.stat(socket_path).st_mode & 0o777
    assert mode == 0o600


def test_prepare_runtime_dir_forces_0700(tmp_path) -> None:
    runtime_dir = tmp_path / "rt"
    runtime_dir.mkdir(mode=0o775)  # pre-existing with a looser mode
    worker_mod.prepare_runtime_dir(runtime_dir)
    assert (runtime_dir.stat().st_mode & 0o777) == 0o700


# --- Health endpoint --------------------------------------------------------


def test_health_over_socket(server) -> None:
    _, _, socket_path = server
    response = _request(socket_path, {"op": "health"})
    assert response["status"] == "ok"
    assert response["model_ready"] is True
    assert response["xpu_ready"] is True
    assert response["device"] == "xpu:0"
    assert response["lifecycle"] == "ready"
    assert "audio" not in response
    assert "text" not in response


def test_live_vad_socket_request_transfers_exactly_one_fd(server) -> None:
    _worker_server, _runtime, socket_path = server
    request = encode_message(
        {
            "id": "live",
            "op": "detect_vad",
            "sample_rate": 16000,
            "session_id": "opaque-session",
            "generation": 1,
            "source_start_ms": 0,
            "source_end_ms": 100,
        }
    ) + b"\n"
    fd = os.open("/dev/null", os.O_RDONLY)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.connect(socket_path)
            sent = conn.sendmsg(
                [request],
                [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        array.array("i", [fd]),
                    )
                ],
            )
            assert sent == len(request)
            response = decode_message(conn.recv(4096).rstrip(b"\n"))
    finally:
        os.close(fd)

    assert response == {
        "id": "live",
        "status": "ok",
        "ranges": [{"start_ms": 0, "end_ms": 100}],
        "error_code": None,
    }


def test_live_ancillary_payload_with_partial_descriptor_is_protocol_error() -> None:
    class MalformedAncillarySocket:
        def recvmsg(self, _size: int, _ancbuf: int):
            return (
                b'{"op":"health"}\n',
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, b"\x01")],
                0,
                None,
            )

    with pytest.raises(ProtocolError, match="malformed ancillary data"):
        worker_mod._read_line_with_fds(MalformedAncillarySocket())


def test_lazy_transcriber_loads_only_for_first_transcription() -> None:
    loaded: list[FakeRuntime] = []

    def load() -> FakeRuntime:
        runtime = FakeRuntime()
        loaded.append(runtime)
        return runtime

    runtime = LazyTranscriber(load, device="xpu:0")
    assert runtime.health().model_ready is False
    assert loaded == []

    first = runtime.transcribe("/tmp/a.wav")
    second = runtime.transcribe("/tmp/b.wav")

    assert first.text == "你好"
    assert second.text == "你好"
    assert len(loaded) == 1


def test_preload_constructs_lazy_runtime_once_then_transcribe_reuses_it() -> None:
    loaded: list[FakeRuntime] = []

    def load() -> FakeRuntime:
        runtime = FakeRuntime()
        loaded.append(runtime)
        return runtime

    worker = Worker(LazyTranscriber(load, device="xpu:0"))

    preload = worker.handle({"id": "p", "op": "preload"})
    transcribe = worker.handle(
        {"id": "t", "op": "transcribe", "audio": "/tmp/a", "sample_rate": 16000}
    )

    assert preload["status"] == "ok"
    assert preload["model_ready"] is True
    assert transcribe["status"] == "ok"
    assert len(loaded) == 1


def test_preload_response_exposes_only_duration_stages() -> None:
    loaded: list[WarmableRuntime] = []

    def load() -> WarmableRuntime:
        runtime = WarmableRuntime()
        loaded.append(runtime)
        return runtime

    response = Worker(LazyTranscriber(load, device="xpu:0")).handle(
        {"id": "p", "op": "preload"}
    )

    assert response["status"] == "ok"
    assert response["warmup_status"] == "ready"
    assert response["warmup_ms"] == 7
    assert isinstance(response["elapsed_ms"], int)
    assert isinstance(response["runtime_load_ms"], int)
    assert loaded[0].warmup_calls == 1
    assert "audio" not in repr(response)
    assert "你好" not in repr(response)


def test_lazy_transcriber_maps_load_failure_to_stable_worker_error() -> None:
    def load() -> FakeRuntime:
        raise RuntimeError("broken checkpoint")

    response = Worker(LazyTranscriber(load, device="xpu:0")).handle(
        {"id": "u1", "op": "transcribe", "audio": "/tmp/a.wav"}
    )
    assert response["status"] == "error"
    assert response["error_code"] == "worker.model_load"


def test_worker_server_idle_monitor_stops_only_after_idle_timeout(tmp_path) -> None:
    socket_path = tmp_path / "idle.sock"
    server = WorkerServer(socket_path, Worker(FakeRuntime()), uid=os.getuid())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        server.start_idle_monitor(0.02)
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    finally:
        server.stop_idle_monitor()
        server.server_close()


# --- Transcribe round trip --------------------------------------------------


def test_transcribe_round_trip(server) -> None:
    _, _, socket_path = server
    request = {
        "id": "u1", "op": "transcribe", "audio": "/tmp/a.wav", "sample_rate": 16000,
    }
    response = _request(socket_path, request)
    assert response["status"] == "ok"
    assert response["id"] == "u1"
    assert response["text"] == "你好"
    assert response["segments"] == [{"start_ms": 0, "end_ms": 100, "text": "你好"}]


def test_long_response_over_socket(server) -> None:
    """A worker response above 64 KiB must survive the wire intact (high-3)."""
    _, runtime, socket_path = server
    long_text = "长" * 40_000  # 120,000 UTF-8 bytes > 64 KiB, < 4 MiB
    runtime.transcription = Transcription(
        text=long_text, segments=(Segment(0, 100, long_text),)
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(socket_path)
        client.sendall(
            encode_message(
                {
                    "id": "u1",
                    "op": "transcribe",
                    "audio": "/tmp/a.wav",
                    "sample_rate": 16000,
                }
            )
            + b"\n"
        )
        data = _read_response(client)
    response = decode_message(data.rstrip(b"\n"), max_bytes=WORKER_RESPONSE_MAX_BYTES)
    assert response["status"] == "ok"
    assert response["text"] == long_text
    assert response["segments"] == [{"start_ms": 0, "end_ms": 100, "text": long_text}]
    assert response["error_code"] is None


def test_server_keeps_listening_after_error(tmp_path) -> None:
    socket_path = tmp_path / "flaky.sock"
    server = WorkerServer(socket_path, Worker(FlakyRuntime()), uid=os.getuid())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first = _request(
            str(socket_path),
            {
                "id": "u1",
                "op": "transcribe",
                "audio": "/tmp/a.wav",
                "sample_rate": 16000,
            },
        )
        assert first["status"] == "error"
        # The same server must still answer after a failed request.
        health = _request(str(socket_path), {"op": "health"})
        assert health["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_worker_main_uses_toml_inference_config(monkeypatch, tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    paths = RuntimePaths(
        runtime_dir=runtime_dir,
        worker_socket=runtime_dir / "worker.sock",
        daemon_socket=runtime_dir / "daemon.sock",
        fcitx_socket=tmp_path / "fcitx.sock",
    )
    cfg = Config(
        inference=InferenceConfig(
            device="xpu:0",
            dtype="bf16",
            gpu_memory_utilization=0.2,
            enforce_eager=False,
        )
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(worker_mod.config, "load_config", lambda: cfg)
    monkeypatch.setattr(worker_mod.config, "resolve_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(worker_mod.config, "build_runtime_paths", lambda _path: paths)
    monkeypatch.setattr(
        worker_mod,
        "load_nano_runtime",
        lambda **kwargs: captured.update(kwargs) or FakeRuntime(),
    )

    def fake_serve(_path, worker, **_kwargs):
        assert worker.handle({"op": "health"})["model_ready"] is False
        assert worker.handle(
            {"id": "u1", "op": "transcribe", "audio": "/tmp/a.wav"}
        )["status"] == "ok"
        return 0

    monkeypatch.setattr(worker_mod, "serve", fake_serve)

    assert worker_mod.main([]) == 0
    assert captured["device"] == "xpu:0"
    assert captured["dtype"] == "bf16"
    assert captured["gpu_memory_utilization"] == 0.2
    assert captured["max_model_len"] == 1536
    assert captured["enforce_eager"] is False


def test_worker_main_rejects_cpu_cli_override(monkeypatch, tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    paths = RuntimePaths(
        runtime_dir=runtime_dir,
        worker_socket=runtime_dir / "worker.sock",
        daemon_socket=runtime_dir / "daemon.sock",
        fcitx_socket=tmp_path / "fcitx.sock",
    )
    called = False

    def _load_runtime(**_kwargs: object) -> FakeRuntime:
        nonlocal called
        called = True
        return FakeRuntime()

    monkeypatch.setattr(worker_mod.config, "load_config", Config)
    monkeypatch.setattr(worker_mod.config, "resolve_runtime_dir", lambda: runtime_dir)
    monkeypatch.setattr(worker_mod.config, "build_runtime_paths", lambda _path: paths)
    monkeypatch.setattr(worker_mod, "load_nano_runtime", _load_runtime)
    monkeypatch.setattr(worker_mod, "serve", lambda _path, _worker: 0)

    assert worker_mod.main(["--device", "cpu"]) == 1
    assert called is False


# --- Framing / protocol errors ----------------------------------------------


def test_oversized_message_returns_protocol_error(server) -> None:
    _, _, socket_path = server
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(socket_path)
        client.sendall(b" " * (MAX_MESSAGE_BYTES + 100) + b"\n")
        data = _read_response(client)
    response = decode_message(data.rstrip(b"\n"))
    assert response["status"] == "error"
    assert response["error_code"] == "worker.protocol"


def test_invalid_json_returns_protocol_error(server) -> None:
    _, _, socket_path = server
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(socket_path)
        client.sendall(b"not json\n")
        data = _read_response(client)
    response = decode_message(data.rstrip(b"\n"))
    assert response["status"] == "error"
    assert response["error_code"] == "worker.protocol"


def test_unknown_op_over_socket(server) -> None:
    _, _, socket_path = server
    response = _request(socket_path, {"op": "nope"})
    assert response["status"] == "error"
    assert response["error_code"] == "worker.protocol"
