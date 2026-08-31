"""Protocol/transport tests for the worker Unix-socket server.

These exercise the real ``WorkerServer`` (socketserver) over an actual AF_UNIX
socket with a fake runtime, plus SO_PEERCRED same-uid gating and the 64 KiB
single-line JSON framing.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
from collections.abc import Iterator

import pytest

from fun_voice import worker as worker_mod
from fun_voice.contracts import (
    MAX_MESSAGE_BYTES,
    Segment,
    Transcription,
    WorkerHealth,
    decode_message,
    encode_message,
)
from fun_voice.worker import Worker, WorkerServer, peer_uid


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
    assert "audio" not in response
    assert "text" not in response


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
