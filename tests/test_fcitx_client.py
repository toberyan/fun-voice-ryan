"""Tests for the Fcitx5 addon client (``fun_voice.fcitx``)."""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from fun_voice.config import FCITX_SOCKET_NAME
from fun_voice.contracts import (
    FCITX_CHUNK_MAX_BYTES,
    MAX_MESSAGE_BYTES,
    parse_commit_frame,
)
from fun_voice.fcitx import FcitxClient, FcitxCommitError, default_socket_path

_LENGTH = struct.Struct(">I")


class FakeAddon:
    """A tiny in-process addon speaking the client's length-prefixed protocol."""

    def __init__(
        self, path: Path, responder: Callable[[bytes], bytes | None]
    ) -> None:
        self.path = path
        self.frames: list[bytes] = []
        self._responder = responder
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
        data = b""
        while len(data) < size:
            part = sock.recv(size - len(data))
            if not part:
                return None
            data += part
        return data

    def _serve(self) -> None:
        self._server.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(5.0)
                while not self._stop.is_set():
                    header = self._recv_exact(conn, _LENGTH.size)
                    if header is None:
                        break
                    (length,) = _LENGTH.unpack(header)
                    payload = self._recv_exact(conn, length)
                    if payload is None:
                        break
                    self.frames.append(payload)
                    reply = self._responder(payload)
                    if reply is None:
                        break
                    conn.sendall(_LENGTH.pack(len(reply)) + reply)

    def close(self) -> None:
        self._stop.set()
        self._server.close()
        self._thread.join(timeout=2.0)


class RawServer:
    """Accept one connection and hand raw socket control to ``handler``."""

    def __init__(
        self, path: Path, handler: Callable[[socket.socket], None]
    ) -> None:
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self._thread = threading.Thread(
            target=self._serve, args=(handler,), daemon=True
        )
        self._thread.start()

    def _serve(self, handler: Callable[[socket.socket], None]) -> None:
        conn, _ = self._server.accept()
        with conn:
            handler(conn)

    def close(self) -> None:
        self._server.close()
        self._thread.join(timeout=2.0)


@pytest.fixture
def make_addon(tmp_path: Path):
    created: list[FakeAddon] = []

    def make(responder: Callable[[bytes], bytes | None]) -> FakeAddon:
        addon = FakeAddon(tmp_path / "fcitx.sock", responder)
        created.append(addon)
        return addon

    yield make
    for addon in created:
        addon.close()


def test_ping_pong(make_addon) -> None:
    fake = make_addon(lambda payload: b"PONG" if payload == b"PING" else b"ERROR")
    client = FcitxClient(fake.path)
    try:
        assert client.ping() is True
    finally:
        client.close()
    assert fake.frames == [b"PING"]


def test_start_focus_returns_token(make_addon) -> None:
    fake = make_addon(
        lambda payload: b"FOCUS 0123456789abcdef0123456789abcdef"
        if payload == b"START_FOCUS"
        else b"ERROR unsupported"
    )
    client = FcitxClient(fake.path)
    try:
        assert client.start_focus() == "0123456789abcdef0123456789abcdef"
    finally:
        client.close()


def test_start_focus_no_input_context(make_addon) -> None:
    fake = make_addon(lambda payload: b"REJECT no-input-context")
    client = FcitxClient(fake.path)
    try:
        assert client.start_focus() is None
    finally:
        client.close()


def test_commit_splits_long_text_into_ordered_chunks(make_addon) -> None:
    text = "你" * 30000  # 90,000 bytes > 64 KiB, forces multi-chunk
    fake = make_addon(lambda payload: b"OK")
    client = FcitxClient(fake.path)
    try:
        result = client.commit("tok-123", text)
    finally:
        client.close()
    assert result.committed is True
    assert len(fake.frames) > 1

    parsed = [parse_commit_frame(frame) for frame in fake.frames]
    assert [frame.sequence for frame in parsed] == list(range(1, len(parsed) + 1))
    assert all(frame.total == len(parsed) for frame in parsed)
    assert all(frame.focus_token == "tok-123" for frame in parsed)
    assert "".join(frame.text for frame in parsed) == text
    assert all(
        len(frame.text.encode("utf-8")) <= FCITX_CHUNK_MAX_BYTES for frame in parsed
    )


def test_commit_single_chunk(make_addon) -> None:
    fake = make_addon(lambda payload: b"OK")
    client = FcitxClient(fake.path)
    try:
        result = client.commit("tok", "你好 world")
    finally:
        client.close()
    assert result.committed is True
    assert fake.frames == [b"COMMIT tok 1 1\n" + "你好 world".encode()]


def test_commit_stops_on_reject(make_addon) -> None:
    calls: list[bytes] = []

    def responder(payload: bytes) -> bytes:
        calls.append(payload)
        return b"REJECT stale-focus"

    fake = make_addon(responder)
    client = FcitxClient(fake.path)
    try:
        result = client.commit("tok", "你" * 30000)  # multi-chunk text
    finally:
        client.close()

    assert result.committed is False
    assert result.error is not None
    assert result.error.category == "fcitx"
    assert result.error.code == "stale-focus"
    assert len(calls) == 1  # stopped after the first reject


def test_commit_never_retries_to_another_input_box(make_addon) -> None:
    def responder(payload: bytes) -> bytes:
        if payload == b"START_FOCUS":
            return b"FOCUS new-token"  # must never be requested
        return b"REJECT stale-focus"

    fake = make_addon(responder)
    client = FcitxClient(fake.path)
    try:
        result = client.commit("old-token", "你好")
    finally:
        client.close()

    assert result.committed is False
    assert fake.frames and all(f.startswith(b"COMMIT ") for f in fake.frames)
    assert b"START_FOCUS" not in fake.frames


def test_commit_empty_text_sends_nothing(make_addon) -> None:
    fake = make_addon(lambda payload: b"OK")
    client = FcitxClient(fake.path)
    try:
        result = client.commit("tok", "")
    finally:
        client.close()
    assert result.committed is True
    assert fake.frames == []


def test_default_socket_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert default_socket_path() == tmp_path / FCITX_SOCKET_NAME


def test_default_socket_path_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    with pytest.raises(FcitxCommitError):
        default_socket_path()


def test_connect_failure_raises(tmp_path: Path) -> None:
    client = FcitxClient(tmp_path / "nonexistent.sock", timeout=0.1)
    with pytest.raises(FcitxCommitError):
        client.ping()


def test_request_reconnects_after_disconnect(make_addon) -> None:
    state = {"attempts": 0}

    def responder(payload: bytes) -> bytes | None:
        state["attempts"] += 1
        if state["attempts"] == 1:
            return None  # drop the connection without a reply
        return b"PONG"

    fake = make_addon(responder)
    client = FcitxClient(fake.path)
    try:
        with pytest.raises(FcitxCommitError):
            client.ping()  # first attempt: connection dropped mid-request
        assert client.ping() is True  # next request opens a fresh connection
    finally:
        client.close()


def test_oversized_reply_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sock"

    def handler(conn: socket.socket) -> None:
        conn.recv(1024)  # consume the request
        conn.sendall(_LENGTH.pack(MAX_MESSAGE_BYTES + 1))  # bogus length, no body

    server = RawServer(path, handler)
    client = FcitxClient(path, timeout=1.0)
    try:
        with pytest.raises(FcitxCommitError):
            client.ping()
    finally:
        client.close()
        server.close()


def test_recv_timeout_raises(tmp_path: Path) -> None:
    path = tmp_path / "sock"

    def handler(conn: socket.socket) -> None:
        conn.recv(1024)  # consume the request, then stay silent
        time.sleep(0.5)

    server = RawServer(path, handler)
    client = FcitxClient(path, timeout=0.1)
    try:
        with pytest.raises(FcitxCommitError):
            client.ping()
    finally:
        client.close()
        server.close()


def test_malformed_reply_raises(tmp_path: Path) -> None:
    path = tmp_path / "sock"

    def handler(conn: socket.socket) -> None:
        conn.recv(1024)  # consume the request
        reply = b"NOT-A-VALID-REPLY"
        conn.sendall(_LENGTH.pack(len(reply)) + reply)

    server = RawServer(path, handler)
    client = FcitxClient(path, timeout=1.0)
    try:
        with pytest.raises(FcitxCommitError):
            client.commit("tok", "你好")
    finally:
        client.close()
        server.close()
