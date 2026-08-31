"""Focus-safe client for the Fcitx5 commit addon.

The addon listens on ``$XDG_RUNTIME_DIR/fun-voice-ryan-fcitx.sock`` (mode
``0600``) and commits transcription text into the currently focused input
context, but only while the focus token issued at recording start is still
valid. This module speaks the addon's length-prefixed frame protocol and never
retries into a different input context.
"""

from __future__ import annotations

import socket
import struct
from pathlib import Path
from typing import Final

from fun_voice.config import FCITX_SOCKET_NAME, get_xdg_runtime_dir
from fun_voice.contracts import (
    MAX_MESSAGE_BYTES,
    CommitResult,
    ErrorCode,
    FcitxResponse,
    build_commit_frames,
    parse_fcitx_response,
)

_LENGTH: Final = struct.Struct(">I")

DEFAULT_TIMEOUT: Final = 1.0


class FcitxCommitError(RuntimeError):
    """The addon could not be reached, or its reply was malformed."""


def default_socket_path() -> Path:
    """Return the addon socket path, or raise when ``XDG_RUNTIME_DIR`` is unset."""
    runtime = get_xdg_runtime_dir()
    if not runtime:
        raise FcitxCommitError("XDG_RUNTIME_DIR is not set")
    return Path(runtime) / FCITX_SOCKET_NAME


class FcitxClient:
    """Single-connection client for the Fcitx5 addon protocol.

    Each request is framed as a 4-byte big-endian length followed by the
    payload, and each reply uses the same framing. ``commit`` splits text on
    Unicode boundaries (via :func:`fun_voice.contracts.build_commit_frames`)
    and stops sending further chunks as soon as one is rejected.
    """

    def __init__(
        self,
        socket_path: Path | str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._path = (
            Path(socket_path) if socket_path is not None else default_socket_path()
        )
        self._timeout = timeout
        self._sock: socket.socket | None = None

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        try:
            sock.connect(str(self._path))
        except OSError as exc:
            sock.close()
            raise FcitxCommitError(f"cannot connect to {self._path}: {exc}") from exc
        self._sock = sock
        return sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _request(self, payload: bytes) -> bytes:
        sock = self._connect()
        try:
            sock.sendall(_LENGTH.pack(len(payload)) + payload)
            header = self._recv_exact(sock, _LENGTH.size)
            (length,) = _LENGTH.unpack(header)
            if length > MAX_MESSAGE_BYTES:
                raise FcitxCommitError(f"addon reply too large: {length} bytes")
            return self._recv_exact(sock, length)
        except (OSError, struct.error) as exc:
            self.close()
            raise FcitxCommitError(f"communication failed: {exc}") from exc

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            part = sock.recv(size - len(chunks))
            if not part:
                raise FcitxCommitError("connection closed by addon")
            chunks.extend(part)
        return bytes(chunks)

    def ping(self) -> bool:
        """Return True when the addon answers ``PING`` with ``PONG``."""
        reply = self._request(b"PING")
        return reply.strip() == b"PONG"

    def start_focus(self) -> str | None:
        """Ask the addon for a focus token, or return ``None`` if no context
        has focus."""
        reply = self._request(b"START_FOCUS").decode("utf-8").strip()
        if reply.startswith("FOCUS "):
            token = reply[len("FOCUS ") :]
            if token:
                return token
        if reply == "REJECT no-input-context":
            return None
        raise FcitxCommitError(f"unexpected START_FOCUS reply: {reply!r}")

    def commit(self, focus_token: str, text: str) -> CommitResult:
        """Commit ``text`` into the focused context using ``focus_token``.

        Returns ``committed=True`` only if every chunk was acknowledged with
        ``OK``. The first reject or error stops the remaining chunks and the
        result reports ``committed=False``; no re-request of focus is made.
        """
        frames = build_commit_frames(focus_token, text)
        for frame in frames:
            response = parse_fcitx_response(self._request(frame))
            if response.status != "ok":
                return CommitResult(
                    committed=False,
                    method="fcitx",
                    error=self._commit_error(response),
                )
        return CommitResult(committed=True, method="fcitx")

    @staticmethod
    def _commit_error(response: FcitxResponse) -> ErrorCode:
        if response.status == "reject":
            return ErrorCode(category="fcitx", code=response.reason or "rejected")
        return ErrorCode(
            category="fcitx", code=f"error-{response.code or 'unknown'}"
        )
