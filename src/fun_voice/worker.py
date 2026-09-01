"""Unix-socket ASR worker: warm Fun-ASR-Nano on XPU, VAD + transcribe per request.

The worker binds ``$XDG_RUNTIME_DIR/fun-voice-ryan/worker.sock`` (directory
``0700``, socket ``0600``) and only serves clients with the same uid (verified
per connection via ``SO_PEERCRED``). It speaks the single-line UTF-8 JSON
protocol from :mod:`fun_voice.contracts` (64 KiB max):

    daemon -> worker  {"id": "uuid", "op": "transcribe",
                       "audio": "path", "sample_rate": 16000}
    worker -> daemon  {"id": "uuid", "status": "ok"|"error", "text": ...,
                       "segments": [...], "elapsed_ms": ..., "error_code": ...}
    daemon -> worker  {"op": "health"}
    worker -> daemon  {"id": ..., "status": "ok", "version": ...,
                       "model_ready": ..., "xpu_ready": ..., "device": ...,
                       "last_error": ...}

Privacy: responses and logs never carry audio paths or transcription text.
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
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from fun_voice import config
from fun_voice.contracts import (
    MAX_MESSAGE_BYTES,
    WORKER_RESPONSE_MAX_BYTES,
    ErrorCode,
    MessageTooLarge,
    ProtocolError,
    Transcription,
    WorkerHealth,
    decode_message,
    encode_message,
)
from fun_voice.nano_runtime import (
    DEFAULT_TIMEOUT_SECONDS,
    DEVICE,
    VERSION,
    NanoRuntime,
    NanoRuntimeError,
    load_nano_runtime,
)

logger = logging.getLogger(__name__)

ERR_INTERNAL = ErrorCode("worker", "internal")
ERR_PROTOCOL = ErrorCode("worker", "protocol")
SOCKET_BACKLOG = 4
DEFAULT_TIMEOUT_MS = int(DEFAULT_TIMEOUT_SECONDS * 1000)


class Transcriber(Protocol):
    """The runtime seam the worker depends on (``NanoRuntime`` implements it)."""

    def transcribe(
        self, audio: str, *, sample_rate: int = 16000, timeout: float | None = None
    ) -> Transcription: ...

    def health(self) -> WorkerHealth: ...

    def close(self) -> None: ...


# --- Response builders ------------------------------------------------------


def _error_response(
    request_id: Any, code: ErrorCode, message: str, elapsed_ms: int = 0
) -> dict[str, Any]:
    return {
        "id": request_id,
        "status": "error",
        "text": "",
        "segments": [],
        "elapsed_ms": elapsed_ms,
        "error_code": str(code),
        "error_message": message,
    }


def _ok_response(
    request_id: Any, transcription: Transcription, elapsed_ms: int
) -> dict[str, Any]:
    return {
        "id": request_id,
        "status": "ok",
        "text": transcription.text,
        "segments": [
            {
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
            }
            for segment in transcription.segments
        ],
        "elapsed_ms": elapsed_ms,
        "error_code": None,
    }


def _error_code_of(exc: BaseException) -> ErrorCode:
    code = getattr(exc, "error_code", None)
    return code if isinstance(code, ErrorCode) else ERR_INTERNAL


# --- Dispatch ---------------------------------------------------------------


class Worker:
    """Dispatches protocol messages to the runtime (pure; no sockets)."""

    def __init__(self, runtime: Transcriber, *, version: str = VERSION) -> None:
        self.runtime = runtime
        self.version = version

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any]:
        op = message.get("op")
        if op == "transcribe":
            return self._transcribe(message)
        if op == "health":
            return self._health(message)
        return _error_response(message.get("id"), ERR_PROTOCOL, f"unknown op: {op!r}")

    def _transcribe(self, message: Mapping[str, Any]) -> dict[str, Any]:
        request_id = message.get("id")
        if not isinstance(request_id, str) or not request_id:
            return _error_response(None, ERR_PROTOCOL, "missing or invalid id")
        audio = message.get("audio")
        if not isinstance(audio, str) or not audio:
            return _error_response(
                request_id, ERR_PROTOCOL, "missing or invalid audio path"
            )
        sample_rate = message.get("sample_rate", 16000)
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            return _error_response(request_id, ERR_PROTOCOL, "invalid sample_rate")
        timeout_ms = message.get("timeout_ms")
        timeout: float | None = None
        if timeout_ms is not None:
            if (
                isinstance(timeout_ms, bool)
                or not isinstance(timeout_ms, (int, float))
                or timeout_ms <= 0
            ):
                return _error_response(request_id, ERR_PROTOCOL, "invalid timeout_ms")
            timeout = float(timeout_ms) / 1000.0

        started = time.perf_counter()
        try:
            transcription = self.runtime.transcribe(
                audio, sample_rate=sample_rate, timeout=timeout
            )
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            code = _error_code_of(exc)
            detail = (
                str(exc) if isinstance(exc, NanoRuntimeError) else type(exc).__name__
            )
            logger.warning("transcribe failed: %s (%s)", code, type(exc).__name__)
            return _error_response(request_id, code, detail, elapsed)
        elapsed = int((time.perf_counter() - started) * 1000)
        return _ok_response(request_id, transcription, elapsed)

    def _health(self, message: Mapping[str, Any]) -> dict[str, Any]:
        health = self.runtime.health()
        return {
            "id": message.get("id"),
            "status": "ok",
            "version": self.version,
            "model_ready": health.model_ready,
            "xpu_ready": health.xpu_ready,
            "device": health.device,
            "last_error": str(health.last_error) if health.last_error else None,
        }

    def close(self) -> None:
        self.runtime.close()


# --- Socket server ----------------------------------------------------------


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


class WorkerServer(socketserver.UnixStreamServer):
    """Single-threaded Unix socket server; rejects non-owner clients on connect."""

    def __init__(
        self, socket_path: Path, worker: Worker, *, uid: int | None = None
    ) -> None:
        self.worker = worker
        self.uid = os.getuid() if uid is None else uid
        self.socket_path = socket_path
        self.allow_reuse_address = True
        self.request_queue_size = SOCKET_BACKLOG
        super().__init__(str(socket_path), WorkerRequestHandler)
        os.chmod(socket_path, config.SOCKET_MODE)

    def client_allowed(self, conn: socket.socket) -> bool:
        return peer_uid(conn) == self.uid


class WorkerRequestHandler(socketserver.StreamRequestHandler):
    """Handle one JSON-line request per connection."""

    def handle(self) -> None:
        server = cast(WorkerServer, self.server)
        conn = self.connection
        if not server.client_allowed(conn):
            logger.warning("rejecting connection from non-owner uid")
            return
        try:
            line = _read_line(conn)
        except MessageTooLarge as exc:
            _send(conn, _error_response(None, ERR_PROTOCOL, str(exc)))
            return
        except OSError:
            return
        if not line:
            return
        try:
            message = decode_message(line)
        except ProtocolError as exc:
            _send(conn, _error_response(None, ERR_PROTOCOL, str(exc)))
            return
        response = server.worker.handle(message)
        _send(conn, response)


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


def _send(conn: socket.socket, response: dict[str, Any]) -> None:
    try:
        payload = encode_message(response, max_bytes=WORKER_RESPONSE_MAX_BYTES) + b"\n"
    except (ProtocolError, ValueError):
        payload = encode_message(
            _error_response(None, ERR_PROTOCOL, "response too large")
        ) + b"\n"
    conn.sendall(payload)


# --- Socket lifecycle -------------------------------------------------------


class _ShutdownRequested(Exception):
    pass


def _unlink_socket(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("could not remove socket %s: %s", path, exc)


def prepare_runtime_dir(path: Path) -> None:
    """Create the private runtime dir and force its mode to ``0700``.

    ``mkdir(..., exist_ok=True)`` leaves an existing directory's mode untouched,
    so the mode is re-applied explicitly: the directory may have been created
    earlier with a looser umask, and it must never be world/group accessible.
    """
    path.mkdir(mode=config.DIRECTORY_MODE, parents=True, exist_ok=True)
    os.chmod(path, config.DIRECTORY_MODE)

def serve(socket_path: Path, worker: Worker, *, uid: int | None = None) -> int:
    _unlink_socket(socket_path)
    server = WorkerServer(socket_path, worker, uid=uid)

    def _stop(signum: int, frame: object) -> None:
        raise _ShutdownRequested(signum)

    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):  # not in the main thread
            previous[signum] = signal.signal(signum, _stop)
    try:
        logger.info("worker listening on %s", socket_path)
        server.serve_forever()
    except _ShutdownRequested:
        logger.info("shutdown requested")
    finally:
        server.server_close()
        worker.close()
        _unlink_socket(socket_path)
        for saved, handler in previous.items():
            signal.signal(saved, handler)
    return 0


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fun-voice-worker",
        description="Warm Fun-ASR-Nano XPU worker over a private Unix socket.",
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
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

    runtime_dir = paths.runtime_dir
    prepare_runtime_dir(runtime_dir)

    logger.info("loading Nano runtime (this may take a few minutes) ...")
    try:
        runtime: NanoRuntime = load_nano_runtime(
            device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            default_timeout=args.timeout_ms / 1000.0,
        )
    except Exception as exc:
        logger.error("failed to load Nano runtime: %s", type(exc).__name__)
        return 1

    worker = Worker(runtime)
    return serve(paths.worker_socket, worker)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
