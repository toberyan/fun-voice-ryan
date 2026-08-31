"""Stable contracts and wire-protocol codecs for Fun Voice Ryan.

This module contains no business logic. It defines the immutable data types,
the daemon state machine, structured error codes, and the byte-level framing
used between the bridge, daemon, worker, and the Fcitx addon.

Protocol summary:
- Bridge/daemon/worker messages are single-line UTF-8 JSON, at most 64 KiB.
- Fcitx uses a header line followed by a UTF-8 body, at most 64 KiB per frame.
- Fcitx COMMIT text is limited to 64 KiB per line; longer text is split on
  Unicode codepoint boundaries into ordered chunks of at most 8 KiB.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast

# --- Size limits ------------------------------------------------------------

MAX_MESSAGE_BYTES = 64 * 1024
# Maximum wire size (bytes) of a JSON message or a Fcitx frame.

FCITX_TEXT_LINE_MAX_BYTES = 64 * 1024
# Maximum UTF-8 byte length of a single Fcitx COMMIT text line.

FCITX_CHUNK_MAX_BYTES = 8 * 1024
# Maximum UTF-8 byte length of one chunk when splitting overlong text.


class ProtocolError(ValueError):
    """Raised when a peer violates the wire protocol."""


class MessageTooLarge(ProtocolError):
    """Raised when a message or frame exceeds the size limit."""


# --- State machine ----------------------------------------------------------

class DaemonState(Enum):
    """Top-level states of the voice daemon."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    COMMITTING = "committing"
    ERROR = "error"


# --- Structured error code --------------------------------------------------

@dataclass(frozen=True)
class ErrorCode:
    """Machine-readable error: a category plus a category-specific code."""

    category: str
    code: str

    def __str__(self) -> str:
        return f"{self.category}.{self.code}"


# --- Data types -------------------------------------------------------------

@dataclass(frozen=True)
class StartRequest:
    """Bridge -> daemon request to start recording if idle."""

    op: str = "start_if_idle"


@dataclass(frozen=True)
class StopRequest:
    """Bridge -> daemon request to stop recording."""

    op: str = "stop"


@dataclass(frozen=True)
class FocusSnapshot:
    """X11 focus state captured at recording start, compared before commit."""

    active_window: int | None
    process_name: str | None
    input_focus: int | None
    monotonic_ns: int
    focus_token: str | None = None


@dataclass(frozen=True)
class CaptureArtifact:
    """Handle to captured audio: a path or memfd plus its format."""

    audio: str
    sample_rate: int = 16000
    channels: int = 1
    format: str = "s16le"
    duration_ms: int | None = None


@dataclass(frozen=True)
class Segment:
    """A single transcribed speech segment with time range."""

    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Transcription:
    """Worker result: raw model text plus time-ordered segments."""

    text: str
    segments: tuple[Segment, ...] = ()
    request_id: str | None = None


@dataclass(frozen=True)
class CommitResult:
    """Outcome of committing text into the focused input context."""

    committed: bool
    method: str
    error: ErrorCode | None = None


@dataclass(frozen=True)
class WorkerHealth:
    """Worker health snapshot; never carries audio or transcription text."""

    version: str
    xpu_ready: bool
    model_ready: bool
    device: str | None = None
    last_error: ErrorCode | None = None


# --- JSON message codec -----------------------------------------------------

def encode_message(message: Mapping[str, Any]) -> bytes:
    """Encode a bridge/daemon/worker message as single-line UTF-8 JSON."""
    data = json.dumps(dict(message), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if b"\n" in data or b"\r" in data:
        raise ProtocolError("JSON message must be a single line")
    if len(data) > MAX_MESSAGE_BYTES:
        raise MessageTooLarge(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    return data


def decode_message(data: bytes) -> dict[str, Any]:
    """Decode a single-line UTF-8 JSON message into a dict.

    Callers are responsible for stripping any line terminator added by the
    transport; embedded newlines are rejected here.
    """
    if len(data) > MAX_MESSAGE_BYTES:
        raise MessageTooLarge(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    if b"\n" in data or b"\r" in data:
        raise ProtocolError("JSON message must be a single line")
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON message: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("JSON message must be a JSON object")
    return cast("dict[str, Any]", obj)


# --- Fcitx frame protocol ---------------------------------------------------

@dataclass(frozen=True)
class FcitxResponse:
    """Reply from the Fcitx addon to a COMMIT (or control) request."""

    status: Literal["ok", "reject", "error"]
    reason: str | None = None
    code: str | None = None


@dataclass(frozen=True)
class CommitFrame:
    """Parsed ``COMMIT <focus-token> <sequence> <total>\\n<utf8-text>`` frame."""

    focus_token: str
    sequence: int
    total: int
    text: str


def encode_commit_frame(
    focus_token: str, sequence: int, total: int, text: str
) -> bytes:
    if sequence < 1 or total < 1 or sequence > total:
        raise ValueError("sequence/total must satisfy 1 <= sequence <= total")
    if any(c in focus_token for c in (" ", "\n", "\r")):
        raise ValueError("focus_token must not contain whitespace")
    body = text.encode("utf-8")
    if len(body) > FCITX_TEXT_LINE_MAX_BYTES:
        raise MessageTooLarge(f"text line exceeds {FCITX_TEXT_LINE_MAX_BYTES} bytes")
    header = f"COMMIT {focus_token} {sequence} {total}\n".encode()
    frame = header + body
    if len(frame) > MAX_MESSAGE_BYTES:
        raise MessageTooLarge(f"frame exceeds {MAX_MESSAGE_BYTES} bytes")
    return frame


def parse_commit_frame(frame: bytes) -> CommitFrame:
    """Parse a Fcitx COMMIT frame into its parts."""
    if len(frame) > MAX_MESSAGE_BYTES:
        raise MessageTooLarge(f"frame exceeds {MAX_MESSAGE_BYTES} bytes")
    header, sep, body = frame.partition(b"\n")
    if not sep:
        raise ProtocolError("COMMIT frame is missing its header line")
    parts = header.decode("utf-8").split(" ", 3)
    if len(parts) != 4 or parts[0] != "COMMIT":
        raise ProtocolError(f"malformed COMMIT header: {header!r}")
    try:
        sequence = int(parts[2])
        total = int(parts[3])
    except ValueError as exc:
        raise ProtocolError(f"malformed COMMIT sequence/total: {header!r}") from exc
    return CommitFrame(
        focus_token=parts[1], sequence=sequence, total=total, text=body.decode("utf-8")
    )


def split_utf8(text: str, max_bytes: int = FCITX_CHUNK_MAX_BYTES) -> list[str]:
    """Split ``text`` into ordered chunks, each at most ``max_bytes`` UTF-8 bytes.

    Chunks are cut only on Unicode codepoint boundaries, so a multi-byte
    character is never split across two chunks. Returns ``[]`` for empty text.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for ch in text:
        ch_size = len(ch.encode("utf-8"))
        if current and current_size + ch_size > max_bytes:
            chunks.append("".join(current))
            current = []
            current_size = 0
        current.append(ch)
        current_size += ch_size
    if current:
        chunks.append("".join(current))
    return chunks


def build_commit_frames(focus_token: str, text: str) -> list[bytes]:
    """Build the ordered COMMIT frames for ``text`` using one focus token.

    A text line that fits the 64 KiB limit is sent as a single frame. Longer
    text (or a frame that would overflow) is split into chunks of at most
    8 KiB on Unicode boundaries, numbered ``1..total``.
    """
    body = text.encode("utf-8")
    if len(body) <= FCITX_TEXT_LINE_MAX_BYTES:
        header = f"COMMIT {focus_token} 1 1\n".encode()
        if len(header) + len(body) <= MAX_MESSAGE_BYTES:
            return [encode_commit_frame(focus_token, 1, 1, text)]
    chunks = split_utf8(text, FCITX_CHUNK_MAX_BYTES)
    total = len(chunks)
    return [
        encode_commit_frame(focus_token, index, total, chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]


def parse_fcitx_response(line: bytes) -> FcitxResponse:
    """Parse a single-line Fcitx reply (OK / REJECT / ERROR)."""
    if len(line) > MAX_MESSAGE_BYTES:
        raise MessageTooLarge(f"response exceeds {MAX_MESSAGE_BYTES} bytes")
    if b"\n" in line or b"\r" in line:
        raise ProtocolError("Fcitx response must be a single line")
    text = line.decode("utf-8").strip()
    if text == "OK":
        return FcitxResponse(status="ok")
    if text.startswith("REJECT"):
        reason = text.split(" ", 1)[1].strip() if " " in text else ""
        return FcitxResponse(status="reject", reason=reason)
    if text.startswith("ERROR"):
        code = text.split(" ", 1)[1].strip() if " " in text else ""
        return FcitxResponse(status="error", code=code)
    raise ProtocolError(f"unknown Fcitx response: {text!r}")
