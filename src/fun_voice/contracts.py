"""Stable contracts and wire-protocol codecs for Fun Voice Ryan.

This module contains no business logic. It defines the immutable data types,
the daemon state machine, structured error codes, and the byte-level framing
used between local control clients, the daemon, worker, and the Fcitx addon.

Protocol summary:
- Control-client/daemon/worker messages are single-line UTF-8 JSON, at most 64 KiB.
- Fcitx uses a header line followed by a UTF-8 body, at most 64 KiB per frame.
- Fcitx COMMIT text is limited to 64 KiB per line; longer text is split on
  Unicode codepoint boundaries into ordered chunks of at most 8 KiB.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Final, Literal, cast

# --- Size limits ------------------------------------------------------------

MAX_MESSAGE_BYTES = 64 * 1024
# Maximum wire size (bytes) of a JSON message or a Fcitx frame.

WORKER_RESPONSE_MAX_BYTES = 4 * 1024 * 1024
# Maximum wire size (bytes) of a worker response. A 30-minute recording can
# yield ~100 KB+ of transcribed text (full text plus time-ranged segments),
# which exceeds the 64 KiB request cap, so responses use a dedicated larger
# limit while requests stay bounded by MAX_MESSAGE_BYTES.

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
    PREPARING = "preparing"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    FINALIZING = "finalizing"
    CORRECTING = "correcting"
    COMMITTING = "committing"
    REHYDRATING = "rehydrating"
    ENRICHING = "enriching"
    ACTIVE_IDLE = "active_idle"
    ERROR = "error"


# --- Structured error code --------------------------------------------------

@dataclass(frozen=True)
class ErrorCode:
    """Machine-readable error: a category plus a category-specific code."""

    category: str
    code: str

    def __str__(self) -> str:
        return f"{self.category}.{self.code}"


class ModelTaskKind(StrEnum):
    """The approved XPU scheduler task classes, in priority order elsewhere."""

    FINAL_TAIL = "final_tail"
    STABLE_SEGMENT = "stable_segment"
    PROVISIONAL_TAIL = "provisional_tail"
    CORRECTION = "correction"
    ENRICHMENT = "enrichment"


@dataclass(frozen=True, slots=True)
class SessionKey:
    """Opaque, process-local identity for work belonging to one recording.

    The identifier is deliberately omitted from ``repr`` so an accidental debug
    log cannot correlate otherwise private session work.  Live worker requests
    carry it only as bounded same-UID socket metadata; it is never returned,
    persisted, or included in telemetry.
    """

    session_id: str = field(repr=False)
    generation: int = 1

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.generation < 1:
            raise ValueError("generation must be positive")


# --- Data types -------------------------------------------------------------

@dataclass(frozen=True)
class StartRequest:
    """Local control-client -> daemon request to start recording if idle."""

    op: str = "start_if_idle"


@dataclass(frozen=True)
class StopRequest:
    """Local control-client -> daemon request to stop recording."""

    op: str = "stop"


@dataclass(frozen=True)
class FocusSnapshot:
    """X11 focus state captured at recording start, compared before commit.

    ``active_window`` is the ``_NET_ACTIVE_WINDOW`` id, ``input_focus`` the
    current X input-focus window, ``window_pid`` the ``_NET_WM_PID`` of the
    active window and ``process_name`` its resolved process name. ``monotonic_ns``
    is a capture timestamp (excluded from equality comparisons).
    """

    active_window: int | None
    process_name: str | None
    input_focus: int | None
    monotonic_ns: int
    window_pid: int | None = None

@dataclass(frozen=True)
class CaptureArtifact:
    """Handle to captured audio: a path or memfd plus its format.

    When ``audio`` is a ``/proc/<pid>/fd/<n>`` path it references an anonymous,
    unlinked tmpfs-backed file whose validity is bound to the recorder that
    produced it: the underlying descriptor stays open only until the recorder is
    reused (``start()``) or torn down (``cleanup()``).  Consumers must finish
    reading the handle before either of those events; there is no stable on-disk
    path.  When ``audio`` is a real path, the file exists until the producer
    deletes it.
    """

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
class AsrStageTiming:
    """Non-sensitive duration stages produced inside an ASR runtime."""

    audio_load_ms: int | None = None
    vad_ms: int | None = None
    generate_ms: int | None = None


@dataclass(frozen=True)
class Transcription:
    """Worker result: raw model text plus time-ordered segments."""

    text: str
    segments: tuple[Segment, ...] = ()
    request_id: str | None = None
    engine: Literal["nano", "sensevoice"] = "nano"
    timing: AsrStageTiming = field(default_factory=AsrStageTiming)
    worker_elapsed_ms: int | None = None


@dataclass(frozen=True)
class PreloadTiming:
    """Non-sensitive duration stages for lazy ASR runtime materialization."""

    worker_elapsed_ms: int | None = None
    runtime_load_ms: int | None = None
    warmup_ms: int | None = None
    warmup_status: Literal["not_requested", "ready", "failed"] = "not_requested"


CORRECTION_REJECTION_REASONS: Final = frozenset(
    {
        "envelope_missing",
        "envelope_malformed",
        "output_empty",
        "output_too_long",
        "similarity",
        "protected_token",
        "input_too_large",
        "model_load",
        "oom",
        "device",
        "protocol",
        "no_output",
        "generation",
        "timeout",
        "unavailable",
        "internal",
    }
)


@dataclass(frozen=True)
class CorrectionTiming:
    """Non-sensitive durations emitted by one isolated Qwen correction call."""

    model_load_ms: int | None = None
    generate_ms: int | None = None
    validate_ms: int | None = None


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
    lifecycle: Literal["loading", "ready", "inactive", "failed"] = "ready"

    def __post_init__(self) -> None:
        if self.lifecycle not in {"loading", "ready", "inactive", "failed"}:
            raise ValueError("worker lifecycle is invalid")


# --- JSON message codec -----------------------------------------------------
def encode_message(
    message: Mapping[str, Any], max_bytes: int = MAX_MESSAGE_BYTES
) -> bytes:
    """Encode a control-client/daemon/worker message as single-line UTF-8 JSON.

    ``max_bytes`` bounds the encoded byte size; worker responses pass the
    larger ``WORKER_RESPONSE_MAX_BYTES``.
    """
    data = json.dumps(dict(message), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if b"\n" in data or b"\r" in data:
        raise ProtocolError("JSON message must be a single line")
    if len(data) > max_bytes:
        raise MessageTooLarge(f"message exceeds {max_bytes} bytes")
    return data


def decode_message(data: bytes, max_bytes: int = MAX_MESSAGE_BYTES) -> dict[str, Any]:
    """Decode a single-line UTF-8 JSON message into a dict.

    Callers are responsible for stripping any line terminator added by the
    transport; embedded newlines are rejected here. ``max_bytes`` bounds the
    accepted byte size; worker responses are decoded with
    ``WORKER_RESPONSE_MAX_BYTES``.
    """
    if len(data) > max_bytes:
        raise MessageTooLarge(f"message exceeds {max_bytes} bytes")
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
    """Encode a single Fcitx COMMIT frame (header line + UTF-8 body)."""
    if sequence < 1 or total < 1 or sequence > total:
        raise ValueError("sequence/total must satisfy 1 <= sequence <= total")
    if not focus_token:
        raise ValueError("focus_token must not be empty")
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
    try:
        header_text = header.decode("utf-8")
        body_text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("COMMIT frame is not valid UTF-8") from exc
    parts = header_text.split(" ", 3)
    if len(parts) != 4 or parts[0] != "COMMIT":
        raise ProtocolError(f"malformed COMMIT header: {header!r}")
    try:
        sequence = int(parts[2])
        total = int(parts[3])
    except ValueError as exc:
        raise ProtocolError(f"malformed COMMIT sequence/total: {header!r}") from exc
    if not (1 <= sequence <= total):
        raise ProtocolError(f"invalid COMMIT sequence/total: {sequence}/{total}")
    return CommitFrame(
        focus_token=parts[1], sequence=sequence, total=total, text=body_text
    )


def split_utf8(text: str, max_bytes: int = FCITX_CHUNK_MAX_BYTES) -> list[str]:
    """Split ``text`` into ordered chunks, each at most ``max_bytes`` UTF-8 bytes.

    Chunks are cut on Unicode codepoint (not grapheme cluster) boundaries, so
    a multi-byte character is never split across two chunks. If a single
    codepoint itself exceeds ``max_bytes``, that chunk will exceed the limit
    (UTF-8 encodes one codepoint in at most 4 bytes, so this cannot happen at
    the default 8 KiB threshold). Returns ``[]`` for empty text.
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
    8 KiB on Unicode boundaries, numbered ``1..total``. Empty ``text`` yields
    no frames.
    """
    if not text:
        return []
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
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("Fcitx response is not valid UTF-8") from exc
    text = text.strip()
    if text == "OK":
        return FcitxResponse(status="ok")
    if text == "REJECT" or text.startswith("REJECT "):
        return FcitxResponse(status="reject", reason=text[7:].strip())
    if text == "ERROR" or text.startswith("ERROR "):
        return FcitxResponse(status="error", code=text[6:].strip())
    raise ProtocolError(f"unknown Fcitx response: {text!r}")
