"""Tests for immutable contract types and wire-protocol codecs."""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from fun_voice.contracts import (
    FCITX_CHUNK_MAX_BYTES,
    FCITX_TEXT_LINE_MAX_BYTES,
    MAX_MESSAGE_BYTES,
    CaptureArtifact,
    CommitResult,
    DaemonState,
    ErrorCode,
    FcitxResponse,
    FocusSnapshot,
    MessageTooLarge,
    ProtocolError,
    Segment,
    StartRequest,
    StopRequest,
    Transcription,
    WorkerHealth,
    build_commit_frames,
    decode_message,
    encode_commit_frame,
    encode_message,
    parse_commit_frame,
    parse_fcitx_response,
    split_utf8,
)

# --- Constants and state ----------------------------------------------------

def test_protocol_size_constants() -> None:
    assert MAX_MESSAGE_BYTES == 64 * 1024
    assert FCITX_TEXT_LINE_MAX_BYTES == 64 * 1024
    assert FCITX_CHUNK_MAX_BYTES == 8 * 1024


def test_daemon_state_members() -> None:
    assert {state.name for state in DaemonState} == {
        "IDLE",
        "RECORDING",
        "TRANSCRIBING",
        "COMMITTING",
        "ERROR",
    }


# --- Immutable types --------------------------------------------------------

def test_data_types_are_frozen() -> None:
    seg = Segment(start_ms=0, end_ms=120, text="你好")
    with pytest.raises(dataclasses.FrozenInstanceError):
        seg.text = "改写"
    snap = FocusSnapshot(
        active_window=1,
        process_name="app",
        input_focus=1,
        monotonic_ns=0,
        window_pid=4242,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.monotonic_ns = 1


def test_fcitx_response_is_frozen() -> None:
    resp = FcitxResponse(status="ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        resp.status = "error"


def test_error_code_is_structured() -> None:
    err = ErrorCode(category="worker", code="oom")
    assert err.category == "worker"
    assert err.code == "oom"
    assert str(err) == "worker.oom"


def test_requests_have_expected_ops() -> None:
    assert StartRequest().op == "start_if_idle"
    assert StopRequest().op == "stop"


def test_typed_records_construct() -> None:
    segments = (Segment(0, 10, "a"), Segment(20, 30, "b"))
    transcription = Transcription(text="ab", segments=segments, request_id="r1")
    assert transcription.text == "ab"
    assert transcription.segments == segments

    health = WorkerHealth(
        version="1.0",
        xpu_ready=True,
        model_ready=False,
        device="xpu:0",
        last_error=ErrorCode("xpu", "not-ready"),
    )
    assert health.xpu_ready is True
    assert health.last_error is not None

    result = CommitResult(committed=True, method="fcitx")
    assert result.committed is True
    assert result.error is None

    artifact = CaptureArtifact(audio="/run/user/1000/fun-voice-ryan/shard-0.pcm")
    assert artifact.sample_rate == 16000


# --- JSON message codec -----------------------------------------------------

def test_encode_message_is_single_line_json() -> None:
    assert encode_message({"op": "start_if_idle"}) == b'{"op":"start_if_idle"}'
    assert encode_message({"op": "stop"}) == b'{"op":"stop"}'
    assert b"\n" not in encode_message({"op": "start_if_idle"})


def test_message_round_trip() -> None:
    msg = {
        "id": str(uuid.uuid4()),
        "op": "transcribe",
        "audio": "/run/user/1000/fun-voice-ryan/shard-0.pcm",
        "sample_rate": 16000,
    }
    assert decode_message(encode_message(msg)) == msg


def test_decode_rejects_multiline() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b'{"op":"start_if_idle"}\n{"op":"stop"}')


def test_decode_rejects_non_object() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b"[1, 2, 3]")


def test_message_size_limit() -> None:
    too_big = {"op": "x" * (MAX_MESSAGE_BYTES + 16)}
    with pytest.raises(MessageTooLarge):
        encode_message(too_big)
    with pytest.raises(MessageTooLarge):
        decode_message(b" " * (MAX_MESSAGE_BYTES + 1))


# --- Fcitx frame protocol ---------------------------------------------------

def test_encode_commit_frame_format() -> None:
    frame = encode_commit_frame("tok", 1, 3, "你好 world")
    assert frame == b"COMMIT tok 1 3\n" + "你好 world".encode()


def test_parse_commit_frame_round_trip() -> None:
    frame = encode_commit_frame("tok-9", 2, 4, "你好")
    parsed = parse_commit_frame(frame)
    assert parsed.focus_token == "tok-9"
    assert parsed.sequence == 2
    assert parsed.total == 4
    assert parsed.text == "你好"


def test_parse_commit_frame_rejects_missing_header() -> None:
    with pytest.raises(ProtocolError):
        parse_commit_frame(b"no newline here")


def test_encode_commit_frame_rejects_bad_sequence() -> None:
    with pytest.raises(ValueError):
        encode_commit_frame("tok", 2, 1, "text")


def test_parse_fcitx_response_ok() -> None:
    assert parse_fcitx_response(b"OK").status == "ok"


def test_parse_fcitx_response_reject() -> None:
    stale = parse_fcitx_response(b"REJECT stale-focus")
    assert stale.status == "reject"
    assert stale.reason == "stale-focus"
    no_ctx = parse_fcitx_response(b"REJECT no-input-context")
    assert no_ctx.reason == "no-input-context"


def test_parse_fcitx_response_error() -> None:
    resp = parse_fcitx_response(b"ERROR 7")
    assert resp.status == "error"
    assert resp.code == "7"


def test_parse_fcitx_response_unknown() -> None:
    with pytest.raises(ProtocolError):
        parse_fcitx_response(b"HELLO")


# --- Text splitting ---------------------------------------------------------

def test_split_utf8_preserves_text_and_boundaries() -> None:
    text = "a你b好c"
    chunks = split_utf8(text, max_bytes=5)
    assert chunks == ["a你b", "好c"]
    assert "".join(chunks) == text
    for chunk in chunks:
        assert len(chunk.encode("utf-8")) <= 5


def test_split_utf8_never_splits_codepoint() -> None:
    text = "你" * 10  # 30 UTF-8 bytes
    chunks = split_utf8(text, max_bytes=7)
    assert "".join(chunks) == text
    for chunk in chunks:
        assert len(chunk.encode("utf-8")) <= 7


def test_split_utf8_empty() -> None:
    assert split_utf8("") == []


# --- Commit frame building --------------------------------------------------

def test_build_commit_frames_single_short_text() -> None:
    frames = build_commit_frames("tok", "你好")
    assert len(frames) == 1
    parsed = parse_commit_frame(frames[0])
    assert parsed.sequence == 1
    assert parsed.total == 1
    assert parsed.text == "你好"


def test_build_commit_frames_splits_long_text() -> None:
    text = "你" * 30000  # 90,000 bytes > 64 KiB
    frames = build_commit_frames("tok", text)
    assert len(frames) > 1
    total = len(frames)
    reassembled: list[str] = []
    for index, frame in enumerate(frames, start=1):
        assert len(frame) <= MAX_MESSAGE_BYTES
        parsed = parse_commit_frame(frame)
        assert parsed.focus_token == "tok"
        assert parsed.sequence == index
        assert parsed.total == total
        assert len(parsed.text.encode("utf-8")) <= FCITX_CHUNK_MAX_BYTES
        reassembled.append(parsed.text)
    assert "".join(reassembled) == text


def test_parse_commit_frame_rejects_invalid_utf8() -> None:
    with pytest.raises(ProtocolError):
        parse_commit_frame(b"COMMIT tok 1 1\n\xff\xfe")


def test_parse_commit_frame_rejects_sequence_out_of_range() -> None:
    with pytest.raises(ProtocolError):
        parse_commit_frame(b"COMMIT tok 3 2\ntext")
    with pytest.raises(ProtocolError):
        parse_commit_frame(b"COMMIT tok 0 1\ntext")


def test_parse_fcitx_response_rejects_invalid_utf8() -> None:
    with pytest.raises(ProtocolError):
        parse_fcitx_response(b"REJECT \xff")


def test_parse_fcitx_response_rejects_prefix_impostors() -> None:
    with pytest.raises(ProtocolError):
        parse_fcitx_response(b"REJECTED")
    with pytest.raises(ProtocolError):
        parse_fcitx_response(b"ERRORS")


def test_split_utf8_handles_astral_codepoints() -> None:
    text = "😀" * 10  # 40 UTF-8 bytes total, 4 bytes per codepoint
    chunks = split_utf8(text, max_bytes=9)
    assert "".join(chunks) == text
    for chunk in chunks:
        assert len(chunk.encode("utf-8")) <= 9


def test_split_utf8_single_codepoint_exceeding_limit_is_kept() -> None:
    # A 4-byte astral codepoint cannot fit in max_bytes=3; it is kept intact
    # (the chunk may exceed the limit) rather than split.
    assert split_utf8("😀a", max_bytes=3) == ["😀", "a"]


def test_encode_commit_frame_rejects_empty_token() -> None:
    with pytest.raises(ValueError):
        encode_commit_frame("", 1, 1, "text")


def test_build_commit_frames_empty_text_returns_no_frames() -> None:
    assert build_commit_frames("tok", "") == []
