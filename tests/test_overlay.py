"""Unit tests for the private, native DTK transient overlay controller."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fun_voice.contracts import DaemonState
from fun_voice.overlay import (
    DtkOverlayController,
    OverlayModel,
    default_overlay_executable,
)


class FakeWriter:
    def __init__(self, *, broken: bool = False) -> None:
        self.broken = broken
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes) -> int:
        if self.broken:
            raise BrokenPipeError
        self.data.extend(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, *, broken: bool = False) -> None:
        self.stdin: FakeWriter | None = FakeWriter(broken=broken)
        self.stdout = None
        self.returncode: int | None = None
        self.terminate_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fun-voice-overlay", timeout)
        return self.returncode


def decode_frames(writer: FakeWriter) -> list[dict[str, object]]:
    raw = bytes(writer.data)
    frames: list[dict[str, object]] = []
    while raw:
        length = int.from_bytes(raw[:4], "big")
        payload = raw[4 : 4 + length]
        assert len(payload) == length
        frames.append(json.loads(payload.decode("utf-8")))
        raw = raw[4 + length :]
    return frames


def test_dtk_controller_starts_lazily_and_writes_one_bounded_show_frame() -> None:
    spawned: list[FakeProcess] = []

    def popen(_argv: list[str]) -> FakeProcess:
        process = FakeProcess()
        spawned.append(process)
        return process

    controller = DtkOverlayController(
        executable=Path("/native/fun-voice-overlay"), popen=popen
    )

    controller.show(OverlayModel(phase=DaemonState.RECORDING, level=42))

    assert len(spawned) == 1
    writer = spawned[0].stdin
    assert writer is not None
    assert decode_frames(writer) == [
        {
            "command": "show",
            "level": 42,
            "phase": "recording",
            "provisional_text": "",
            "stable_text": "",
        }
    ]


def test_dtk_controller_clear_replaces_transient_text_with_a_text_free_command(
) -> None:
    process = FakeProcess()
    controller = DtkOverlayController(
        executable=Path("overlay"), popen=lambda _argv: process
    )

    controller.show(
        OverlayModel(phase=DaemonState.RECORDING, stable_text="私密文本")
    )
    controller.clear()

    writer = process.stdin
    assert writer is not None
    assert decode_frames(writer)[-1] == {"command": "clear"}


def test_dtk_controller_does_not_spawn_for_an_oversized_transient_model() -> None:
    spawned: list[FakeProcess] = []
    controller = DtkOverlayController(
        executable=Path("overlay"),
        popen=lambda _argv: spawned.append(FakeProcess()) or spawned[-1],
    )

    controller.show(
        OverlayModel(phase=DaemonState.RECORDING, stable_text="a" * (64 * 1024))
    )

    assert spawned == []


def test_dtk_controller_recovers_from_a_broken_pipe_on_the_next_show() -> None:
    failed = FakeProcess(broken=True)
    recovered = FakeProcess()
    processes = iter((failed, recovered))
    controller = DtkOverlayController(
        executable=Path("overlay"), popen=lambda _argv: next(processes)
    )

    controller.show(OverlayModel(phase=DaemonState.RECORDING))
    controller.show(OverlayModel(phase=DaemonState.FINALIZING))

    writer = recovered.stdin
    assert writer is not None
    assert decode_frames(writer) == [
        {
            "command": "show",
            "phase": "finalizing",
            "provisional_text": "",
            "stable_text": "",
        }
    ]
    assert failed.terminate_calls == 1


def test_dtk_controller_shutdowns_the_owned_process() -> None:
    process = FakeProcess()
    controller = DtkOverlayController(
        executable=Path("overlay"), popen=lambda _argv: process
    )
    controller.show(OverlayModel(phase=DaemonState.RECORDING))
    controller.close()

    writer = process.stdin
    assert writer is not None
    assert decode_frames(writer)[-1] == {"command": "shutdown"}
    assert process.terminate_calls == 1


def test_default_overlay_binary_uses_the_user_scoped_install_location() -> None:
    assert default_overlay_executable() == (
        Path.home() / ".local/lib/fun-voice-ryan/fun-voice-overlay"
    )
