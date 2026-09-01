"""Unit tests for the PipeWire capture adapter.

Uses fake subprocesses and a fake monotonic clock so no real ``pw-record``,
PipeWire server, or wall-clock waiting is required.
"""

from __future__ import annotations

import errno
import io
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from fun_voice.capture import (
    MAX_RECORDING_MINUTES,
    MEMORY_THRESHOLD_BYTES,
    NOTIFY_AT_MINUTES,
    SAMPLE_FORMAT,
    SAMPLE_RATE,
    SHARD_BYTES,
    CaptureConfig,
    CaptureError,
    PipeWireRecorder,
)
from fun_voice.contracts import CaptureArtifact


def pcm_bytes(ms: int) -> bytes:
    """Return ``ms`` milliseconds of raw s16le silence."""
    n_samples = int(ms * SAMPLE_RATE / 1000)
    return b"\x00\x00" * n_samples


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeProcess:
    """A controllable stand-in for ``subprocess.Popen``."""

    def __init__(self, data: bytes = b"", exit_code: int = 0, *, exited: bool = False):
        self.stdout: io.BytesIO = io.BytesIO(data)
        self._exit_code = exit_code
        self.returncode: int | None = exit_code if exited else None
        self.sent_signals: list[int] = []
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        assert self.returncode is not None
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.sent_signals.append(sig)
        if self.returncode is None:
            self.returncode = self._exit_code

    def terminate(self) -> None:
        self.terminated = True
        if self.returncode is None:
            self.returncode = self._exit_code

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = self._exit_code


def make_recorder(
    tmp_path: Path,
    *,
    data: bytes = b"",
    exit_code: int = 0,
    exited: bool = False,
    clock: Callable[[], float] | None = None,
    notifier: Callable[[str], None] | None = None,
    on_auto_stop: Callable[[], None] | None = None,
    memory_threshold_bytes: int = MEMORY_THRESHOLD_BYTES,
) -> tuple[PipeWireRecorder, list[list[str]], dict[str, FakeProcess]]:
    argv_log: list[list[str]] = []
    holder: dict[str, FakeProcess] = {}

    def spawn(argv: list[str]) -> FakeProcess:
        argv_log.append(argv)
        holder["proc"] = FakeProcess(data, exit_code, exited=exited)
        return holder["proc"]

    recorder = PipeWireRecorder(
        clock=clock if clock is not None else FakeClock(),
        notifier=notifier,
        on_auto_stop=on_auto_stop,
        spawn=spawn,
        runtime_dir=tmp_path,
        memory_threshold_bytes=memory_threshold_bytes,
    )
    return recorder, argv_log, holder


def read_artifact(artifact: CaptureArtifact) -> bytes:
    with open(artifact.audio, "rb") as f:
        return f.read()


def wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)


# --- Argument construction ---------------------------------------------------


def test_start_passes_expected_argv(tmp_path: Path) -> None:
    recorder, argv_log, _ = make_recorder(tmp_path, data=pcm_bytes(1000))
    recorder.start()
    assert argv_log[0] == [
        "pw-record",
        "--rate",
        "16000",
        "--channels",
        "1",
        "--format",
        "s16",
        "--media-type",
        "Audio",
        "--raw",
        "--target",
        "default",
        "-",
    ]
    recorder.cancel()


def test_source_is_passed_verbatim(tmp_path: Path) -> None:
    recorder, argv_log, _ = make_recorder(tmp_path, data=pcm_bytes(1000))
    recorder.start(CaptureConfig(source="effect_output.rnnoise"))
    assert argv_log[0][argv_log[0].index("--target") + 1] == "effect_output.rnnoise"
    recorder.cancel()


# --- Stop / exit codes --------------------------------------------------------


def test_stop_sends_sigint(tmp_path: Path) -> None:
    recorder, _, holder = make_recorder(tmp_path, data=pcm_bytes(1000))
    recorder.start()
    artifact = recorder.stop()
    proc = holder["proc"]
    assert proc.sent_signals and proc.sent_signals[-1] == signal.SIGINT
    assert artifact.duration_ms == 1000


@pytest.mark.parametrize("exit_code", [0, 130, -signal.SIGINT, 1])
def test_normal_exit_codes_accepted(tmp_path: Path, exit_code: int) -> None:
    data = pcm_bytes(1000)
    recorder, _, holder = make_recorder(tmp_path, data=data, exit_code=exit_code)
    recorder.start()
    artifact = recorder.stop()
    assert holder["proc"].returncode == exit_code
    assert read_artifact(artifact) == data
    assert artifact.sample_rate == SAMPLE_RATE
    assert artifact.format == SAMPLE_FORMAT
    assert artifact.duration_ms == 1000


def test_already_exited_normally_accepted(tmp_path: Path) -> None:
    data = pcm_bytes(500)
    recorder, _, holder = make_recorder(
        tmp_path, data=data, exit_code=0, exited=True
    )
    recorder.start()
    artifact = recorder.stop()
    assert holder["proc"].sent_signals == []  # already gone; no signal needed
    assert read_artifact(artifact) == data


def test_abnormal_exit_code_raises(tmp_path: Path) -> None:
    notifications: list[str] = []
    recorder, _, _ = make_recorder(
        tmp_path, data=pcm_bytes(1000), exit_code=42, notifier=notifications.append
    )
    recorder.start()
    with pytest.raises(CaptureError, match="exited abnormally"):
        recorder.stop()
    assert notifications == []


# --- Memory vs shard storage --------------------------------------------------


def test_memory_only_recording_never_creates_directory(tmp_path: Path) -> None:
    data = pcm_bytes(500)
    recorder, _, _ = make_recorder(tmp_path, data=data)
    recorder.start()
    artifact = recorder.stop()
    assert read_artifact(artifact) == data
    assert not (tmp_path / "capture").exists()


def test_shards_reassemble_losslessly(tmp_path: Path) -> None:
    data = bytes(i % 256 for i in range(1024 + 2 * SHARD_BYTES + 1234))
    recorder, _, _ = make_recorder(tmp_path, data=data, memory_threshold_bytes=1024)
    recorder.start()
    artifact = recorder.stop()
    assert read_artifact(artifact) == data
    # Cleanup removed the directory after materialization.
    assert not (tmp_path / "capture").exists()


def test_shard_layout_before_finalize(tmp_path: Path) -> None:
    """Shards are 60s-aligned, monotonic, 0600, in a 0700 dir, bounded size."""
    data = bytes(i % 256 for i in range(1024 + 2 * SHARD_BYTES + 1234))
    recorder, _, _ = make_recorder(tmp_path, data=data, memory_threshold_bytes=1024)
    recorder.start()
    wait_for(lambda: recorder._bytes == len(data))  # let the reader drain

    shard_dir = tmp_path / "capture"
    assert shard_dir.is_dir()
    assert (shard_dir.stat().st_mode & 0o777) == 0o700
    names = sorted(p.name for p in shard_dir.glob("shard-*.pcm"))
    assert names == ["shard-000000.pcm", "shard-000060.pcm", "shard-000120.pcm"]

    shards = sorted(shard_dir.glob("shard-*.pcm"))
    for shard in shards:
        assert (shard.stat().st_mode & 0o777) == 0o600
        assert shard.stat().st_size <= SHARD_BYTES

    assert shards[0].stat().st_size == SHARD_BYTES
    assert shards[1].stat().st_size == SHARD_BYTES
    assert shards[2].stat().st_size == 1234

    recorder.cancel()


def test_live_snapshot_uses_exact_pcm_bounds_and_survives_a_new_recording(
    tmp_path: Path,
) -> None:
    data = bytes(index % 256 for index in range(1000 * 32))
    recorder, _, _ = make_recorder(tmp_path, data=data)
    recorder.start()
    wait_for(lambda: recorder._bytes == len(data))  # noqa: SLF001 - live boundary

    first = recorder.snapshot(100, 400)
    overlap = recorder.snapshot(250, 500)
    retained = first.retain()
    first.release()

    assert read_artifact(retained.artifact) == data[3200:12800]
    assert read_artifact(overlap.artifact) == data[8000:16000]
    assert retained.artifact.duration_ms == 300
    assert overlap.artifact.duration_ms == 250

    recorder.stop()
    recorder.start()
    assert read_artifact(retained.artifact) == data[3200:12800]
    retained.release()
    retained.release()
    overlap.release()
    recorder.cancel()


# --- Watchdog: notify and auto-stop ------------------------------------------


def test_notify_fires_once_at_25_minutes(tmp_path: Path) -> None:
    clock = FakeClock()
    notifications: list[str] = []
    recorder, _, _ = make_recorder(
        tmp_path, data=pcm_bytes(1000), clock=clock, notifier=notifications.append
    )
    recorder.start()
    clock.now = NOTIFY_AT_MINUTES * 60 + 1
    wait_for(lambda: bool(notifications))
    assert len(notifications) == 1
    recorder.cancel()


def test_auto_stop_at_30_minutes_returns_data(tmp_path: Path) -> None:
    clock = FakeClock()
    data = pcm_bytes(1000)
    recorder, _, holder = make_recorder(tmp_path, data=data, clock=clock)
    recorder.start()
    clock.now = MAX_RECORDING_MINUTES * 60 + 1
    wait_for(lambda: bool(holder["proc"].sent_signals))
    assert holder["proc"].sent_signals[-1] == signal.SIGINT
    artifact = recorder.stop()
    assert read_artifact(artifact) == data


# --- Error conditions ---------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "data", "match"),
    [
        ("no-bytes", b"", "no audio bytes"),
        ("too-short", pcm_bytes(100), "too short"),
        ("container", b"RIFF\x00\x00\x00\x00WAVE" + pcm_bytes(1000), "container"),
    ],
)
def test_error_conditions_raise_and_never_notify(
    tmp_path: Path, label: str, data: bytes, match: str
) -> None:
    clock = FakeClock()
    notifications: list[str] = []
    recorder, _, _ = make_recorder(
        tmp_path, data=data, clock=clock, notifier=notifications.append
    )
    recorder.start()
    with pytest.raises(CaptureError, match=match):
        recorder.stop()
    assert notifications == []


def test_subprocess_start_error_raises(tmp_path: Path) -> None:
    def spawn(argv: list[str]) -> FakeProcess:
        raise OSError("no such file")

    recorder = PipeWireRecorder(spawn=spawn, runtime_dir=tmp_path)
    with pytest.raises(CaptureError, match="failed to start"):
        recorder.start()


def test_stop_before_start_raises(tmp_path: Path) -> None:
    recorder = PipeWireRecorder(runtime_dir=tmp_path)
    with pytest.raises(CaptureError, match="not started"):
        recorder.stop()


# --- Cleanup ------------------------------------------------------------------


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    data = bytes(i % 256 for i in range(1024 + SHARD_BYTES + 5))
    recorder, _, _ = make_recorder(tmp_path, data=data, memory_threshold_bytes=1024)
    recorder.start()
    wait_for(lambda: recorder._bytes == len(data))
    assert (tmp_path / "capture").exists()
    recorder.cancel()
    assert not (tmp_path / "capture").exists()
    # Second cleanup must be a harmless no-op.
    recorder.cleanup()
    assert not (tmp_path / "capture").exists()


def test_startup_leftover_directory_is_cleaned(tmp_path: Path) -> None:
    shard_dir = tmp_path / "capture"
    shard_dir.mkdir(parents=True)
    (shard_dir / "shard-000600.pcm").write_bytes(b"stale")
    recorder, _, _ = make_recorder(tmp_path, data=pcm_bytes(1000))
    recorder.start()
    assert not shard_dir.exists()
    recorder.cancel()
def test_artifact_handle_readable_across_process(tmp_path: Path) -> None:
    data = pcm_bytes(1000)
    recorder, _, _ = make_recorder(tmp_path, data=data)
    recorder.start()
    artifact = recorder.stop()
    # A separate process opens the /proc/<pid>/fd/<n> handle and reads it back.
    result = subprocess.run(
        [sys.executable, "-c", "import sys; print(len(open(sys.argv[1], 'rb').read()))",
         artifact.audio],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert int(result.stdout.strip()) == len(data)


def test_await_exit_escalates_terminate_then_kill(tmp_path: Path) -> None:
    data = pcm_bytes(1000)

    class StallingProcess(FakeProcess):
        def __init__(self) -> None:
            super().__init__(data, exit_code=0)
            self._wait_calls = 0

        def wait(self, timeout: float | None = None) -> int:
            self._wait_calls += 1
            if self._wait_calls < 3:
                raise subprocess.TimeoutExpired(cmd="pw-record", timeout=timeout or 0)
            return 0

    holder: dict[str, StallingProcess] = {}

    def spawn(argv: list[str]) -> StallingProcess:
        holder["proc"] = StallingProcess()
        return holder["proc"]

    recorder = PipeWireRecorder(spawn=spawn, runtime_dir=tmp_path)
    recorder.start()
    artifact = recorder.stop()
    assert holder["proc"].terminated
    assert holder["proc"].killed
    assert read_artifact(artifact) == data


def test_auto_stop_fires_on_auto_stop_callback(tmp_path: Path) -> None:
    clock = FakeClock()
    data = pcm_bytes(1000)
    auto_stops: list[None] = []
    recorder, _, _ = make_recorder(
        tmp_path, data=data, clock=clock, on_auto_stop=lambda: auto_stops.append(None)
    )
    recorder.start()
    clock.now = MAX_RECORDING_MINUTES * 60 + 1
    wait_for(lambda: bool(auto_stops))
    assert len(auto_stops) == 1
    artifact = recorder.stop()
    assert read_artifact(artifact) == data


def test_shard_parent_dir_is_private(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "fun-voice-ryan"
    data = bytes(i % 256 for i in range(1024 + SHARD_BYTES + 5))
    recorder = PipeWireRecorder(
        spawn=lambda argv: FakeProcess(data, 0),
        runtime_dir=runtime_dir,
        memory_threshold_bytes=1024,
    )
    recorder.start()
    wait_for(lambda: recorder._bytes == len(data))
    assert (runtime_dir.stat().st_mode & 0o777) == 0o700
    assert (runtime_dir / "capture").is_dir()
    recorder.cancel()


def test_materialize_write_error_raises_capture_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _NoSpaceFile:
        def write(self, data: bytes) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        def flush(self) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self) -> None:
            pass

    recorder, _, _ = make_recorder(tmp_path, data=pcm_bytes(1000))
    recorder.start()
    monkeypatch.setattr(
        "fun_voice.capture._create_memory_backed_file", lambda: _NoSpaceFile()
    )
    with pytest.raises(CaptureError, match="materialize"):
        recorder.stop()
