"""PipeWire capture adapter: bounded PCM recording via ``pw-record``.

The recorder runs ``pw-record`` with a fixed raw s16le mono 16 kHz stream and
accumulates the bytes subject to a hard memory bound:

- The first ``MEMORY_THRESHOLD_BYTES`` (~10 minutes) live in an in-memory
  ``bytearray`` only; nothing touches disk.  This keeps short push-to-talk
  recordings fully off the filesystem (privacy red line: audio is never
  persisted for recordings under the memory threshold).
- Once the threshold is crossed, PCM spills into 60-second shard files under a
  private ``capture`` subdirectory of the runtime dir (``0700``), each shard
  ``0600`` and bounded to ``SHARD_BYTES``.  The directory only ever holds the
  current task's shards and is removed when the artifact is handed off.

``stop()`` finalizes by concatenating memory and shards into an anonymous memfd
(never a named file) and returns a :class:`~fun_voice.contracts.CaptureArtifact`.

Privacy red lines (never violated):

- Audio and transcription text are never logged.
- The runtime directory is only ever removed at the exact ``capture`` shard
  subdirectory we created; ``XDG_RUNTIME_DIR`` is never broadly deleted.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from fun_voice.config import (
    DIRECTORY_MODE,
    FILE_MODE,
    ConfigError,
    resolve_runtime_dir,
)
from fun_voice.contracts import CaptureArtifact

# --- Fixed stream format -----------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_FORMAT = "s16le"
BYTES_PER_FRAME = 2  # one s16 mono sample
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * BYTES_PER_FRAME  # 32_000

# --- Capture limits ----------------------------------------------------------

MEMORY_THRESHOLD_MINUTES = 10
"""PCM stays in memory (no disk) for the first 10 minutes."""

MEMORY_THRESHOLD_BYTES = MEMORY_THRESHOLD_MINUTES * 60 * BYTES_PER_SECOND

MAX_RECORDING_MINUTES = 30
"""Non-configurable safety upper bound; capture always stops at 30 minutes."""

MAX_RECORDING_SECONDS = MAX_RECORDING_MINUTES * 60

NOTIFY_AT_MINUTES = 25
"""Fire the (injected) notification callback at 25 minutes."""

SHARD_SECONDS = 60
SHARD_BYTES = SHARD_SECONDS * BYTES_PER_SECOND

MIN_DURATION_MS = 300
"""Minimum valid recording duration."""

SHARD_DIR_NAME = "capture"
"""Subdirectory of the runtime dir that only ever holds this task's shards."""

# pw-record 1.6.4 exits ``1`` on normal completion (verified empirically: both
# reaching ``-n`` and being stopped by SIGINT return 1).  The task-specified
# codes ``0`` / ``130`` / negative-SIGINT remain accepted for other versions.
_NORMAL_EXIT_CODES = frozenset({0, 1, 130, -signal.SIGINT})

_CONTAINER_MAGICS = (b"RIFF", b"FORM")  # WAV / AIFF container headers


class CaptureError(RuntimeError):
    """Raised when capture fails, yields no valid audio, or misbehaves."""


@dataclass(frozen=True)
class CaptureConfig:
    """Per-recording capture settings.

    ``source`` is passed verbatim to ``pw-record --target``.  There is no
    implicit fallback between the default source and any effect source such as
    the rnnoise effect output: only a user-configured value selects it.
    """

    source: str = "default"


class CaptureProcess(Protocol):
    """Minimal subprocess surface the recorder needs (testable via fakes)."""

    stdout: BinaryIO | None
    returncode: int | None

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def send_signal(self, sig: int) -> None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


def _default_spawn(argv: Sequence[str]) -> CaptureProcess:
    """Spawn ``pw-record`` with stdout piped and stderr discarded."""
    return cast(
        CaptureProcess,
        subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL),
    )


def _create_memory_backed_file() -> BinaryIO:
    """Return an anonymous memory-backed (tmpfs) temp file, else anonymous."""
    shm = "/dev/shm"
    if os.path.isdir(shm):
        try:
            return tempfile.TemporaryFile(dir=shm)
        except OSError:
            pass
    return tempfile.TemporaryFile()



class PipeWireRecorder:
    """Runs ``pw-record`` and accumulates raw s16le PCM with bounded memory."""

    def __init__(
        self,
        *,
        pw_record: str = "pw-record",
        notifier: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Sequence[str]], CaptureProcess] | None = None,
        runtime_dir: Path | None = None,
        memory_threshold_bytes: int = MEMORY_THRESHOLD_BYTES,
    ) -> None:
        self._pw_record = pw_record
        self._notifier = notifier
        self._clock = clock
        self._spawn = spawn if spawn is not None else _default_spawn
        self._runtime_dir = runtime_dir
        self._memory_threshold_bytes = memory_threshold_bytes

        self._proc: CaptureProcess | None = None
        self._reader: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._memory = bytearray()
        self._shard_dir: Path | None = None
        self._shards: list[Path] = []
        self._current_shard_file: BinaryIO | None = None
        self._current_shard_size = 0
        self._bytes = 0
        self._started_mono = 0.0
        self._notified = False
        self._auto_stopped = False

        self._backing_files: list[BinaryIO] = []

    # --- Public lifecycle ----------------------------------------------------

    def start(self, config: CaptureConfig | None = None) -> None:
        """Begin recording.  Raises :class:`CaptureError` on failure."""
        if self._proc is not None:
            raise CaptureError("capture already in progress")
        self._close_backing_files()  # previous artifacts are superseded
        self._cleanup_leftover()
        self._reset_capture_state()

        cfg = config if config is not None else CaptureConfig()
        self._started_mono = self._clock()
        try:
            self._proc = self._spawn(self._build_argv(cfg))
        except OSError as exc:
            self._proc = None
            raise CaptureError(f"failed to start {self._pw_record}: {exc}") from exc

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._reader.start()
        self._watchdog.start()

    def stop(self) -> CaptureArtifact:
        """Stop recording and return the collected audio artifact."""
        if self._proc is None:
            raise CaptureError("capture not started")
        proc = self._proc
        self._request_stop()
        self._await_exit(proc)
        if self._reader is not None:
            self._reader.join(timeout=5.0)
        return self._finalize(proc)

    def cancel(self) -> None:
        """Abort recording and clean up without producing an artifact."""
        if self._proc is None:
            return
        self.cleanup()

    def cleanup(self) -> None:
        """Terminate capture and remove exactly this task's files.  Idempotent."""
        self._stop_event.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.terminate()
            self._await_exit(proc)
        if self._reader is not None:
            self._reader.join(timeout=5.0)
        self._close_current_shard()
        self._cleanup_shards()
        self._close_backing_files()
        self._proc = None
        self._reader = None
        self._watchdog = None

    # --- Command construction -------------------------------------------------

    def _build_argv(self, config: CaptureConfig) -> list[str]:
        return [
            self._pw_record,
            "--rate",
            str(SAMPLE_RATE),
            "--channels",
            str(CHANNELS),
            "--format",
            "s16",
            "--media-type",
            "Audio",
            "--raw",
            "--target",
            config.source,
            "-",
        ]

    # --- Reader thread --------------------------------------------------------

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        stdout = proc.stdout
        if stdout is None:
            return
        try:
            while True:
                chunk = stdout.read(8192)
                if not chunk:
                    break
                self._write_audio(chunk)
        except (OSError, ValueError):
            # stdout closed or read error mid-stream; whatever we have stands.
            pass

    def _write_audio(self, chunk: bytes) -> None:
        while chunk:
            if self._shard_dir is None and self._bytes < self._memory_threshold_bytes:
                room = self._memory_threshold_bytes - self._bytes
                take = min(room, len(chunk))
                self._memory.extend(chunk[:take])
                self._bytes += take
                chunk = chunk[take:]
            else:
                if self._shard_dir is None:
                    self._ensure_shard_dir()
                self._append_to_shard(chunk)
                return

    def _append_to_shard(self, chunk: bytes) -> None:
        while chunk:
            if (
                self._current_shard_file is None
                or self._current_shard_size >= SHARD_BYTES
            ):
                self._rotate_shard()
            room = SHARD_BYTES - self._current_shard_size
            take = min(room, len(chunk))
            assert self._current_shard_file is not None
            self._current_shard_file.write(chunk[:take])
            self._current_shard_size += take
            self._bytes += take
            chunk = chunk[take:]

    def _rotate_shard(self) -> None:
        self._close_current_shard()
        assert self._shard_dir is not None
        # 60-second timestamp naming: monotonic, no gaps, no overlaps.
        start_seconds = self._bytes // BYTES_PER_SECOND
        path = self._shard_dir / f"shard-{start_seconds:06d}.pcm"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
        self._current_shard_file = os.fdopen(fd, "wb", buffering=0)
        self._current_shard_size = 0
        self._shards.append(path)

    def _close_current_shard(self) -> None:
        if self._current_shard_file is not None:
            self._current_shard_file.close()
            self._current_shard_file = None
            self._current_shard_size = 0

    # --- Shard directory ------------------------------------------------------

    def _shard_dir_path(self) -> Path:
        if self._runtime_dir is not None:
            return self._runtime_dir / SHARD_DIR_NAME
        try:
            runtime_dir = resolve_runtime_dir()
        except ConfigError as exc:
            raise CaptureError(f"cannot resolve runtime directory: {exc}") from exc
        return runtime_dir / SHARD_DIR_NAME

    def _ensure_shard_dir(self) -> None:
        if self._shard_dir is not None:
            return
        shard_dir = self._shard_dir_path()
        shard_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(shard_dir, DIRECTORY_MODE)
        self._shard_dir = shard_dir

    # --- Watchdog thread ------------------------------------------------------

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            elapsed = self._clock() - self._started_mono
            if elapsed >= MAX_RECORDING_SECONDS:
                self._auto_stop()
                return
            if not self._notified and elapsed >= NOTIFY_AT_MINUTES * 60:
                self._notified = True
                self._fire_notifier(elapsed)
            self._stop_event.wait(0.05)

    def _fire_notifier(self, elapsed: float) -> None:
        if self._notifier is None:
            return
        remaining = MAX_RECORDING_MINUTES - int(elapsed // 60)
        message = f"录音已持续 {int(elapsed // 60)} 分钟,{remaining} 分钟后将自动停止"
        with contextlib.suppress(Exception):
            self._notifier(message)

    def _auto_stop(self) -> None:
        self._auto_stopped = True
        self._stop_event.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.send_signal(signal.SIGINT)

    # --- Stop / finalize ------------------------------------------------------

    def _request_stop(self) -> None:
        self._stop_event.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.send_signal(signal.SIGINT)

    def _await_exit(self, proc: CaptureProcess) -> None:
        try:
            proc.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, OSError):
            with contextlib.suppress(OSError):
                proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, OSError):
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    proc.wait(timeout=2.0)

    def _finalize(self, proc: CaptureProcess) -> CaptureArtifact:
        try:
            self._close_current_shard()
            exit_code = proc.returncode
            duration_ms = int(self._bytes * 1000 / BYTES_PER_SECOND)
            self._validate(exit_code, duration_ms)
            return self._materialize(duration_ms)
        finally:
            self._cleanup_shards()
            self._proc = None
            self._reader = None
            self._watchdog = None

    def _validate(self, exit_code: int | None, duration_ms: int) -> None:
        if self._bytes == 0:
            raise CaptureError("captured no audio bytes")
        if duration_ms < MIN_DURATION_MS:
            raise CaptureError(
                f"capture too short: {duration_ms}ms < {MIN_DURATION_MS}ms"
            )
        if self._is_container_wrapped():
            raise CaptureError("unexpected container wrapper, expected raw s16le PCM")
        if exit_code not in _NORMAL_EXIT_CODES:
            raise CaptureError(f"pw-record exited abnormally (code {exit_code})")

    def _is_container_wrapped(self) -> bool:
        return bytes(self._memory[:4]) in _CONTAINER_MAGICS

    def _materialize(self, duration_ms: int) -> CaptureArtifact:
        out = _create_memory_backed_file()
        try:
            out.write(self._memory)
            for path in self._shards:
                with path.open("rb") as shard:
                    shutil.copyfileobj(shard, out)
            out.flush()
            fd = out.fileno()
        except BaseException:
            out.close()
            raise
        self._backing_files.append(out)
        return CaptureArtifact(
            audio=f"/proc/{os.getpid()}/fd/{fd}",
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=SAMPLE_FORMAT,
            duration_ms=duration_ms,
        )

    # --- Cleanup --------------------------------------------------------------

    def _cleanup_shards(self) -> None:
        self._close_current_shard()
        shard_dir = self._shard_dir
        self._shard_dir = None
        self._shards = []
        if shard_dir is None:
            return
        self._purge_shard_dir(shard_dir)

    def _purge_shard_dir(self, shard_dir: Path) -> None:
        """Remove shard files then the directory; never touches parent dirs."""
        for entry in shard_dir.glob("shard-*.pcm"):
            with contextlib.suppress(OSError):
                entry.unlink()
        with contextlib.suppress(OSError):
            shard_dir.rmdir()

    def _cleanup_leftover(self) -> None:
        """Remove a leftover capture directory from a previous crashed run."""
        try:
            shard_dir = self._shard_dir_path()
        except CaptureError:
            return
        if shard_dir.is_dir():
            self._purge_shard_dir(shard_dir)

    def _close_backing_files(self) -> None:
        for f in self._backing_files:
            with contextlib.suppress(OSError):
                f.close()
        self._backing_files.clear()

    def _reset_capture_state(self) -> None:
        self._stop_event.clear()
        self._memory = bytearray()
        self._shard_dir = None
        self._shards = []
        self._close_current_shard()
        self._bytes = 0
        self._notified = False
        self._auto_stopped = False

