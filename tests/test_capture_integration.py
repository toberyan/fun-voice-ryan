"""Integration test: real ``pw-record`` capture, gated behind ``CI_AUDIO=1``.

Only runs when ``pw-record`` is on ``PATH`` and ``CI_AUDIO=1`` is set in the
environment; every other environment is skipped.  Records one second and
validates the raw s16le mono 16 kHz sample format.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from fun_voice.capture import (
    BYTES_PER_SECOND,
    SAMPLE_FORMAT,
    SAMPLE_RATE,
    PipeWireRecorder,
)

_PW_RECORD = shutil.which("pw-record") is not None
_CI_AUDIO = os.environ.get("CI_AUDIO") == "1"

pytestmark = pytest.mark.skipif(
    not (_PW_RECORD and _CI_AUDIO),
    reason="requires pw-record on PATH and CI_AUDIO=1",
)


def test_record_one_second_validates_format(tmp_path: Path) -> None:
    recorder = PipeWireRecorder(runtime_dir=tmp_path)
    recorder.start()
    time.sleep(1.0)
    artifact = recorder.stop()

    assert artifact.sample_rate == SAMPLE_RATE
    assert artifact.channels == 1
    assert artifact.format == SAMPLE_FORMAT

    assert artifact.duration_ms is not None
    # ~1 second, allowing a generous margin for SIGINT teardown latency.
    assert 400 <= artifact.duration_ms <= 2500

    with open(artifact.audio, "rb") as f:
        data = f.read()
    assert data
    assert len(data) % 2 == 0  # s16le frame alignment
    assert not data.startswith(b"RIFF")  # raw PCM, not a WAV container
    # Byte count is consistent with the declared duration (no wild mismatch).
    assert abs(len(data) - artifact.duration_ms * BYTES_PER_SECOND // 1000) < 4096
