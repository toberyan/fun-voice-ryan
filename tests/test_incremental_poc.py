"""Tests for the private, aggregate-only incremental Nano POC gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fun_voice import config as config_module
from fun_voice import incremental_poc as poc_module
from fun_voice import nano_runtime as nano_module
from fun_voice.config import Config
from fun_voice.contracts import Segment, Transcription, WorkerHealth
from fun_voice.incremental_poc import (
    DEVICE,
    NANO_MODEL_ID,
    POC_REPORT_NAME,
    IncrementalPocGate,
    IncrementalPocReport,
    _default_final_tail_probe,
    _default_runtime_factory,
    run_incremental_poc,
    run_local_incremental_poc,
    write_poc_report,
)
from fun_voice.nano_runtime import EmptySpeechError
from fun_voice.runtime_selection import RuntimeSelection


def _passing_report(*, created_at: int = 100) -> IncrementalPocReport:
    return IncrementalPocReport(
        created_at=created_at,
        model_id=NANO_MODEL_ID,
        model_revision="master",
        nano_device=DEVICE,
        vad_device=DEVICE,
        full_segment_count=2,
        incremental_segment_count=4,
        incremental_window_count=3,
        duplicate_boundary_count=0,
        reconciled_duplicate_boundary_count=1,
        final_text_equal=True,
        cer_milli=0,
        peak_xpu_memory_bytes=1024,
        final_tail_preemption_passed=True,
        timed_out=False,
        deadlocked=False,
    )


def _xpu_selection() -> RuntimeSelection:
    return RuntimeSelection(
        schema_version=1,
        backend="xpu",
        python=Path("/selected-runtime/bin/python"),
        device=DEVICE,
        dtype="bf16",
        primary_asr_profile="nano",
        fallback_asr_profile="sensevoice",
        enhanced_enabled=True,
        speaker_enabled=True,
        model_revisions={
            "nano": "master",
            "sensevoice": "master",
            "vad": "master",
            "qwen": "master",
            "campplus": "master",
        },
        probe_status="pass",
        selected_at=1,
    )


def _cpu_selection() -> RuntimeSelection:
    return RuntimeSelection(
        schema_version=1,
        backend="cpu",
        python=Path("/selected-runtime/bin/python"),
        device="cpu",
        dtype="float32",
        primary_asr_profile="sensevoice",
        fallback_asr_profile=None,
        enhanced_enabled=False,
        speaker_enabled=False,
        model_revisions={"sensevoice": "master", "vad": "master"},
        probe_status="pass",
        selected_at=1,
    )


def test_default_poc_runtime_factory_uses_selected_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _xpu_selection()
    runtime = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        poc_module, "load_runtime_selection", lambda: selection, raising=False
    )
    monkeypatch.setattr(
        poc_module,
        "config",
        SimpleNamespace(
            load_config=Config,
            effective_runtime_config=config_module.effective_runtime_config,
        ),
        raising=False,
    )

    def load_nano(*, selection: object, inference: object) -> object:
        captured["selection"] = selection
        captured["inference"] = inference
        return runtime

    monkeypatch.setattr(
        nano_module,
        "load_nano_runtime",
        load_nano,
    )

    assert _default_runtime_factory() is runtime
    assert captured["selection"] == selection
    assert captured["inference"].device == DEVICE
    assert captured["inference"].dtype == "bf16"


def test_default_poc_runtime_factory_rejects_non_xpu_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False
    monkeypatch.setattr(
        poc_module, "load_runtime_selection", _cpu_selection, raising=False
    )

    def load_nano(**_kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(nano_module, "load_nano_runtime", load_nano)

    with pytest.raises(RuntimeError, match="incremental POC requires selected XPU"):
        _default_runtime_factory()

    assert called is False


def test_poc_gate_allows_only_fresh_report_for_exact_model_device_and_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / POC_REPORT_NAME
    write_poc_report(path, _passing_report())

    gate = IncrementalPocGate(
        path,
        expected_model_id=NANO_MODEL_ID,
        expected_revision="master",
        expected_device=DEVICE,
        freshness_seconds=60,
        clock=lambda: 150,
    )

    assert gate.is_approved() is True
    assert gate.allows_provisional(config_enabled=True) is True


def test_failed_or_invalid_poc_reports_cannot_enable_provisional_transcription(
    tmp_path: Path,
) -> None:
    path = tmp_path / POC_REPORT_NAME
    failed = _passing_report()
    payload = failed.to_dict()
    payload["final_text_equal"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    gate = IncrementalPocGate(
        path,
        expected_model_id=NANO_MODEL_ID,
        expected_revision="master",
        expected_device=DEVICE,
        freshness_seconds=60,
        clock=lambda: 120,
    )

    assert gate.is_approved() is False
    assert gate.allows_provisional(config_enabled=True) is False
    assert gate.allows_provisional(config_enabled=False) is False


def test_poc_gate_rejects_stale_or_mismatched_report_without_exposing_contents(
    tmp_path: Path,
) -> None:
    path = tmp_path / POC_REPORT_NAME
    write_poc_report(path, _passing_report(created_at=10))

    stale = IncrementalPocGate(
        path,
        expected_model_id=NANO_MODEL_ID,
        expected_revision="master",
        expected_device=DEVICE,
        freshness_seconds=60,
        clock=lambda: 100,
    )
    wrong_revision = IncrementalPocGate(
        path,
        expected_model_id=NANO_MODEL_ID,
        expected_revision="another-revision",
        expected_device=DEVICE,
        freshness_seconds=120,
        clock=lambda: 100,
    )

    assert stale.is_approved() is False
    assert wrong_revision.is_approved() is False


def test_poc_report_is_aggregate_only_and_written_owner_only(tmp_path: Path) -> None:
    path = tmp_path / POC_REPORT_NAME
    write_poc_report(path, _passing_report())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["nano_device"] == DEVICE
    assert payload["vad_device"] == DEVICE
    assert payload["full_segment_count"] == 2
    assert payload["incremental_segment_count"] == 4
    assert payload["incremental_window_count"] == 3
    assert payload["duplicate_boundary_count"] == 0
    assert payload["reconciled_duplicate_boundary_count"] == 1
    assert {"text", "transcript", "audio", "path", "session_id"}.isdisjoint(payload)
    assert (path.stat().st_mode & 0o777) == 0o600


def test_incremental_poc_publishes_only_the_fake_runner_aggregate(
    tmp_path: Path,
) -> None:
    path = tmp_path / POC_REPORT_NAME
    calls: list[None] = []

    report = run_incremental_poc(
        path,
        runner=lambda: calls.append(None) or _passing_report(created_at=123),
    )

    assert calls == [None]
    assert report.ready is True
    assert json.loads(path.read_text(encoding="utf-8"))["created_at"] == 123


def test_local_poc_runner_uses_fake_runtime_and_keeps_text_out_of_report(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.pcm").touch()

    class FakeRuntime:
        device = DEVICE

        def __init__(self) -> None:
            self.closed = False

        def health(self) -> WorkerHealth:
            return WorkerHealth(
                version="test", xpu_ready=True, model_ready=True, device=DEVICE
            )

        def device_evidence(self) -> tuple[str, str]:
            return DEVICE, DEVICE

        def transcribe(self, _audio: str, *, sample_rate: int) -> Transcription:
            return Transcription("#", segments=(Segment(0, 100, "#"),))

        def transcribe_samples(
            self, _samples: list[int], *, sample_rate: int
        ) -> Transcription:
            return Transcription("#", segments=(Segment(0, 100, "#"),))

        def close(self) -> None:
            self.closed = True

    runtime = FakeRuntime()
    report = run_local_incremental_poc(
        tmp_path,
        model_revision="master",
        runtime_factory=lambda: runtime,
        audio_loader=lambda _path, _rate: [0] * (16000 * 3),
        peak_memory_bytes=lambda: 4096,
        final_tail_probe=lambda _runtime, _samples: (True, False, False),
        clock=lambda: 200,
    )

    assert report.ready is True
    assert report.full_segment_count == 1
    assert report.incremental_window_count == 3
    assert report.incremental_segment_count == 3
    assert report.reconciled_duplicate_boundary_count == 2
    assert report.duplicate_boundary_count == 0
    assert runtime.closed is True
    assert "#" not in json.dumps(report.to_dict())


def test_local_poc_skips_silent_incremental_windows(tmp_path: Path) -> None:
    (tmp_path / "sample.pcm").touch()

    class FakeRuntime:
        device = DEVICE

        def __init__(self) -> None:
            self.window_calls = 0

        def health(self) -> WorkerHealth:
            return WorkerHealth(
                version="test", xpu_ready=True, model_ready=True, device=DEVICE
            )

        def device_evidence(self) -> tuple[str, str]:
            return DEVICE, DEVICE

        def transcribe(self, _audio: str, *, sample_rate: int) -> Transcription:
            return Transcription("#", segments=(Segment(0, 100, "#"),))

        def transcribe_samples(
            self, _samples: list[int], *, sample_rate: int
        ) -> Transcription:
            self.window_calls += 1
            if self.window_calls == 2:
                raise EmptySpeechError("silent window")
            return Transcription("#", segments=(Segment(0, 100, "#"),))

        def close(self) -> None:
            pass

    probe_inputs: list[list[int]] = []
    report = run_local_incremental_poc(
        tmp_path,
        model_revision="master",
        runtime_factory=FakeRuntime,
        audio_loader=lambda _path, _rate: [0] * (16000 * 3),
        peak_memory_bytes=lambda: 4096,
        final_tail_probe=lambda _runtime, samples: probe_inputs.append(samples)
        or (True, False, False),
        clock=lambda: 200,
    )

    assert report.ready is True
    assert report.incremental_window_count == 2
    assert report.incremental_segment_count == 2
    assert [len(samples) for samples in probe_inputs] == [24_000]
    assert "#" not in json.dumps(report.to_dict())


def test_local_poc_skips_whole_utterances_without_speech(tmp_path: Path) -> None:
    (tmp_path / "silent.pcm").touch()

    class FakeRuntime:
        device = DEVICE

        def __init__(self) -> None:
            self.closed = False

        def health(self) -> WorkerHealth:
            return WorkerHealth(
                version="test", xpu_ready=True, model_ready=True, device=DEVICE
            )

        def device_evidence(self) -> tuple[str, str]:
            return DEVICE, DEVICE

        def transcribe(self, _audio: str, *, sample_rate: int) -> Transcription:
            raise EmptySpeechError("silent utterance")

        def transcribe_samples(
            self, _samples: list[int], *, sample_rate: int
        ) -> Transcription:
            raise AssertionError("silent utterance must not reach incremental decoding")

        def close(self) -> None:
            self.closed = True

    runtime = FakeRuntime()
    report = run_local_incremental_poc(
        tmp_path,
        model_revision="master",
        runtime_factory=lambda: runtime,
        audio_loader=lambda _path, _rate: [0] * 16_000,
        peak_memory_bytes=lambda: 4096,
        final_tail_probe=lambda _runtime, _samples: (True, False, False),
        clock=lambda: 200,
    )

    assert report.ready is False
    assert report.full_segment_count == 0
    assert report.incremental_window_count == 0
    assert runtime.closed is True


def test_local_poc_runner_rejects_unverified_vad_device_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.pcm").touch()

    class BadEvidenceRuntime:
        device = DEVICE

        def __init__(self) -> None:
            self.closed = False

        def health(self) -> WorkerHealth:
            return WorkerHealth(
                version="test", xpu_ready=True, model_ready=True, device=DEVICE
            )

        def device_evidence(self) -> tuple[str, str]:
            return DEVICE, "cpu:0"

        def close(self) -> None:
            self.closed = True

    runtime = BadEvidenceRuntime()
    with pytest.raises(RuntimeError, match="VAD"):
        run_local_incremental_poc(
            tmp_path,
            model_revision="master",
            runtime_factory=lambda: runtime,
            audio_loader=lambda _path, _rate: [0],
            peak_memory_bytes=lambda: 4096,
            final_tail_probe=lambda _runtime, _samples: (True, False, False),
        )

    assert runtime.closed is True


def test_poc_main_publishes_the_injected_local_runner_aggregate(
    monkeypatch, tmp_path: Path
) -> None:
    report_path = tmp_path / POC_REPORT_NAME
    captured: dict[str, object] = {}

    def fake_local_runner(corpus: Path, **kwargs: object) -> IncrementalPocReport:
        captured["corpus"] = corpus
        captured.update(kwargs)
        return _passing_report(created_at=321)

    monkeypatch.setattr(poc_module, "run_local_incremental_poc", fake_local_runner)

    assert (
        poc_module.main(
            [
                "--report",
                str(report_path),
                "--corpus",
                str(tmp_path),
                "--revision",
                "master",
            ]
        )
        == 0
    )
    assert captured["corpus"] == tmp_path
    assert captured["model_revision"] == "master"
    assert json.loads(report_path.read_text(encoding="utf-8"))["ready"] is True


def test_final_tail_probe_runs_final_before_queued_provisional_window() -> None:
    class FakeRuntime:
        calls = 0

        def transcribe_samples(self, _samples: list[int], *, sample_rate: int) -> None:
            self.calls += 1
            raise AssertionError("scheduler probe must not decode user audio")

    runtime = FakeRuntime()

    assert _default_final_tail_probe(runtime, [0] * 16_000) == (True, False, False)
    assert runtime.calls == 0
