"""Aggregate-only scoring tests for the explicit local benchmark command."""

from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fun_voice import benchmark as benchmark_module
from fun_voice.benchmark import (
    _build_selected_benchmark_runtime,
    _write_report,
    aggregate_scores,
    score_text,
)
from fun_voice.contracts import CaptureArtifact, Transcription
from fun_voice.runtime_selection import RuntimeSelection


def test_score_text_calculates_cer_terms_and_punctuation_without_text() -> None:
    score = score_text("运行 git commit。", "运行 git commit", ("git", "commit"))

    assert score["cer"] > 0
    assert score["term_exact"] == 1.0
    assert score["punctuation_f1"] == 0.0
    assert "运行" not in repr(score)


def test_aggregate_groups_category_without_audio_or_reference() -> None:
    report = aggregate_scores([("mixed", {"cer": 0.1, "term_exact": 1.0})])

    assert report["categories"]["mixed"]["count"] == 1
    assert "audio" not in repr(report)
    assert "reference" not in repr(report)


def test_explicit_report_is_owner_only_and_contains_aggregates(tmp_path: Path) -> None:
    report_path = tmp_path / "benchmark.json"
    _write_report(report_path, {"count": 1, "categories": {"mixed": {"count": 1}}})

    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert report_path.read_text(encoding="utf-8") == (
        '{"categories": {"mixed": {"count": 1}}, "count": 1}\n'
    )


def _selection(backend: str) -> RuntimeSelection:
    cpu = backend == "cpu"
    return RuntimeSelection(
        schema_version=1,
        backend=backend,  # type: ignore[arg-type]
        python=Path("/runtime/bin/python"),
        device="cpu" if cpu else f"{backend}:0",
        dtype="float32" if cpu else "bf16",
        primary_asr_profile="sensevoice" if cpu else "nano",
        fallback_asr_profile=None if cpu else "sensevoice",
        enhanced_enabled=not cpu,
        speaker_enabled=not cpu,
        model_revisions=(
            {"sensevoice": "master", "vad": "master"}
            if cpu
            else {
                "nano": "master",
                "sensevoice": "master",
                "vad": "master",
                "qwen": "master",
                "campplus": "master",
            }
        ),
        probe_status="pass",
        selected_at=1,
    )


@pytest.mark.parametrize(
    ("backend", "profile", "socket_name"),
    [
        ("cpu", "sensevoice", "worker-sensevoice.sock"),
        ("cuda", "nano", "worker.sock"),
    ],
)
def test_benchmark_uses_selected_profile_and_daemon_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    profile: str,
    socket_name: str,
) -> None:
    workers: list[Any] = []
    scheduler_calls: list[tuple[str, object]] = []

    class FakeWorker:
        def __init__(self, socket_path: Path, **kwargs: object) -> None:
            self.socket_path = Path(socket_path)
            self.profile = kwargs["profile"]
            self.auto_start_service = kwargs["auto_start_service"]
            workers.append(self)

        def health(self) -> object:
            return SimpleNamespace(lifecycle="ready")

        def transcribe(self, artifact: CaptureArtifact) -> Transcription:
            return Transcription(text="ok")

        def close(self) -> None:
            pass

    class FakeSupervisor:
        def __init__(self, **kwargs: object) -> None:
            self.workers = kwargs["workers"]

        def start_profile(self, _profile: str) -> bool:
            return True

        def stop_profile(self, _profile: str) -> bool:
            return True

        def health_profile(self, _profile: str) -> object:
            return SimpleNamespace(value="inactive")

        def transport_profile(self, _profile: str) -> object:
            return SimpleNamespace(value="ready")

    class FakeHandle:
        def wait(self, timeout: float | None = None) -> bool:
            return True

        def result(self) -> object:
            return Transcription(text="ok")

    class FakeScheduler:
        def __init__(self, **kwargs: object) -> None:
            scheduler_calls.append(("allowed", kwargs["allowed_profiles"]))

        def activate(self, key: object) -> None:
            scheduler_calls.append(("activate", key))

        def run_asr(
            self, key: object, selected_profile: str, fn: object
        ) -> FakeHandle:
            scheduler_calls.append((selected_profile, key))
            assert callable(fn)
            fn()
            return FakeHandle()

        def close(self) -> None:
            scheduler_calls.append(("close", None))

    monkeypatch.setattr(benchmark_module, "SocketWorkerClient", FakeWorker)
    monkeypatch.setattr(
        benchmark_module, "SystemdModelProfileSupervisor", FakeSupervisor
    )
    monkeypatch.setattr(benchmark_module, "ModelScheduler", FakeScheduler)

    paths = SimpleNamespace(
        runtime_dir=tmp_path, worker_socket=tmp_path / "worker.sock"
    )
    runtime = _build_selected_benchmark_runtime(
        _selection(backend),
        paths=paths,
        user_config=benchmark_module.config.Config(),
    )
    try:
        result = runtime.transcribe(CaptureArtifact(audio="/private/audio.wav"))
    finally:
        runtime.close()

    assert result.text == "ok"
    primary = next(worker for worker in workers if worker.profile == profile)
    assert primary.socket_path.name == socket_name
    assert primary.auto_start_service is False
    assert scheduler_calls[0] == (
        "allowed",
        ("sensevoice",) if backend == "cpu" else ("nano", "sensevoice"),
    )
    assert any(call[0] == profile for call in scheduler_calls)
