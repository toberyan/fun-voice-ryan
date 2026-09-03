from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fun_voice.backend_probe import ProbeResult
from fun_voice.bootstrap import (
    CommandResult,
    InitializationError,
    InitializationOptions,
    candidate_backends,
    run_initialization,
)
from fun_voice.runtime_selection import (
    RuntimeSelection,
    load_runtime_selection,
    selection_path,
    write_runtime_selection,
)


class FakeRunner:
    def __init__(self, results: dict[str, ProbeResult]) -> None:
        self.results = results
        self.calls: list[tuple[tuple[str, ...], dict[str, str] | None]] = []
        self.probed: list[str] = []

    @classmethod
    def all_fail(cls) -> FakeRunner:
        return cls({name: fail("unavailable") for name in candidate_backends("auto")})

    def run(
        self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
    ) -> CommandResult:
        self.calls.append((argv, env))
        if argv[:3] == ("systemctl", "--user", "show"):
            return CommandResult(0, "inactive\n")
        if argv[:3] == ("systemctl", "--user", "is-enabled"):
            return CommandResult(1, "disabled\n")
        if "create-runtime-env.sh" in argv[0]:
            backend = argv[argv.index("--backend") + 1]
            runtime = Path(argv[argv.index("--runtime-dir") + 1])
            (runtime / "bin").mkdir(parents=True, mode=0o700, exist_ok=True)
            (runtime.parent).chmod(0o700)
            runtime.chmod(0o700)
            (runtime / "bin").chmod(0o700)
            python = runtime / "bin/python"
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o700)
            return CommandResult(0, "")
        if "fun_voice.backend_probe" in argv:
            backend = argv[argv.index("--backend") + 1]
            self.probed.append(backend)
            result = replace(self.results[backend], backend=backend)
            models_root = Path(argv[argv.index("--models-root") + 1])
            cache_names = {
                "nano": "FunAudioLLM--Fun-ASR-Nano-2512",
                "sensevoice": "iic--SenseVoiceSmall",
                "vad": "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "qwen": "Qwen--Qwen3.5-0.8B",
                "campplus": "iic--speech_campplus_sv_zh-cn_16k-common",
            }
            for key in result.models:
                snapshot = (
                    models_root
                    / "models"
                    / cache_names[key]
                    / "snapshots/master"
                )
                snapshot.mkdir(parents=True, mode=0o700, exist_ok=True)
                (snapshot / "configuration.json").write_text("{}", encoding="utf-8")
            return CommandResult(0 if result.status == "pass" else 1, result.to_json())
        return CommandResult(0, "")


def passed(backend: str, dtype: str) -> ProbeResult:
    models = (
        {"sensevoice": "master", "vad": "master"}
        if backend == "cpu"
        else {
            "nano": "master",
            "sensevoice": "master",
            "vad": "master",
            "qwen": "master",
            "campplus": "master",
        }
    )
    return ProbeResult(
        backend=backend,
        status="pass",
        error_category=None,
        dtype=dtype,
        models=models,
        tensor_ms=1,
        asr_ms=2,
    )


def fail(category: str) -> ProbeResult:
    return ProbeResult(
        backend="cpu",
        status="fail",
        error_category=category,
        dtype=None,
        models={},
        tensor_ms=0,
        asr_ms=0,
    )


def _options(
    backend: str,
    *,
    force: bool = False,
    root: Path,
    dry_run: bool = False,
) -> InitializationOptions:
    return InitializationOptions(
        backend=backend,  # type: ignore[arg-type]
        force_reselect=force,
        dry_run=dry_run,
        data_root=root,
        project_root=Path(__file__).resolve().parents[1],
    )


@pytest.fixture
def desktop_prerequisites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("uv", "cmake", "pkg-config", "pw-cli", "fcitx5-remote"):
        executable = tools / name
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o700)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "Deepin;DDE")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "data" / "fun-voice-ryan"


def test_auto_tries_cuda_xpu_cpu_in_priority_order(
    desktop_prerequisites: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = FakeRunner(
        {"cuda": fail("tensor"), "xpu": fail("asr"), "cpu": passed("cpu", "float32")}
    )
    selected = run_initialization(
        _options("auto", root=desktop_prerequisites), runner=runner
    )
    assert runner.probed == ["cuda", "xpu", "cpu"]
    assert selected.backend == "cpu"
    output = capsys.readouterr().out
    assert '"error_category":"tensor"' in output
    assert '"error_category":"asr"' in output


@pytest.mark.parametrize("backend", ["cuda", "xpu", "cpu"])
def test_explicit_backend_does_not_fall_through(
    backend: str, desktop_prerequisites: Path
) -> None:
    result = fail("unavailable")
    runner = FakeRunner({backend: result})
    with pytest.raises(InitializationError, match="selected backend failed"):
        run_initialization(_options(backend, root=desktop_prerequisites), runner=runner)
    assert runner.probed == [backend]


def _write_selection(root: Path, backend: str) -> RuntimeSelection:
    runtime = root / "runtimes" / backend / "bin"
    runtime.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    (root / "runtimes").chmod(0o700)
    (root / "runtimes" / backend).chmod(0o700)
    runtime.chmod(0o700)
    python = runtime / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o700)
    selection = RuntimeSelection(
        schema_version=1,
        backend=backend,  # type: ignore[arg-type]
        python=python,
        device="cpu" if backend == "cpu" else f"{backend}:0",
        dtype="float32" if backend == "cpu" else "bf16",
        primary_asr_profile="sensevoice" if backend == "cpu" else "nano",
        fallback_asr_profile=None if backend == "cpu" else "sensevoice",
        enhanced_enabled=backend != "cpu",
        speaker_enabled=backend != "cpu",
        model_revisions=(
            {"sensevoice": "master", "vad": "master"}
            if backend == "cpu"
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
    write_runtime_selection(selection, root)
    return selection


def test_failed_force_reselect_keeps_existing_manifest(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "xpu")
    with pytest.raises(InitializationError):
        run_initialization(
            _options("auto", force=True, root=desktop_prerequisites),
            runner=FakeRunner.all_fail(),
        )
    assert load_runtime_selection(desktop_prerequisites) == previous


def test_existing_selection_is_returned_without_subprocess(
    desktop_prerequisites: Path,
) -> None:
    expected = _write_selection(desktop_prerequisites, "cpu")
    runner = FakeRunner.all_fail()
    assert (
        run_initialization(
            _options("auto", root=desktop_prerequisites), runner=runner
        )
        == expected
    )
    assert runner.calls == []


def test_historical_group_writable_app_root_is_migrated_before_lock(
    desktop_prerequisites: Path,
) -> None:
    expected = _write_selection(desktop_prerequisites, "cpu")
    desktop_prerequisites.chmod(0o775)
    runner = FakeRunner.all_fail()

    selected = run_initialization(
        _options("auto", root=desktop_prerequisites), runner=runner
    )

    assert selected == expected
    assert desktop_prerequisites.stat().st_mode & 0o777 == 0o700
    assert runner.calls == []


def test_data_root_permission_migration_never_follows_a_symlink(
    desktop_prerequisites: Path,
) -> None:
    outside = desktop_prerequisites.parent / "outside-data"
    outside.mkdir(parents=True, mode=0o775)
    outside.chmod(0o775)
    desktop_prerequisites.symlink_to(outside, target_is_directory=True)
    runner = FakeRunner.all_fail()

    with pytest.raises(InitializationError, match="^lock$"):
        run_initialization(
            _options("auto", root=desktop_prerequisites), runner=runner
        )

    assert outside.stat().st_mode & 0o777 == 0o775
    assert runner.calls == []


def test_accelerator_and_cpu_probe_model_policies(
    desktop_prerequisites: Path,
) -> None:
    for backend, dtype in (("cuda", "bf16"), ("xpu", "bf16"), ("cpu", "float32")):
        root = desktop_prerequisites.parent / backend
        runner = FakeRunner({backend: passed(backend, dtype)})
        selected = run_initialization(_options(backend, root=root), runner=runner)
        probe_argv, probe_env = next(
            call for call in runner.calls if "fun_voice.backend_probe" in call[0]
        )
        assert "--models-root" in probe_argv
        if backend == "cpu":
            combined = " ".join(probe_argv) + json.dumps(probe_env or {})
            assert "qwen" not in combined.lower()
            assert "campplus" not in combined.lower()
            assert set(selected.model_revisions) == {"sensevoice", "vad"}
        else:
            assert set(selected.model_revisions) == {
                "nano",
                "sensevoice",
                "vad",
                "qwen",
                "campplus",
            }
        assert all(
            "audio" not in key.lower() and "text" not in key.lower()
            for key in (probe_env or {})
        )


def test_dry_run_only_returns_candidate_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = FakeRunner.all_fail()
    result = run_initialization(_options("auto", root=tmp_path, dry_run=True), runner)
    assert result == ("cuda", "xpu", "cpu")
    assert json.loads(capsys.readouterr().out) == {
        "candidates": ["cuda", "xpu", "cpu"]
    }
    assert runner.calls == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DISPLAY", None),
        ("XDG_SESSION_TYPE", "wayland"),
        ("XDG_CURRENT_DESKTOP", "GNOME"),
        ("XDG_RUNTIME_DIR", None),
    ],
)
def test_missing_session_prerequisite_fails_before_subprocess(
    key: str,
    value: str | None,
    desktop_prerequisites: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    runner = FakeRunner.all_fail()
    with pytest.raises(InitializationError, match="^desktop_prerequisite$"):
        run_initialization(_options("auto", root=desktop_prerequisites), runner)
    assert runner.calls == []


@pytest.mark.parametrize(
    "missing", ["uv", "cmake", "pkg-config", "pw-cli", "fcitx5-remote"]
)
def test_missing_executable_fails_before_subprocess(
    missing: str,
    desktop_prerequisites: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_dir = Path(os.environ["PATH"])
    (tool_dir / missing).unlink()
    runner = FakeRunner.all_fail()
    with pytest.raises(InitializationError, match="^desktop_prerequisite$"):
        run_initialization(_options("auto", root=desktop_prerequisites), runner)
    assert runner.calls == []


def test_pipewire_socket_can_replace_pw_cli(
    desktop_prerequisites: Path,
) -> None:
    tool_dir = Path(os.environ["PATH"])
    (tool_dir / "pw-cli").unlink()
    runtime_dir = Path(os.environ["XDG_RUNTIME_DIR"])
    pipewire = socket.socket(socket.AF_UNIX)
    pipewire.bind(str(runtime_dir / "pipewire-0"))
    try:
        runner = FakeRunner({"cpu": passed("cpu", "float32")})
        selected = run_initialization(
            _options("cpu", root=desktop_prerequisites), runner
        )
        assert selected.backend == "cpu"
    finally:
        pipewire.close()


def test_insecure_runtime_directory_fails_before_subprocess(
    desktop_prerequisites: Path,
) -> None:
    Path(os.environ["XDG_RUNTIME_DIR"]).chmod(0o755)
    runner = FakeRunner.all_fail()
    with pytest.raises(InitializationError, match="^desktop_prerequisite$"):
        run_initialization(_options("auto", root=desktop_prerequisites), runner)
    assert runner.calls == []


def test_unwritable_data_parent_fails_before_subprocess(
    desktop_prerequisites: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fun_voice.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap.tempfile, "NamedTemporaryFile", _deny_tempfile)
    runner = FakeRunner.all_fail()
    with pytest.raises(InitializationError, match="^desktop_prerequisite$"):
        run_initialization(_options("auto", root=desktop_prerequisites), runner)
    assert runner.calls == []


def _deny_tempfile(*args: Any, **kwargs: Any) -> Any:
    raise PermissionError


def test_install_failure_restores_exact_previous_manifest(
    desktop_prerequisites: Path,
) -> None:
    _write_selection(desktop_prerequisites, "xpu")
    manifest = selection_path(desktop_prerequisites)
    previous_bytes = manifest.read_bytes()
    previous_mode = manifest.stat().st_mode & 0o777

    class InstallFailureRunner(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "install-user.sh" in argv[0]:
                return CommandResult(1, "")
            return result

    runner = InstallFailureRunner({"cpu": passed("cpu", "float32")})
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )
    assert manifest.read_bytes() == previous_bytes
    assert manifest.stat().st_mode & 0o777 == previous_mode
    assert not any(
        call[0]
        == ("systemctl", "--user", "restart", "fun-voice-daemon.service")
        for call in runner.calls
    )


def test_accelerator_to_cpu_reselection_stops_both_workers_before_daemon(
    desktop_prerequisites: Path,
) -> None:
    _write_selection(desktop_prerequisites, "xpu")
    worker_units = {
        "fun-voice-worker@nano.service",
        "fun-voice-worker@sensevoice.service",
    }

    class WorkerTransitionRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.active_workers = set(worker_units)
            self.daemon_active = True
            self.active_at_daemon_restart: set[str] | None = None
            self.daemon_was_quiesced = False

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if argv[:3] == ("systemctl", "--user", "stop") and len(argv) == 4:
                if argv[3] == "fun-voice-daemon.service":
                    self.daemon_active = False
                else:
                    self.active_workers.discard(argv[3])
            elif argv[:3] == ("systemctl", "--user", "show"):
                unit = argv[-1]
                active = (
                    self.daemon_active
                    if unit == "fun-voice-daemon.service"
                    else unit in self.active_workers
                )
                state = "active" if active else "inactive"
                return CommandResult(0, state + "\n")
            elif argv == (
                "systemctl",
                "--user",
                "restart",
                "fun-voice-daemon.service",
            ):
                self.active_at_daemon_restart = set(self.active_workers)
                self.daemon_was_quiesced = not self.daemon_active
                self.daemon_active = True
            return result

    runner = WorkerTransitionRunner()
    selected = run_initialization(
        _options("cpu", force=True, root=desktop_prerequisites), runner
    )

    assert selected.backend == "cpu"
    assert runner.active_at_daemon_restart == set()
    assert runner.daemon_was_quiesced
    commands = [argv for argv, _ in runner.calls]
    restart_index = commands.index(
        ("systemctl", "--user", "restart", "fun-voice-daemon.service")
    )
    daemon_stop = (
        "systemctl",
        "--user",
        "stop",
        "fun-voice-daemon.service",
    )
    daemon_confirm = (
        "systemctl",
        "--user",
        "show",
        "--property=ActiveState",
        "--value",
        "fun-voice-daemon.service",
    )
    daemon_stop_index = commands.index(daemon_stop)
    daemon_confirm_index = commands.index(daemon_confirm, daemon_stop_index + 1)
    assert daemon_stop_index < daemon_confirm_index
    for unit in sorted(worker_units):
        stop = ("systemctl", "--user", "stop", unit)
        confirm = (
            "systemctl",
            "--user",
            "show",
            "--property=ActiveState",
            "--value",
            unit,
        )
        assert (
            daemon_confirm_index
            < commands.index(stop)
            < commands.index(confirm, commands.index(stop) + 1)
            < restart_index
        )


def test_same_backend_new_generation_quiesces_both_workers_before_restart(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "xpu")

    class GenerationRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"xpu": passed("xpu", "bf16")})
            self.running = {
                "fun-voice-daemon.service",
                "fun-voice-worker@nano.service",
                "fun-voice-worker@sensevoice.service",
            }
            self.install_generation: str | None = None
            self.restart_running: set[str] | None = None

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "install-user.sh" in argv[0]:
                self.install_generation = load_runtime_selection(
                    desktop_prerequisites
                ).python.parent.parent.name
            elif argv[:3] == ("systemctl", "--user", "stop"):
                self.running.discard(argv[-1])
            elif argv[:3] == ("systemctl", "--user", "show"):
                return CommandResult(
                    0, "active\n" if argv[-1] in self.running else "inactive\n"
                )
            elif argv == (
                "systemctl",
                "--user",
                "restart",
                "fun-voice-daemon.service",
            ):
                self.restart_running = set(self.running)
                self.running.add("fun-voice-daemon.service")
            return result

    runner = GenerationRunner()
    selected = run_initialization(
        _options("xpu", force=True, root=desktop_prerequisites), runner
    )

    assert selected.python != previous.python
    assert runner.install_generation == selected.python.parent.parent.name
    assert runner.restart_running == set()
    commands = [argv for argv, _ in runner.calls]
    install_index = next(
        index for index, argv in enumerate(commands) if "install-user.sh" in argv[0]
    )
    restart_index = commands.index(
        ("systemctl", "--user", "restart", "fun-voice-daemon.service")
    )
    for unit in (
        "fun-voice-daemon.service",
        "fun-voice-worker@nano.service",
        "fun-voice-worker@sensevoice.service",
    ):
        stop = ("systemctl", "--user", "stop", unit)
        confirm = (
            "systemctl",
            "--user",
            "show",
            "--property=ActiveState",
            "--value",
            unit,
        )
        stop_index = commands.index(stop)
        confirm_index = commands.index(confirm, stop_index + 1)
        assert stop_index < confirm_index < install_index < restart_index


def test_restart_failure_restores_old_binding_manifest_and_active_state(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "xpu")

    class RestartFailureRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.daemon_active = True
            self.worker_active = {"fun-voice-worker@nano.service"}
            self.install_bindings: list[Path] = []
            self.restart_attempts = 0

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "install-user.sh" in argv[0]:
                self.install_bindings.append(
                    load_runtime_selection(desktop_prerequisites).python
                )
            elif argv[:3] == ("systemctl", "--user", "show"):
                unit = argv[-1]
                active = (
                    self.daemon_active
                    if unit == "fun-voice-daemon.service"
                    else unit in self.worker_active
                )
                return CommandResult(0, "active\n" if active else "inactive\n")
            elif argv[:3] == ("systemctl", "--user", "stop"):
                unit = argv[-1]
                if unit == "fun-voice-daemon.service":
                    self.daemon_active = False
                else:
                    self.worker_active.discard(unit)
            elif argv[:3] == ("systemctl", "--user", "start"):
                unit = argv[-1]
                if unit == "fun-voice-daemon.service":
                    self.daemon_active = True
                else:
                    self.worker_active.add(unit)
            elif argv == (
                "systemctl",
                "--user",
                "restart",
                "fun-voice-daemon.service",
            ):
                self.restart_attempts += 1
                if self.restart_attempts == 1:
                    return CommandResult(1, "")
                self.daemon_active = True
            return result

    runner = RestartFailureRunner()
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    assert load_runtime_selection(desktop_prerequisites) == previous
    assert len(runner.install_bindings) == 1
    assert runner.install_bindings[0] != previous.python
    assert runner.daemon_active is True
    assert runner.worker_active == {"fun-voice-worker@nano.service"}


def test_failed_restart_restores_previously_inactive_daemon_state(
    desktop_prerequisites: Path,
) -> None:
    class ActiveDespiteRestartFailure(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.daemon_active = False

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if argv[:3] == ("systemctl", "--user", "show"):
                active = argv[-1] == "fun-voice-daemon.service" and self.daemon_active
                return CommandResult(0, "active\n" if active else "inactive\n")
            if (
                argv[:3] == ("systemctl", "--user", "stop")
                and argv[-1] == "fun-voice-daemon.service"
            ):
                self.daemon_active = False
            if argv == (
                "systemctl",
                "--user",
                "restart",
                "fun-voice-daemon.service",
            ):
                self.daemon_active = True
                return CommandResult(1, "")
            return result

    runner = ActiveDespiteRestartFailure()
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", root=desktop_prerequisites), runner
        )

    assert runner.daemon_active is False
    assert not selection_path(desktop_prerequisites).exists()


def test_service_snapshot_failure_aborts_before_deployment_transaction(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "xpu")

    class SnapshotFailure(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if (
                argv[:3] == ("systemctl", "--user", "show")
                and argv[-1] == "fun-voice-daemon.service"
            ):
                return CommandResult(1, "")
            return result

    runner = SnapshotFailure({"cpu": passed("cpu", "float32")})
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    assert load_runtime_selection(desktop_prerequisites) == previous
    assert not (desktop_prerequisites / ".initialization-transaction.json").exists()
    assert not any(
        argv[:3] == ("systemctl", "--user", "stop") for argv, _ in runner.calls
    )


def test_failed_upgrade_restores_active_enabled_legacy_worker(
    desktop_prerequisites: Path,
) -> None:
    _write_selection(desktop_prerequisites, "xpu")
    legacy_unit = Path(os.environ["HOME"]) / (
        ".config/systemd/user/fun-voice-worker.service"
    )
    legacy_unit.parent.mkdir(parents=True, mode=0o700)
    legacy_unit.write_text("old legacy unit\n", encoding="utf-8")
    legacy_unit.chmod(0o600)

    class LegacyUpgradeFailure(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.legacy_active = True
            self.legacy_enabled = True

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if argv[:3] == ("systemctl", "--user", "show"):
                active = (
                    argv[-1] == "fun-voice-worker.service" and self.legacy_active
                )
                return CommandResult(0, "active\n" if active else "inactive\n")
            if argv[:3] == ("systemctl", "--user", "is-enabled"):
                enabled = (
                    argv[-1] == "fun-voice-worker.service" and self.legacy_enabled
                )
                return CommandResult(
                    0 if enabled else 1,
                    "enabled\n" if enabled else "disabled\n",
                )
            if "install-user.sh" in argv[0]:
                legacy_unit.unlink()
                self.legacy_active = False
                self.legacy_enabled = False
                return CommandResult(1, "")
            if (
                argv[:3] == ("systemctl", "--user", "enable")
                and argv[-1] == "fun-voice-worker.service"
            ):
                self.legacy_enabled = True
            if (
                argv[:3] == ("systemctl", "--user", "start")
                and argv[-1] == "fun-voice-worker.service"
            ):
                self.legacy_active = True
            return result

    runner = LegacyUpgradeFailure()
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    assert legacy_unit.read_text(encoding="utf-8") == "old legacy unit\n"
    assert runner.legacy_enabled is True
    assert runner.legacy_active is True


@pytest.mark.parametrize(
    "original_state",
    ["enabled", "enabled-runtime", "linked", "linked-runtime", "disabled"],
)
def test_failed_install_restores_exact_raw_daemon_enablement_state(
    desktop_prerequisites: Path,
    tmp_path: Path,
    original_state: str,
) -> None:
    _write_selection(desktop_prerequisites, "xpu")
    unit = "fun-voice-daemon.service"
    fragment = tmp_path / "unit-source" / unit
    fragment.parent.mkdir(mode=0o700)
    fragment.write_text("[Unit]\n", encoding="utf-8")

    class RawEnablementRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.states = {
                unit: original_state,
                "fun-voice-worker.service": "disabled",
            }

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if argv[:3] == ("systemctl", "--user", "is-enabled"):
                state = self.states[argv[-1]]
                return CommandResult(
                    0 if state in {"enabled", "enabled-runtime"} else 1,
                    state + "\n",
                )
            if argv[:3] == ("systemctl", "--user", "show"):
                if "--property=FragmentPath" in argv:
                    return CommandResult(0, str(fragment) + "\n")
                return CommandResult(0, "inactive\n")
            if "install-user.sh" in argv[0]:
                self.states[unit] = "disabled"
                return CommandResult(1, "")
            if argv[:3] == ("systemctl", "--user", "enable"):
                self.states[argv[-1]] = (
                    "enabled-runtime" if "--runtime" in argv else "enabled"
                )
            if argv[:3] == ("systemctl", "--user", "link"):
                self.states[unit] = (
                    "linked-runtime" if "--runtime" in argv else "linked"
                )
            return result

    runner = RawEnablementRunner()
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    assert runner.states[unit] == original_state
    commands = [argv for argv, _ in runner.calls]
    if original_state == "enabled-runtime":
        assert ("systemctl", "--user", "enable", "--runtime", unit) in commands
        assert ("systemctl", "--user", "enable", unit) not in commands
    elif original_state == "linked":
        assert ("systemctl", "--user", "link", str(fragment)) in commands
        assert not any(
            argv[:3] == ("systemctl", "--user", "enable")
            for argv in commands
        )
    elif original_state == "linked-runtime":
        assert (
            "systemctl",
            "--user",
            "link",
            "--runtime",
            str(fragment),
        ) in commands
        assert not any(
            argv[:3] == ("systemctl", "--user", "enable")
            for argv in commands
        )


def test_persistently_failing_install_restores_exact_launcher_and_daemon_state(
    desktop_prerequisites: Path,
) -> None:
    _write_selection(desktop_prerequisites, "xpu")
    launcher = Path(os.environ["HOME"]) / ".local/bin/fun-voice-daemon"
    launcher.parent.mkdir(parents=True, mode=0o700)
    launcher.write_bytes(b"#!/bin/sh\nexec /old/runtime\n")
    launcher.chmod(0o700)
    previous_bytes = launcher.read_bytes()

    class PersistentInstallFailure(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.daemon_active = True

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "install-user.sh" in argv[0]:
                launcher.write_bytes(b"#!/bin/sh\nexec /new/runtime\n")
                launcher.chmod(0o700)
                return CommandResult(1, "")
            if argv[:3] == ("systemctl", "--user", "show"):
                active = argv[-1] == "fun-voice-daemon.service" and self.daemon_active
                return CommandResult(0, "active\n" if active else "inactive\n")
            if (
                argv[:3] == ("systemctl", "--user", "stop")
                and argv[-1] == "fun-voice-daemon.service"
            ):
                self.daemon_active = False
            if (
                argv[:3] == ("systemctl", "--user", "start")
                and argv[-1] == "fun-voice-daemon.service"
            ):
                self.daemon_active = True
            return result

    runner = PersistentInstallFailure()
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    assert launcher.read_bytes() == previous_bytes
    assert launcher.stat().st_mode & 0o777 == 0o700
    assert runner.daemon_active is True


def test_unconfirmed_worker_quiescence_keeps_old_selection_unpublished(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "xpu")

    class QuiesceFailure(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"xpu": passed("xpu", "bf16")})
            self.install_calls = 0

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if argv == (
                "systemctl",
                "--user",
                "stop",
                "fun-voice-worker@sensevoice.service",
            ):
                return CommandResult(1, "")
            if "install-user.sh" in argv[0]:
                self.install_calls += 1
            return result

    runner = QuiesceFailure()
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("xpu", force=True, root=desktop_prerequisites), runner
        )

    assert load_runtime_selection(desktop_prerequisites) == previous
    assert runner.install_calls == 0


def test_legacy_worker_stop_failure_aborts_before_new_selection_is_published(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "xpu")

    class LegacyStopFailure(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.install_calls = 0

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if argv == (
                "systemctl",
                "--user",
                "stop",
                "fun-voice-worker.service",
            ):
                return CommandResult(1, "")
            if "install-user.sh" in argv[0]:
                self.install_calls += 1
            return result

    runner = LegacyStopFailure()
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    assert load_runtime_selection(desktop_prerequisites) == previous
    assert runner.install_calls == 0


def test_interrupt_during_install_rolls_back_before_next_run_can_accept_selection(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "xpu")
    previous_runtime = previous.python.parent.parent

    class InterruptRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.install_calls = 0

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "install-user.sh" in argv[0]:
                self.install_calls += 1
                if self.install_calls == 1:
                    raise KeyboardInterrupt
            return result

    runner = InterruptRunner()
    with pytest.raises(KeyboardInterrupt):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    assert runner.install_calls == 1
    assert load_runtime_selection(desktop_prerequisites) == previous
    assert set((desktop_prerequisites / "runtimes").iterdir()) == {
        previous_runtime
    }
    next_runner = FakeRunner.all_fail()
    assert run_initialization(
        _options("auto", root=desktop_prerequisites), next_runner
    ) == previous
    assert next_runner.calls == []


def test_failed_install_removes_the_prejournaled_promoted_runtime(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "cpu")

    class JournalObservingInstallFailure(FakeRunner):
        def __init__(self) -> None:
            super().__init__({"cpu": passed("cpu", "float32")})
            self.published_runtime: Path | None = None
            self.journal: dict[str, Any] | None = None

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "install-user.sh" in argv[0]:
                self.published_runtime = load_runtime_selection(
                    desktop_prerequisites
                ).python.parent.parent
                self.journal = json.loads(
                    (
                        desktop_prerequisites
                        / ".initialization-transaction.json"
                    ).read_text(encoding="utf-8")
                )
                return CommandResult(1, "")
            return result

    runner = JournalObservingInstallFailure()
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    assert runner.published_runtime is not None
    assert runner.published_runtime != previous.python.parent.parent
    assert runner.journal is not None
    assert runner.journal["runtime_promotion"]["destination"] == (
        runner.published_runtime.name
    )
    assert not runner.published_runtime.exists()
    assert load_runtime_selection(desktop_prerequisites) == previous


def test_next_run_recovers_model_snapshot_left_by_abrupt_interruption(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "xpu")
    manifest = selection_path(desktop_prerequisites)
    previous_bytes = manifest.read_bytes()
    previous_mode = manifest.stat().st_mode & 0o777
    token = "0123456789abcdef0123456789abcdef"
    snapshots = (
        desktop_prerequisites
        / "models/models/iic--SenseVoiceSmall/snapshots"
    )
    current = snapshots / "master"
    backup = snapshots / f".previous-master-{token}"
    current.mkdir(parents=True, mode=0o700)
    backup.mkdir(mode=0o700)
    (current / "configuration.json").write_text("new", encoding="utf-8")
    (backup / "configuration.json").write_text("old", encoding="utf-8")
    _write_selection(desktop_prerequisites, "cpu")

    import fun_voice.bootstrap as bootstrap

    bindings = [
        {
            "relative_path": relative,
            "kind": "missing",
            "content": None,
            "mode": None,
        }
        for relative in bootstrap._DEPLOYMENT_BINDINGS
    ]
    journal = {
        "version": 3,
        "previous_manifest": base64.b64encode(previous_bytes).decode("ascii"),
        "previous_mode": previous_mode,
        "active_units": [],
        "unit_files": [
            {
                "unit": "fun-voice-daemon.service",
                "state": "disabled",
                "fragment_path": None,
            },
            {
                "unit": "fun-voice-worker.service",
                "state": "disabled",
                "fragment_path": None,
            },
        ],
        "bindings": bindings,
        "model_promotions": [
            {
                "key": "sensevoice",
                "token": token,
                "had_destination": True,
            }
        ],
        "runtime_promotion": {
            "backend": "cpu",
            "token": token,
            "destination": f"cpu-{token}",
        },
    }
    journal_path = desktop_prerequisites / ".initialization-transaction.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    journal_path.chmod(0o600)

    runner = FakeRunner.all_fail()
    selected = run_initialization(
        _options("auto", root=desktop_prerequisites), runner
    )

    assert selected == previous
    assert (current / "configuration.json").read_text(encoding="utf-8") == "old"
    assert not backup.exists()
    assert not journal_path.exists()
    assert not any("create-runtime-env.sh" in argv[0] for argv, _ in runner.calls)


def test_native_build_failure_precedes_runtime_and_model_work(
    desktop_prerequisites: Path,
) -> None:
    class NativeFailureRunner(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "build-native-artifacts.sh" in argv[0]:
                return CommandResult(1, "")
            return result

    runner = NativeFailureRunner({"cpu": passed("cpu", "float32")})
    with pytest.raises(InitializationError, match="^native_prerequisite$"):
        run_initialization(_options("cpu", root=desktop_prerequisites), runner)

    commands = [argv for argv, _ in runner.calls]
    assert any("build-native-artifacts.sh" in argv[0] for argv in commands)
    assert not any("create-runtime-env.sh" in argv[0] for argv in commands)
    assert not selection_path(desktop_prerequisites).exists()


def test_failed_accelerator_model_candidates_do_not_pollute_cpu_cache(
    desktop_prerequisites: Path,
) -> None:
    unrelated = desktop_prerequisites / "models/models/user--private/snapshots/master"
    unrelated.mkdir(parents=True, mode=0o700)
    desktop_prerequisites.chmod(0o700)
    (unrelated / "keep").write_text("keep", encoding="utf-8")

    class CandidateCacheRunner(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            if "fun_voice.backend_probe" in argv:
                backend = argv[argv.index("--backend") + 1]
                models_root = Path(argv[argv.index("--models-root") + 1])
                names = (
                    (
                        "iic--SenseVoiceSmall",
                        "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    )
                    if backend == "cpu"
                    else (
                        "FunAudioLLM--Fun-ASR-Nano-2512",
                        "iic--SenseVoiceSmall",
                        "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        "Qwen--Qwen3.5-0.8B",
                        "iic--speech_campplus_sv_zh-cn_16k-common",
                    )
                )
                for name in names:
                    snapshot = models_root / "models" / name / "snapshots/master"
                    snapshot.mkdir(parents=True, mode=0o700)
                    (snapshot / "configuration.json").write_text(
                        "{}", encoding="utf-8"
                    )
            return super().run(argv, env=env)

    runner = CandidateCacheRunner(
        {
            "cuda": fail("tensor"),
            "xpu": fail("asr"),
            "cpu": passed("cpu", "float32"),
        }
    )
    selected = run_initialization(
        _options("auto", root=desktop_prerequisites), runner
    )

    assert selected.backend == "cpu"
    final_models = desktop_prerequisites / "models/models"
    assert (unrelated / "keep").read_text(encoding="utf-8") == "keep"
    assert {path.name for path in final_models.iterdir()} == {
        "user--private",
        "iic--SenseVoiceSmall",
        "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    }
    candidates = desktop_prerequisites / "model-candidates"
    assert not candidates.exists() or list(candidates.iterdir()) == []


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_probe_base_exception_cleans_candidate_runtime_and_model_cache(
    desktop_prerequisites: Path,
    interruption: BaseException,
) -> None:
    class InterruptedProbe(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "fun_voice.backend_probe" in argv:
                raise interruption
            return result

    with pytest.raises(type(interruption)):
        run_initialization(
            _options("cpu", root=desktop_prerequisites),
            InterruptedProbe({"cpu": passed("cpu", "float32")}),
        )

    runtimes = desktop_prerequisites / "runtimes"
    candidates = desktop_prerequisites / "model-candidates"
    assert not runtimes.exists() or list(runtimes.iterdir()) == []
    assert not candidates.exists() or list(candidates.iterdir()) == []


def test_parse_interruption_cleans_candidate_runtime_and_model_cache(
    desktop_prerequisites: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fun_voice.bootstrap as bootstrap

    def interrupt_parse(payload: str, expected: str) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(bootstrap, "_parse_probe", interrupt_parse)

    with pytest.raises(KeyboardInterrupt):
        run_initialization(
            _options("cpu", root=desktop_prerequisites),
            FakeRunner({"cpu": passed("cpu", "float32")}),
        )

    runtimes = desktop_prerequisites / "runtimes"
    candidates = desktop_prerequisites / "model-candidates"
    assert not runtimes.exists() or list(runtimes.iterdir()) == []
    assert not candidates.exists() or list(candidates.iterdir()) == []


def test_cpu_initialization_sweeps_stale_candidate_models(
    desktop_prerequisites: Path,
) -> None:
    stale = desktop_prerequisites / (
        "model-candidates/0123456789abcdef0123456789abcdef/models/"
        "Qwen--Qwen3.5-0.8B/snapshots/master"
    )
    stale.mkdir(parents=True, mode=0o700)
    desktop_prerequisites.chmod(0o700)
    (stale / "configuration.json").write_text("stale", encoding="utf-8")

    selected = run_initialization(
        _options("cpu", root=desktop_prerequisites),
        FakeRunner({"cpu": passed("cpu", "float32")}),
    )

    assert selected.backend == "cpu"
    candidates = desktop_prerequisites / "model-candidates"
    assert not candidates.exists() or list(candidates.iterdir()) == []


def test_stale_candidate_symlink_is_unlinked_without_touching_its_target(
    desktop_prerequisites: Path,
    tmp_path: Path,
) -> None:
    expected = _write_selection(desktop_prerequisites, "cpu")
    outside = tmp_path / "outside-candidate"
    outside.mkdir(mode=0o700)
    sentinel = outside / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    candidates = desktop_prerequisites / "model-candidates"
    candidates.mkdir(mode=0o700)
    stale = candidates / "0123456789abcdef0123456789abcdef"
    stale.symlink_to(outside, target_is_directory=True)

    runner = FakeRunner.all_fail()
    selected = run_initialization(
        _options("auto", root=desktop_prerequisites), runner
    )

    assert selected == expected
    assert not stale.exists() and not stale.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert runner.calls == []


def test_model_promotion_rejects_symlinked_canonical_cache_ancestors(
    desktop_prerequisites: Path,
    tmp_path: Path,
) -> None:
    desktop_prerequisites.mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside-cache"
    master = outside / "models/iic--SenseVoiceSmall/snapshots/master"
    master.mkdir(parents=True, mode=0o700)
    sentinel = master / "configuration.json"
    sentinel.write_text("outside-user-cache", encoding="utf-8")
    (desktop_prerequisites / "models").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", root=desktop_prerequisites),
            FakeRunner({"cpu": passed("cpu", "float32")}),
        )

    assert sentinel.read_text(encoding="utf-8") == "outside-user-cache"
    assert not selection_path(desktop_prerequisites).exists()


def test_successful_candidate_is_atomically_promoted_to_immutable_generation(
    desktop_prerequisites: Path,
) -> None:
    runner = FakeRunner({"cpu": passed("cpu", "float32")})

    selected = run_initialization(
        _options("cpu", root=desktop_prerequisites), runner
    )

    build_argv = next(
        argv for argv, _ in runner.calls if "create-runtime-env.sh" in argv[0]
    )
    candidate = Path(build_argv[build_argv.index("--runtime-dir") + 1])
    promoted = selected.python.parent.parent
    assert candidate.name.startswith(".candidate-cpu-")
    assert promoted.name.startswith("cpu-")
    assert len(promoted.name.removeprefix("cpu-")) == 32
    assert promoted.is_dir()
    assert not candidate.exists()
    assert load_runtime_selection(desktop_prerequisites) == selected


@pytest.mark.parametrize(
    ("dtype", "models"),
    [
        ("bf16", {"sensevoice": "master", "vad": "master"}),
        (
            "float32",
            {"sensevoice": "master", "vad": "master", "qwen": "master"},
        ),
    ],
    ids=("bad-dtype", "bad-model-policy"),
)
def test_schema_valid_policy_failure_is_not_promoted(
    desktop_prerequisites: Path,
    dtype: str,
    models: dict[str, str],
) -> None:
    result = ProbeResult(
        backend="cpu",
        status="pass",
        error_category=None,
        dtype=dtype,
        models=models,
        tensor_ms=1,
        asr_ms=1,
    )
    runner = FakeRunner({"cpu": result})

    with pytest.raises(InitializationError, match="^selected backend failed$"):
        run_initialization(_options("cpu", root=desktop_prerequisites), runner)

    runtimes = desktop_prerequisites / "runtimes"
    assert runtimes.is_dir()
    assert list(runtimes.iterdir()) == []
    assert not selection_path(desktop_prerequisites).exists()


def test_failed_same_backend_reselection_never_mutates_working_runtime(
    desktop_prerequisites: Path,
) -> None:
    previous = _write_selection(desktop_prerequisites, "cpu")
    working_runtime = previous.python.parent.parent
    marker = working_runtime / "working.marker"
    marker.write_text("working", encoding="utf-8")
    previous_bytes = selection_path(desktop_prerequisites).read_bytes()

    class DestructiveInstallFailureRunner(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            if "create-runtime-env.sh" in argv[0]:
                target = Path(argv[argv.index("--runtime-dir") + 1])
                if target == working_runtime:
                    marker.write_text("mutated", encoding="utf-8")
            result = super().run(argv, env=env)
            if "install-user.sh" in argv[0]:
                return CommandResult(1, "")
            return result

    runner = DestructiveInstallFailureRunner({"cpu": passed("cpu", "float32")})
    with pytest.raises(InitializationError, match="^install$"):
        run_initialization(
            _options("cpu", force=True, root=desktop_prerequisites), runner
        )

    build_argv = next(
        argv for argv, _ in runner.calls if "create-runtime-env.sh" in argv[0]
    )
    candidate = Path(build_argv[build_argv.index("--runtime-dir") + 1])
    assert candidate != working_runtime
    assert marker.read_text(encoding="utf-8") == "working"
    assert selection_path(desktop_prerequisites).read_bytes() == previous_bytes
    assert load_runtime_selection(desktop_prerequisites) == previous


def test_concurrent_initializers_are_serialized_before_any_candidate_build(
    desktop_prerequisites: Path,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_called = threading.Event()
    results: list[RuntimeSelection] = []
    errors: list[BaseException] = []

    class BlockingRunner(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if "create-runtime-env.sh" in argv[0]:
                first_started.set()
                if not release_first.wait(timeout=5):
                    raise AssertionError("first initializer was not released")
            return result

    class ObservingRunner(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            second_called.set()
            return super().run(argv, env=env)

    first_runner = BlockingRunner({"cpu": passed("cpu", "float32")})
    second_runner = ObservingRunner({"cpu": passed("cpu", "float32")})

    def initialize(runner: FakeRunner) -> None:
        try:
            result = run_initialization(
                _options("cpu", root=desktop_prerequisites), runner
            )
            assert isinstance(result, RuntimeSelection)
            results.append(result)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=initialize, args=(first_runner,))
    second = threading.Thread(target=initialize, args=(second_runner,))
    first.start()
    assert first_started.wait(timeout=5)
    second.start()
    serialized = not second_called.wait(timeout=0.25)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert serialized
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    assert second_runner.calls == []


def test_initializer_rejects_symlink_lock_without_running_commands(
    desktop_prerequisites: Path,
) -> None:
    desktop_prerequisites.mkdir(parents=True, mode=0o700)
    target = desktop_prerequisites.parent / "outside.lock"
    target.write_text("unrelated", encoding="utf-8")
    (desktop_prerequisites / ".initialize.lock").symlink_to(target)
    runner = FakeRunner({"cpu": passed("cpu", "float32")})

    with pytest.raises(InitializationError, match="^lock$"):
        run_initialization(_options("cpu", root=desktop_prerequisites), runner)

    assert runner.calls == []
    assert target.read_text(encoding="utf-8") == "unrelated"


def test_probe_entrypoint_cannot_be_shadowed_by_current_directory_package(
    desktop_prerequisites: Path,
    tmp_path: Path,
) -> None:
    forged_cwd = tmp_path / "forged-cwd"
    forged_package = forged_cwd / "fun_voice"
    forged_package.mkdir(parents=True)
    (forged_package / "__init__.py").write_text(
        "raise RuntimeError('forged package imported')\n", encoding="utf-8"
    )

    class EntrypointRunner(FakeRunner):
        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            if "fun_voice.backend_probe" in argv:
                module_index = argv.index("-m")
                command = (sys.executable, *argv[1 : module_index + 2], "--help")
                completed = subprocess.run(
                    command,
                    cwd=forged_cwd,
                    env=env,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if completed.returncode != 0:
                    return CommandResult(completed.returncode, "")
            return super().run(argv, env=env)

    runner = EntrypointRunner({"cpu": passed("cpu", "float32")})
    selected = run_initialization(
        _options("cpu", root=desktop_prerequisites), runner
    )
    probe_argv = next(
        argv for argv, _ in runner.calls if "fun_voice.backend_probe" in argv
    )

    assert probe_argv[1:4] == ("-P", "-m", "fun_voice.backend_probe")
    assert selected.backend == "cpu"
