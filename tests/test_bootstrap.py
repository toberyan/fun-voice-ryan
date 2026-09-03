from __future__ import annotations

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
    assert not any(call[0][0] == "systemctl" for call in runner.calls)


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
            self.active_at_daemon_restart: set[str] | None = None

        def run(
            self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
        ) -> CommandResult:
            result = super().run(argv, env=env)
            if argv[:3] == ("systemctl", "--user", "stop") and len(argv) == 4:
                self.active_workers.discard(argv[3])
            elif argv[:3] == ("systemctl", "--user", "show"):
                unit = argv[-1]
                state = "active" if unit in self.active_workers else "inactive"
                return CommandResult(0, state + "\n")
            elif argv == (
                "systemctl",
                "--user",
                "restart",
                "fun-voice-daemon.service",
            ):
                self.active_at_daemon_restart = set(self.active_workers)
            return result

    runner = WorkerTransitionRunner()
    selected = run_initialization(
        _options("cpu", force=True, root=desktop_prerequisites), runner
    )

    assert selected.backend == "cpu"
    assert runner.active_at_daemon_restart == set()
    commands = [argv for argv, _ in runner.calls]
    restart_index = commands.index(
        ("systemctl", "--user", "restart", "fun-voice-daemon.service")
    )
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
        assert commands.index(stop) < commands.index(confirm) < restart_index


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
