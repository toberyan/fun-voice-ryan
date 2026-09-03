"""First-run selection of a verified, isolated model runtime."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from fun_voice.runtime_selection import (
    Backend,
    RuntimeSelection,
    RuntimeSelectionError,
    load_runtime_selection,
    selection_path,
    write_runtime_selection,
)
from fun_voice.runtime_selection import data_root as default_data_root

_PROBE_KEYS = frozenset(
    {
        "backend",
        "status",
        "error_category",
        "dtype",
        "models",
        "tensor_ms",
        "asr_ms",
    }
)
_PROBE_ERRORS = frozenset(
    {
        "environment",
        "import",
        "availability",
        "tensor",
        "dtype",
        "model_download",
        "asr",
        "internal",
    }
)
_MODEL_POLICY = {
    "cpu": frozenset({"sensevoice", "vad"}),
    "cuda": frozenset({"nano", "sensevoice", "vad", "qwen", "campplus"}),
    "xpu": frozenset({"nano", "sensevoice", "vad", "qwen", "campplus"}),
}
_ASR_WORKER_UNITS = (
    "fun-voice-worker@nano.service",
    "fun-voice-worker@sensevoice.service",
)


class InitializationError(RuntimeError):
    """A fixed-category first-run initialization failure."""


@dataclass(frozen=True, slots=True)
class InitializationOptions:
    backend: Literal["auto", "cuda", "xpu", "cpu"] = "auto"
    force_reselect: bool = False
    dry_run: bool = False
    data_root: Path | None = None
    project_root: Path | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str


class CommandRunner(Protocol):
    def run(
        self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    def run(
        self, argv: tuple[str, ...], *, env: dict[str, str] | None = None
    ) -> CommandResult:
        try:
            completed = subprocess.run(
                argv,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return CommandResult(127, "")
        return CommandResult(completed.returncode, completed.stdout)


def candidate_backends(
    requested: str,
) -> tuple[Backend] | tuple[Backend, Backend] | tuple[Backend, Backend, Backend]:
    if requested == "auto":
        return ("cuda", "xpu", "cpu")
    if requested in {"cuda", "xpu", "cpu"}:
        return (cast(Backend, requested),)
    raise InitializationError("invalid backend")


def _owned_private_directory(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(details.st_mode)
        and not stat.S_ISLNK(details.st_mode)
        and details.st_uid == os.geteuid()
        and not details.st_mode & 0o077
    )


def _owned_pipewire_socket(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return stat.S_ISSOCK(details.st_mode) and details.st_uid == os.geteuid()


def validate_desktop_prerequisites(
    options: InitializationOptions,
    *,
    environment: Mapping[str, str],
    which: Callable[[str], str | None],
) -> None:
    """Validate the desktop boundary without exposing rejected values."""
    lookup = which
    if not environment.get("DISPLAY"):
        raise InitializationError("desktop_prerequisite")
    session_type = environment.get("XDG_SESSION_TYPE")
    if session_type is not None and session_type.lower() != "x11":
        raise InitializationError("desktop_prerequisite")
    desktop = environment.get("XDG_CURRENT_DESKTOP")
    if desktop is not None and "dde" not in desktop.lower():
        raise InitializationError("desktop_prerequisite")
    runtime_value = environment.get("XDG_RUNTIME_DIR")
    if not runtime_value or not _owned_private_directory(Path(runtime_value)):
        raise InitializationError("desktop_prerequisite")
    if any(lookup(tool) is None for tool in ("uv", "cmake", "pkg-config")):
        raise InitializationError("desktop_prerequisite")
    if lookup("pw-cli") is None and not _owned_pipewire_socket(
        Path(runtime_value) / "pipewire-0"
    ):
        raise InitializationError("desktop_prerequisite")
    if lookup("fcitx5-remote") is None:
        raise InitializationError("desktop_prerequisite")

    root = options.data_root or default_data_root(environment)
    parent = root.parent
    probe_path: Path | None = None
    try:
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=parent, delete=False) as probe:
            probe_path = Path(probe.name)
        os.chmod(probe_path, 0o600)
        probe_path.unlink()
        probe_path = None
    except OSError as exc:
        if probe_path is not None:
            with suppress(OSError):
                probe_path.unlink(missing_ok=True)
        raise InitializationError("desktop_prerequisite") from exc


@dataclass(frozen=True, slots=True)
class _ParsedProbe:
    backend: Backend
    status: Literal["pass", "fail"]
    error_category: str | None
    dtype: str | None
    models: dict[str, str]


def _parse_probe(payload: str, expected: Backend) -> _ParsedProbe:
    try:
        raw = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise InitializationError("probe") from exc
    if not isinstance(raw, dict) or frozenset(raw) != _PROBE_KEYS:
        raise InitializationError("probe")
    backend = raw["backend"]
    status_value = raw["status"]
    category = raw["error_category"]
    dtype = raw["dtype"]
    models = raw["models"]
    timings = (raw["tensor_ms"], raw["asr_ms"])
    if (
        backend != expected
        or status_value not in {"pass", "fail"}
        or not all(type(value) is int and value >= 0 for value in timings)
        or not isinstance(models, dict)
        or not all(
            isinstance(key, str)
            and isinstance(value, str)
            and bool(value)
            and value.isascii()
            for key, value in models.items()
        )
    ):
        raise InitializationError("probe")
    if status_value == "pass":
        if category is not None or not isinstance(dtype, str):
            raise InitializationError("probe")
    elif category not in _PROBE_ERRORS or (
        dtype is not None and not isinstance(dtype, str)
    ):
        raise InitializationError("probe")
    return _ParsedProbe(
        backend=expected,
        status=cast(Literal["pass", "fail"], status_value),
        error_category=cast(str | None, category),
        dtype=cast(str | None, dtype),
        models=cast(dict[str, str], models),
    )


def _selection_from_probe(
    backend: Backend, runtime: Path, probe: _ParsedProbe
) -> RuntimeSelection:
    expected_models = _MODEL_POLICY[backend]
    if frozenset(probe.models) != expected_models:
        raise InitializationError("probe")
    if backend == "cpu":
        if probe.dtype != "float32":
            raise InitializationError("probe")
        device = "cpu"
    elif backend == "cuda":
        if probe.dtype not in {"bf16", "fp16"}:
            raise InitializationError("probe")
        device = "cuda:0"
    else:
        if probe.dtype != "bf16":
            raise InitializationError("probe")
        device = "xpu:0"
    return RuntimeSelection(
        schema_version=1,
        backend=backend,
        python=runtime / "bin/python",
        device=device,
        dtype=probe.dtype,
        primary_asr_profile="sensevoice" if backend == "cpu" else "nano",
        fallback_asr_profile=None if backend == "cpu" else "sensevoice",
        enhanced_enabled=backend != "cpu",
        speaker_enabled=backend != "cpu",
        model_revisions=probe.models,
        probe_status="pass",
        selected_at=max(1, int(time.time())),
    )


def _record_candidate_failure(backend: Backend, category: str) -> None:
    print(
        json.dumps(
            {
                "backend": backend,
                "status": "fail",
                "error_category": category,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _probe_environment(project_root: Path, models_root: Path) -> dict[str, str]:
    environment = {
        "PYTHONPATH": str(project_root / "src"),
        "MODELSCOPE_CACHE": str(models_root),
    }
    for key in (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "CUDA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
    ):
        value = os.environ.get(key)
        if value is not None:
            environment[key] = value
    return environment


def _restore_manifest(
    path: Path, previous_bytes: bytes | None, previous_mode: int | None
) -> None:
    if previous_bytes is None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise InitializationError("install") from exc
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(previous_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, cast(int, previous_mode))
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise InitializationError("install") from exc


def _candidate_runtime_paths(root: Path, backend: Backend) -> tuple[Path, Path]:
    token = secrets.token_hex(16)
    runtimes = root / "runtimes"
    return (
        runtimes / f".candidate-{backend}-{token}",
        runtimes / f"{backend}-{token}",
    )


def _discard_candidate(candidate: Path) -> None:
    try:
        if candidate.is_symlink():
            candidate.unlink()
        elif candidate.exists():
            shutil.rmtree(candidate)
    except OSError:
        pass


def _stop_accelerator_workers(runner: CommandRunner) -> None:
    """Stop both accelerator-era workers and prove neither remains active."""
    for unit in _ASR_WORKER_UNITS:
        stopped = runner.run(("systemctl", "--user", "stop", unit))
        if stopped.returncode != 0:
            raise InitializationError("install")
    for unit in _ASR_WORKER_UNITS:
        state = runner.run(
            (
                "systemctl",
                "--user",
                "show",
                "--property=ActiveState",
                "--value",
                unit,
            )
        )
        if state.returncode != 0 or state.stdout.strip() not in {"inactive", "failed"}:
            raise InitializationError("install")


@contextmanager
def _initialization_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".initialize.lock"
    descriptor: int | None = None
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_details = root.lstat()
        if (
            stat.S_ISLNK(root_details.st_mode)
            or not stat.S_ISDIR(root_details.st_mode)
            or root_details.st_uid != os.geteuid()
            or root_details.st_mode & 0o022
        ):
            raise InitializationError("lock")

        flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        details = os.fstat(descriptor)
        path_details = lock_path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
            or (details.st_dev, details.st_ino)
            != (path_details.st_dev, path_details.st_ino)
        ):
            raise InitializationError("lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        path_details = lock_path.lstat()
        if (details.st_dev, details.st_ino) != (
            path_details.st_dev,
            path_details.st_ino,
        ):
            raise InitializationError("lock")
        yield
    except InitializationError:
        raise
    except OSError as exc:
        raise InitializationError("lock") from exc
    finally:
        if descriptor is not None:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(descriptor)


def run_initialization(
    options: InitializationOptions, runner: CommandRunner
) -> RuntimeSelection | tuple[Backend] | tuple[Backend, Backend] | tuple[
    Backend, Backend, Backend
]:
    candidates = candidate_backends(options.backend)
    if options.dry_run:
        print(json.dumps({"candidates": list(candidates)}, separators=(",", ":")))
        return candidates

    validate_desktop_prerequisites(
        options, environment=os.environ, which=shutil.which
    )
    root = options.data_root or default_data_root()
    project_root = options.project_root or Path(__file__).resolve().parents[2]
    with _initialization_lock(root):
        return _run_locked_initialization(
            options, runner, candidates, root, project_root
        )


def _run_locked_initialization(
    options: InitializationOptions,
    runner: CommandRunner,
    candidates: Sequence[Backend],
    root: Path,
    project_root: Path,
) -> RuntimeSelection:
    manifest = selection_path(root)
    previous: RuntimeSelection | None = None
    previous_bytes: bytes | None = None
    previous_mode: int | None = None
    if manifest.exists() or manifest.is_symlink():
        try:
            previous = load_runtime_selection(root)
            previous_bytes = manifest.read_bytes()
            previous_mode = stat.S_IMODE(manifest.stat().st_mode)
        except (OSError, RuntimeSelectionError) as exc:
            raise InitializationError("selection") from exc
        if not options.force_reselect:
            print("already_selected")
            return previous

    models_root = root / "models"
    successful: RuntimeSelection | None = None
    for backend in candidates:
        candidate, runtime = _candidate_runtime_paths(root, backend)
        build = runner.run(
            (
                str(project_root / "scripts/create-runtime-env.sh"),
                "--backend",
                backend,
                "--runtime-dir",
                str(candidate),
                "--models-root",
                str(models_root),
            )
        )
        if build.returncode != 0:
            _discard_candidate(candidate)
            _record_candidate_failure(backend, "environment")
            if options.backend != "auto":
                raise InitializationError("selected backend failed")
            continue
        probe_result = runner.run(
            (
                str(candidate / "bin/python"),
                "-P",
                "-m",
                "fun_voice.backend_probe",
                "--backend",
                backend,
                "--models-root",
                str(models_root),
                "--json",
            ),
            env=_probe_environment(project_root, models_root),
        )
        try:
            parsed = _parse_probe(probe_result.stdout, backend)
            if probe_result.returncode == 0 and parsed.status == "pass":
                pending = _selection_from_probe(backend, runtime, parsed)
                try:
                    os.replace(candidate, runtime)
                except OSError:
                    category = "environment"
                else:
                    successful = pending
                    break
            else:
                category = parsed.error_category or "internal"
        except InitializationError:
            category = "internal"
        _discard_candidate(candidate)
        _record_candidate_failure(backend, category)
        if options.backend != "auto":
            raise InitializationError("selected backend failed")

    if successful is None:
        raise InitializationError("no backend available")

    try:
        published = write_runtime_selection(successful, root)
        install = runner.run(
            (
                str(project_root / "scripts/install-user.sh"),
                "--runtime-selection",
                str(published),
            )
        )
        if install.returncode != 0:
            raise InitializationError("install")
        if (
            previous is not None
            and previous.backend in {"cuda", "xpu"}
            and successful.backend == "cpu"
        ):
            _stop_accelerator_workers(runner)
        reload_result = runner.run(("systemctl", "--user", "daemon-reload"))
        if reload_result.returncode != 0:
            raise InitializationError("install")
        restart = runner.run(
            ("systemctl", "--user", "restart", "fun-voice-daemon.service")
        )
        if restart.returncode != 0:
            raise InitializationError("install")
    except (OSError, RuntimeSelectionError, InitializationError) as exc:
        _restore_manifest(manifest, previous_bytes, previous_mode)
        raise InitializationError("install") from exc
    return successful


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="initialize portable model runtime")
    parser.add_argument(
        "--backend", choices=("auto", "cuda", "xpu", "cpu"), default="auto"
    )
    parser.add_argument("--force-reselect", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = InitializationOptions(
        backend=cast(Literal["auto", "cuda", "xpu", "cpu"], args.backend),
        force_reselect=args.force_reselect,
        dry_run=args.dry_run,
    )
    try:
        run_initialization(options, SubprocessCommandRunner())
    except InitializationError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
