"""First-run selection of a verified, isolated model runtime."""

from __future__ import annotations

import argparse
import base64
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
from typing import Any, Literal, Protocol, cast

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
_MANAGED_UNITS = ("fun-voice-daemon.service", *_ASR_WORKER_UNITS)
_TRANSACTION_FILE = ".initialization-transaction.json"
_MODEL_CACHE_NAMES = {
    "nano": "FunAudioLLM--Fun-ASR-Nano-2512",
    "sensevoice": "iic--SenseVoiceSmall",
    "vad": "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "qwen": "Qwen--Qwen3.5-0.8B",
    "campplus": "iic--speech_campplus_sv_zh-cn_16k-common",
}
_DEPLOYMENT_BINDINGS = (
    ".local/bin/fun-voice-daemon",
    ".local/bin/fun-voice-worker",
    ".local/bin/fun-voice-preflight",
    ".local/bin/fun-voice-selftest",
    ".local/bin/fun-voice-corrector",
    ".local/bin/fun-voice-benchmark",
    ".local/bin/fun-voice-bridge",
    ".config/systemd/user/fun-voice-daemon.service",
    ".config/systemd/user/fun-voice-worker@.service",
    ".config/systemd/user/fun-voice-worker.service",
    ".config/autostart/fun-voice-session.desktop",
    ".local/lib/fcitx5/fcitx5-fun-voice.so",
    ".local/share/fcitx5/addon/fcitx5-fun-voice.conf",
    ".local/lib/fun-voice-ryan/fun-voice-overlay",
)
_MAX_BINDING_BYTES = 4 * 1024 * 1024


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


@dataclass(frozen=True, slots=True)
class _ServiceSnapshot:
    active_units: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ModelPromotion:
    destination: Path
    backup: Path | None


@dataclass(frozen=True, slots=True)
class _ModelPromotionPlan:
    key: str
    token: str
    had_destination: bool


@dataclass(frozen=True, slots=True)
class _BindingSnapshot:
    relative_path: str
    kind: Literal["missing", "file", "symlink"]
    content: bytes | None
    mode: int | None


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


def _candidate_models_path(root: Path, runtime: Path) -> Path:
    token = runtime.name.rsplit("-", 1)[-1]
    return root / "model-candidates" / token


def _discard_candidate(candidate: Path) -> None:
    try:
        if candidate.is_symlink() or candidate.is_file():
            candidate.unlink()
        elif candidate.exists():
            shutil.rmtree(candidate)
    except OSError:
        pass


def _stop_service(runner: CommandRunner, unit: str) -> None:
    stopped = runner.run(("systemctl", "--user", "stop", unit))
    if stopped.returncode != 0:
        raise InitializationError("install")
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


def _quiesce_model_services(runner: CommandRunner) -> None:
    """Quiesce the daemon, then prove both possible workers have exited."""
    _stop_service(runner, "fun-voice-daemon.service")
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


def _snapshot_service_state(runner: CommandRunner) -> _ServiceSnapshot:
    active: set[str] = set()
    for unit in _MANAGED_UNITS:
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
        if state.returncode == 0 and state.stdout.strip() in {
            "active",
            "activating",
            "reloading",
        }:
            active.add(unit)
    return _ServiceSnapshot(frozenset(active))


def _restore_service_state(
    runner: CommandRunner, snapshot: _ServiceSnapshot
) -> None:
    for unit in (*_ASR_WORKER_UNITS, "fun-voice-daemon.service"):
        if unit not in snapshot.active_units:
            continue
        started = runner.run(("systemctl", "--user", "start", unit))
        if started.returncode != 0:
            raise InitializationError("install")


def _stop_managed_services_for_restore(runner: CommandRunner) -> None:
    for unit in _MANAGED_UNITS:
        runner.run(("systemctl", "--user", "stop", unit))
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
        if state.returncode != 0 or state.stdout.strip() not in {
            "inactive",
            "failed",
        }:
            raise InitializationError("install")


def _write_transaction_journal(
    root: Path,
    *,
    previous_bytes: bytes | None,
    previous_mode: int | None,
    services: _ServiceSnapshot,
    bindings: Sequence[_BindingSnapshot],
    model_promotions: Sequence[_ModelPromotionPlan],
) -> Path:
    path = root / _TRANSACTION_FILE
    payload = {
        "version": 2,
        "previous_manifest": (
            None
            if previous_bytes is None
            else base64.b64encode(previous_bytes).decode("ascii")
        ),
        "previous_mode": previous_mode,
        "active_units": sorted(services.active_units),
        "bindings": [
            {
                "relative_path": item.relative_path,
                "kind": item.kind,
                "content": (
                    None
                    if item.content is None
                    else base64.b64encode(item.content).decode("ascii")
                ),
                "mode": item.mode,
            }
            for item in bindings
        ],
        "model_promotions": [
            {
                "key": item.key,
                "token": item.token,
                "had_destination": item.had_destination,
            }
            for item in model_promotions
        ],
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(
                (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        return path
    except OSError as exc:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise InitializationError("install") from exc


def _read_transaction_journal(
    root: Path,
) -> tuple[
    bytes | None,
    int | None,
    _ServiceSnapshot,
    tuple[_BindingSnapshot, ...],
    tuple[_ModelPromotionPlan, ...],
] | None:
    path = root / _TRANSACTION_FILE
    if not path.exists() and not path.is_symlink():
        return None
    try:
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_size > 64 * 1024 * 1024
        ):
            raise InitializationError("install")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {
            "version",
            "previous_manifest",
            "previous_mode",
            "active_units",
            "bindings",
            "model_promotions",
        }:
            raise InitializationError("install")
        encoded = raw["previous_manifest"]
        mode = raw["previous_mode"]
        units = raw["active_units"]
        bindings_raw = raw["bindings"]
        promotions_raw = raw["model_promotions"]
        if (
            raw["version"] != 2
            or (encoded is not None and not isinstance(encoded, str))
            or (mode is not None and mode != 0o600)
            or not isinstance(units, list)
            or not all(
                isinstance(unit, str) and unit in _MANAGED_UNITS for unit in units
            )
            or len(set(units)) != len(units)
            or not isinstance(bindings_raw, list)
            or not isinstance(promotions_raw, list)
        ):
            raise InitializationError("install")
        previous = (
            None
            if encoded is None
            else base64.b64decode(encoded.encode("ascii"), validate=True)
        )
        bindings: list[_BindingSnapshot] = []
        seen_paths: set[str] = set()
        for item in bindings_raw:
            if not isinstance(item, dict) or set(item) != {
                "relative_path",
                "kind",
                "content",
                "mode",
            }:
                raise InitializationError("install")
            relative = item["relative_path"]
            kind = item["kind"]
            content_value = item["content"]
            binding_mode = item["mode"]
            if (
                relative not in _DEPLOYMENT_BINDINGS
                or relative in seen_paths
                or kind not in {"missing", "file", "symlink"}
                or (content_value is not None and not isinstance(content_value, str))
                or (binding_mode is not None and not isinstance(binding_mode, int))
            ):
                raise InitializationError("install")
            content = (
                None
                if content_value is None
                else base64.b64decode(content_value.encode("ascii"), validate=True)
            )
            if (
                (
                    kind == "missing"
                    and (content is not None or binding_mode is not None)
                )
                or (kind == "file" and (content is None or binding_mode is None))
                or (kind == "symlink" and (content is None or binding_mode is not None))
                or (content is not None and len(content) > _MAX_BINDING_BYTES)
            ):
                raise InitializationError("install")
            seen_paths.add(relative)
            bindings.append(
                _BindingSnapshot(relative, cast(Any, kind), content, binding_mode)
            )
        if seen_paths != set(_DEPLOYMENT_BINDINGS):
            raise InitializationError("install")
        promotions: list[_ModelPromotionPlan] = []
        seen_keys: set[str] = set()
        for item in promotions_raw:
            if not isinstance(item, dict) or set(item) != {
                "key",
                "token",
                "had_destination",
            }:
                raise InitializationError("install")
            key = item["key"]
            token = item["token"]
            had_destination = item["had_destination"]
            if (
                not isinstance(key, str)
                or key not in _MODEL_CACHE_NAMES
                or key in seen_keys
                or not isinstance(token, str)
                or len(token) != 32
                or any(character not in "0123456789abcdef" for character in token)
                or not isinstance(had_destination, bool)
            ):
                raise InitializationError("install")
            seen_keys.add(key)
            promotions.append(_ModelPromotionPlan(key, token, had_destination))
    except (OSError, UnicodeError, ValueError) as exc:
        raise InitializationError("install") from exc
    return (
        previous,
        cast(int | None, mode),
        _ServiceSnapshot(frozenset(units)),
        tuple(bindings),
        tuple(promotions),
    )


def _snapshot_deployment_bindings() -> tuple[_BindingSnapshot, ...]:
    home = Path(os.environ.get("HOME", str(Path.home())))
    if not home.is_absolute():
        raise InitializationError("install")
    snapshots: list[_BindingSnapshot] = []
    for relative in _DEPLOYMENT_BINDINGS:
        path = home / relative
        try:
            details = path.lstat()
        except FileNotFoundError:
            snapshots.append(_BindingSnapshot(relative, "missing", None, None))
            continue
        except OSError as exc:
            raise InitializationError("install") from exc
        if stat.S_ISLNK(details.st_mode):
            try:
                target = os.readlink(path).encode("utf-8")
            except (OSError, UnicodeError) as exc:
                raise InitializationError("install") from exc
            if len(target) > 4096:
                raise InitializationError("install")
            snapshots.append(_BindingSnapshot(relative, "symlink", target, None))
        elif stat.S_ISREG(details.st_mode) and details.st_size <= _MAX_BINDING_BYTES:
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise InitializationError("install") from exc
            snapshots.append(
                _BindingSnapshot(
                    relative, "file", content, stat.S_IMODE(details.st_mode)
                )
            )
        else:
            raise InitializationError("install")
    return tuple(snapshots)


def _restore_deployment_bindings(bindings: Sequence[_BindingSnapshot]) -> None:
    home = Path(os.environ.get("HOME", str(Path.home())))
    for item in bindings:
        path = home / item.relative_path
        try:
            if path.exists() or path.is_symlink():
                if path.is_dir() and not path.is_symlink():
                    raise InitializationError("install")
                path.unlink()
            if item.kind == "missing":
                continue
            path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            temporary = path.parent / f".{path.name}.restore-{secrets.token_hex(8)}"
            if item.kind == "file":
                assert item.content is not None and item.mode is not None
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    item.mode,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(item.content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, item.mode)
            else:
                assert item.content is not None
                os.symlink(item.content.decode("utf-8"), temporary)
            os.replace(temporary, path)
        except (OSError, UnicodeError) as exc:
            raise InitializationError("install") from exc


def _clear_transaction_journal(root: Path) -> None:
    try:
        (root / _TRANSACTION_FILE).unlink(missing_ok=True)
    except OSError as exc:
        raise InitializationError("install") from exc


def _restore_deployment(
    root: Path,
    runner: CommandRunner,
) -> None:
    journal = _read_transaction_journal(root)
    if journal is None:
        return
    previous_bytes, previous_mode, services, bindings, promotions = journal
    manifest = selection_path(root)
    _stop_managed_services_for_restore(runner)
    _restore_model_promotions(root, promotions)
    _restore_manifest(manifest, previous_bytes, previous_mode)
    _restore_deployment_bindings(bindings)
    reload_result = runner.run(("systemctl", "--user", "daemon-reload"))
    if reload_result.returncode != 0:
        raise InitializationError("install")
    _restore_service_state(runner, services)
    _clear_transaction_journal(root)


def _model_promotion_paths(
    root: Path, plan: _ModelPromotionPlan
) -> tuple[Path, Path, Path]:
    name = _MODEL_CACHE_NAMES[plan.key]
    source = (
        root
        / "model-candidates"
        / plan.token
        / "models"
        / name
        / "snapshots/master"
    )
    snapshots = root / "models" / "models" / name / "snapshots"
    return source, snapshots / "master", snapshots / f".previous-master-{plan.token}"


def _plan_candidate_models(
    root: Path,
    candidate_root: Path,
    model_keys: Mapping[str, str],
) -> tuple[_ModelPromotionPlan, ...]:
    token = candidate_root.name
    if (
        candidate_root != root / "model-candidates" / token
        or len(token) != 32
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise InitializationError("install")
    plans: list[_ModelPromotionPlan] = []
    try:
        for key in model_keys:
            if key not in _MODEL_CACHE_NAMES:
                raise InitializationError("install")
            plan = _ModelPromotionPlan(key, token, False)
            source, destination, backup = _model_promotion_paths(root, plan)
            if not source.is_dir() or source.is_symlink():
                raise InitializationError("install")
            if backup.exists() or backup.is_symlink():
                raise InitializationError("install")
            plans.append(
                _ModelPromotionPlan(
                    key,
                    token,
                    destination.exists() or destination.is_symlink(),
                )
            )
    except OSError as exc:
        raise InitializationError("install") from exc
    return tuple(plans)


def _promote_candidate_models(
    root: Path,
    plans: Sequence[_ModelPromotionPlan],
) -> list[_ModelPromotion]:
    promotions: list[_ModelPromotion] = []
    try:
        for plan in plans:
            source, destination, backup_path = _model_promotion_paths(root, plan)
            snapshots = destination.parent
            snapshots.mkdir(parents=True, mode=0o700, exist_ok=True)
            if not source.is_dir() or source.is_symlink():
                raise InitializationError("install")
            destination_exists = destination.exists() or destination.is_symlink()
            if destination_exists != plan.had_destination:
                raise InitializationError("install")
            backup: Path | None = None
            if plan.had_destination:
                backup = backup_path
                if backup.exists() or backup.is_symlink():
                    raise InitializationError("install")
                os.replace(destination, backup)
            os.replace(source, destination)
            promotions.append(_ModelPromotion(destination, backup))
    except (OSError, KeyError, InitializationError) as exc:
        _rollback_model_promotion(promotions)
        raise InitializationError("install") from exc
    return promotions


def _restore_model_promotions(
    root: Path, plans: Sequence[_ModelPromotionPlan]
) -> None:
    for plan in reversed(plans):
        source, destination, backup = _model_promotion_paths(root, plan)
        if plan.had_destination:
            if not backup.exists() and not backup.is_symlink():
                continue
            _discard_candidate(destination)
            try:
                os.replace(backup, destination)
            except OSError as exc:
                raise InitializationError("install") from exc
        elif not source.exists() and not source.is_symlink():
            _discard_candidate(destination)


def _rollback_model_promotion(promotions: Sequence[_ModelPromotion]) -> None:
    for promotion in reversed(promotions):
        _discard_candidate(promotion.destination)
        if promotion.backup is not None:
            try:
                os.replace(promotion.backup, promotion.destination)
            except OSError as exc:
                raise InitializationError("install") from exc


def _commit_model_promotion(promotions: Sequence[_ModelPromotion]) -> None:
    for promotion in promotions:
        if promotion.backup is not None:
            _discard_candidate(promotion.backup)


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
    if _read_transaction_journal(root) is not None:
        _restore_deployment(root, runner)
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

    native = runner.run((str(project_root / "scripts/build-native-artifacts.sh"),))
    if native.returncode != 0:
        raise InitializationError("native_prerequisite")

    successful: RuntimeSelection | None = None
    successful_models: Path | None = None
    for backend in candidates:
        candidate, runtime = _candidate_runtime_paths(root, backend)
        candidate_models = _candidate_models_path(root, runtime)
        build = runner.run(
            (
                str(project_root / "scripts/create-runtime-env.sh"),
                "--backend",
                backend,
                "--runtime-dir",
                str(candidate),
                "--models-root",
                str(candidate_models),
            )
        )
        if build.returncode != 0:
            _discard_candidate(candidate)
            _discard_candidate(candidate_models)
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
                str(candidate_models),
                "--json",
            ),
            env=_probe_environment(project_root, candidate_models),
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
                    successful_models = candidate_models
                    break
            else:
                category = parsed.error_category or "internal"
        except InitializationError:
            category = "internal"
        _discard_candidate(candidate)
        _discard_candidate(candidate_models)
        _record_candidate_failure(backend, category)
        if options.backend != "auto":
            raise InitializationError("selected backend failed")

    if successful is None or successful_models is None:
        raise InitializationError("no backend available")

    services = _snapshot_service_state(runner)
    bindings = _snapshot_deployment_bindings()
    model_plans = _plan_candidate_models(
        root, successful_models, successful.model_revisions
    )
    _write_transaction_journal(
        root,
        previous_bytes=previous_bytes,
        previous_mode=previous_mode,
        services=services,
        bindings=bindings,
        model_promotions=model_plans,
    )
    promotions: list[_ModelPromotion] = []
    try:
        if previous is not None:
            _quiesce_model_services(runner)
        promotions = _promote_candidate_models(root, model_plans)
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
        if previous is None:
            _quiesce_model_services(runner)
        reload_result = runner.run(("systemctl", "--user", "daemon-reload"))
        if reload_result.returncode != 0:
            raise InitializationError("install")
        restart = runner.run(
            ("systemctl", "--user", "restart", "fun-voice-daemon.service")
        )
        if restart.returncode != 0:
            raise InitializationError("install")
        _clear_transaction_journal(root)
    except BaseException as exc:
        rollback_error: BaseException | None = None
        try:
            _rollback_model_promotion(promotions)
            _restore_deployment(root, runner)
        except BaseException as restore_exc:  # noqa: BLE001 - preserve rollback proof
            rollback_error = restore_exc
        finally:
            _discard_candidate(successful_models)
        if rollback_error is not None:
            raise InitializationError("install") from rollback_error
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise InitializationError("install") from exc
    _commit_model_promotion(promotions)
    _discard_candidate(successful_models)
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
