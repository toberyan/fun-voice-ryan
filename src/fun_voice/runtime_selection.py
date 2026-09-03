"""Safe persistence for the selected portable model runtime.

This module deliberately uses only the Python standard library so launchers can
validate a selection before importing any selected backend's dependencies.
"""

from __future__ import annotations

import grp
import hashlib
import json
import os
import pwd
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

Backend = Literal["cuda", "xpu", "cpu"]
AsrProfile = Literal["nano", "sensevoice"]

SELECTION_SCHEMA_VERSION = 1
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
APP_DATA_DIR_NAME = "fun-voice-ryan"
SELECTION_DIRECTORY_NAME = "runtime"
SELECTION_FILE_NAME = "selection.json"
ACCELERATOR_MODELS = frozenset({"nano", "sensevoice", "vad", "qwen", "campplus"})
CPU_MODELS = frozenset({"sensevoice", "vad"})


class RuntimeSelectionError(RuntimeError):
    """Raised when a runtime selection is absent, unsafe, or incompatible."""


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """The deployment-owned settings a user preference cannot override."""

    backend: Backend
    device: str
    dtype: str
    primary_asr_profile: AsrProfile
    fallback_asr_profile: AsrProfile | None
    enhanced_enabled: bool
    speaker_enabled: bool

    @property
    def allowed_profiles(self) -> tuple[AsrProfile] | tuple[AsrProfile, AsrProfile]:
        if self.fallback_asr_profile is None:
            return (self.primary_asr_profile,)
        return (self.primary_asr_profile, self.fallback_asr_profile)


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """An immutable record of one completely probed runtime environment."""

    schema_version: int
    backend: Backend
    python: Path
    device: str
    dtype: str
    primary_asr_profile: AsrProfile
    fallback_asr_profile: AsrProfile | None
    enhanced_enabled: bool
    speaker_enabled: bool
    model_revisions: Mapping[str, str]
    probe_status: Literal["pass"]
    selected_at: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_revisions, Mapping):
            raise RuntimeSelectionError("invalid selection schema")
        object.__setattr__(
            self,
            "model_revisions",
            MappingProxyType(dict(self.model_revisions)),
        )

    def policy(self) -> RuntimePolicy:
        return RuntimePolicy(
            self.backend,
            self.device,
            self.dtype,
            self.primary_asr_profile,
            self.fallback_asr_profile,
            self.enhanced_enabled,
            self.speaker_enabled,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "python": str(self.python),
            "device": self.device,
            "dtype": self.dtype,
            "primary_asr_profile": self.primary_asr_profile,
            "fallback_asr_profile": self.fallback_asr_profile,
            "enhanced_enabled": self.enhanced_enabled,
            "speaker_enabled": self.speaker_enabled,
            "model_revisions": dict(self.model_revisions),
            "probe": {"status": self.probe_status, "selected_at": self.selected_at},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RuntimeSelection:
        probe = raw.get("probe")
        revisions = raw.get("model_revisions")
        if not isinstance(probe, Mapping) or not isinstance(revisions, Mapping):
            raise RuntimeSelectionError("invalid selection schema")

        python = raw.get("python")
        if not isinstance(python, str):
            raise RuntimeSelectionError("invalid selection schema")

        return cls(
            schema_version=cast(int, raw.get("schema_version")),
            backend=cast(Backend, raw.get("backend")),
            python=Path(python),
            device=cast(str, raw.get("device")),
            dtype=cast(str, raw.get("dtype")),
            primary_asr_profile=cast(AsrProfile, raw.get("primary_asr_profile")),
            fallback_asr_profile=cast(
                AsrProfile | None, raw.get("fallback_asr_profile")
            ),
            enhanced_enabled=cast(bool, raw.get("enhanced_enabled")),
            speaker_enabled=cast(bool, raw.get("speaker_enabled")),
            model_revisions=cast(Mapping[str, str], revisions),
            probe_status=cast(Literal["pass"], probe.get("status")),
            selected_at=cast(int, probe.get("selected_at")),
        )


def selection_fingerprint(selection: RuntimeSelection) -> str:
    """Return the canonical SHA-256 identity for persisted selection data."""
    payload = json.dumps(
        selection.to_dict(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def data_root(env: Mapping[str, str] | None = None) -> Path:
    """Return the application data root, respecting ``XDG_DATA_HOME``."""
    values = os.environ if env is None else env
    xdg_data_home = values.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / APP_DATA_DIR_NAME


def selection_path(root: Path | None = None) -> Path:
    """Return the canonical runtime-selection manifest path without loading it."""
    base = data_root() if root is None else Path(root)
    return base / SELECTION_DIRECTORY_NAME / SELECTION_FILE_NAME


def _current_uid() -> int:
    return os.geteuid()


def _stat(path: Path) -> os.stat_result:
    """Stat one expected path without leaking arbitrary selection contents."""
    try:
        return path.stat()
    except OSError as exc:
        raise RuntimeSelectionError("cannot inspect runtime selection") from exc


def _lstat(path: Path) -> os.stat_result:
    """Inspect one path without following an unsafe symlink."""
    try:
        return path.lstat()
    except OSError as exc:
        raise RuntimeSelectionError("selected interpreter is unsafe") from exc


def _check_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeSelectionError("runtime selection directory is unsafe")
    details = _stat(path)
    if not stat.S_ISDIR(details.st_mode):
        raise RuntimeSelectionError("runtime selection directory is unsafe")
    if (
        details.st_uid != _current_uid()
        or stat.S_IMODE(details.st_mode) != DIRECTORY_MODE
    ):
        raise RuntimeSelectionError("runtime selection directory is unsafe")


def _selection_parent(root: Path) -> Path:
    parent = selection_path(root).parent
    try:
        parent.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeSelectionError(
            "cannot create runtime selection directory"
        ) from exc
    _check_private_directory(parent)
    return parent


def _check_selection_file(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeSelectionError("runtime selection file is unsafe")
    details = _stat(path)
    if not stat.S_ISREG(details.st_mode):
        raise RuntimeSelectionError("runtime selection file is unsafe")
    if details.st_uid != _current_uid() or stat.S_IMODE(details.st_mode) != FILE_MODE:
        raise RuntimeSelectionError("runtime selection file is unsafe")


def _safe_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and all("!" <= character <= "~" for character in value)
    )


def _validate_python_path(selection: RuntimeSelection, root: Path) -> None:
    if not isinstance(selection.python, Path) or not selection.python.is_absolute():
        raise RuntimeSelectionError("selected interpreter is unsafe")
    try:
        resolved_root = root.resolve(strict=True)
        allowed_root = resolved_root / "runtimes"
    except OSError as exc:
        raise RuntimeSelectionError("selected interpreter is unsafe") from exc

    runtime_name = selection.python.parent.parent.name
    generation_prefix = f"{selection.backend}-"
    generation_suffix = runtime_name.removeprefix(generation_prefix)
    valid_generation = (
        runtime_name.startswith(generation_prefix)
        and len(generation_suffix) == 32
        and all(character in "0123456789abcdef" for character in generation_suffix)
    )
    expected_python = allowed_root / runtime_name / "bin" / "python"
    if (
        selection.python != expected_python
        or (runtime_name != selection.backend and not valid_generation)
    ):
        raise RuntimeSelectionError("selected interpreter is unsafe")

    components = [allowed_root, expected_python.parent.parent, expected_python.parent]
    for component in components:
        _validate_runtime_directory(component)
    _validate_runtime_interpreter(expected_python)


def _validate_runtime_directory(path: Path) -> None:
    details = _lstat(path)
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != _current_uid()
        or details.st_mode & 0o022
    ):
        raise RuntimeSelectionError("selected interpreter is unsafe")


def _target_permissions_are_safe(details: os.stat_result) -> bool:
    if details.st_mode & 0o002 or not details.st_mode & 0o111:
        return False
    if not details.st_mode & 0o020:
        return True
    if details.st_uid != _current_uid() or details.st_gid != os.getegid():
        return False
    try:
        current_name = pwd.getpwuid(_current_uid()).pw_name
        group = grp.getgrgid(details.st_gid)
        primary_users = {
            account.pw_name
            for account in pwd.getpwall()
            if account.pw_gid == details.st_gid
        }
    except KeyError:
        return False
    return group.gr_mem in ([], [current_name]) and primary_users == {current_name}


def _validate_runtime_interpreter(path: Path) -> None:
    details = _lstat(path)
    if details.st_uid != _current_uid() or not os.access(path, os.X_OK):
        raise RuntimeSelectionError("selected interpreter is unsafe")
    if stat.S_ISREG(details.st_mode):
        if details.st_mode & 0o022 or not details.st_mode & 0o111:
            raise RuntimeSelectionError("selected interpreter is unsafe")
        return
    if not stat.S_ISLNK(details.st_mode):
        raise RuntimeSelectionError("selected interpreter is unsafe")

    try:
        target = path.resolve(strict=True)
        target_details = target.stat()
    except OSError as exc:
        raise RuntimeSelectionError("selected interpreter is unsafe") from exc
    if (
        not stat.S_ISREG(target_details.st_mode)
        or target_details.st_uid not in {0, _current_uid()}
        or not _target_permissions_are_safe(target_details)
    ):
        raise RuntimeSelectionError("selected interpreter is unsafe")

    config = path.parent.parent / "pyvenv.cfg"
    config_details = _lstat(config)
    if (
        not stat.S_ISREG(config_details.st_mode)
        or config_details.st_uid != _current_uid()
        or config_details.st_mode & 0o022
        or config_details.st_size > 16_384
    ):
        raise RuntimeSelectionError("selected interpreter is unsafe")
    try:
        settings = dict(
            (key.strip(), value.strip())
            for line in config.read_text(encoding="utf-8").splitlines()
            if "=" in line
            for key, value in (line.split("=", 1),)
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeSelectionError("selected interpreter is unsafe") from exc
    if (
        settings.get("implementation") != "CPython"
        or not settings.get("version_info", "").startswith("3.12.")
        or settings.get("include-system-site-packages", "").lower() != "false"
    ):
        raise RuntimeSelectionError("selected interpreter is unsafe")


def _validate_common_fields(selection: RuntimeSelection) -> frozenset[str]:
    if (
        type(selection.schema_version) is not int
        or selection.schema_version != SELECTION_SCHEMA_VERSION
        or selection.probe_status != "pass"
        or type(selection.selected_at) is not int
        or selection.selected_at <= 0
    ):
        raise RuntimeSelectionError("invalid runtime selection schema")
    if (
        type(selection.enhanced_enabled) is not bool
        or type(selection.speaker_enabled) is not bool
        or not isinstance(selection.backend, str)
        or selection.backend not in {"cuda", "xpu", "cpu"}
        or not isinstance(selection.device, str)
        or not isinstance(selection.dtype, str)
        or not isinstance(selection.primary_asr_profile, str)
        or selection.primary_asr_profile not in {"nano", "sensevoice"}
        or not (
            selection.fallback_asr_profile is None
            or (
                isinstance(selection.fallback_asr_profile, str)
                and selection.fallback_asr_profile in {"nano", "sensevoice"}
            )
        )
    ):
        raise RuntimeSelectionError("invalid runtime selection schema")

    revisions = selection.model_revisions
    if not isinstance(revisions, Mapping):
        raise RuntimeSelectionError("invalid runtime model revisions")
    try:
        items = tuple(revisions.items())
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeSelectionError("invalid runtime model revisions") from exc
    if not all(
        isinstance(name, str) and _safe_revision(revision)
        for name, revision in items
    ):
        raise RuntimeSelectionError("invalid runtime model revisions")
    return frozenset(name for name, _ in items)


def _validate_selection(selection: RuntimeSelection, root: Path) -> None:
    """Reject all selection states outside the fixed deployment policy."""
    model_names = _validate_common_fields(selection)
    _validate_python_path(selection, root)

    if selection.backend == "cpu":
        if (
            selection.device != "cpu"
            or selection.dtype != "float32"
            or selection.primary_asr_profile != "sensevoice"
            or selection.fallback_asr_profile is not None
            or selection.enhanced_enabled
            or selection.speaker_enabled
            or model_names != CPU_MODELS
        ):
            raise RuntimeSelectionError("CPU runtime policy is invalid")
        return

    if selection.backend == "cuda":
        valid_dtype = selection.dtype in {"bf16", "fp16"}
        expected_device = "cuda:0"
    elif selection.backend == "xpu":
        valid_dtype = selection.dtype == "bf16"
        expected_device = "xpu:0"
    else:
        raise RuntimeSelectionError("invalid runtime selection schema")

    if (
        selection.device != expected_device
        or not valid_dtype
        or selection.primary_asr_profile != "nano"
        or selection.fallback_asr_profile != "sensevoice"
        or not selection.enhanced_enabled
        or not selection.speaker_enabled
        or model_names != ACCELERATOR_MODELS
    ):
        raise RuntimeSelectionError("accelerator runtime policy is invalid")


def write_runtime_selection(
    selection: RuntimeSelection, root: Path | None = None
) -> Path:
    """Validate and atomically publish one safe runtime-selection manifest."""
    base = data_root() if root is None else Path(root)
    parent = _selection_parent(base)
    _validate_selection(selection, base)
    path = parent / SELECTION_FILE_NAME
    temporary_path: Path | None = None

    try:
        payload = json.dumps(selection.to_dict(), ensure_ascii=False, sort_keys=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=".selection.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, FILE_MODE)
        os.replace(temporary_path, path)
        temporary_path = None
    except Exception as exc:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
        raise RuntimeSelectionError("cannot write runtime selection") from exc

    return path


def load_runtime_selection(root: Path | None = None) -> RuntimeSelection:
    """Load a manifest only after its path, ownership, mode, and policy validate."""
    base = data_root() if root is None else Path(root)
    parent = _selection_parent(base)
    path = parent / SELECTION_FILE_NAME
    _check_selection_file(path)
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeSelectionError("cannot load runtime selection") from exc
    if not isinstance(raw, Mapping):
        raise RuntimeSelectionError("invalid selection schema")

    try:
        selection = RuntimeSelection.from_dict(cast(Mapping[str, object], raw))
        _validate_selection(selection, base)
    except (TypeError, ValueError, RuntimeSelectionError) as exc:
        raise RuntimeSelectionError("invalid runtime selection") from exc
    return selection
