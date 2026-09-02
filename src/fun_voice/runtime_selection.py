"""Safe persistence for the selected portable model runtime.

This module deliberately uses only the Python standard library so launchers can
validate a selection before importing any selected backend's dependencies.
"""

from __future__ import annotations

import json
import os
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
        resolved_python = selection.python.resolve(strict=True)
    except OSError as exc:
        raise RuntimeSelectionError("selected interpreter is unsafe") from exc

    if (
        resolved_python in (resolved_root, allowed_root)
        or not resolved_python.is_relative_to(allowed_root)
        or not resolved_python.is_file()
        or not os.access(resolved_python, os.X_OK)
    ):
        raise RuntimeSelectionError("selected interpreter is unsafe")


def _validate_common_fields(selection: RuntimeSelection) -> None:
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
        or selection.backend not in {"cuda", "xpu", "cpu"}
        or not isinstance(selection.device, str)
        or not isinstance(selection.dtype, str)
        or selection.primary_asr_profile not in {"nano", "sensevoice"}
        or selection.fallback_asr_profile not in {None, "nano", "sensevoice"}
    ):
        raise RuntimeSelectionError("invalid runtime selection schema")

    revisions = selection.model_revisions
    if not all(
        isinstance(name, str) and _safe_revision(revision)
        for name, revision in revisions.items()
    ):
        raise RuntimeSelectionError("invalid runtime model revisions")


def _validate_selection(selection: RuntimeSelection, root: Path) -> None:
    """Reject all selection states outside the fixed deployment policy."""
    _validate_common_fields(selection)
    _validate_python_path(selection, root)
    model_names = frozenset(selection.model_revisions)

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
