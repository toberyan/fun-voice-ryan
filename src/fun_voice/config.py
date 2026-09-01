"""Typed configuration and private runtime path resolution.

The runtime directory lives under ``$XDG_RUNTIME_DIR`` (a per-user tmpfs) and is
created with mode ``0700``; sockets and files beneath it are ``0600``. Before
returning any path we verify that ``XDG_RUNTIME_DIR`` exists and is owned by the
current user, otherwise the caller must refuse to start.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# --- Permission policy ------------------------------------------------------

DIRECTORY_MODE = 0o700
"""Mode for the private runtime directory."""

FILE_MODE = 0o600
"""Mode for temporary PCM shards."""

SOCKET_MODE = 0o600
"""Mode for the worker, daemon and Fcitx Unix sockets."""

# --- Path names -------------------------------------------------------------

RUNTIME_DIR_NAME = "fun-voice-ryan"
WORKER_SOCKET_NAME = "worker.sock"
DAEMON_SOCKET_NAME = "daemon.sock"
CONFIG_DIR_NAME = "fun-voice-ryan"
CONFIG_FILE_NAME = "config.toml"
FCITX_SOCKET_NAME = "fun-voice-ryan-fcitx.sock"


class ConfigError(RuntimeError):
    """Raised when the configuration or runtime environment cannot be used safely."""

@dataclass(frozen=True)
class InferenceConfig:
    """Inference (worker) settings; defaults mirror the preflight constants."""

    device: str = "xpu:0"
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.35
    enforce_eager: bool = True


@dataclass(frozen=True)
class Config:
    """Typed application configuration with safe defaults.

    Loaded from the single TOML file under
    ``${XDG_CONFIG_HOME:-~/.config}/fun-voice-ryan/config.toml`` (see
    :func:`load_config`). The non-configurable safety bounds
    (``max_recording_minutes`` / ``memory_threshold_minutes``) are intentionally
    absent: they are capture-side invariants, never user-tunable.
    """

    audio_source: str = "default"
    fcitx_commit_timeout_ms: float = 0.5
    allow_x11_paste_fallback: bool = True
    inference: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved private runtime paths under the user's XDG_RUNTIME_DIR.

    ``runtime_dir`` doubles as the directory for temporary PCM shards and is
    created with mode ``DIRECTORY_MODE``; ``worker_socket``, ``daemon_socket``
    and ``fcitx_socket`` are created with mode ``SOCKET_MODE``.
    """

    runtime_dir: Path
    worker_socket: Path
    daemon_socket: Path
    fcitx_socket: Path


def get_xdg_runtime_dir(env: Mapping[str, str] | None = None) -> str | None:
    """Return ``XDG_RUNTIME_DIR`` from ``env`` (defaults to ``os.environ``)."""
    if env is None:
        env = os.environ
    value = env.get("XDG_RUNTIME_DIR")
    return value or None


def resolve_runtime_dir(
    *, xdg_runtime_dir: str | Path | None = None, uid: int | None = None
) -> Path:
    """Return ``<XDG_RUNTIME_DIR>/fun-voice-ryan`` after safety checks.

    Raises :class:`ConfigError` when ``XDG_RUNTIME_DIR`` is unset, does not
    exist, or is not owned by ``uid`` (defaults to the current user).
    """
    if xdg_runtime_dir is None:
        xdg_runtime_dir = get_xdg_runtime_dir()
    if xdg_runtime_dir is None or str(xdg_runtime_dir) == "":
        raise ConfigError("XDG_RUNTIME_DIR is not set")

    base = Path(xdg_runtime_dir)
    if not base.is_dir():
        raise ConfigError(f"XDG_RUNTIME_DIR does not exist: {base}")

    if uid is None:
        uid = os.getuid()
    try:
        owner = base.stat().st_uid
    except OSError as exc:
        raise ConfigError(f"cannot stat XDG_RUNTIME_DIR {base}: {exc}") from exc
    if owner != uid:
        raise ConfigError(
            f"XDG_RUNTIME_DIR {base} is owned by uid {owner}, expected {uid}"
        )

    return base / RUNTIME_DIR_NAME


def build_runtime_paths(runtime_dir: Path) -> RuntimePaths:
    """Build the runtime paths rooted at ``runtime_dir``.

    The worker and daemon sockets live inside the private runtime directory;
    the Fcitx socket is a sibling at the XDG_RUNTIME_DIR top level, matching the
    addon contract (``fun-voice-ryan-fcitx.sock``).
    """
    return RuntimePaths(
        runtime_dir=runtime_dir,
        worker_socket=runtime_dir / WORKER_SOCKET_NAME,
        daemon_socket=runtime_dir / DAEMON_SOCKET_NAME,
        fcitx_socket=runtime_dir.parent / FCITX_SOCKET_NAME,
    )


def default_config_path() -> Path:
    """Return the default TOML config path (respects ``XDG_CONFIG_HOME``)."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def _table(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def load_config(path: str | Path | None = None) -> Config:
    """Load the single TOML configuration, or return the safe defaults.

    ``path`` overrides the default location
    (``${XDG_CONFIG_HOME:-~/.config}/fun-voice-ryan/config.toml``). A missing
    file yields :class:`Config` defaults; a corrupt or unreadable file raises
    :class:`ConfigError`. Unknown keys and sections are ignored for forward
    compatibility.
    """
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        return Config()
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot load config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config {config_path} must be a TOML table")

    audio = _table(raw.get("audio"))
    input_method = _table(raw.get("input_method"))
    inference = _table(raw.get("inference"))

    return Config(
        audio_source=_str(audio.get("source"), "default"),
        fcitx_commit_timeout_ms=_float(
            input_method.get("fcitx_commit_timeout_ms"), 0.5
        ),
        allow_x11_paste_fallback=_bool(
            input_method.get("allow_x11_paste_fallback"), True
        ),
        inference=InferenceConfig(
            device=_str(inference.get("device"), "xpu:0"),
            dtype=_str(inference.get("dtype"), "bf16"),
            gpu_memory_utilization=_float(
                inference.get("gpu_memory_utilization"), 0.35
            ),
            enforce_eager=_bool(inference.get("enforce_eager"), True),
        ),
    )
