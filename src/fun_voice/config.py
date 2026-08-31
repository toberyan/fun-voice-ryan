"""Typed configuration and private runtime path resolution.

The runtime directory lives under ``$XDG_RUNTIME_DIR`` (a per-user tmpfs) and is
created with mode ``0700``; sockets and files beneath it are ``0600``. Before
returning any path we verify that ``XDG_RUNTIME_DIR`` exists and is owned by the
current user, otherwise the caller must refuse to start.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# --- Permission policy ------------------------------------------------------

DIRECTORY_MODE = 0o700
"""Mode for the private runtime directory."""

FILE_MODE = 0o600
"""Mode for temporary PCM shards."""

SOCKET_MODE = 0o600
"""Mode for the worker and Fcitx Unix sockets."""

# --- Path names -------------------------------------------------------------

RUNTIME_DIR_NAME = "fun-voice-ryan"
WORKER_SOCKET_NAME = "worker.sock"
FCITX_SOCKET_NAME = "fun-voice-ryan-fcitx.sock"


class ConfigError(RuntimeError):
    """Raised when the runtime environment cannot be used safely."""


@dataclass(frozen=True)
class Config:
    """Typed application configuration with safe defaults.

    Mirrors the TOML configuration described in the design document; loading
    and persistence are out of scope for this module.
    """

    hotkey: str = "<Super>C"
    audio_source: str = "default"
    memory_threshold_minutes: int = 10
    max_recording_minutes: int = 30
    input_method: str = "fcitx5"
    commit_timeout_ms: int = 500
    allow_x11_paste_fallback: bool = True
    model: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    device: str = "xpu:0"
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.35
    enforce_eager: bool = True
    keep_warm_until_logout: bool = True
    retain_history: bool = False


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved private runtime paths under the user's XDG_RUNTIME_DIR.

    ``runtime_dir`` doubles as the directory for temporary PCM shards and is
    created with mode ``DIRECTORY_MODE``; ``worker_socket`` and ``fcitx_socket``
    are created with mode ``SOCKET_MODE``.
    """

    runtime_dir: Path
    worker_socket: Path
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

    The worker socket lives inside the private runtime directory; the Fcitx
    socket is a sibling at the XDG_RUNTIME_DIR top level, matching the addon
    contract (``fun-voice-ryan-fcitx.sock``).
    """
    return RuntimePaths(
        runtime_dir=runtime_dir,
        worker_socket=runtime_dir / WORKER_SOCKET_NAME,
        fcitx_socket=runtime_dir.parent / FCITX_SOCKET_NAME,
    )
