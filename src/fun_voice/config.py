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
from dataclasses import dataclass, field, replace
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
XPU_DEVICE = "xpu:0"


class ConfigError(RuntimeError):
    """Raised when the configuration or runtime environment cannot be used safely."""

@dataclass(frozen=True)
class InferenceConfig:
    """Bounded XPU ASR settings for a transient worker process.

    The vLLM allocation fraction and context length are deliberately capped so
    a configuration edit cannot recreate the former multi-gibibyte idle KV
    cache reservation.
    """

    device: str = XPU_DEVICE
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.15
    max_model_len: int = 1536
    idle_unload_seconds: int = 120
    allow_sensevoice_fallback: bool = True
    enforce_eager: bool = True


@dataclass(frozen=True)
class EnhancedInferenceConfig:
    """Fixed safety bounds for enhanced XPU-only voice processing.

    The individual model services read this one value object.  Keeping the
    supported devices and model identifier fixed prevents a configuration typo
    from silently selecting a CPU or an unvalidated correction model.
    """

    enabled: bool = True
    result_ttl_seconds: int = 600
    result_max_entries: int = 8
    correction_model: str = "Qwen/Qwen3.5-0.8B"
    correction_device: str = XPU_DEVICE
    correction_dtype: str = "bf16"
    correction_max_source_characters: int = 512
    correction_max_new_tokens: int = 512
    correction_timeout_seconds: int = 30
    correction_protected_terms: tuple[str, ...] = ()
    correction_enable_thinking: bool = False
    identity_enabled: bool = False
    identity_device: str = XPU_DEVICE


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
    fcitx_commit_timeout_ms: int = 500
    allow_x11_paste_fallback: bool = True
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    enhanced: EnhancedInferenceConfig = field(default_factory=EnhancedInferenceConfig)


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


def _positive_int(value: object, *, key: str, default: int) -> int:
    """Return a configured positive integer, rejecting invalid known keys."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer")
    return value


def _protected_terms(value: object) -> tuple[str, ...]:
    """Parse bounded local technical terms without accepting arbitrary objects."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(term, str) for term in value):
        raise ConfigError("correction.protected_terms must be a list of strings")
    if len(value) > 64:
        raise ConfigError("correction.protected_terms must contain at most 64 terms")
    normalized: list[str] = []
    for term in value:
        candidate = term.strip()
        if (
            not candidate
            or len(candidate) > 128
            or "\n" in candidate
            or "\r" in candidate
        ):
            raise ConfigError("correction.protected_terms contains an invalid term")
        if candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def validate_inference_config(inference: InferenceConfig) -> InferenceConfig:
    """Validate the only inference settings compatible with this XPU-only app."""
    if inference.device != XPU_DEVICE:
        raise ConfigError(f"inference.device must be {XPU_DEVICE!r}")
    if not 0.10 <= inference.gpu_memory_utilization <= 0.20:
        raise ConfigError(
            "inference.gpu_memory_utilization must be in [0.10, 0.20]"
        )
    if not 1024 <= inference.max_model_len <= 1536:
        raise ConfigError("inference.max_model_len must be in [1024, 1536]")
    if not 30 <= inference.idle_unload_seconds <= 300:
        raise ConfigError("inference.idle_unload_seconds must be in [30, 300]")
    return inference


def validate_enhanced_inference_config(
    value: EnhancedInferenceConfig,
) -> EnhancedInferenceConfig:
    """Reject unsupported enhanced-service settings before startup.

    The result API intentionally has fixed capacity and lifetime: allowing a
    longer retention period would change the local privacy boundary.
    """
    if value.correction_device != XPU_DEVICE:
        raise ConfigError("correction.device must be 'xpu:0'")
    if value.identity_device != XPU_DEVICE:
        raise ConfigError("speaker_identity.device must be 'xpu:0'")
    if value.correction_model != "Qwen/Qwen3.5-0.8B":
        raise ConfigError("correction.model must be 'Qwen/Qwen3.5-0.8B'")
    if value.correction_dtype != "bf16":
        raise ConfigError("correction.dtype must be 'bf16'")
    if not 1 <= value.correction_max_source_characters <= 512:
        raise ConfigError("correction.max_source_characters must be in [1, 512]")
    if not 1 <= value.correction_max_new_tokens <= 512:
        raise ConfigError("correction.max_new_tokens must be in [1, 512]")
    if not 1 <= value.correction_timeout_seconds <= 60:
        raise ConfigError("correction.timeout_seconds must be in [1, 60]")
    protected_terms = _protected_terms(list(value.correction_protected_terms))
    if value.correction_enable_thinking:
        raise ConfigError("correction.enable_thinking must be false")
    if (
        not 1 <= value.result_max_entries <= 8
        or value.result_ttl_seconds != 600
    ):
        raise ConfigError(
            "enhanced result retention is fixed at 8 entries / 600 seconds"
        )
    return replace(value, correction_protected_terms=protected_terms)


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
    enhanced = _table(raw.get("enhanced"))
    correction = _table(raw.get("correction"))
    speaker_identity = _table(raw.get("speaker_identity"))

    inference_config = validate_inference_config(
        InferenceConfig(
            device=_str(inference.get("device"), XPU_DEVICE),
            dtype=_str(inference.get("dtype"), "bf16"),
            gpu_memory_utilization=_float(
                inference.get("gpu_memory_utilization"), 0.15
            ),
            max_model_len=_positive_int(
                inference.get("max_model_len"),
                key="inference.max_model_len",
                default=1536,
            ),
            idle_unload_seconds=_positive_int(
                inference.get("idle_unload_seconds"),
                key="inference.idle_unload_seconds",
                default=120,
            ),
            allow_sensevoice_fallback=_bool(
                inference.get("allow_sensevoice_fallback"), True
            ),
            enforce_eager=_bool(inference.get("enforce_eager"), True),
        )
    )
    enhanced_config = validate_enhanced_inference_config(
        EnhancedInferenceConfig(
            enabled=_bool(enhanced.get("enabled"), True),
            result_ttl_seconds=_positive_int(
                enhanced.get("result_ttl_seconds"),
                key="enhanced.result_ttl_seconds",
                default=600,
            ),
            result_max_entries=_positive_int(
                enhanced.get("result_max_entries"),
                key="enhanced.result_max_entries",
                default=8,
            ),
            correction_model=_str(
                correction.get("model"), "Qwen/Qwen3.5-0.8B"
            ),
            correction_device=_str(correction.get("device"), XPU_DEVICE),
            correction_dtype=_str(correction.get("dtype"), "bf16"),
            correction_max_source_characters=_positive_int(
                correction.get("max_source_characters"),
                key="correction.max_source_characters",
                default=512,
            ),
            correction_max_new_tokens=_positive_int(
                correction.get("max_new_tokens"),
                key="correction.max_new_tokens",
                default=512,
            ),
            correction_timeout_seconds=_positive_int(
                correction.get("timeout_seconds"),
                key="correction.timeout_seconds",
                default=30,
            ),
            correction_protected_terms=_protected_terms(
                correction.get("protected_terms")
            ),
            correction_enable_thinking=_bool(
                correction.get("enable_thinking"), False
            ),
            identity_enabled=_bool(speaker_identity.get("enabled"), False),
            identity_device=_str(speaker_identity.get("device"), XPU_DEVICE),
        )
    )
    return Config(
        audio_source=_str(audio.get("source"), "default"),
        fcitx_commit_timeout_ms=_positive_int(
            input_method.get("commit_timeout_ms"),
            key="input_method.commit_timeout_ms",
            default=500,
        ),
        allow_x11_paste_fallback=_bool(
            input_method.get("allow_x11_paste_fallback"), True
        ),
        inference=inference_config,
        enhanced=enhanced_config,
    )
