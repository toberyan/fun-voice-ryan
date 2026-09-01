"""Tests for runtime path resolution and permission policy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import fun_voice.config as config_module
from fun_voice.config import (
    DAEMON_SOCKET_NAME,
    DIRECTORY_MODE,
    FCITX_SOCKET_NAME,
    FILE_MODE,
    RUNTIME_DIR_NAME,
    SOCKET_MODE,
    WORKER_SOCKET_NAME,
    Config,
    ConfigError,
    InferenceConfig,
    build_runtime_paths,
    load_config,
    resolve_runtime_dir,
)


def test_default_runtime_dir_uses_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert resolve_runtime_dir(uid=os.getuid()) == tmp_path / RUNTIME_DIR_NAME


def test_missing_xdg_runtime_dir_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    with pytest.raises(ConfigError):
        resolve_runtime_dir()


def test_nonexistent_xdg_runtime_dir_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ConfigError):
        resolve_runtime_dir(xdg_runtime_dir=missing)


def test_xdg_runtime_dir_not_owned_by_uid_rejected(tmp_path: Path) -> None:
    other_uid = os.getuid() + 1
    with pytest.raises(ConfigError):
        resolve_runtime_dir(xdg_runtime_dir=tmp_path, uid=other_uid)


def test_permission_policy_constants() -> None:
    assert DIRECTORY_MODE == 0o700
    assert FILE_MODE == 0o600
    assert SOCKET_MODE == 0o600


def test_runtime_paths_layout(tmp_path: Path) -> None:
    runtime_dir = tmp_path / RUNTIME_DIR_NAME
    paths = build_runtime_paths(runtime_dir)
    assert paths.runtime_dir == runtime_dir
    assert paths.worker_socket == runtime_dir / WORKER_SOCKET_NAME
    assert paths.daemon_socket == runtime_dir / DAEMON_SOCKET_NAME
    assert paths.fcitx_socket == tmp_path / FCITX_SOCKET_NAME


def test_resolved_paths_are_private_and_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    paths = build_runtime_paths(resolve_runtime_dir(uid=os.getuid()))
    assert paths.worker_socket == tmp_path / RUNTIME_DIR_NAME / WORKER_SOCKET_NAME
    assert paths.daemon_socket == tmp_path / RUNTIME_DIR_NAME / DAEMON_SOCKET_NAME
    assert paths.fcitx_socket == tmp_path / FCITX_SOCKET_NAME


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.audio_source == "default"
    assert cfg.fcitx_commit_timeout_ms == 500
    assert cfg.allow_x11_paste_fallback is True
    assert cfg.inference == InferenceConfig()
    assert cfg.inference.gpu_memory_utilization == 0.15
    assert cfg.inference.max_model_len == 1536
    assert cfg.inference.idle_unload_seconds == 120
    assert cfg.inference.allow_sensevoice_fallback is True


def test_load_config_defaults_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such" / "config.toml"
    assert load_config(missing) == Config()


def test_load_config_parses_toml_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(
            [
                "[audio]",
                'source = "alsa_input.custom"',
                "",
                "[input_method]",
                "commit_timeout_ms = 2500",
                "allow_x11_paste_fallback = false",
                "",
                "[inference]",
                'device = "xpu:0"',
                'dtype = "bf16"',
                "gpu_memory_utilization = 0.2",
                "max_model_len = 1024",
                "idle_unload_seconds = 90",
                "allow_sensevoice_fallback = false",
                "enforce_eager = false",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.audio_source == "alsa_input.custom"
    assert cfg.fcitx_commit_timeout_ms == 2500
    assert cfg.allow_x11_paste_fallback is False
    assert cfg.inference == InferenceConfig(
        device="xpu:0",
        dtype="bf16",
        gpu_memory_utilization=0.2,
        max_model_len=1024,
        idle_unload_seconds=90,
        allow_sensevoice_fallback=False,
        enforce_eager=False,
    )


def test_load_config_rejects_non_xpu_inference_device(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[inference]\ndevice = "cpu"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="xpu:0"):
        load_config(path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("gpu_memory_utilization", "0.09", "0.10"),
        ("gpu_memory_utilization", "0.21", "0.20"),
        ("max_model_len", "1023", "1024"),
        ("max_model_len", "1537", "1536"),
        ("idle_unload_seconds", "29", "30"),
        ("idle_unload_seconds", "301", "300"),
    ],
)
def test_load_config_rejects_unsafe_model_lifecycle_bounds(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[inference]\n{key} = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_enhanced_inference_rejects_non_xpu_corrector() -> None:
    with pytest.raises(ConfigError, match="correction.device must be 'xpu:0'"):
        config_module.validate_enhanced_inference_config(
            config_module.EnhancedInferenceConfig(correction_device="cpu")
        )


def test_enhanced_inference_uses_low_kv_bounds_for_qwen() -> None:
    value = config_module.validate_enhanced_inference_config(
        config_module.EnhancedInferenceConfig()
    )

    assert value.correction_gpu_memory_utilization == 0.15
    assert value.correction_max_model_len == 1536
    assert value.enabled is True

    with pytest.raises(ConfigError, match="correction.gpu_memory_utilization"):
        config_module.validate_enhanced_inference_config(
            config_module.EnhancedInferenceConfig(
                correction_gpu_memory_utilization=0.25
            )
        )


def test_load_config_rejects_non_positive_commit_timeout(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[input_method]\ncommit_timeout_ms = 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="commit_timeout_ms"):
        load_config(path)


def test_load_config_ignores_unknown_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[shortcut]\nhotkey = "<Super>X"\n', encoding="utf-8")
    assert load_config(path) == Config()


def test_load_config_corrupt_toml_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("not [ valid toml", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_default_config_path_respects_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from fun_voice.config import default_config_path

    assert default_config_path() == tmp_path / "fun-voice-ryan" / "config.toml"
