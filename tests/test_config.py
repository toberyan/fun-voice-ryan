"""Tests for runtime path resolution and permission policy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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
    assert cfg.fcitx_commit_timeout_ms == 0.5
    assert cfg.allow_x11_paste_fallback is True
    assert cfg.inference == InferenceConfig()


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
                "fcitx_commit_timeout_ms = 2.5",
                "allow_x11_paste_fallback = false",
                "",
                "[inference]",
                'device = "cpu"',
                'dtype = "fp32"',
                "gpu_memory_utilization = 0.8",
                "enforce_eager = false",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.audio_source == "alsa_input.custom"
    assert cfg.fcitx_commit_timeout_ms == 2.5
    assert cfg.allow_x11_paste_fallback is False
    assert cfg.inference == InferenceConfig(
        device="cpu", dtype="fp32", gpu_memory_utilization=0.8, enforce_eager=False
    )


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
