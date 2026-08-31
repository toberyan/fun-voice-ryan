"""Tests for runtime path resolution and permission policy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fun_voice.config import (
    DIRECTORY_MODE,
    FCITX_SOCKET_NAME,
    FILE_MODE,
    RUNTIME_DIR_NAME,
    SOCKET_MODE,
    WORKER_SOCKET_NAME,
    ConfigError,
    build_runtime_paths,
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
    assert paths.fcitx_socket == tmp_path / FCITX_SOCKET_NAME


def test_resolved_paths_are_private_and_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    paths = build_runtime_paths(resolve_runtime_dir(uid=os.getuid()))
    assert paths.worker_socket == tmp_path / RUNTIME_DIR_NAME / WORKER_SOCKET_NAME
    assert paths.fcitx_socket == tmp_path / FCITX_SOCKET_NAME
