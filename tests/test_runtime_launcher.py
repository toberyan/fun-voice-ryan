"""Closed-dispatch tests for the selected portable runtime launcher."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from fun_voice import runtime_launcher
from fun_voice.runtime_selection import RuntimeSelectionError, selection_path


class _ExecCapture:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict[str, str]]] = []

    def __call__(
        self, path: str, argv: list[str], environment: dict[str, str]
    ) -> None:
        self.calls.append((path, argv, environment))


def _selection(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        python=tmp_path / "runtimes/cpu-0123456789abcdef0123456789abcdef/bin/python"
    )


def _data_root(tmp_path: Path) -> Path:
    return tmp_path / "custom-data/fun-voice-ryan"


def test_launcher_execs_manifest_python_with_fixed_daemon_module(
    monkeypatch, tmp_path: Path
) -> None:
    data_root = _data_root(tmp_path)
    selection = _selection(data_root)
    capture = _ExecCapture()
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-development-package")
    monkeypatch.setenv("PYTHONHOME", "/tmp/untrusted-python-home")
    monkeypatch.setenv("PYTHONUSERBASE", "/tmp/untrusted-user-site")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/untrusted-virtualenv")
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/untrusted-conda")
    monkeypatch.setattr(
        runtime_launcher, "load_runtime_selection", lambda root: selection
    )
    monkeypatch.setattr(os, "execvpe", capture)

    assert runtime_launcher.main(
        [
            "--runtime-selection",
            str(selection_path(data_root)),
            "fun-voice-daemon",
            "--log-level",
            "DEBUG",
        ]
    ) == 0

    assert len(capture.calls) == 1
    path, argv, environment = capture.calls[0]
    assert path == str(selection.python)
    assert argv == [
        str(selection.python),
        "-P",
        "-m",
        "fun_voice.daemon",
        "--log-level",
        "DEBUG",
    ]
    assert environment["PYTHONPATH"].endswith("/src")
    assert "/tmp/untrusted-development-package" not in environment["PYTHONPATH"]
    for key in ("PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX"):
        assert key not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"


def test_launcher_rejects_unknown_binary_without_exec(
    monkeypatch, tmp_path: Path
) -> None:
    capture = _ExecCapture()
    monkeypatch.setattr(os, "execvpe", capture)

    assert runtime_launcher.main(
        [
            "--runtime-selection",
            str(selection_path(_data_root(tmp_path))),
            "fun-voice-arbitrary",
        ]
    ) == 2
    assert capture.calls == []


def test_launcher_rejects_unsafe_selection_without_echoing_path(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    capture = _ExecCapture()

    def reject_selection(root: Path) -> None:
        raise RuntimeSelectionError("/private/unsafe/runtime")

    monkeypatch.setattr(runtime_launcher, "load_runtime_selection", reject_selection)
    monkeypatch.setattr(os, "execvpe", capture)

    assert runtime_launcher.main(
        [
            "--runtime-selection",
            str(selection_path(_data_root(tmp_path))),
            "fun-voice-worker",
        ]
    ) == 2
    assert capture.calls == []
    output = capsys.readouterr()
    assert "/private/unsafe/runtime" not in output.out
    assert "/private/unsafe/runtime" not in output.err


def test_launcher_exposes_only_the_six_public_commands() -> None:
    assert runtime_launcher.ENTRYPOINTS == {
        "fun-voice-daemon": "fun_voice.daemon",
        "fun-voice-worker": "fun_voice.worker",
        "fun-voice-preflight": "fun_voice.preflight",
        "fun-voice-selftest": "fun_voice.selftest",
        "fun-voice-corrector": "fun_voice.corrector",
        "fun-voice-benchmark": "fun_voice.benchmark",
    }


def test_launcher_binds_explicit_selection_and_models_root(
    monkeypatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "custom-data/fun-voice-ryan"
    manifest = selection_path(data_root)
    selection = _selection(data_root)
    loaded_roots: list[Path] = []
    capture = _ExecCapture()

    def load_bound_selection(root: Path) -> SimpleNamespace:
        loaded_roots.append(root)
        return selection

    monkeypatch.setattr(
        runtime_launcher, "load_runtime_selection", load_bound_selection
    )
    monkeypatch.setattr(os, "execvpe", capture)
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/wrong-data-root")
    monkeypatch.setenv("MODELSCOPE_CACHE", "/tmp/wrong-model-cache")
    monkeypatch.setenv("FUN_VOICE_MODELS_ROOT", "/tmp/wrong-model-root")

    assert runtime_launcher.main(
        [
            "--runtime-selection",
            str(manifest),
            "fun-voice-worker",
            "--profile",
            "sensevoice",
        ]
    ) == 0

    assert loaded_roots == [data_root]
    _, _, environment = capture.calls[0]
    assert environment["XDG_DATA_HOME"] == str(data_root.parent)
    assert environment["MODELSCOPE_CACHE"] == str(data_root / "models")
    assert environment["FUN_VOICE_MODELS_ROOT"] == str(data_root / "models")
