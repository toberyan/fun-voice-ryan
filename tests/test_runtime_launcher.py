"""Closed-dispatch tests for the selected portable runtime launcher."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from fun_voice import runtime_launcher
from fun_voice.runtime_selection import RuntimeSelectionError


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


def test_launcher_execs_manifest_python_with_fixed_daemon_module(
    monkeypatch, tmp_path: Path
) -> None:
    selection = _selection(tmp_path)
    capture = _ExecCapture()
    monkeypatch.setattr(
        runtime_launcher, "load_runtime_selection", lambda: selection
    )
    monkeypatch.setattr(os, "execvpe", capture)

    assert runtime_launcher.main(
        ["fun-voice-daemon", "--log-level", "DEBUG"]
    ) == 0

    assert len(capture.calls) == 1
    path, argv, environment = capture.calls[0]
    assert path == str(selection.python)
    assert argv == [
        str(selection.python),
        "-m",
        "fun_voice.daemon",
        "--log-level",
        "DEBUG",
    ]
    assert environment["PYTHONPATH"].split(os.pathsep)[0].endswith("/src")


def test_launcher_rejects_unknown_binary_without_exec(monkeypatch) -> None:
    capture = _ExecCapture()
    monkeypatch.setattr(os, "execvpe", capture)

    assert runtime_launcher.main(["fun-voice-arbitrary"]) == 2
    assert capture.calls == []


def test_launcher_rejects_unsafe_selection_without_echoing_path(
    monkeypatch, capsys
) -> None:
    capture = _ExecCapture()

    def reject_selection() -> None:
        raise RuntimeSelectionError("/private/unsafe/runtime")

    monkeypatch.setattr(runtime_launcher, "load_runtime_selection", reject_selection)
    monkeypatch.setattr(os, "execvpe", capture)

    assert runtime_launcher.main(["fun-voice-worker"]) == 2
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
