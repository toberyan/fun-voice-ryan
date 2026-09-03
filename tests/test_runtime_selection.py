"""Tests for the safe, portable runtime-selection manifest."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import fun_voice.runtime_selection as runtime_selection
from fun_voice.runtime_selection import (
    RuntimeSelection,
    RuntimeSelectionError,
    load_runtime_selection,
    selection_path,
    write_runtime_selection,
)


def _selection(root: Path, backend: str = "cpu") -> RuntimeSelection:
    python = root / "runtimes" / backend / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    for component in (root / "runtimes", root / "runtimes" / backend, python.parent):
        component.chmod(0o755)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o700)
    if backend == "cpu":
        return RuntimeSelection(
            schema_version=1,
            backend="cpu",
            python=python,
            device="cpu",
            dtype="float32",
            primary_asr_profile="sensevoice",
            fallback_asr_profile=None,
            enhanced_enabled=False,
            speaker_enabled=False,
            model_revisions={"sensevoice": "master", "vad": "master"},
            probe_status="pass",
            selected_at=1,
        )
    return RuntimeSelection(
        schema_version=1,
        backend=backend,
        python=python,
        device=f"{backend}:0",
        dtype="bf16",
        primary_asr_profile="nano",
        fallback_asr_profile="sensevoice",
        enhanced_enabled=True,
        speaker_enabled=True,
        model_revisions={
            "nano": "master",
            "sensevoice": "master",
            "vad": "master",
            "qwen": "master",
            "campplus": "master",
        },
        probe_status="pass",
        selected_at=1,
    )


def test_cpu_manifest_round_trip_forbids_accelerator_models(tmp_path: Path) -> None:
    root = tmp_path / "data"
    expected = _selection(root)

    path = write_runtime_selection(expected, root)

    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert load_runtime_selection(root) == expected
    assert dict(expected.model_revisions) == {"sensevoice": "master", "vad": "master"}
    with pytest.raises(TypeError):
        expected.model_revisions["qwen"] = "master"  # type: ignore[index]


def test_selection_fingerprint_is_a_canonical_manifest_digest(tmp_path: Path) -> None:
    selection = _selection(tmp_path / "data")
    reordered = dataclasses.replace(
        selection,
        model_revisions={"vad": "master", "sensevoice": "master"},
    )
    expected = hashlib.sha256(
        json.dumps(
            selection.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert runtime_selection.selection_fingerprint(selection) == expected
    assert runtime_selection.selection_fingerprint(reordered) == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda selection: dataclasses.replace(selection, enhanced_enabled=True),
        lambda selection: dataclasses.replace(selection, speaker_enabled=True),
        lambda selection: dataclasses.replace(
            selection,
            model_revisions={
                "sensevoice": "master",
                "vad": "master",
                "qwen": "master",
            },
        ),
    ],
)
def test_cpu_rejects_qwen_and_speaker_enablement(
    tmp_path: Path, mutate: Callable[[RuntimeSelection], RuntimeSelection]
) -> None:
    root = tmp_path / "data"
    invalid = mutate(_selection(root))

    with pytest.raises(RuntimeSelectionError, match="CPU runtime"):
        write_runtime_selection(invalid, root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda selection, root: dataclasses.replace(
            selection, python=Path("/usr/bin/python3")
        ),
        lambda selection, root: dataclasses.replace(selection, dtype="bf16"),
        lambda selection, root: dataclasses.replace(
            selection, fallback_asr_profile="nano"
        ),
    ],
)
def test_selection_rejects_unsafe_interpreter_or_cpu_policy(
    tmp_path: Path, mutate: Callable[[RuntimeSelection, Path], RuntimeSelection]
) -> None:
    root = tmp_path / "data"

    with pytest.raises(RuntimeSelectionError):
        write_runtime_selection(mutate(_selection(root), root), root)


@pytest.mark.parametrize(
    ("unsafe_path", "mode"),
    [
        ("runtimes/cpu/bin", 0o775),
        ("runtimes/cpu/bin/python", 0o722),
        ("runtimes/cpu/bin/python", 0o600),
    ],
)
def test_selection_rejects_unsafe_runtime_path_modes_without_leaking_paths(
    tmp_path: Path, unsafe_path: str, mode: int
) -> None:
    root = tmp_path / "data"
    selection = _selection(root)
    _write_valid_selection(root)
    (root / unsafe_path).chmod(mode)

    with pytest.raises(RuntimeSelectionError) as error:
        write_runtime_selection(selection, root)

    assert str(root) not in str(error.value)
    with pytest.raises(RuntimeSelectionError) as error:
        load_runtime_selection(root)

    assert str(root) not in str(error.value)


def test_selection_rejects_unowned_interpreter_without_leaking_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    selection = _selection(root)
    _write_valid_selection(root)
    original_lstat = Path.lstat

    def _unowned_lstat(path: Path) -> os.stat_result:
        details = original_lstat(path)
        if path == selection.python:
            return os.stat_result(
                (
                    details.st_mode,
                    details.st_ino,
                    details.st_dev,
                    details.st_nlink,
                    details.st_uid + 1,
                    details.st_gid,
                    details.st_size,
                    details.st_atime,
                    details.st_mtime,
                    details.st_ctime,
                )
            )
        return details

    monkeypatch.setattr(Path, "lstat", _unowned_lstat)

    with pytest.raises(RuntimeSelectionError) as error:
        write_runtime_selection(selection, root)

    assert str(root) not in str(error.value)
    with pytest.raises(RuntimeSelectionError) as error:
        load_runtime_selection(root)

    assert str(root) not in str(error.value)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("backend", []),
        ("primary_asr_profile", []),
        ("fallback_asr_profile", []),
    ],
)
def test_malformed_selection_values_raise_runtime_selection_error_on_write_and_load(
    tmp_path: Path, field: str, invalid: object
) -> None:
    root = tmp_path / "data"
    selection = dataclasses.replace(_selection(root), **{field: invalid})

    with pytest.raises(RuntimeSelectionError):
        write_runtime_selection(selection, root)

    path = _write_valid_selection(root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[field] = invalid
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RuntimeSelectionError):
        load_runtime_selection(root)


def test_malformed_model_map_raises_runtime_selection_error_on_write_and_load(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    selection = _selection(root)
    object.__setattr__(selection, "model_revisions", [])

    with pytest.raises(RuntimeSelectionError):
        write_runtime_selection(selection, root)

    path = _write_valid_selection(root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model_revisions"] = []
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RuntimeSelectionError):
        load_runtime_selection(root)


def test_invalid_replacement_keeps_previous_valid_selection(tmp_path: Path) -> None:
    root = tmp_path / "data"
    previous = _selection(root)
    path = write_runtime_selection(previous, root)
    before = path.read_bytes()
    invalid = dataclasses.replace(_selection(root, "cuda"), dtype="float32")

    with pytest.raises(RuntimeSelectionError):
        write_runtime_selection(invalid, root)

    assert path.read_bytes() == before
    assert load_runtime_selection(root) == previous


def _write_valid_selection(root: Path) -> Path:
    return write_runtime_selection(_selection(root), root)


@pytest.mark.parametrize(
    "case",
    [
        "malformed-json",
        "schema-version",
        "file-mode",
        "parent-mode",
        "non-owner-file",
        "missing-interpreter",
        "symlink-escape",
    ],
)
def test_load_rejects_unsafe_or_incompatible_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    root = tmp_path / "data"
    path = _write_valid_selection(root)

    if case == "malformed-json":
        path.write_text('{"secret": "do-not-echo"', encoding="utf-8")
    elif case == "schema-version":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["schema_version"] = 2
        path.write_text(json.dumps(raw), encoding="utf-8")
    elif case == "file-mode":
        path.chmod(0o644)
    elif case == "parent-mode":
        path.parent.chmod(0o755)
    elif case == "non-owner-file":
        actual_stat = runtime_selection._stat

        def _non_owner(candidate: Path) -> os.stat_result:
            result = actual_stat(candidate)
            if candidate == path:
                return os.stat_result(
                    (
                        result.st_mode,
                        result.st_ino,
                        result.st_dev,
                        result.st_nlink,
                        result.st_uid + 1,
                        result.st_gid,
                        result.st_size,
                        result.st_atime,
                        result.st_mtime,
                        result.st_ctime,
                    )
                )
            return result

        monkeypatch.setattr(runtime_selection, "_stat", _non_owner)
    elif case == "missing-interpreter":
        _selection(root).python.unlink()
    elif case == "symlink-escape":
        outside = tmp_path / "outside-python"
        outside.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        outside.chmod(0o700)
        interpreter = _selection(root).python
        interpreter.unlink()
        interpreter.symlink_to(outside)
    else:
        raise AssertionError(f"unknown case: {case}")

    with pytest.raises(RuntimeSelectionError) as error:
        load_runtime_selection(root)

    assert "do-not-echo" not in str(error.value)


@pytest.mark.parametrize(
    "backend,dtype",
    [("cuda", "bf16"), ("cuda", "fp16"), ("xpu", "bf16")],
)
def test_accelerators_require_their_exact_safe_policy(
    tmp_path: Path, backend: str, dtype: str
) -> None:
    root = tmp_path / "data"
    selection = dataclasses.replace(_selection(root, backend), dtype=dtype)

    assert write_runtime_selection(selection, root) == selection_path(root)
    assert load_runtime_selection(root) == selection


def test_runtime_selection_module_imports_stdlib_only() -> None:
    module_path = Path(runtime_selection.__file__ or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }

    assert imported_roots <= sys.stdlib_module_names


def test_data_root_honors_xdg_data_home() -> None:
    assert runtime_selection.data_root({"XDG_DATA_HOME": "/safe/data"}) == Path(
        "/safe/data/fun-voice-ryan"
    )
