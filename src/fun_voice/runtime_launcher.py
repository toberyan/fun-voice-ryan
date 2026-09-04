"""Dispatch public commands through the validated portable runtime."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from fun_voice.runtime_selection import (
    APP_DATA_DIR_NAME,
    RuntimeSelectionError,
    load_runtime_selection,
    selection_path,
)

ENTRYPOINTS = {
    "fun-voice-daemon": "fun_voice.daemon",
    "fun-voice-worker": "fun_voice.worker",
    "fun-voice-preflight": "fun_voice.preflight",
    "fun-voice-selftest": "fun_voice.selftest",
    "fun-voice-corrector": "fun_voice.corrector",
    "fun-voice-benchmark": "fun_voice.benchmark",
}

_UNTRUSTED_PYTHON_ENVIRONMENT = (
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "PYTHONWARNINGS",
    "PYTHONBREAKPOINT",
    "PYTHONEXECUTABLE",
    "PYTHONPLATLIBDIR",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "_PYTHON_SYSCONFIGDATA_NAME",
    "__PYVENV_LAUNCHER__",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Replace this process with one fixed module in the selected interpreter."""
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) < 3 or values[0] != "--runtime-selection":
        return 2
    manifest = Path(values[1])
    data_root = manifest.parent.parent
    if (
        not manifest.is_absolute()
        or manifest.name != "selection.json"
        or manifest.parent.name != "runtime"
        or data_root.name != APP_DATA_DIR_NAME
        or selection_path(data_root) != manifest
    ):
        return 2
    command = values[2]
    if command not in ENTRYPOINTS:
        return 2
    try:
        selection = load_runtime_selection(data_root)
    except RuntimeSelectionError:
        return 2

    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    for key in _UNTRUSTED_PYTHON_ENVIRONMENT:
        environment.pop(key, None)
    environment["PYTHONPATH"] = str(source_root / "src")
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    environment["XDG_DATA_HOME"] = str(data_root.parent)
    models_root = str(data_root / "models")
    environment["MODELSCOPE_CACHE"] = models_root
    environment["FUN_VOICE_MODELS_ROOT"] = models_root
    python = str(selection.python)
    os.execvpe(
        python,
        [python, "-P", "-m", ENTRYPOINTS[command], *values[3:]],
        environment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
