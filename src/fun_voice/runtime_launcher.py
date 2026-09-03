"""Dispatch public commands through the validated portable runtime."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from fun_voice.runtime_selection import RuntimeSelectionError, load_runtime_selection

ENTRYPOINTS = {
    "fun-voice-daemon": "fun_voice.daemon",
    "fun-voice-worker": "fun_voice.worker",
    "fun-voice-preflight": "fun_voice.preflight",
    "fun-voice-selftest": "fun_voice.selftest",
    "fun-voice-corrector": "fun_voice.corrector",
    "fun-voice-benchmark": "fun_voice.benchmark",
}


def main(argv: Sequence[str] | None = None) -> int:
    """Replace this process with one fixed module in the selected interpreter."""
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] not in ENTRYPOINTS:
        return 2
    try:
        selection = load_runtime_selection()
    except RuntimeSelectionError:
        return 2

    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root / "src") + (
        os.pathsep + existing_path if existing_path else ""
    )
    python = str(selection.python)
    os.execvpe(
        python,
        [python, "-m", ENTRYPOINTS[values[0]], *values[1:]],
        environment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
