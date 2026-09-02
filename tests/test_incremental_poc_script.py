"""Regression checks for the explicit incremental Nano POC entry point."""

from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run-incremental-nano-poc.sh"


def test_incremental_poc_uses_file_backed_module_and_private_runtime_report() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "umask 077" in script
    assert "-m fun_voice.incremental_poc" in script
    assert "incremental-poc-report.json" in script
    assert "XDG_RUNTIME_DIR" in script
    assert "python -" not in script
    assert "MODEL_REVISION=\"master\"" in script
    assert "--revision \"${MODEL_REVISION}\"" in script
    assert "MODELSCOPE_OFFLINE=1" in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "rm -f \"${REPORT}\"" in script
