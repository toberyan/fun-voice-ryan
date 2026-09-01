"""Regression checks for the XPU POC shell harness contract."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run-nano-xpu-poc.sh"


def test_poc_script_keeps_samples_private_and_records_source_metadata() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "umask 077" in script
    assert "comp+=(" in script
    assert '"source"' in script
    assert '"language"' in script
    assert '"duration_s"' in script
