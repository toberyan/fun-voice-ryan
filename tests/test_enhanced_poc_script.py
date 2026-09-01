"""Regression checks for the enhanced XPU POC shell harness contract."""

from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run-enhanced-xpu-poc.sh"
MODULE = Path(__file__).parents[1] / "src" / "fun_voice" / "enhanced_poc.py"


def test_enhanced_poc_uses_qwen35_transformers_xpu_without_cpu_fallback() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    module = MODULE.read_text(encoding="utf-8")
    assert "Qwen/Qwen3.5-0.8B" in script
    assert "Qwen3_5ForConditionalGeneration" in module
    assert ".to(DEVICE)" in module
    assert 'device="cpu"' not in script + module
    assert "fallback_to_cpu" not in script + module


def test_enhanced_poc_covers_sensevoice_and_low_kv_models() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    module = MODULE.read_text(encoding="utf-8")
    assert "iic/SenseVoiceSmall" in script
    assert "--sensevoice-dir" in script
    assert "sensevoice_xpu" in module
    assert "load_nano_engine" in module
    assert "max_new_tokens=4" in module


def test_enhanced_poc_releases_nano_before_loading_qwen() -> None:
    module = MODULE.read_text(encoding="utf-8")
    assert "del nano" in module
    assert "torch.xpu.empty_cache()" in module
    assert module.index("del nano") < module.index("gates.extend(_qwen_gates")


def test_enhanced_poc_accepts_funasr_yaml_snapshot_metadata() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "config.yaml" in script
    assert "compgen -G" not in script


def test_enhanced_poc_waits_for_snapshot_metadata_before_loading() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "wait_for_snapshot_metadata" in script
    assert "SNAPSHOT_READY_TIMEOUT_SECONDS" in script


def test_enhanced_poc_uses_modelscope_camplus_cache_name() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "iic--speech_campplus_sv_zh-cn_16k-common/snapshots" in script


def test_enhanced_poc_runs_xpu_loader_from_importable_module() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "-m fun_voice.enhanced_poc" in script
