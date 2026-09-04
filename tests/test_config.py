"""Tests for runtime path resolution and permission policy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import fun_voice.config as config_module
from fun_voice.config import (
    DAEMON_SOCKET_NAME,
    DIRECTORY_MODE,
    FCITX_SOCKET_NAME,
    FILE_MODE,
    RUNTIME_DIR_NAME,
    SOCKET_MODE,
    WORKER_SOCKET_NAME,
    ActiveSessionConfig,
    Config,
    ConfigError,
    EffectiveRuntimeConfig,
    InferenceConfig,
    OverlayConfig,
    ResourcePolicy,
    build_runtime_paths,
    effective_runtime_config,
    load_config,
    resolve_runtime_dir,
    validate_active_session_config,
    validate_inference_config,
    validate_overlay_config,
)
from fun_voice.runtime_selection import RuntimeSelection


def _cpu_selection(root: Path) -> RuntimeSelection:
    python = root / "runtimes" / "cpu" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o700)
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
    assert cfg.fcitx_commit_timeout_ms == 500
    assert cfg.allow_x11_paste_fallback is True
    assert cfg.inference == InferenceConfig()
    assert cfg.inference.gpu_memory_utilization == 0.15
    assert cfg.inference.max_model_len == 1536
    assert cfg.inference.worker_failsafe_idle_seconds == 1800
    assert cfg.inference.allow_sensevoice_fallback is True
    assert cfg.active_session == ActiveSessionConfig()
    assert cfg.overlay == OverlayConfig()


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
                "commit_timeout_ms = 2500",
                "allow_x11_paste_fallback = false",
                "",
                "[inference]",
                'device = "xpu:0"',
                'dtype = "bf16"',
                "gpu_memory_utilization = 0.2",
                "max_model_len = 1024",
                "idle_unload_seconds = 90",
                "allow_sensevoice_fallback = false",
                "enforce_eager = false",
                "",
                "[active_session]",
                'policy = "memory_saver"',
                "provisional_enabled = true",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.audio_source == "alsa_input.custom"
    assert cfg.fcitx_commit_timeout_ms == 2500
    assert cfg.allow_x11_paste_fallback is False
    assert cfg.inference == InferenceConfig(
        device="xpu:0",
        dtype="bf16",
        gpu_memory_utilization=0.2,
        max_model_len=1024,
        worker_failsafe_idle_seconds=1800,
        allow_sensevoice_fallback=False,
        enforce_eager=False,
    )
    assert cfg.active_session == ActiveSessionConfig(
        policy=ResourcePolicy.MEMORY_SAVER,
        active_idle_seconds=120,
        worker_failsafe_idle_seconds=1800,
        provisional_enabled=True,
    )


def test_load_config_discards_legacy_runtime_device_and_dtype_preferences(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[inference]\ndevice = 'cpu'\ndtype = 'float32'\n"
        "[active_session]\ndevice = 'cpu'\n"
        "[correction]\ndevice = 'cpu'\ndtype = 'float32'\n"
        "[speaker_identity]\ndevice = 'cpu'\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.inference.device == "xpu:0"
    assert config.inference.dtype == "bf16"
    assert config.active_session.device == "xpu:0"
    assert config.enhanced.correction_device == "xpu:0"
    assert config.enhanced.correction_dtype == "bf16"
    assert config.enhanced.identity_device == "xpu:0"


def test_inference_validation_uses_legacy_xpu_without_policy_or_cpu_policy(
    tmp_path: Path,
) -> None:
    cpu = InferenceConfig(
        device="cpu", dtype="float32", allow_sensevoice_fallback=False
    )

    with pytest.raises(ConfigError, match="xpu:0"):
        validate_inference_config(cpu)

    assert validate_inference_config(cpu, _cpu_selection(tmp_path).policy()) == cpu


def test_effective_runtime_config_cpu_overrides_toml_devices(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[inference]\ndevice = 'xpu:0'\n[enhanced]\nenabled = true\n"
        "[correction]\ndevice = 'xpu:0'\ndtype = 'bf16'\n",
        encoding="utf-8",
    )

    effective = effective_runtime_config(load_config(path), _cpu_selection(tmp_path))

    assert isinstance(effective, EffectiveRuntimeConfig)
    assert effective.inference.device == "cpu"
    assert effective.inference.dtype == "float32"
    assert effective.primary_asr_profile == "sensevoice"
    assert effective.fallback_asr_profile is None
    assert effective.enhanced.enabled is False
    assert effective.speaker_enabled is False


def test_load_config_parses_overlay_layout(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[overlay]\nvertical_center_ratio = 0.85\nwidth_px = 900\nfont_scale = 1.2\n",
        encoding="utf-8",
    )

    assert load_config(path).overlay == OverlayConfig(
        vertical_center_ratio=0.85,
        width_px=900,
        font_scale=1.2,
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (OverlayConfig(vertical_center_ratio=0.49), "overlay.vertical_center_ratio"),
        (OverlayConfig(vertical_center_ratio=0.86), "overlay.vertical_center_ratio"),
        (OverlayConfig(width_px=419), "overlay.width_px"),
        (OverlayConfig(width_px=1001), "overlay.width_px"),
        (OverlayConfig(font_scale=0.79), "overlay.font_scale"),
        (OverlayConfig(font_scale=1.81), "overlay.font_scale"),
    ],
)
def test_overlay_config_rejects_out_of_range_values(
    value: OverlayConfig, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        validate_overlay_config(value)


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ("width_px = true", "overlay.width_px"),
        ('vertical_center_ratio = "low"', "overlay.vertical_center_ratio"),
        ("font_scale = nan", "overlay.font_scale"),
    ],
)
def test_load_config_rejects_invalid_overlay_field_types(
    tmp_path: Path, toml: str, message: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[overlay]\n{toml}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("gpu_memory_utilization", "0.09", "0.10"),
        ("gpu_memory_utilization", "0.21", "0.20"),
        ("max_model_len", "1023", "1024"),
        ("max_model_len", "1537", "1536"),
        ("worker_failsafe_idle_seconds", "1799", "1800"),
        ("worker_failsafe_idle_seconds", "1801", "1800"),
    ],
)
def test_load_config_rejects_unsafe_model_lifecycle_bounds(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[inference]\n{key} = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("policy", "active_idle_seconds"),
    [
        (ResourcePolicy.MEMORY_SAVER, 480),
        (ResourcePolicy.BALANCED, 120),
        (ResourcePolicy.SUSTAINED, 480),
    ],
)
def test_active_session_rejects_window_not_fixed_for_policy(
    policy: ResourcePolicy, active_idle_seconds: int
) -> None:
    with pytest.raises(ConfigError, match="active_session.active_idle_seconds"):
        validate_active_session_config(
            ActiveSessionConfig(
                policy=policy,
                active_idle_seconds=active_idle_seconds,
            )
        )


def test_active_session_rejects_non_policy_device_and_nonfixed_failsafe(
    tmp_path: Path,
) -> None:
    policy = _cpu_selection(tmp_path).policy()
    with pytest.raises(ConfigError, match="active_session.device"):
        validate_active_session_config(ActiveSessionConfig(), policy)
    with pytest.raises(ConfigError, match="worker_failsafe_idle_seconds"):
        validate_active_session_config(
            ActiveSessionConfig(device="cpu", worker_failsafe_idle_seconds=1799),
            policy,
        )


def test_enhanced_inference_rejects_non_policy_corrector(tmp_path: Path) -> None:
    policy = _cpu_selection(tmp_path).policy()
    with pytest.raises(
        ConfigError, match="correction.device must match selected runtime"
    ):
        config_module.validate_enhanced_inference_config(
            config_module.EnhancedInferenceConfig(), policy
        )


def test_enhanced_inference_uses_low_kv_bounds_for_qwen(tmp_path: Path) -> None:
    policy = _cpu_selection(tmp_path).policy()
    value = config_module.validate_enhanced_inference_config(
        config_module.EnhancedInferenceConfig(
            enabled=False,
            correction_device="cpu",
            correction_dtype="float32",
            identity_device="cpu",
        ),
        policy,
    )

    assert value.correction_max_source_characters == 512
    assert value.correction_max_new_tokens == 512
    assert value.enabled is False

    with pytest.raises(ConfigError, match="correction.max_new_tokens"):
        config_module.validate_enhanced_inference_config(
            config_module.EnhancedInferenceConfig(
                enabled=False,
                correction_device="cpu",
                correction_dtype="float32",
                identity_device="cpu",
                correction_max_new_tokens=513
            ),
            policy,
        )


def test_live_qwen_limits_are_loaded_from_config(tmp_path: Path) -> None:
    policy = _cpu_selection(tmp_path).policy()
    value = config_module.validate_enhanced_inference_config(
        config_module.EnhancedInferenceConfig(
            enabled=False,
            correction_device="cpu",
            correction_dtype="float32",
            identity_device="cpu",
        ),
        policy,
    )

    assert value.correction_timeout_seconds == 30
    assert value.correction_max_source_characters == 512
    assert value.correction_max_new_tokens == 512


def test_load_config_parses_live_qwen_limits_and_protected_terms(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(
            [
                "[correction]",
                "max_source_characters = 400",
                "max_new_tokens = 256",
                "timeout_seconds = 20",
                'protected_terms = ["OpenAI SDK", "FunASR"]',
            ]
        ),
        encoding="utf-8",
    )

    correction = load_config(path).enhanced

    assert correction.correction_max_source_characters == 400
    assert correction.correction_max_new_tokens == 256
    assert correction.correction_timeout_seconds == 20
    assert correction.correction_protected_terms == ("OpenAI SDK", "FunASR")


def test_load_config_rejects_non_positive_commit_timeout(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[input_method]\ncommit_timeout_ms = 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="commit_timeout_ms"):
        load_config(path)


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
