"""Static contract tests for X11 hotkey deployment and legacy DDE retirement."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_new_install_has_no_dde_registration_or_bridge_console_script() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "AddCustomShortcut" not in install
    assert "register-dde-shortcut.sh" not in install
    assert "fun-voice-bridge =" not in project


def test_installer_retires_only_verified_legacy_bridge_shortcut() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    assert "GetShortcutCommand" in install
    assert "fun-voice-bridge" in install
    assert "DeleteCustomShortcut" in install
    assert install.index("GetShortcutCommand") < install.index("DeleteCustomShortcut")


def test_dde_and_bridge_source_assets_are_removed() -> None:
    assert not (ROOT / "src/fun_voice/bridge.py").exists()
    assert not (ROOT / "scripts/register-dde-shortcut.sh").exists()
    assert not (ROOT / "scripts/unregister-dde-shortcut.sh").exists()
    assert not (ROOT / "scripts/start-session-bridge.sh").exists()


def test_session_importer_only_restarts_daemon_and_desktop_uses_it() -> None:
    session = (ROOT / "scripts/import-session-environment.sh").read_text(
        encoding="utf-8"
    )
    desktop = (ROOT / "systemd/fun-voice-session.desktop").read_text(
        encoding="utf-8"
    )
    assert "import-environment DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS" in session
    assert "restart fun-voice-daemon.service" in session
    assert "fun-voice-bridge" not in session
    assert "start_if_idle" not in session
    assert "import-session-environment.sh" in desktop


def test_uninstaller_has_no_dde_or_bridge_runtime_path() -> None:
    uninstall = (ROOT / "scripts/uninstall-user.sh").read_text(encoding="utf-8")
    assert "fun-voice-bridge" not in uninstall
    assert "unregister-dde-shortcut" not in uninstall


def test_daemon_does_not_restart_after_hotkey_grab_failure() -> None:
    service = (ROOT / "systemd/fun-voice-daemon.service").read_text(encoding="utf-8")
    assert "RestartPreventExitStatus=2" in service


def test_installer_requires_the_current_xpu_runtime_not_only_a_stale_report() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    assert "RUNTIME_MODULES=(torch funasr modelscope transformers Xlib)" in install
    assert "Nano POC backend is not native_funasr_pytorch" in install
    assert "XPU runtime imports verified" in install
    assert "uv sync --inexact" in install


def test_xpu_environment_uses_native_funasr_without_vllm_runtime() -> None:
    environment = (ROOT / "scripts/create-xpu-env.sh").read_text(encoding="utf-8")
    lock = (ROOT / "requirements-xpu.lock").read_text(encoding="utf-8")
    assert "vllm-xpu-kernels" not in environment
    assert "wheels.vllm.ai" not in environment
    assert "vllm==" not in lock
    assert "vllm-xpu-kernels==" not in lock
    assert "cuda-python==" not in lock
    assert "flashinfer-python==" not in lock
    assert "torch==2.13.0+xpu" in lock
    assert "torchaudio==2.11.0+xpu" in lock
    assert "transformers==5.16.1" in lock
    assert "python-xlib==" in lock


def test_xpu_environment_syncs_the_hashed_lock_before_deployment() -> None:
    environment = (ROOT / "scripts/create-xpu-env.sh").read_text(encoding="utf-8")
    lock = (ROOT / "requirements-xpu.lock").read_text(encoding="utf-8")
    assert 'LOCK_FILE="${ROOT_DIR}/requirements-xpu.lock"' in environment
    assert 'FUNASR_TARBALL_SHA256="' in environment
    assert '"${UV}" pip sync' in environment
    assert "--require-hashes" in environment
    assert '"${LOCK_FILE}"' in environment
    assert "--hash=sha256:" in lock
    assert "funasr @" not in lock
    assert "--no-deps" in environment
    assert '"${FUNASR_SRC}"' in environment


def test_xpu_environment_rebuilds_funasr_from_the_verified_archive() -> None:
    environment = (ROOT / "scripts/create-xpu-env.sh").read_text(encoding="utf-8")
    assert 'FUNASR_DOWNLOAD="$(mktemp' in environment
    assert 'FUNASR_STAGE="$(mktemp -d' in environment
    assert 'tar xzf "${FUNASR_TARBALL}" -C "${FUNASR_STAGE}"' in environment
    assert 'mv "${FUNASR_STAGE}" "${FUNASR_SRC}"' in environment


def test_xpu_poc_doc_records_the_verified_native_run() -> None:
    document = (ROOT / "docs/xpu-poc.md").read_text(encoding="utf-8")
    assert "状态:**已通过（真实运行）**" in document
    assert "状态:**需要重新运行**" not in document
    assert "`nano_decoder_xpu` | pass | `backend=native_funasr_pytorch`" in document


def test_installer_defers_daemon_start_until_graphical_session_import() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    assert "fun-voice-worker@.service" in install
    assert "disable --now fun-voice-worker.service" in install
    assert "enable --now fun-voice-worker.service" not in install
    assert "disable --now fun-voice-daemon.service" in install
    assert "enable fun-voice-daemon.service" not in install
    assert "restart fun-voice-daemon.service" not in install
    assert not (ROOT / "systemd/fun-voice-worker.service").exists()


def test_current_user_docs_describe_x11_not_dde_bridge() -> None:
    for relative in (
        "README.md",
        "docs/operations.md",
        "docs/acceptance-checklist.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "DDE 快捷键" not in text
        assert "bridge_hold_timing" not in text
        assert "X11" in text


def test_operations_document_benchmark_preload_and_serial_qwen() -> None:
    text = (ROOT / "docs/operations.md").read_text(encoding="utf-8")
    assert "fun-voice-benchmark" in text
    assert "预加载" in text
    assert "停止 Nano" in text


def test_install_and_uninstall_include_all_public_console_scripts() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    uninstall = (ROOT / "scripts/uninstall-user.sh").read_text(encoding="utf-8")
    for script in ("fun-voice-corrector", "fun-voice-benchmark"):
        assert script in install
        assert script in uninstall


def test_historical_design_links_to_the_x11_replacement() -> None:
    design = (
        ROOT
        / "docs/superpowers/specs"
        / "2026-08-31-fun-asr-nano-intel-xpu-voice-assistant-design.md"
    ).read_text(encoding="utf-8")
    assert "2026-09-01-x11-hotkey-replacement-design.md" in design


def test_installer_validates_and_installs_the_private_dtk_overlay_binary() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    assert 'OVERLAY_BIN="${ROOT}/build/dtk-overlay/fun-voice-overlay"' in install
    assert 'OVERLAY_INSTALL_DIR="${HOME}/.local/lib/fun-voice-ryan"' in install
    assert (
        'install_file "${OVERLAY_BIN}" '
        '"${OVERLAY_INSTALL_DIR}/fun-voice-overlay" 755' in install
    )
    assert "enable --now fun-voice-overlay" not in install


def test_uninstaller_removes_only_the_owned_overlay_binary() -> None:
    uninstall = (ROOT / "scripts/uninstall-user.sh").read_text(encoding="utf-8")
    assert 'OVERLAY_INSTALL_DIR="${HOME}/.local/lib/fun-voice-ryan"' in uninstall
    assert 'remove_file "${OVERLAY_INSTALL_DIR}/fun-voice-overlay"' in uninstall


def test_user_docs_cover_dtk_build_fallback_and_visual_acceptance() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    operations = (ROOT / "docs/operations.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs/acceptance-checklist.md").read_text(encoding="utf-8")
    assert "libdtk6gui-dev" in readme
    assert "libdtk6widget-dev" in readme
    assert "native/dtk-overlay" in readme
    assert "无悬浮窗" in operations
    for expected in ("底部居中", "深色", "浅色", "中文", "圆角", "焦点"):
        assert expected in checklist
