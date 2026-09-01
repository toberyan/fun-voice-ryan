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
    assert "RUNTIME_MODULES=(torch vllm funasr modelscope)" in install
    assert "XPU runtime imports verified" in install
    assert "uv sync --inexact" in install


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


def test_historical_design_links_to_the_x11_replacement() -> None:
    design = (
        ROOT
        / "docs/superpowers/specs"
        / "2026-08-31-fun-asr-nano-intel-xpu-voice-assistant-design.md"
    ).read_text(encoding="utf-8")
    assert "2026-09-01-x11-hotkey-replacement-design.md" in design
