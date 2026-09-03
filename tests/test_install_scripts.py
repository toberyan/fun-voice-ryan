"""Static contract tests for X11 hotkey deployment and legacy DDE retirement."""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from fun_voice.runtime_selection import RuntimeSelection, write_runtime_selection

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


def test_installer_requires_and_validates_the_selected_runtime() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    assert "--runtime-selection" in install
    assert "load_runtime_selection" in install
    assert "RuntimeSelectionError" in install
    assert "import torch, funasr, modelscope, transformers, Xlib" in install
    assert "runtime_selection_invalid" in install
    assert "runtime_import_failed" in install
    for obsolete in (
        "POC_REPORT",
        "poc-report.json",
        "Nano POC backend",
        "uv sync --inexact",
        '${ROOT}/.venv/bin/fun-voice-',
    ):
        assert obsolete not in install


def test_installer_writes_all_launchers_through_one_closed_function() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    assert "install_launcher()" in install
    assert "RUNTIME_DATA_ROOT" in install
    assert "SELECTION_PYTHON" in install
    assert install.count('install_launcher "${script}"') == 1
    assert 'src="${ROOT}/.venv/bin/${script}"' not in install


def test_selected_runtime_adapter_only_delegates_to_closed_python_launcher() -> None:
    adapter = (ROOT / "scripts/run-selected-runtime.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in adapter
    assert '"${SELECTED_PYTHON}" -P -m fun_voice.runtime_launcher' in adapter
    assert "exec python3" not in adapter
    assert "selection.json" not in adapter
    assert "load_runtime_selection" not in adapter
    assert "torch" not in adapter


def _portable_install_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    project = tmp_path / "project"
    shutil.copytree(ROOT / "src", project / "src")
    for directory in ("scripts", "systemd", "native/fcitx5-fun-voice"):
        (project / directory).mkdir(parents=True, exist_ok=True)
    for relative in (
        "scripts/install-user.sh",
        "scripts/run-selected-runtime.sh",
        "systemd/fun-voice-worker@.service",
        "systemd/fun-voice-daemon.service",
        "systemd/fun-voice-session.desktop",
        "native/fcitx5-fun-voice/fcitx5-fun-voice.conf",
    ):
        shutil.copy2(ROOT / relative, project / relative)
    for relative in (
        "build/fcitx/fcitx5-fun-voice.so",
        "build/dtk-overlay/fun-voice-overlay",
    ):
        artifact = project / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("test artifact", encoding="utf-8")
        artifact.chmod(0o700)

    data_root = tmp_path / "non-default-data/fun-voice-ryan"
    runtime = data_root / "runtimes/cpu-0123456789abcdef0123456789abcdef"
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True, mode=0o700)
    for directory in (data_root, data_root / "runtimes", runtime, python.parent):
        directory.chmod(0o700)
    capture = tmp_path / "selected-process.env"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == '-P' && \"${2:-}\" == '-c' ]]; then exit 0; fi\n"
        "if [[ \"${1:-}\" == '-P' && \"${2:-}\" == '-m' "
        "&& \"${3:-}\" == 'fun_voice.runtime_launcher' ]]; then\n"
        f"  exec {sys.executable} \"$@\"\n"
        "fi\n"
        "if [[ \"${1:-}\" == '-P' && \"${2:-}\" == '-m' "
        "&& \"${3:-}\" == 'fun_voice.worker' ]]; then\n"
        "  {\n"
        "    printf 'PYTHONPATH=%s\\n' \"${PYTHONPATH-}\"\n"
        "    printf 'PYTHONHOME=%s\\n' \"${PYTHONHOME-}\"\n"
        "    printf 'XDG_DATA_HOME=%s\\n' \"${XDG_DATA_HOME-}\"\n"
        "    printf 'MODELSCOPE_CACHE=%s\\n' \"${MODELSCOPE_CACHE-}\"\n"
        "    printf 'FUN_VOICE_MODELS_ROOT=%s\\n' \"${FUN_VOICE_MODELS_ROOT-}\"\n"
        "    printf 'ARGV=%s\\n' \"$*\"\n"
        "  } > \"${FAKE_SELECTED_CAPTURE}\"\n"
        "  exit 0\n"
        "fi\n"
        "exit 91\n",
        encoding="utf-8",
    )
    python.chmod(0o700)
    selection = RuntimeSelection(
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
    manifest = write_runtime_selection(selection, data_root)

    home = tmp_path / "home"
    home.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    systemctl = fake_bin / "systemctl"
    systemctl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    systemctl.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "XDG_DATA_HOME": str(tmp_path / "ambient-wrong-data"),
            "XDG_RUNTIME_DIR": str(runtime_dir),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FAKE_SELECTED_CAPTURE": str(capture),
        }
    )
    return project, manifest, capture, environment


def test_install_to_launcher_to_worker_uses_verified_non_default_data_root(
    tmp_path: Path,
) -> None:
    project, manifest, capture, environment = _portable_install_fixture(tmp_path)
    installed = subprocess.run(
        [
            str(project / "scripts/install-user.sh"),
            "--runtime-selection",
            str(manifest),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr

    launch_environment = environment.copy()
    launch_environment.update(
        {
            "PYTHONPATH": "/tmp/malicious-development-package",
            "PYTHONHOME": "/tmp/malicious-python-home",
            "XDG_DATA_HOME": "/tmp/malicious-data-home",
            "MODELSCOPE_CACHE": "/tmp/malicious-model-cache",
            "FUN_VOICE_MODELS_ROOT": "/tmp/malicious-model-root",
        }
    )
    launched = subprocess.run(
        [
            str(Path(environment["HOME"]) / ".local/bin/fun-voice-worker"),
            "--profile",
            "sensevoice",
        ],
        env=launch_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert launched.returncode == 0, launched.stderr

    values = dict(
        line.split("=", 1)
        for line in capture.read_text(encoding="utf-8").splitlines()
    )
    data_root = manifest.parent.parent
    assert values["PYTHONPATH"] == str(project / "src")
    assert values["PYTHONHOME"] == ""
    assert values["XDG_DATA_HOME"] == str(data_root.parent)
    assert values["MODELSCOPE_CACHE"] == str(data_root / "models")
    assert values["FUN_VOICE_MODELS_ROOT"] == str(data_root / "models")
    assert values["ARGV"] == "-P -m fun_voice.worker --profile sensevoice"
    installed_worker_unit = Path(environment["HOME"]) / (
        ".config/systemd/user/fun-voice-worker@.service"
    )
    unit_text = installed_worker_unit.read_text(encoding="utf-8")
    assert "%h/.local/share/fun-voice-ryan/models" not in unit_text


def test_legacy_worker_disable_failure_preserves_its_unit_file(
    tmp_path: Path,
) -> None:
    project, manifest, _, environment = _portable_install_fixture(tmp_path)
    legacy = Path(environment["HOME"]) / (
        ".config/systemd/user/fun-voice-worker.service"
    )
    legacy.parent.mkdir(parents=True, mode=0o700)
    legacy.write_text("legacy unit\n", encoding="utf-8")
    fake_systemctl = Path(environment["PATH"].split(":", 1)[0]) / "systemctl"
    fake_systemctl.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == '--user disable --now fun-voice-worker.service' ]]; "
        "then exit 1; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_systemctl.chmod(0o700)

    completed = subprocess.run(
        [
            str(project / "scripts/install-user.sh"),
            "--runtime-selection",
            str(manifest),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ERROR(systemd): cannot stop and disable retired warm worker" in (
        completed.stderr
    )
    assert legacy.read_text(encoding="utf-8") == "legacy unit\n"


def test_uninstaller_preserves_portable_runtime_and_model_state() -> None:
    uninstall = (ROOT / "scripts/uninstall-user.sh").read_text(encoding="utf-8")
    assert "MODELS_DIR=" not in uninstall
    assert "--purge" not in uninstall
    assert 'rm -rf "${MODELS_DIR}"' not in uninstall
    assert "/runtimes" not in uninstall
    assert "/selection.json" not in uninstall


@pytest.mark.parametrize(
    "unsafe_kind",
    (
        "shared-xdg",
        "symlink-xdg",
        "dangling-symlink-xdg",
        "symlink-runtime",
        "symlink-capture",
    ),
)
def test_uninstaller_rejects_unsafe_runtime_tree_before_any_deletion(
    tmp_path: Path, unsafe_kind: str
) -> None:
    home = tmp_path / "home"
    launcher = home / ".local/bin/fun-voice-daemon"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("owned launcher", encoding="utf-8")

    real_xdg = tmp_path / "real-runtime"
    real_xdg.mkdir(mode=0o700)
    xdg_runtime = real_xdg
    if unsafe_kind == "shared-xdg":
        real_xdg.chmod(0o755)
    elif unsafe_kind == "symlink-xdg":
        xdg_runtime = tmp_path / "runtime-link"
        xdg_runtime.symlink_to(real_xdg, target_is_directory=True)
    elif unsafe_kind == "dangling-symlink-xdg":
        xdg_runtime = tmp_path / "runtime-link"
        xdg_runtime.symlink_to(tmp_path / "missing-runtime", target_is_directory=True)

    real_app = real_xdg / "fun-voice-ryan-real"
    app_runtime = real_xdg / "fun-voice-ryan"
    if unsafe_kind == "symlink-runtime":
        real_app.mkdir(mode=0o700)
        app_runtime.symlink_to(real_app, target_is_directory=True)
    else:
        app_runtime.mkdir(mode=0o700)
        real_app = app_runtime

    real_capture = tmp_path / "capture-real"
    capture = real_app / "capture"
    if unsafe_kind == "symlink-capture":
        real_capture.mkdir(mode=0o700)
        capture.symlink_to(real_capture, target_is_directory=True)
    else:
        capture.mkdir(mode=0o700)
        real_capture = capture
    shard = real_capture / "keep.pcm"
    shard.write_text("not application-authorized for deletion", encoding="utf-8")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    systemctl = fake_bin / "systemctl"
    systemctl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    systemctl.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "XDG_RUNTIME_DIR": str(xdg_runtime),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )

    completed = subprocess.run(
        [str(ROOT / "scripts/uninstall-user.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ERROR(runtime-safety)" in completed.stderr
    assert launcher.read_text(encoding="utf-8") == "owned launcher"
    assert shard.read_text(encoding="utf-8") == (
        "not application-authorized for deletion"
    )


def test_installer_rejects_an_invalid_selection_before_user_writes(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    data_home = tmp_path / "data"
    home.mkdir()
    data_home.mkdir()
    missing = data_home / "fun-voice-ryan/runtime/selection.json"
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "XDG_DATA_HOME": str(data_home),
            "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
        }
    )

    completed = subprocess.run(
        [
            str(ROOT / "scripts/install-user.sh"),
            "--runtime-selection",
            str(missing),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ERROR(runtime_selection_invalid)" in completed.stderr
    assert list(home.iterdir()) == []


def test_xpu_environment_uses_native_funasr_without_vllm_runtime() -> None:
    environment = (ROOT / "scripts/create-runtime-env.sh").read_text(encoding="utf-8")
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
    environment = (ROOT / "scripts/create-runtime-env.sh").read_text(encoding="utf-8")
    lock = (ROOT / "requirements-xpu.lock").read_text(encoding="utf-8")
    assert 'LOCK_FILE="${ROOT_DIR}/requirements-${BACKEND}.lock"' in environment
    assert 'FUNASR_TARBALL_SHA256="' in environment
    assert '"${UV}" pip sync' in environment
    assert "--require-hashes" in environment
    assert '"${LOCK_FILE}"' in environment
    assert "--hash=sha256:" in lock
    assert "funasr @" not in lock
    assert "--no-deps" in environment
    assert "--no-build-isolation" in environment
    assert '"${FUNASR_SRC}"' in environment


def test_funasr_install_uses_locked_environment_without_network(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    python312 = shutil.which("python3.12")
    if uv is None or python312 is None:
        pytest.skip("uv and Python 3.12 are required for the builder regression")

    data_home = tmp_path / "data"
    runtime = data_home / "fun-voice-ryan/runtimes/cpu"
    models = data_home / "fun-voice-ryan/models"
    runtime.parent.mkdir(parents=True, mode=0o700)
    subprocess.run(
        [uv, "venv", str(runtime), "--python", python312],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    runtime.chmod(0o700)
    (runtime / "bin").chmod(0o700)
    (runtime / "pyvenv.cfg").chmod(0o600)
    purelib = Path(
        subprocess.check_output(
            [
                str(runtime / "bin/python"),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            text=True,
        ).strip()
    )
    for module in ("torch", "funasr", "modelscope", "transformers", "Xlib"):
        package = purelib / module
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")

    archive_source = tmp_path / "archive-source"
    archive_source.mkdir()
    (archive_source / "README").write_text("verified source", encoding="utf-8")
    archive = runtime / ".funasr-src.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(archive_source / "README", arcname="FunASR-pinned/README")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    uv_log = tmp_path / "uv.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"${FAKE_UV_LOG}\"\n"
        "if [[ \"$1 $2\" == 'pip install' "
        "&& \" $* \" != *' --no-build-isolation '* ]]; then exit 42; fi\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o700)
    fake_sha = fake_bin / "sha256sum"
    fake_sha.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_sha.chmod(0o700)
    network_marker = tmp_path / "network-attempted"
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        f"#!/usr/bin/env bash\ntouch {network_marker}\nexit 99\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o700)

    environment = os.environ.copy()
    environment.update(
        {
            "XDG_DATA_HOME": str(data_home),
            "HOME": str(tmp_path / "home"),
            "UV": str(fake_uv),
            "FAKE_UV_LOG": str(uv_log),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    completed = subprocess.run(
        [
            str(ROOT / "scripts/create-runtime-env.sh"),
            "--backend",
            "cpu",
            "--runtime-dir",
            str(runtime),
            "--models-root",
            str(models),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "pip install" in uv_log.read_text(encoding="utf-8")
    assert "--no-build-isolation" in uv_log.read_text(encoding="utf-8")
    assert not network_marker.exists()


def test_xpu_environment_rebuilds_funasr_from_the_verified_archive() -> None:
    environment = (ROOT / "scripts/create-runtime-env.sh").read_text(encoding="utf-8")
    assert 'FUNASR_DOWNLOAD="$(mktemp' in environment
    assert 'FUNASR_STAGE="$(mktemp -d' in environment
    assert 'tar xzf "${FUNASR_TARBALL}" -C "${FUNASR_STAGE}"' in environment
    assert 'mv "${FUNASR_STAGE}" "${FUNASR_SRC}"' in environment


def test_runtime_inputs_pin_exact_backend_distributions() -> None:
    expected = {
        "cuda": (
            "https://download.pytorch.org/whl/cu130",
            "torch==2.13.0+cu130",
            "torchaudio==2.11.0+cu130",
        ),
        "xpu": (
            "https://download.pytorch.org/whl/xpu",
            "torch==2.13.0+xpu",
            "torchaudio==2.11.0+xpu",
        ),
        "cpu": (
            "https://download.pytorch.org/whl/cpu",
            "torch==2.13.0+cpu",
            "torchaudio==2.11.0+cpu",
        ),
    }
    for backend, values in expected.items():
        source = (ROOT / f"requirements-{backend}.in").read_text(encoding="utf-8")
        assert "--index-url https://pypi.tuna.tsinghua.edu.cn/simple" in source
        assert "--index-strategy unsafe-best-match" in source
        assert f"--extra-index-url {values[0]}" in source
        assert values[1] in source
        assert values[2] in source
        assert "modelscope==1.39.1" in source
        assert "transformers==5.16.1" in source


def test_all_backend_locks_are_hashed_and_exclude_heavy_unapproved_runtimes() -> None:
    for backend in ("cuda", "xpu", "cpu"):
        lock = (ROOT / f"requirements-{backend}.lock").read_text(encoding="utf-8")
        assert "--hash=sha256:" in lock
        assert f"torch==2.13.0+{backend if backend != 'cuda' else 'cu130'}" in lock
        assert "modelscope==1.39.1" in lock
        assert "transformers==5.16.1" in lock
        for forbidden in (
            "vllm==",
            "vllm-xpu-kernels==",
            "cuda-python==",
            "flashinfer-python==",
        ):
            assert forbidden not in lock


def test_lock_compiler_verifies_source_and_compiles_every_backend() -> None:
    compiler = (ROOT / "scripts/compile-runtime-locks.sh").read_text(encoding="utf-8")
    assert "8cd758c0ced576516b05a749194e6a94cdd38f99" in compiler
    assert (
        "f8b2c9b9954c463b5c0e433bd1f2706b5c6c28f16f755f55ec66365960c06da0"
        in compiler
    )
    assert "mktemp -d" in compiler
    assert "trap cleanup EXIT" in compiler
    assert "--generate-hashes" in compiler
    assert "--python-version 3.12" in compiler
    assert "--no-emit-package funasr" in compiler
    assert '"${UV}" pip compile' in compiler
    assert '--extra-index-url "${EXTRA_INDEX}"' in compiler
    assert "for BACKEND in cuda xpu cpu" in compiler
    assert "find" in compiler and "-lname '/*' -delete" in compiler


def test_generic_runtime_builder_is_isolated_and_hash_locked() -> None:
    builder = (ROOT / "scripts/create-runtime-env.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in builder
    assert "umask 077" in builder
    assert "trap cleanup EXIT" in builder
    assert "--backend" in builder and "--runtime-dir" in builder
    assert "--models-root" in builder
    assert (
        'RUNTIMES_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/'
        'fun-voice-ryan/runtimes"' in builder
    )
    assert "--allow-project-venv" in builder
    assert '"${UV}" venv "${RUNTIME_DIR}" --python 3.12' in builder
    assert '"${UV}" pip sync' in builder
    assert "--require-hashes" in builder
    assert "uv sync" not in builder
    assert "--editable" not in builder
    assert 'FUNASR_SRC="${RUNTIME_DIR}/.funasr-src"' in builder
    assert "snapshot_download" not in builder
    for module in ("torch", "funasr", "modelscope", "transformers", "Xlib"):
        assert f'"{module}"' in builder


def test_xpu_builder_is_only_an_explicit_project_venv_wrapper() -> None:
    wrapper = (ROOT / "scripts/create-xpu-env.sh").read_text(encoding="utf-8")
    assert "create-runtime-env.sh" in wrapper
    assert "--backend xpu" in wrapper
    assert "--allow-project-venv" in wrapper
    assert 'FUN_VOICE_VENV_DIR:-${ROOT_DIR}/.venv' in wrapper
    assert "torch.xpu.is_available" in wrapper
    assert "torch.xpu.total_memory" in wrapper
    assert "curl" not in wrapper


@pytest.mark.parametrize("target_name", ["home", "project", "unrelated"])
def test_xpu_wrapper_rejects_non_project_venv_override(
    tmp_path: Path,
    target_name: str,
) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/create-xpu-env.sh", scripts / "create-xpu-env.sh")
    shutil.copy2(
        ROOT / "scripts/create-runtime-env.sh", scripts / "create-runtime-env.sh"
    )
    shutil.copy2(ROOT / "requirements-xpu.lock", project / "requirements-xpu.lock")
    home = tmp_path / "home"
    home.mkdir()
    unrelated = tmp_path / "unrelated/.venv"
    unrelated.mkdir(parents=True)
    targets = {"home": home, "project": project, "unrelated": unrelated}
    data_home = tmp_path / "data"
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "XDG_DATA_HOME": str(data_home),
            "FUN_VOICE_VENV_DIR": str(targets[target_name]),
            "UV": "/bin/false",
        }
    )

    completed = subprocess.run(
        [str(scripts / "create-xpu-env.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "only the repository .venv" in completed.stderr


def test_first_run_wrapper_contains_only_bootstrap_entrypoint() -> None:
    wrapper = (ROOT / "scripts/initialize-first-run.sh").read_text(encoding="utf-8")
    assert 'PYTHONPATH="${ROOT_DIR}/src"' in wrapper
    assert 'python3 -P -m fun_voice.bootstrap "$@"' in wrapper
    for forbidden in (
        "Fun-ASR",
        "SenseVoice",
        "Qwen",
        "cuda:0",
        "xpu:0",
        "curl",
        "rm ",
    ):
        assert forbidden not in wrapper


def test_first_run_entrypoint_cannot_be_shadowed_by_current_directory_package(
    tmp_path: Path,
) -> None:
    forged_package = tmp_path / "fun_voice"
    forged_package.mkdir()
    (forged_package / "__init__.py").write_text(
        "raise RuntimeError('forged package imported')\n", encoding="utf-8"
    )

    completed = subprocess.run(
        [str(ROOT / "scripts/initialize-first-run.sh"), "--dry-run"],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == '{"candidates":["cuda","xpu","cpu"]}'


def test_runtime_builder_rejects_unknown_option_without_installing(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [str(ROOT / "scripts/create-runtime-env.sh"), "--unknown"],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "unknown option" in completed.stderr


def test_runtime_builder_rejects_path_outside_xdg_runtime_root(
    tmp_path: Path,
) -> None:
    data_home = tmp_path / "data"
    home = tmp_path / "home"
    data_home.mkdir()
    home.mkdir()
    environment = os.environ.copy()
    environment.update({"XDG_DATA_HOME": str(data_home), "HOME": str(home)})
    completed = subprocess.run(
        [
            str(ROOT / "scripts/create-runtime-env.sh"),
            "--backend",
            "cpu",
            "--runtime-dir",
            str(tmp_path / "outside"),
            "--models-root",
            str(data_home / "fun-voice-ryan/models"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "outside the application runtimes root" in completed.stderr
    assert not (tmp_path / "outside").exists()


def test_runtime_builder_rejects_symlink_runtime_without_installing(
    tmp_path: Path,
) -> None:
    data_home = tmp_path / "data"
    runtimes = data_home / "fun-voice-ryan/runtimes"
    real_runtime = runtimes / "real"
    real_runtime.mkdir(parents=True)
    link = runtimes / "cpu"
    link.symlink_to(real_runtime, target_is_directory=True)
    environment = os.environ.copy()
    environment.update({"XDG_DATA_HOME": str(data_home), "HOME": str(tmp_path)})
    completed = subprocess.run(
        [
            str(ROOT / "scripts/create-runtime-env.sh"),
            "--backend",
            "cpu",
            "--runtime-dir",
            str(link),
            "--models-root",
            str(data_home / "fun-voice-ryan/models"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "must not be a symlink" in completed.stderr


def _run_builder_against_existing_runtime(
    tmp_path: Path,
    runtime: Path,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    data_home = tmp_path / "data"
    fake_uv = tmp_path / "recording-uv"
    uv_marker = tmp_path / "uv-invoked"
    fake_uv.write_text(
        f"#!/usr/bin/env bash\ntouch {uv_marker}\nexit 1\n", encoding="utf-8"
    )
    fake_uv.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "XDG_DATA_HOME": str(data_home),
            "HOME": str(tmp_path / "home"),
            "UV": str(fake_uv),
        }
    )
    completed = subprocess.run(
        [
            str(ROOT / "scripts/create-runtime-env.sh"),
            "--backend",
            "cpu",
            "--runtime-dir",
            str(runtime),
            "--models-root",
            str(data_home / "fun-voice-ryan/models"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, uv_marker


def _make_runtime_parent_private(runtime: Path) -> None:
    runtime.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    current = runtime.parent
    while current.name != "data":
        current.chmod(0o700)
        current = current.parent


def test_runtime_builder_rejects_existing_python311_before_sync(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    python311 = shutil.which("python3.11")
    if uv is None or python311 is None:
        pytest.skip("uv and Python 3.11 are required for the validation regression")
    runtime = tmp_path / "data/fun-voice-ryan/runtimes/cpu"
    _make_runtime_parent_private(runtime)
    subprocess.run(
        [uv, "venv", str(runtime), "--python", python311],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    runtime.chmod(0o700)
    (runtime / "bin").chmod(0o700)
    (runtime / "pyvenv.cfg").chmod(0o600)

    completed, uv_marker = _run_builder_against_existing_runtime(
        tmp_path, runtime
    )

    assert completed.returncode != 0
    assert "secure Python 3.12 virtual environment" in completed.stderr
    assert not uv_marker.exists()


def test_runtime_builder_rejects_forged_python_executable_before_sync(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "data/fun-voice-ryan/runtimes/cpu"
    _make_runtime_parent_private(runtime)
    (runtime / "bin").mkdir(parents=True, mode=0o700)
    runtime.chmod(0o700)
    forged = runtime / "bin/python"
    forged.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    forged.chmod(0o700)
    (runtime / "pyvenv.cfg").write_text(
        "implementation = CPython\n"
        "version_info = 3.12.0\n"
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    (runtime / "pyvenv.cfg").chmod(0o600)

    completed, uv_marker = _run_builder_against_existing_runtime(
        tmp_path, runtime
    )

    assert completed.returncode != 0
    assert "secure Python 3.12 virtual environment" in completed.stderr
    assert not uv_marker.exists()


def test_runtime_builder_rejects_zero_exit_non_python_symlink_before_sync(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "data/fun-voice-ryan/runtimes/cpu"
    _make_runtime_parent_private(runtime)
    (runtime / "bin").mkdir(parents=True, mode=0o700)
    runtime.chmod(0o700)
    (runtime / "bin/python").symlink_to("/bin/true")
    (runtime / "pyvenv.cfg").write_text(
        "implementation = CPython\n"
        "version_info = 3.12.0\n"
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    (runtime / "pyvenv.cfg").chmod(0o600)

    completed, uv_marker = _run_builder_against_existing_runtime(
        tmp_path, runtime
    )

    assert completed.returncode != 0
    assert "secure Python 3.12 virtual environment" in completed.stderr
    assert not uv_marker.exists()


def test_runtime_builder_rejects_non_venv_path_before_sync(tmp_path: Path) -> None:
    python312 = shutil.which("python3.12")
    if python312 is None:
        pytest.skip("Python 3.12 is required for the validation regression")
    runtime = tmp_path / "data/fun-voice-ryan/runtimes/cpu"
    _make_runtime_parent_private(runtime)
    (runtime / "bin").mkdir(parents=True, mode=0o700)
    runtime.chmod(0o700)
    (runtime / "bin/python").symlink_to(python312)

    completed, uv_marker = _run_builder_against_existing_runtime(
        tmp_path, runtime
    )

    assert completed.returncode != 0
    assert "secure Python 3.12 virtual environment" in completed.stderr
    assert not uv_marker.exists()


def test_runtime_builder_rejects_group_writable_external_interpreter(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    python312 = shutil.which("python3.12")
    alternate_groups = [group for group in os.getgroups() if group != os.getegid()]
    if uv is None or python312 is None or not alternate_groups:
        pytest.skip("uv, Python 3.12, and a secondary group are required")
    runtime = tmp_path / "data/fun-voice-ryan/runtimes/cpu"
    _make_runtime_parent_private(runtime)
    subprocess.run(
        [uv, "venv", str(runtime), "--python", python312],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    runtime.chmod(0o700)
    (runtime / "bin").chmod(0o700)
    (runtime / "pyvenv.cfg").chmod(0o600)
    external_python = tmp_path / "python3.12"
    shutil.copy2(Path(python312).resolve(), external_python)
    os.chown(external_python, -1, alternate_groups[0])
    external_python.chmod(0o770)
    (runtime / "bin/python").unlink()
    (runtime / "bin/python").symlink_to(external_python)

    completed, uv_marker = _run_builder_against_existing_runtime(
        tmp_path, runtime
    )

    assert completed.returncode != 0
    assert "secure Python 3.12 virtual environment" in completed.stderr
    assert not uv_marker.exists()


def test_runtime_builder_rejects_interpreter_under_world_writable_parent(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    python312 = shutil.which("python3.12")
    if uv is None or python312 is None:
        pytest.skip("uv and Python 3.12 are required for the symlink regression")
    runtime = tmp_path / "data/fun-voice-ryan/runtimes/cpu"
    _make_runtime_parent_private(runtime)
    subprocess.run(
        [uv, "venv", str(runtime), "--python", python312],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    runtime.chmod(0o700)
    (runtime / "bin").chmod(0o700)
    (runtime / "pyvenv.cfg").chmod(0o600)
    replaceable = tmp_path / "replaceable"
    replaceable.mkdir(mode=0o777)
    replaceable.chmod(0o777)
    external_python = replaceable / "python3.12"
    shutil.copy2(Path(python312).resolve(), external_python)
    external_python.chmod(0o755)
    (runtime / "bin/python").unlink()
    (runtime / "bin/python").symlink_to(external_python)

    completed, uv_marker = _run_builder_against_existing_runtime(
        tmp_path, runtime
    )

    assert completed.returncode != 0
    assert "secure Python 3.12 virtual environment" in completed.stderr
    assert not uv_marker.exists()


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


def test_installer_fails_closed_before_deleting_the_legacy_worker_unit() -> None:
    install = (ROOT / "scripts/install-user.sh").read_text(encoding="utf-8")
    disable = "systemctl --user disable --now fun-voice-worker.service"
    remove = 'rm -f "${SYSTEMD_USER_DIR}/fun-voice-worker.service"'

    assert f"{disable} 2>/dev/null || true" not in install
    assert disable in install
    assert '|| die "systemd" "cannot stop and disable retired warm worker"' in install
    assert install.index(disable) < install.index(remove)


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
    for expected in ("中下部", "深色", "浅色", "中文", "圆角", "焦点"):
        assert expected in checklist


def test_user_docs_describe_portable_first_run_selection() -> None:
    for relative in ("README.md", "docs/operations.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for command in (
            "scripts/initialize-first-run.sh",
            "scripts/initialize-first-run.sh --backend cpu",
            "scripts/initialize-first-run.sh --force-reselect",
            "fun-voice-selftest --format json",
        ):
            assert command in text
        for policy in (
            "CUDA → Intel XPU → CPU",
            "SenseVoice-only",
            "selection.json",
            "0700",
            "0600",
            "BF16",
            "FP16",
            "不回退",
        ):
            assert policy in text
        assert "仓库 `.venv`" in text


def test_example_config_leaves_backend_policy_to_first_initialization() -> None:
    example = (ROOT / "scripts/config.example.toml").read_text(encoding="utf-8")
    assert 'device = "xpu:0"' not in example
    assert 'dtype = "bf16"' not in example
    assert "首次初始化" in example
    assert "CPU" in example and "不生效" in example
    for knob in (
        "max_source_characters",
        "max_new_tokens",
        "timeout_seconds",
        "protected_terms",
    ):
        assert knob in example


def test_acceptance_checklist_has_mutually_exclusive_backend_sections() -> None:
    checklist = (ROOT / "docs/acceptance-checklist.md").read_text(
        encoding="utf-8"
    )
    for heading in ("CUDA 机器", "Intel XPU 机器", "纯 CPU 机器"):
        assert heading in checklist
    for required in (
        "fun-voice-worker@nano",
        "SenseVoice-only",
        "Qwen",
        "CAM++",
        "Super+C",
        "clipboard",
        "BF16",
    ):
        assert required in checklist


def test_current_docs_do_not_claim_unimplemented_speaker_or_structured_api() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    operations = (ROOT / "docs/operations.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs/acceptance-checklist.md").read_text(
        encoding="utf-8"
    )

    assert "当前版本未实现 CAM++ 加载、说话人分离/身份或结构化结果接口" in readme
    assert "CAM++ 按需" not in readme
    assert "CAM++ 按需" not in operations
    assert "结构化结果只经接口提供" not in operations
    assert "分别比较 Nano 原始结果与串行 Qwen 修正" not in operations
    assert "说话人请求才按需运行 CAM++" not in checklist
    assert "CAM++ 只在说话人能力实际请求时启动" not in checklist


def test_native_builder_reports_an_actionable_category_when_cmake_fails(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    script = project / "scripts/build-native-artifacts.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/build-native-artifacts.sh", script)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    cmake = fake_bin / "cmake"
    cmake.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    cmake.chmod(0o700)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"

    completed = subprocess.run(
        [str(script)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "native_prerequisite" in completed.stderr


def test_clean_machine_acceptance_initializes_before_daemon_status_check() -> None:
    checklist = (ROOT / "docs/acceptance-checklist.md").read_text(
        encoding="utf-8"
    )
    first_initialization = checklist.index("scripts/initialize-first-run.sh --backend")
    daemon_status = checklist.index(
        "systemctl --user status fun-voice-daemon.service --no-pager"
    )
    assert first_initialization < daemon_status


def test_xpu_poc_is_documented_as_optional_explicit_diagnostic() -> None:
    document = (ROOT / "docs/xpu-poc.md").read_text(encoding="utf-8")
    assert "显式 Intel XPU 诊断" in document
    assert "不能阻断 CUDA 或 CPU 初始化" in document
    assert "桌面服务上线的**硬门**" not in document
