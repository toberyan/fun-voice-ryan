"""Static contract tests for X11 hotkey deployment and legacy DDE retirement."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

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


def test_runtime_builder_rejects_non_venv_path_before_sync(tmp_path: Path) -> None:
    python312 = shutil.which("python3.12")
    if python312 is None:
        pytest.skip("Python 3.12 is required for the validation regression")
    runtime = tmp_path / "data/fun-voice-ryan/runtimes/cpu"
    _make_runtime_parent_private(runtime)
    (runtime / "bin").mkdir(parents=True, mode=0o700)
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
    for expected in ("中下部", "深色", "浅色", "中文", "圆角", "焦点"):
        assert expected in checklist
