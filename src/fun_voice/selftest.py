"""Post-install self-test for the selected Fun Voice Ryan runtime.

Every check returns a ``name``/``status``/``detail`` triple; any failing check
makes the process exit non-zero. JSON output carries no sensitive data: no
audio paths, no transcription text, no focus tokens, or model cache layout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fun_voice.config import get_xdg_runtime_dir
from fun_voice.fcitx import FcitxClient, FcitxCommitError, default_socket_path
from fun_voice.preflight import (
    STATUS_FAIL,
    STATUS_PASS,
    CheckResult,
    probe_worker_health,
)
from fun_voice.runtime_selection import (
    AsrProfile,
    RuntimeSelection,
    RuntimeSelectionError,
    load_runtime_selection,
)

DAEMON_DIAGNOSTICS_TIMEOUT_SECONDS = 1.0
WORKER_SYSTEMD_TIMEOUT_SECONDS = 1.0
WORKER_HEALTH_TIMEOUT_SECONDS = 5.0
WORKER_HEALTH_MAX_BYTES = 64 * 1024

# Check names in report order (stable, used by tests and the human view).
CHECK_NAMES_SELFTEST: tuple[str, ...] = (
    "x11_hotkey",
    "pipewire",
    "fcitx_ping",
    "clipboard",
    "xtest_eligibility",
    "worker_health",
    "runtime_selection",
)


@dataclass(frozen=True)
class SelfTestResult:
    """Outcome of one self-test check (metrics only, no sensitive data)."""

    name: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelfTestReport:
    """Aggregate of all self-test checks plus the overall pass verdict."""

    checks: tuple[SelfTestResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.status == STATUS_PASS for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


# --- X11 hotkey check --------------------------------------------------------


def probe_hotkey_state(
    socket_path: str | Path | None = None,
    *,
    timeout: float = DAEMON_DIAGNOSTICS_TIMEOUT_SECONDS,
) -> dict[str, bool] | None:
    """Read the daemon's privacy-preserving X11 hotkey diagnostics.

    The response schema is deliberately closed: any extra, missing, or
    non-boolean fields are treated as unavailable rather than echoed in the
    self-test report.
    """
    if socket_path is None:
        xdg = get_xdg_runtime_dir()
        if not xdg:
            return None
        socket_path = Path(xdg) / "fun-voice-ryan" / "daemon.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(socket_path))
            conn.sendall(b'{"op":"diagnostics"}\n')
            data = bytearray()
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data.extend(chunk)
        response = json.loads(bytes(data).decode("utf-8"))
    except Exception:  # noqa: BLE001 - IPC failures map to an actionable check
        return None
    expected_keys = {"status", "hotkey_registered", "hotkey_press_seen"}
    if not isinstance(response, dict) or set(response) != expected_keys:
        return None
    registered = response.get("hotkey_registered")
    press_seen = response.get("hotkey_press_seen")
    if (
        response.get("status") != "ok"
        or not isinstance(registered, bool)
        or not isinstance(press_seen, bool)
    ):
        return None
    return {"hotkey_registered": registered, "hotkey_press_seen": press_seen}


def check_x11_hotkey(
    probe: Callable[[], dict[str, bool] | None] = probe_hotkey_state,
) -> SelfTestResult:
    """Verify that the daemon owns X11 ``Super+C`` and has seen one press."""
    try:
        state = probe()
    except Exception:  # noqa: BLE001 - injectable self-test probe must not crash
        state = None
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("hotkey_registered"), bool)
        or not isinstance(state.get("hotkey_press_seen"), bool)
        or set(state) != {"hotkey_registered", "hotkey_press_seen"}
    ):
        return SelfTestResult(
            "x11_hotkey", STATUS_FAIL, {"reason": "daemon diagnostics unavailable"}
        )
    detail = {
        "registered": state["hotkey_registered"],
        "press_seen": state["hotkey_press_seen"],
    }
    return SelfTestResult(
        "x11_hotkey", STATUS_PASS if all(detail.values()) else STATUS_FAIL, detail
    )


# --- Audio backend / runtime checks ------------------------------------------


def check_pipewire(
    *,
    which: Callable[[str], str | None] = shutil.which,
    runtime_dir: str | None = None,
) -> SelfTestResult:
    """Verify the PipeWire audio backend is reachable (socket or CLI)."""
    xdg = runtime_dir if runtime_dir is not None else get_xdg_runtime_dir()
    socket_ok = Path(xdg, "pipewire-0").exists() if xdg else False
    cli_ok = which("pw-cli") is not None or which("pipewire") is not None
    ok = socket_ok or cli_ok
    return SelfTestResult(
        "pipewire",
        STATUS_PASS if ok else STATUS_FAIL,
        {"socket": socket_ok, "cli": cli_ok},
    )


def _fcitx_socket_path() -> Path | None:
    try:
        return default_socket_path()
    except FcitxCommitError:
        return None


def check_fcitx_ping(client: FcitxClient | None = None) -> SelfTestResult:
    """Ping the Fcitx5 addon socket and expect ``PONG``.

    A missing socket (addon not loaded) is reported as a fail with an
    actionable hint rather than a bare connection error.
    """
    try:
        active = client if client is not None else FcitxClient()
        pong = active.ping()
    except FcitxCommitError as exc:
        detail: dict[str, Any] = {"error_class": type(exc).__name__}
        socket_path = _fcitx_socket_path()
        if socket_path is not None and not socket_path.exists():
            detail["hint"] = "fcitx5 addon not loaded (socket missing)"
        return SelfTestResult("fcitx_ping", STATUS_FAIL, detail)
    return SelfTestResult(
        "fcitx_ping", STATUS_PASS if pong else STATUS_FAIL, {"pong": pong}
    )


def check_clipboard(
    which: Callable[[str], str | None] = shutil.which,
) -> SelfTestResult:
    """Verify a clipboard tool (xclip or xsel) is available for the mirror."""
    for tool in ("xclip", "xsel"):
        if which(tool) is not None:
            return SelfTestResult("clipboard", STATUS_PASS, {"tool": tool})
    return SelfTestResult(
        "clipboard", STATUS_FAIL, {"hint": "install xclip or xsel"}
    )


def check_xtest_eligibility(
    make_display: Callable[[], Any] | None = None,
) -> SelfTestResult:
    """Verify the XTEST fallback is eligible: python-xlib importable + X display.

    ``make_display`` is injectable for tests; the default opens the real X11
    display via python-xlib and checks the XTEST extension is present.
    """
    try:
        from Xlib import display as xdisplay
        from Xlib.ext import xtest  # noqa: F401
    except ImportError as exc:
        return SelfTestResult(
            "xtest_eligibility", STATUS_FAIL, {"error_class": type(exc).__name__}
        )
    try:
        display = xdisplay.Display() if make_display is None else make_display()
    except Exception as exc:
        return SelfTestResult(
            "xtest_eligibility", STATUS_FAIL, {"error_class": type(exc).__name__}
        )
    try:
        ext = display.query_extension("XTEST")
        present = bool(getattr(ext, "present", 0)) if ext is not None else False
    except Exception as exc:
        display.close()
        return SelfTestResult(
            "xtest_eligibility", STATUS_FAIL, {"error_class": type(exc).__name__}
        )
    display.close()
    if not present:
        return SelfTestResult(
            "xtest_eligibility", STATUS_FAIL, {"reason": "XTEST extension absent"}
        )
    return SelfTestResult("xtest_eligibility", STATUS_PASS, {"extension": "xtest"})


# --- Worker / selected-runtime checks ---------------------------------------


def _selected_worker_socket_path(
    profile: AsrProfile, *, runtime_dir: str | None = None
) -> Path | None:
    """Return the selected profile's socket below the private runtime root."""
    xdg = runtime_dir if runtime_dir is not None else get_xdg_runtime_dir()
    if not xdg:
        return None
    socket_name = "worker.sock" if profile == "nano" else "worker-sensevoice.sock"
    return Path(xdg) / "fun-voice-ryan" / socket_name


def probe_selected_worker_health(
    socket_path: Path | None,
    *,
    timeout: float = WORKER_HEALTH_TIMEOUT_SECONDS,
) -> CheckResult:
    """Probe one selected worker without imposing the XPU POC health gate."""
    if socket_path is None:
        return CheckResult(
            "worker_health", STATUS_FAIL, {"reason": "runtime unavailable"}
        )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(socket_path))
            conn.sendall(b'{"op":"health"}\n')
            data = bytearray()
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > WORKER_HEALTH_MAX_BYTES:
                    raise RuntimeError("worker health response too large")
        line, _separator, _rest = bytes(data).partition(b"\n")
        response = json.loads(line.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - health failures are self-test data
        return CheckResult(
            "worker_health", STATUS_FAIL, {"error_class": type(exc).__name__}
        )
    if not isinstance(response, dict):
        return CheckResult(
            "worker_health", STATUS_FAIL, {"error_class": "ProtocolError"}
        )
    lifecycle_value = response.get("lifecycle")
    lifecycle = (
        lifecycle_value
        if lifecycle_value in {"loading", "ready", "inactive", "failed"}
        else "unknown"
    )
    model_ready = response.get("model_ready") is True
    ok = (
        response.get("status") == "ok"
        and model_ready
        and lifecycle == "ready"
    )
    return CheckResult(
        "worker_health",
        STATUS_PASS if ok else STATUS_FAIL,
        {"model_ready": model_ready, "lifecycle": lifecycle},
    )


def probe_worker_unit_state(profile: str) -> tuple[str, str] | None:
    """Return the loaded systemd unit's ``(LoadState, ActiveState)``.

    A missing worker socket is healthy only after both explicitly on-demand
    worker units are loaded and inactive.  Do not infer that state from a
    socket error alone: it would hide a missing unit or a worker crash.
    """
    if profile not in {"nano", "sensevoice"}:
        raise ValueError(f"unsupported worker profile: {profile}")
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                f"fun-voice-worker@{profile}.service",
                "--property=LoadState",
                "--property=ActiveState",
                "--value",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=WORKER_SYSTEMD_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = result.stdout.splitlines()
    if len(values) != 2:
        return None
    load_state, active_state = values
    if load_state != "loaded" or active_state not in {
        "active",
        "activating",
        "deactivating",
        "failed",
        "inactive",
        "reloading",
    }:
        return None
    return load_state, active_state


def check_worker_health(
    result: CheckResult | None = None,
    *,
    profiles: tuple[AsrProfile] | tuple[AsrProfile, AsrProfile],
    worker_state_probe: Callable[[str], tuple[str, str] | None] = (
        probe_worker_unit_state
    ),
) -> SelfTestResult:
    """Report a ready worker or the explicit, healthy on-demand idle state."""
    probe = result if result is not None else probe_worker_health()
    if (
        probe.status == STATUS_FAIL
        and probe.detail.get("error_class") == "FileNotFoundError"
    ):
        try:
            states = {
                profile: worker_state_probe(profile) for profile in profiles
            }
        except Exception:  # noqa: BLE001 - self-test must fail closed on probe errors
            states = {}
        if all(state == ("loaded", "inactive") for state in states.values()):
            return SelfTestResult(
                "worker_health",
                STATUS_PASS,
                {
                    "lifecycle": "on_demand_idle",
                    "profiles": {
                        profile: "inactive" for profile in profiles
                    },
                },
            )
    return SelfTestResult("worker_health", probe.status, probe.detail)


def _check_selected_worker_health(
    results: dict[AsrProfile, CheckResult],
    *,
    profiles: tuple[AsrProfile] | tuple[AsrProfile, AsrProfile],
    worker_state_probe: Callable[[str], tuple[str, str] | None],
) -> SelfTestResult:
    """Require each selected profile to be ready or explicitly on-demand idle."""
    statuses: dict[AsrProfile, str] = {}
    missing_profiles: list[AsrProfile] = []
    for profile in profiles:
        result = results[profile]
        if result.status == STATUS_PASS:
            statuses[profile] = "ready"
        elif result.detail.get("error_class") == "FileNotFoundError":
            missing_profiles.append(profile)
        else:
            statuses[profile] = "unavailable"

    for profile in missing_profiles:
        try:
            state = worker_state_probe(profile)
        except Exception:  # noqa: BLE001 - self-test must fail closed on probe errors
            state = None
        statuses[profile] = (
            "inactive" if state == ("loaded", "inactive") else "unavailable"
        )

    return SelfTestResult(
        "worker_health",
        (
            STATUS_PASS
            if all(value != "unavailable" for value in statuses.values())
            else STATUS_FAIL
        ),
        {"profiles": statuses},
    )


def check_runtime_selection(
    loader: Callable[[], RuntimeSelection] = load_runtime_selection,
) -> SelfTestResult:
    """Report the validated runtime that owns this installed service."""
    try:
        selection = loader()
    except RuntimeSelectionError:
        return SelfTestResult(
            "runtime_selection", STATUS_FAIL, {"reason": "invalid_or_missing"}
        )
    return SelfTestResult(
        "runtime_selection",
        STATUS_PASS,
        {
            "backend": selection.backend,
            "primary_profile": selection.primary_asr_profile,
            "enhanced": selection.enhanced_enabled,
        },
    )


# --- Orchestration ------------------------------------------------------------


def run_selftest(
    *,
    fcitx_client: FcitxClient | None = None,
    worker_probe: Callable[[Path | None], CheckResult] = probe_selected_worker_health,
    worker_state_probe: Callable[[str], tuple[str, str] | None] = (
        probe_worker_unit_state
    ),
    which: Callable[[str], str | None] = shutil.which,
    runtime_dir: str | None = None,
    make_display: Callable[[], Any] | None = None,
    hotkey_probe: Callable[[], dict[str, bool] | None] = probe_hotkey_state,
    selection_loader: Callable[[], RuntimeSelection] = load_runtime_selection,
) -> SelfTestReport:
    """Run every self-test check and aggregate the results."""
    try:
        selection = selection_loader()
    except RuntimeSelectionError:
        selection = None
    runtime_check = check_runtime_selection(
        (lambda: selection) if selection is not None else selection_loader
    )
    if selection is None:
        worker_check = SelfTestResult(
            "worker_health", STATUS_FAIL, {"reason": "invalid_or_missing"}
        )
    else:
        profiles = selection.policy().allowed_profiles
        results: dict[AsrProfile, CheckResult] = {}
        for profile in profiles:
            try:
                results[profile] = worker_probe(
                    _selected_worker_socket_path(profile, runtime_dir=runtime_dir)
                )
            except Exception as exc:  # noqa: BLE001 - injectable probes fail closed
                results[profile] = CheckResult(
                    "worker_health", STATUS_FAIL, {"error_class": type(exc).__name__}
                )
        worker_check = _check_selected_worker_health(
            results, profiles=profiles, worker_state_probe=worker_state_probe
        )
    checks: list[SelfTestResult] = [
        check_x11_hotkey(hotkey_probe),
        check_pipewire(which=which, runtime_dir=runtime_dir),
        check_fcitx_ping(fcitx_client),
        check_clipboard(which=which),
        check_xtest_eligibility(make_display=make_display),
        worker_check,
        runtime_check,
    ]
    return SelfTestReport(tuple(checks))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fun-voice-selftest",
        description="Run the Fun Voice Ryan post-install self-test.",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="output format (default: human-readable lines)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_selftest()
    if args.format == "json":
        print(result.to_json())
    else:
        for check in result.checks:
            print(f"{check.name}: {check.status}")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
