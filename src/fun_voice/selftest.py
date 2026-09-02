"""Post-install self-test for Fun Voice Ryan.

Runs a battery of cheap desktop/runtime checks plus the persisted XPU hard-gate
report. Every check returns a ``name``/``status``/``detail`` triple; any failing
check makes the process exit non-zero. JSON output carries no sensitive data:
no audio paths, no transcription text, no focus tokens, no model cache layout.

The XPU hard gate reuses :func:`fun_voice.preflight.run_preflight`'s persisted
report (produced by a real run during the POC) and re-checks its nine gates via
:data:`fun_voice.preflight.CHECK_NAMES`; the worker-health check reuses
:func:`fun_voice.preflight.probe_worker_health`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fun_voice.config import get_xdg_runtime_dir
from fun_voice.fcitx import FcitxClient, FcitxCommitError, default_socket_path
from fun_voice.preflight import (
    CHECK_NAMES,
    NANO_BACKEND,
    STATUS_FAIL,
    STATUS_PASS,
    CheckResult,
    PreflightReport,
    probe_worker_health,
)

# The persisted POC report produced by a real ``run_preflight`` run.
REPORT_RELATIVE_PATH = "fun-voice-ryan/poc-report.json"
DAEMON_DIAGNOSTICS_TIMEOUT_SECONDS = 1.0

# Check names in report order (stable, used by tests and the human view).
CHECK_NAMES_SELFTEST: tuple[str, ...] = (
    "x11_hotkey",
    "pipewire",
    "fcitx_ping",
    "clipboard",
    "xtest_eligibility",
    "worker_health",
    "xpu_hard_gate",
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


# --- Worker / XPU checks (reuse preflight) -----------------------------------


def check_worker_health(result: CheckResult | None = None) -> SelfTestResult:
    """Reuse ``probe_worker_health`` to report the worker socket health."""
    probe = result if result is not None else probe_worker_health()
    return SelfTestResult("worker_health", probe.status, probe.detail)


def check_xpu_hard_gate(report: PreflightReport | None) -> SelfTestResult:
    """Re-check the nine XPU hard gates from a ``run_preflight`` report.

    ``report`` is the persisted output of a real :func:`run_preflight` (the
    POC); ``None`` means no report is available and the gate fails.
    """
    if report is None:
        return SelfTestResult(
            "xpu_hard_gate", STATUS_FAIL, {"reason": "no XPU POC report available"}
        )
    gates = {check.name: check.status for check in report.checks}
    missing = [name for name in CHECK_NAMES if name not in gates]
    failed = [name for name in CHECK_NAMES if gates.get(name) != STATUS_PASS]
    detail: dict[str, Any] = {
        "ready": report.ready,
        "device": report.device,
        "gates": gates,
    }
    decoder = next(
        (check for check in report.checks if check.name == "nano_decoder_xpu"), None
    )
    backend = decoder.detail.get("backend") if decoder is not None else None
    detail["backend"] = backend
    if missing:
        detail["missing_gates"] = missing
    ok = report.ready and not missing and not failed and backend == NANO_BACKEND
    return SelfTestResult(
        "xpu_hard_gate", STATUS_PASS if ok else STATUS_FAIL, detail
    )


# --- Persisted report loading ------------------------------------------------


def default_report_path() -> Path | None:
    """Return the persisted POC report path, or ``None`` without a runtime dir."""
    xdg = get_xdg_runtime_dir()
    if not xdg:
        return None
    return Path(xdg) / REPORT_RELATIVE_PATH


def load_preflight_report(path: str | Path | None = None) -> PreflightReport | None:
    """Load and validate the persisted ``run_preflight`` report, else ``None``."""
    report_path = Path(path) if path is not None else default_report_path()
    if report_path is None or not report_path.is_file():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    raw_checks = data.get("checks")
    if not isinstance(raw_checks, list):
        return None
    checks: list[CheckResult] = []
    for raw in raw_checks:
        if not isinstance(raw, dict):
            return None
        name = raw.get("name")
        status = raw.get("status")
        detail = raw.get("detail")
        if not isinstance(name, str) or not isinstance(status, str):
            return None
        if not isinstance(detail, dict):
            detail = {}
        checks.append(CheckResult(name=name, status=status, detail=detail))
    return PreflightReport(
        device=str(data.get("device") or ""),
        checks=tuple(checks),
        ready=bool(data.get("ready")),
    )


# --- Orchestration ------------------------------------------------------------


def run_selftest(
    *,
    report: PreflightReport | None = None,
    fcitx_client: FcitxClient | None = None,
    worker_probe: Callable[[], CheckResult] = probe_worker_health,
    which: Callable[[str], str | None] = shutil.which,
    runtime_dir: str | None = None,
    make_display: Callable[[], Any] | None = None,
    hotkey_probe: Callable[[], dict[str, bool] | None] = probe_hotkey_state,
) -> SelfTestReport:
    """Run every self-test check and aggregate the results."""
    checks: list[SelfTestResult] = [
        check_x11_hotkey(hotkey_probe),
        check_pipewire(which=which, runtime_dir=runtime_dir),
        check_fcitx_ping(fcitx_client),
        check_clipboard(which=which),
        check_xtest_eligibility(make_display=make_display),
        check_worker_health(worker_probe()),
        check_xpu_hard_gate(report),
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
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="override the persisted XPU preflight report path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = load_preflight_report(args.report)
    result = run_selftest(report=report)
    if args.format == "json":
        print(result.to_json())
    else:
        for check in result.checks:
            print(f"{check.name}: {check.status}")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
