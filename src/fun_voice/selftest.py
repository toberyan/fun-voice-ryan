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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from fun_voice.config import get_xdg_runtime_dir
from fun_voice.desktop import (
    DEFAULT_HOTKEY,
    DEFAULT_SHORTCUT_NAME,
    DdeKeybindingClient,
    DdeKeybindingError,
    HotkeyBridge,
    X11FocusGuard,
)
from fun_voice.fcitx import FcitxClient, FcitxCommitError, default_socket_path
from fun_voice.preflight import (
    CHECK_NAMES,
    STATUS_FAIL,
    STATUS_PASS,
    CheckResult,
    PreflightReport,
    probe_worker_health,
)

# The persisted POC report produced by a real ``run_preflight`` run.
REPORT_RELATIVE_PATH = "fun-voice-ryan/poc-report.json"

# Check names in report order (stable, used by tests and the human view).
CHECK_NAMES_SELFTEST: tuple[str, ...] = (
    "dde_service",
    "super_c_conflict",
    "bridge_hold_timing",
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


# --- Desktop / DDE checks ----------------------------------------------------


def check_dde_service(client: DdeKeybindingClient) -> SelfTestResult:
    """Verify the DDE Keybinding1 service answers on the session bus."""
    try:
        client.lookup_conflict(DEFAULT_HOTKEY)
    except DdeKeybindingError as exc:
        return SelfTestResult(
            "dde_service", STATUS_FAIL, {"error_class": type(exc).__name__}
        )
    return SelfTestResult("dde_service", STATUS_PASS, {"service": client.SERVICE})


def check_super_c_conflict(client: DdeKeybindingClient) -> SelfTestResult:
    """Verify ``Super+C`` is not owned by a *foreign* shortcut.

    Our own registration (from install) is not a conflict: after install the
    ``LookupConflictShortcut`` answer names our shortcut, which must still pass.
    """
    try:
        owner = client.lookup_conflict(DEFAULT_HOTKEY)
    except DdeKeybindingError as exc:
        return SelfTestResult(
            "super_c_conflict", STATUS_FAIL, {"error_class": type(exc).__name__}
        )
    foreign = owner is not None and owner != DEFAULT_SHORTCUT_NAME
    return SelfTestResult(
        "super_c_conflict",
        STATUS_FAIL if foreign else STATUS_PASS,
        {"hotkey": DEFAULT_HOTKEY, "owner": owner},
    )


# --- Bridge hold-timing POC --------------------------------------------------


def check_bridge_timing() -> SelfTestResult:
    """POC: the bridge maps C-held -> start_if_idle and C-released -> stop.

    The automated half covers the bridge's key-state mapping and the daemon's
    key-state read plus its 500 ms press confirmation. The real DDE hold-phase
    trigger (whether DDE actually invokes the action while the key is held) can
    only be confirmed by a human against the daemon journal.
    """

    class _FakeGuard:
        def __init__(self, down: bool) -> None:
            self._down = down

        def c_is_down(self) -> bool:
            return self._down

    def _op(down: bool) -> str | None:
        sent: list[bytes] = []
        bridge = HotkeyBridge(cast(X11FocusGuard, _FakeGuard(down)), sent.append)
        bridge.handle()
        if not sent:
            return None
        payload = json.loads(sent[0].decode("utf-8"))
        return str(payload.get("op")) if isinstance(payload, dict) else None

    detail: dict[str, Any] = {
        "held": _op(True),
        "released": _op(False),
        "automated": "bridge key-state mapping + 500 ms C-key confirmation",
        "manual_verify": (
            "hold Super+C and confirm `journalctl --user -u fun-voice-daemon` "
            "shows `c_pressed_at_trigger=true`"
        ),
        "docs": "docs/operations.md",
    }
    ok = detail["held"] == "start_if_idle" and detail["released"] == "stop"
    return SelfTestResult(
        "bridge_hold_timing", STATUS_PASS if ok else STATUS_FAIL, detail
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
    if missing:
        detail["missing_gates"] = missing
    ok = report.ready and not missing and not failed
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
    dde_client_factory: Callable[[], DdeKeybindingClient] = DdeKeybindingClient,
    fcitx_client: FcitxClient | None = None,
    worker_probe: Callable[[], CheckResult] = probe_worker_health,
    which: Callable[[str], str | None] = shutil.which,
    runtime_dir: str | None = None,
    make_display: Callable[[], Any] | None = None,
) -> SelfTestReport:
    """Run every self-test check and aggregate the results."""
    checks: list[SelfTestResult] = [
        check_dde_service(dde_client_factory()),
        check_super_c_conflict(dde_client_factory()),
        check_bridge_timing(),
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
