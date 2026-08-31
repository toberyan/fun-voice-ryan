"""Unit tests for the post-install self-test (fake DDE / X11 / sockets)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from fun_voice import selftest
from fun_voice.desktop import DdeKeybindingClient, DdeKeybindingError
from fun_voice.fcitx import FcitxCommitError
from fun_voice.preflight import (
    CHECK_NAMES,
    STATUS_FAIL,
    STATUS_PASS,
    CheckResult,
    PreflightReport,
)
from fun_voice.selftest import (
    SelfTestReport,
    SelfTestResult,
    check_bridge_timing,
    check_clipboard,
    check_fcitx_ping,
    check_pipewire,
    check_super_c_conflict,
    check_worker_health,
    check_xpu_hard_gate,
    check_xtest_eligibility,
    load_preflight_report,
    run_selftest,
)

# --- Fakes -------------------------------------------------------------------


class _FakeDdeClient(DdeKeybindingClient):
    """DDE client stub: answers lookup_conflict without touching busctl."""

    def __init__(self, owner: str | None = None, *, error: bool = False) -> None:
        super().__init__(runner=None)
        self._owner = owner
        self._error = error

    def lookup_conflict(self, hotkey: str) -> str | None:
        if self._error:
            raise DdeKeybindingError("org.deepin.dde.Keybinding1 not found")
        return self._owner


class _FakeFcitx:
    def __init__(
        self, pong: bool | None = None, error: FcitxCommitError | None = None
    ) -> None:
        self._pong = pong
        self._error = error

    def ping(self) -> bool:
        if self._error is not None:
            raise self._error
        return bool(self._pong)


class _FakeExt:
    def __init__(self, present: int) -> None:
        self.present = present


class _FakeDisplay:
    def __init__(self, present: int = 1, *, raise_on_query: bool = False) -> None:
        self._present = present
        self._raise = raise_on_query
        self.closed = False

    def query_extension(self, name: str) -> _FakeExt:
        if self._raise:
            raise RuntimeError("query failed")
        return _FakeExt(self._present)

    def close(self) -> None:
        self.closed = True


def _pass_report() -> PreflightReport:
    checks = tuple(CheckResult(name, STATUS_PASS, {}) for name in CHECK_NAMES)
    return PreflightReport(device="xpu:0", checks=checks, ready=True)


# --- Individual checks -------------------------------------------------------


def test_dde_service_pass() -> None:
    result = selftest.check_dde_service(_FakeDdeClient())
    assert result.name == "dde_service"
    assert result.status == STATUS_PASS
    assert result.detail["service"] == "org.deepin.dde.Keybinding1"


def test_dde_service_fail_when_service_missing() -> None:
    result = selftest.check_dde_service(_FakeDdeClient(error=True))
    assert result.status == STATUS_FAIL
    assert result.detail["error_class"] == "DdeKeybindingError"


def test_super_c_conflict_pass_when_owned_by_self() -> None:
    from fun_voice.desktop import DEFAULT_SHORTCUT_NAME

    result = check_super_c_conflict(_FakeDdeClient(owner=DEFAULT_SHORTCUT_NAME))
    assert result.status == STATUS_PASS
    assert result.detail["owner"] == DEFAULT_SHORTCUT_NAME


def test_super_c_conflict_pass_when_free() -> None:
    result = check_super_c_conflict(_FakeDdeClient(owner=None))
    assert result.status == STATUS_PASS
    assert result.detail["hotkey"] == "<Super>C"


def test_super_c_conflict_fail_when_owned() -> None:
    result = check_super_c_conflict(_FakeDdeClient(owner="Other App"))
    assert result.status == STATUS_FAIL
    assert result.detail["owner"] == "Other App"


def test_bridge_hold_timing_maps_key_state() -> None:
    result = check_bridge_timing()
    assert result.status == STATUS_PASS
    assert result.detail["held"] == "start_if_idle"
    assert result.detail["released"] == "stop"


def test_pipewire_pass_via_socket(tmp_path: Any) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "pipewire-0").touch()
    result = check_pipewire(which=lambda _name: None, runtime_dir=str(runtime))
    assert result.status == STATUS_PASS
    assert result.detail["socket"] is True


def test_pipewire_pass_via_cli() -> None:
    result = check_pipewire(which=lambda name: "/usr/bin/" + name, runtime_dir=None)
    assert result.status == STATUS_PASS
    assert result.detail["cli"] is True

def test_pipewire_fail_when_neither(tmp_path: Any) -> None:
    result = check_pipewire(
        which=lambda _name: None, runtime_dir=str(tmp_path / "nope")
    )
    assert result.status == STATUS_FAIL


def test_fcitx_ping_pass() -> None:
    result = check_fcitx_ping(_FakeFcitx(pong=True))  # type: ignore[arg-type]
    assert result.status == STATUS_PASS
    assert result.detail["pong"] is True


def test_fcitx_ping_fail_when_no_pong() -> None:
    result = check_fcitx_ping(_FakeFcitx(pong=False))  # type: ignore[arg-type]
    assert result.status == STATUS_FAIL


def test_fcitx_ping_fail_reports_error_class() -> None:
    error = FcitxCommitError("cannot connect")
    result = check_fcitx_ping(_FakeFcitx(error=error))  # type: ignore[arg-type]
    assert result.status == STATUS_FAIL
    assert result.detail["error_class"] == "FcitxCommitError"


def test_clipboard_pass_prefers_xclip() -> None:
    result = check_clipboard(which=lambda name: "/usr/bin/" + name)
    assert result.status == STATUS_PASS
    assert result.detail["tool"] == "xclip"


def test_clipboard_fail_when_no_tool() -> None:
    result = check_clipboard(which=lambda _name: None)
    assert result.status == STATUS_FAIL
    assert result.detail["hint"]


def test_xtest_eligibility_pass() -> None:
    result = check_xtest_eligibility(make_display=lambda: _FakeDisplay(present=1))
    assert result.status == STATUS_PASS
    assert result.detail["extension"] == "xtest"


def test_xtest_eligibility_fail_when_display_raises() -> None:
    def _boom() -> _FakeDisplay:
        raise RuntimeError("cannot open display")

    result = check_xtest_eligibility(make_display=_boom)
    assert result.status == STATUS_FAIL
    assert result.detail["error_class"] == "RuntimeError"


def test_xtest_eligibility_fail_when_extension_absent() -> None:
    result = check_xtest_eligibility(make_display=lambda: _FakeDisplay(present=0))
    assert result.status == STATUS_FAIL
    assert result.detail["reason"] == "XTEST extension absent"


def test_worker_health_converts_preflight_result() -> None:
    probe = CheckResult("worker_health", STATUS_PASS, {"device": "xpu"})
    result = check_worker_health(probe)
    assert result.name == "worker_health"
    assert result.status == STATUS_PASS
    assert result.detail["device"] == "xpu"


# --- XPU hard gate (reuses preflight report) ---------------------------------


def test_xpu_hard_gate_pass_when_all_nine_pass() -> None:
    result = check_xpu_hard_gate(_pass_report())
    assert result.status == STATUS_PASS
    assert result.detail["ready"] is True
    assert set(result.detail["gates"]) == set(CHECK_NAMES)


def test_xpu_hard_gate_fail_when_report_missing() -> None:
    result = check_xpu_hard_gate(None)
    assert result.status == STATUS_FAIL
    assert result.detail["reason"] == "no XPU POC report available"


def test_xpu_hard_gate_fail_when_gate_missing() -> None:
    checks = tuple(
        CheckResult(name, STATUS_PASS, {}) for name in CHECK_NAMES[:-1]
    )
    report = PreflightReport(device="xpu:0", checks=checks, ready=True)
    result = check_xpu_hard_gate(report)
    assert result.status == STATUS_FAIL
    assert result.detail["missing_gates"] == [CHECK_NAMES[-1]]


def test_xpu_hard_gate_fail_when_not_ready() -> None:
    checks = tuple(CheckResult(name, STATUS_PASS, {}) for name in CHECK_NAMES)
    report = PreflightReport(device="xpu:0", checks=checks, ready=False)
    result = check_xpu_hard_gate(report)
    assert result.status == STATUS_FAIL
    assert result.detail["ready"] is False


# --- Report / aggregation ----------------------------------------------------


def _all_pass_report() -> SelfTestReport:
    return run_selftest(
        report=_pass_report(),
        dde_client_factory=lambda: _FakeDdeClient(),
        fcitx_client=_FakeFcitx(pong=True),  # type: ignore[arg-type]
        worker_probe=lambda: CheckResult("worker_health", STATUS_PASS, {}),
        which=lambda name: "/usr/bin/" + name,
        runtime_dir=None,
        make_display=lambda: _FakeDisplay(present=1),
    )


def test_report_ok_when_all_pass() -> None:
    assert _all_pass_report().ok is True


def test_report_not_ok_when_one_check_fails() -> None:
    report = run_selftest(
        report=_pass_report(),
        dde_client_factory=lambda: _FakeDdeClient(owner="Other App"),
        fcitx_client=_FakeFcitx(pong=True),  # type: ignore[arg-type]
        worker_probe=lambda: CheckResult("worker_health", STATUS_PASS, {}),
        which=lambda name: "/usr/bin/" + name,
        runtime_dir=None,
        make_display=lambda: _FakeDisplay(present=1),
    )
    assert report.ok is False
    by_name = {check.name: check.status for check in report.checks}
    assert by_name["super_c_conflict"] == STATUS_FAIL


def test_every_result_has_name_status_detail() -> None:
    report = _all_pass_report()
    for check in report.checks:
        assert isinstance(check.name, str) and check.name
        assert check.status in (STATUS_PASS, STATUS_FAIL)
        assert isinstance(check.detail, dict)


def test_report_json_omits_sensitive_data() -> None:
    report = _all_pass_report()
    payload = report.to_json()
    assert "wav" not in payload
    assert "transcription" not in payload
    assert "model.pt" not in payload
    assert "/run/user" not in payload
    assert "SECRET" not in payload
    # The payload still contains all nine check names and their statuses.
    data = json.loads(payload)
    assert data["ok"] is True
    names = [check["name"] for check in data["checks"]]
    assert names == list(selftest.CHECK_NAMES_SELFTEST)
    for check in data["checks"]:
        assert set(check) == {"name", "status", "detail"}


# --- Persisted report loading ------------------------------------------------


def test_load_preflight_report_round_trips(tmp_path: Any) -> None:
    path = tmp_path / "poc-report.json"
    path.write_text(_pass_report().to_json() + "\n", encoding="utf-8")
    report = load_preflight_report(path)
    assert report is not None
    assert report.ready is True
    assert {c.name for c in report.checks} == set(CHECK_NAMES)


def test_load_preflight_report_missing_file() -> None:
    assert load_preflight_report("/nonexistent/poc-report.json") is None


def test_load_preflight_report_malformed(tmp_path: Any) -> None:
    path = tmp_path / "poc-report.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_preflight_report(path) is None


# --- CLI exit code -----------------------------------------------------------


def test_main_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failing = SelfTestReport(
        (SelfTestResult("dde_service", STATUS_FAIL, {"error_class": "X"}),)
    )
    monkeypatch.setattr(selftest, "load_preflight_report", lambda _p=None: None)
    monkeypatch.setattr(selftest, "run_selftest", lambda **_kw: failing)
    rc = selftest.main(["--format", "json"])
    assert rc == 1
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["ok"] is False


def test_main_exits_zero_when_all_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    passing = SelfTestReport(
        (SelfTestResult("dde_service", STATUS_PASS, {}),)
    )
    monkeypatch.setattr(selftest, "load_preflight_report", lambda _p=None: None)
    monkeypatch.setattr(selftest, "run_selftest", lambda **_kw: passing)
    rc = selftest.main(["--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
