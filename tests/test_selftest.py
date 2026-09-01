"""Unit tests for the post-install self-test (fake X11/runtime sockets)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from fun_voice import selftest
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
    check_clipboard,
    check_fcitx_ping,
    check_pipewire,
    check_worker_health,
    check_x11_hotkey,
    check_xpu_hard_gate,
    check_xtest_eligibility,
    load_preflight_report,
    probe_hotkey_state,
    run_selftest,
)


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


def _probe_reply(
    monkeypatch: pytest.MonkeyPatch, response: object
) -> tuple[dict[str, bytes], dict[str, bool] | None]:
    sent: dict[str, bytes] = {}
    payload = json.dumps(response).encode("utf-8") + b"\n"

    class _FakeSocket:
        def __enter__(self) -> _FakeSocket:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def settimeout(self, timeout: float) -> None:
            return None

        def connect(self, path: str) -> None:
            return None

        def sendall(self, data: bytes) -> None:
            sent["request"] = data

        def recv(self, size: int) -> bytes:
            return payload

    monkeypatch.setattr(selftest.socket, "socket", lambda *_args: _FakeSocket())
    return sent, probe_hotkey_state("/tmp/fun-voice-daemon.sock")


# --- X11 hotkey --------------------------------------------------------------


def test_x11_hotkey_passes_after_real_press() -> None:
    result = check_x11_hotkey(
        lambda: {"hotkey_registered": True, "hotkey_press_seen": True}
    )
    assert result.status == STATUS_PASS
    assert result.detail == {"registered": True, "press_seen": True}


def test_x11_hotkey_fails_before_any_press_without_sensitive_data() -> None:
    result = check_x11_hotkey(
        lambda: {"hotkey_registered": True, "hotkey_press_seen": False}
    )
    assert result.status == STATUS_FAIL
    assert result.detail == {"registered": True, "press_seen": False}


def test_x11_hotkey_fails_when_daemon_cannot_be_reached() -> None:
    result = check_x11_hotkey(lambda: None)
    assert result.status == STATUS_FAIL
    assert result.detail["reason"] == "daemon diagnostics unavailable"


def test_hotkey_probe_reads_only_expected_boolean_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent, result = _probe_reply(
        monkeypatch,
        {"status": "ok", "hotkey_registered": True, "hotkey_press_seen": False},
    )
    assert sent["request"] == b'{"op":"diagnostics"}\n'
    assert result == {"hotkey_registered": True, "hotkey_press_seen": False}


@pytest.mark.parametrize(
    "response",
    [
        {"status": "ok", "hotkey_registered": 1, "hotkey_press_seen": True},
        {
            "status": "ok",
            "hotkey_registered": True,
            "hotkey_press_seen": False,
            "extra": 1,
        },
        {"status": "error", "hotkey_registered": True, "hotkey_press_seen": True},
    ],
)
def test_hotkey_probe_rejects_malformed_or_extra_payload(
    monkeypatch: pytest.MonkeyPatch, response: object
) -> None:
    _sent, result = _probe_reply(monkeypatch, response)
    assert result is None


# --- Remaining runtime checks -----------------------------------------------


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
    checks = tuple(CheckResult(name, STATUS_PASS, {}) for name in CHECK_NAMES[:-1])
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
        fcitx_client=_FakeFcitx(pong=True),  # type: ignore[arg-type]
        worker_probe=lambda: CheckResult("worker_health", STATUS_PASS, {}),
        which=lambda name: "/usr/bin/" + name,
        runtime_dir=None,
        make_display=lambda: _FakeDisplay(present=1),
        hotkey_probe=lambda: {"hotkey_registered": True, "hotkey_press_seen": True},
    )


def test_report_ok_when_all_pass() -> None:
    assert _all_pass_report().ok is True


def test_report_not_ok_when_x11_hotkey_not_seen() -> None:
    report = run_selftest(
        report=_pass_report(),
        fcitx_client=_FakeFcitx(pong=True),  # type: ignore[arg-type]
        worker_probe=lambda: CheckResult("worker_health", STATUS_PASS, {}),
        which=lambda name: "/usr/bin/" + name,
        runtime_dir=None,
        make_display=lambda: _FakeDisplay(present=1),
        hotkey_probe=lambda: {"hotkey_registered": True, "hotkey_press_seen": False},
    )
    assert report.ok is False
    by_name = {check.name: check.status for check in report.checks}
    assert by_name["x11_hotkey"] == STATUS_FAIL


def test_run_selftest_has_one_x11_hotkey_check_and_no_dde_checks() -> None:
    names = [check.name for check in _all_pass_report().checks]
    assert names == list(selftest.CHECK_NAMES_SELFTEST)
    assert names.count("x11_hotkey") == 1
    assert {"dde_service", "super_c_conflict", "bridge_hold_timing"}.isdisjoint(names)


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
    data = json.loads(payload)
    assert data["ok"] is True
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
    failing = SelfTestReport((SelfTestResult("x11_hotkey", STATUS_FAIL, {}),))
    monkeypatch.setattr(selftest, "load_preflight_report", lambda _p=None: None)
    monkeypatch.setattr(selftest, "run_selftest", lambda **_kw: failing)
    rc = selftest.main(["--format", "json"])
    assert rc == 1
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is False


def test_main_exits_zero_when_all_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    passing = SelfTestReport((SelfTestResult("x11_hotkey", STATUS_PASS, {}),))
    monkeypatch.setattr(selftest, "load_preflight_report", lambda _p=None: None)
    monkeypatch.setattr(selftest, "run_selftest", lambda **_kw: passing)
    rc = selftest.main(["--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
