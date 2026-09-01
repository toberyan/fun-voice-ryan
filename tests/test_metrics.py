"""Privacy boundaries and aggregation behavior for runtime metrics."""

from __future__ import annotations

from fun_voice.metrics import MetricsLedger


def test_metrics_summary_has_percentiles_but_no_sensitive_values() -> None:
    ledger = MetricsLedger(max_entries=2)
    first = ledger.begin()
    ledger.record(first, asr_ms=20, asr_profile="nano")
    second = ledger.begin()
    ledger.record(second, asr_ms=40, asr_profile="sensevoice")

    summary = ledger.summary()

    assert summary["asr_ms"] == {"p50": 30, "p95": 39}
    assert summary["asr_profile"] == {"nano": 1, "sensevoice": 1}
    assert "text" not in repr(summary)
    assert "audio" not in repr(summary)


def test_metrics_ledger_is_bounded_and_rejects_unknown_fields() -> None:
    ledger = MetricsLedger(max_entries=1)
    discarded = ledger.begin()
    retained = ledger.begin()
    ledger.record(discarded, asr_ms=20)
    ledger.record(retained, asr_ms=40)

    assert ledger.summary()["count"] == 1
    assert ledger.summary()["asr_ms"] == {"p50": 40, "p95": 40}

    try:
        ledger.record(retained, text="must never be retained")
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unknown metric field must be rejected")
