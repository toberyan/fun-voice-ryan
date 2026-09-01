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


def test_metrics_aggregate_private_stage_durations_and_warmup_state() -> None:
    ledger = MetricsLedger()
    row = ledger.begin()
    ledger.record(
        row,
        preload_runtime_load_ms=21,
        preload_warmup_ms=8,
        asr_worker_ms=12,
        asr_generate_ms=7,
        asr_release_ms=3,
        nano_warmup="ready",
    )

    summary = ledger.summary()

    assert summary["preload_runtime_load_ms"] == {"p50": 21, "p95": 21}
    assert summary["preload_warmup_ms"] == {"p50": 8, "p95": 8}
    assert summary["asr_worker_ms"] == {"p50": 12, "p95": 12}
    assert summary["asr_generate_ms"] == {"p50": 7, "p95": 7}
    assert summary["asr_release_ms"] == {"p50": 3, "p95": 3}
    assert summary["nano_warmup"] == {"ready": 1}
    assert "worker_response" not in repr(summary)


def test_metrics_aggregate_only_fixed_active_session_categories() -> None:
    ledger = MetricsLedger()
    row = ledger.begin()
    ledger.record(
        row,
        active_idle_ms=480_000,
        session_policy="balanced",
        session_transition="active_idle",
        risk_gate="mixed_technical",
        nano_rehydration="ready",
        background_enrichment="cancelled",
    )

    summary = ledger.summary()

    assert summary["active_idle_ms"] == {"p50": 480_000, "p95": 480_000}
    assert summary["session_policy"] == {"balanced": 1}
    assert summary["session_transition"] == {"active_idle": 1}
    assert summary["risk_gate"] == {"mixed_technical": 1}
    assert summary["nano_rehydration"] == {"ready": 1}
    assert summary["background_enrichment"] == {"cancelled": 1}
