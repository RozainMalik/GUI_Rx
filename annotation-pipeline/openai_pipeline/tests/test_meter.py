import json

import pytest

from annotator.calling import Call, Reply, TransportError
from oai import meter as m
from tests.conftest import run

LUNA = dict(model="gpt-5.6-luna", effort="none", retries=0, reasoning=0)


def usage(input=10_000, cached=0, output=1_000, **kw):
    return {**LUNA, "input": input, "cached": cached, "output": output, **kw}


def test_record_appends_a_priced_line_and_updates_totals(tmp_path):
    meter = m.Meter(tmp_path / "ledger.jsonl", 1, m.Caps(10, 50), log=lambda *_: None)
    usd = meter.record(app="A", frame="A_001.png", kind="ground", usage=usage(retries=2))
    assert usd == pytest.approx((10_000 * 0.20 + 1_000 * 1.20) / 1e6)
    rows = m.read_ledger(tmp_path / "ledger.jsonl")
    assert len(rows) == 1 and rows[0]["batch"] == 1 and rows[0]["kind"] == "ground" and rows[0]["retries"] == 2 and rows[0]["usd"] == usd
    assert set(rows[0]) == {"at", "batch", "app", "frame", "kind", "model", "effort", "input", "cached", "output", "reasoning", "retries", "usd", "transport"}
    assert rows[0]["transport"] == "sync"
    assert meter.batch_total == usd and meter.run_total == usd and meter.calls == 1 and meter.tokens["input"] == 10_000


def test_prior_ledger_counts_toward_run_and_resumed_batch(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = m.Meter(path, 1, m.Caps(10, 50), log=lambda *_: None)
    first.record(app="A", frame="f", kind="ground", usage=usage())
    second = m.Meter(path, 2, m.Caps(10, 50), log=lambda *_: None)
    assert second.run_total == first.batch_total and second.batch_total == 0
    resumed = m.Meter(path, 1, m.Caps(10, 50), log=lambda *_: None)
    assert resumed.batch_total == first.batch_total


def test_check_raises_only_after_a_cap_is_crossed(tmp_path):
    meter = m.Meter(tmp_path / "l.jsonl", 1, m.Caps(batch_usd=0.003, run_usd=50), log=lambda *_: None)
    meter.check()
    meter.record(app="A", frame="f", kind="ground", usage=usage())          # 0.0032 > 0.003
    with pytest.raises(m.BudgetExceeded, match="batch cap"):
        meter.check()
    assert meter.stop_reason == "budget" and isinstance(m.BudgetExceeded("x"), TransportError)
    run_meter = m.Meter(tmp_path / "l2.jsonl", 1, m.Caps(batch_usd=10, run_usd=0.003), log=lambda *_: None)
    run_meter.record(app="A", frame="f", kind="ground", usage=usage())
    with pytest.raises(m.BudgetExceeded, match="run cap"):
        run_meter.check()


def test_anomaly_guard_logs_the_first_and_stops_on_the_second(tmp_path):
    logs = []
    meter = m.Meter(tmp_path / "l.jsonl", 1, m.Caps(100, 100), log=logs.append)
    for i in range(4):                                                    # four normal frames at ~$0.0032
        meter.record(app="A", frame=f"f{i}", kind="ground", usage=usage())
        meter.frame_done("A", f"f{i}")
    for _ in range(6):                                                    # a fifth costing 6x the median
        meter.record(app="A", frame="big", kind="describe", usage=usage())
    assert ("A", "big") in meter.anomalous and meter.stop_reason is None and any(l.startswith("ANOMALY A/big") for l in logs)
    meter.check()                                                         # the first anomaly finishes
    for _ in range(6):
        meter.record(app="B", frame="big2", kind="describe", usage=usage())
    assert meter.stop_reason == "anomaly"
    with pytest.raises(m.AnomalyStop):
        meter.check()


def test_anomaly_needs_a_median_first(tmp_path):
    meter = m.Meter(tmp_path / "l.jsonl", 1, m.Caps(100, 100), log=lambda *_: None)
    for _ in range(20):
        meter.record(app="A", frame="only", kind="ground", usage=usage())
    assert not meter.anomalous


def test_metered_checks_before_and_records_after_each_call(tmp_path):
    meter = m.Meter(tmp_path / "l.jsonl", 3, m.Caps(0.005, 50), log=lambda *_: None)

    class Gen:
        async def generate(self, call):
            return Reply(content="{}", usage=usage())
    wrapped = m.Metered(Gen(), meter, "A", "A_001.png")
    run(wrapped.generate(Call(system="", user_text="", kind="ground")))
    assert meter.calls == 1 and m.read_ledger(tmp_path / "l.jsonl")[0]["app"] == "A"
    run(wrapped.generate(Call(system="", user_text="", kind="ground")))    # crosses 0.005
    with pytest.raises(m.BudgetExceeded):
        run(wrapped.generate(Call(system="", user_text="", kind="ground")))
    assert meter.calls == 2                                               # the refused call was never made


def test_reprice_and_totals_by_batch(tmp_path):
    rows = [{"batch": 1, "input": 1_000_000, "cached": 0, "output": 0, "model": "gpt-5.6-luna", "usd": 0.5},
            {"batch": 2, "input": 0, "cached": 0, "output": 1_000_000, "model": "gpt-5.6-luna", "usd": 0.5}]
    assert m.ledger_total(rows) == 1.0 and m.ledger_total(rows, 2) == 0.5
    assert m.reprice(rows) == pytest.approx(0.20 + 1.20)
    (tmp_path / "l.jsonl").write_text('{"batch": 1, "usd": 0.25}\n\nnot json\n', encoding="utf-8")
    assert m.read_ledger(tmp_path / "l.jsonl") == [{"batch": 1, "usd": 0.25}] and m.read_ledger(tmp_path / "none.jsonl") == []


def test_batch_rows_are_priced_at_the_batch_rate_and_carry_the_job(tmp_path):
    meter = m.Meter(tmp_path / "ledger.jsonl", 9, m.Caps(10, 50), log=lambda *_: None)
    usd = meter.record(app="A", frame="A_001.png", kind="ground", usage=usage(cached=4_000), transport="batch", job="batch_abc")
    assert usd == pytest.approx((10_000 * 0.10 + 1_000 * 0.60) / 1e6)            # cached tokens: no discount in batch
    row = m.read_ledger(tmp_path / "ledger.jsonl")[0]
    assert row["transport"] == "batch" and row["job"] == "batch_abc" and row["cached"] == 4_000
    assert m.reprice([row]) == pytest.approx(usd)                                  # repriced at the row's own transport
    assert m.reprice([{**row, "transport": "sync"}]) == pytest.approx((6_000 * 0.20 + 4_000 * 0.02 + 1_000 * 1.20) / 1e6)


def test_rate_per_kind_is_the_mean_per_call_at_the_given_transport():
    rows = [{"kind": "ground", "model": "gpt-5.6-luna", "input": 2_000, "cached": 0, "output": 100, "transport": "sync"},
            {"kind": "ground", "model": "gpt-5.6-luna", "input": 4_000, "cached": 0, "output": 100, "transport": "batch"},
            {"kind": "propose", "model": "gpt-5.6-luna", "input": 1_000, "cached": 0, "output": 500}]
    rates = m.rate_per_kind(rows)                                                   # batch-priced whatever the row was
    assert rates["ground"] == pytest.approx(((2_000 + 4_000) * 0.10 + 200 * 0.60) / 2 / 1e6)
    assert rates["propose"] == pytest.approx((1_000 * 0.10 + 500 * 0.60) / 1e6)
    assert m.rate_per_kind([]) == {}
