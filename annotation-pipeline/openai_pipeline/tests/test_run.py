import json

import pytest

from annotator.prompts import prompt_hash
from annotator.records import load_record, record_path
from oai import batches as b
from oai.client import OpenAIGenerator
from oai.config import config_hash, load_config
from oai.meter import Caps, Meter, ledger_total, read_ledger
from oai.run import remaining, run_batch
from tests.conftest import FakeOpenAI, happy_router, make_data, nosleep, run, write_config


def setup(tmp_path, n=2):
    cfg = load_config(write_config(tmp_path))
    make_data(cfg.data, apps=("Alpha", "Beta"), n=n)
    plan = b.load_plan(cfg.work / "batches.json", cfg.seed)
    return cfg, plan, b.plan_batch(plan, b.corpus_order(cfg.data, cfg.seed), 2)


def gen_for(fake, cfg):
    return OpenAIGenerator(cfg.models, cfg.efforts, "sk-test", transport=fake.transport(), sleep=nosleep, concurrency=cfg.concurrency)


def meter_for(cfg, batch, batch_usd=10.0, run_usd=50.0, log=lambda *_: None):
    return Meter(cfg.work / "ledger.jsonl", batch.id, Caps(batch_usd, run_usd), log=log)


def test_run_batch_happy_path_writes_records_ledger_overlays_and_marker(tmp_path):
    cfg, plan, batch = setup(tmp_path)
    fake = FakeOpenAI(router=happy_router())
    logs = []
    assert len(remaining(cfg, batch)) == 2
    out = run(run_batch(cfg, batch, gen_for(fake, cfg), meter_for(cfg, batch), log=logs.append))
    assert out.state == "done" and out.cost_usd > 0 and out.usd_per_frame == pytest.approx(out.cost_usd / 2) and out.finished_at
    assert out.models == cfg.models and out.prompt_hash == prompt_hash("pixels") and out.config_hash == config_hash(cfg)
    assert any(l.startswith("BATCH_DONE 1 frames=2 shipped=2 lowconf=0 abstained=0 unreadable=0 cost=$") and "usd_per_frame=" in l for l in logs)
    for app, name, index in batch.frames:
        rec = load_record(record_path(cfg.work, app, index))
        prov = rec["provenance"]
        assert rec["status"] == "shipped" and prov["provider"] == "openai" and prov["batch"] == 1 and prov["run_id"] == "batch1"
        assert prov["coord_mode"] == "pixels" and prov["models"] == cfg.models and prov["effort"] == cfg.efforts
        assert (cfg.reports / "overlays" / "1" / f"{app}_{index:03d}_shipped.png").exists()
    rows = read_ledger(cfg.work / "ledger.jsonl")
    assert len(rows) == 12 == len(fake.requests) and {r["kind"] for r in rows} == {"propose", "describe", "ground", "verify"}
    assert all(r["batch"] == 1 and r["app"] in ("Alpha", "Beta") for r in rows)
    stats = json.loads((cfg.reports / "qc_stats.batch1.json").read_text(encoding="utf-8"))
    assert stats["frames"] == 2 and stats["status"] == {"shipped": 2} and stats["calls"] == 12 and stats["refusals"] == 0
    assert stats["tokens"]["input"] == 12 * 2448 and remaining(cfg, out) == []


def test_budget_stop_leaves_the_frames_resumable(tmp_path):
    cfg, plan, batch = setup(tmp_path)
    fake = FakeOpenAI(router=happy_router())
    logs = []
    out = run(run_batch(cfg, batch, gen_for(fake, cfg), meter_for(cfg, batch, batch_usd=0.0001), log=logs.append))
    assert out.state == "running" and out.stop_reason == "budget"
    assert any(l.startswith("BATCH_STOPPED 1 reason=budget frames_done=0/2") for l in logs)
    assert any(l.startswith("HINT: batch 1 spent") for l in logs)
    assert 1 <= len(fake.requests) <= 2                                   # only the propose(s) in flight when the cap was crossed
    for app, _, i in batch.frames:
        rec = load_record(record_path(cfg.work, app, i))
        assert rec is None or rec["stages_done"] == ["propose"]          # saved after propose, nothing after it
    out = run(run_batch(cfg, out, gen_for(fake, cfg), meter_for(cfg, out), log=logs.append))
    assert out.state == "done" and len(fake.requests) == 12              # finished stages were not repeated
    assert out.stop_reason is None
    assert out.cost_usd == pytest.approx(ledger_total(read_ledger(cfg.work / "ledger.jsonl"), 1))


def test_transport_failure_leaves_the_batch_incomplete_with_the_body(tmp_path):
    cfg, plan, batch = setup(tmp_path)
    fake = FakeOpenAI(router=lambda body: (400, {"error": {"message": "invalid image"}}, {}))
    logs = []
    out = run(run_batch(cfg, batch, gen_for(fake, cfg), meter_for(cfg, batch), log=logs.append))
    assert out.state == "running" and any(l.startswith("BATCH_INCOMPLETE 1 frames_done=0/2") for l in logs)
    assert any("invalid image" in l for l in logs) and not list((cfg.work / "records").rglob("*.json"))


def test_rate_limit_hint_when_429s_exceed_a_fifth_of_calls(tmp_path):
    cfg, plan, batch = setup(tmp_path)
    happy, seen = happy_router(), {"n": 0}

    def flaky(body):
        seen["n"] += 1
        return (429, {}, {}) if seen["n"] % 2 else happy(body)
    fake = FakeOpenAI(router=flaky)
    logs = []
    out = run(run_batch(cfg, batch, gen_for(fake, cfg), meter_for(cfg, batch), log=logs.append))
    assert out.state == "done" and any(l.startswith("HINT: 429") for l in logs) and len(fake.requests) > 12
