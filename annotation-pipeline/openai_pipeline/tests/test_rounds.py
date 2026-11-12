import json
from collections import Counter

import pytest

from annotator.calling import TransportError
from annotator.records import load_record, record_path
from oai import batchapi as ba
from oai import batches as b
from oai import rounds as rd
from oai.batchclient import BatchClient
from oai.config import load_config
from oai.meter import Caps, Meter, read_ledger
from oai.pricing import cost_usd
from tests.conftest import BAD_REPLIES, DESCRIBE_MARK, FakeBatchOpenAI, happy_router, make_data, user_text_of, write_config

SYNC_CALL_USD = (2448 * 0.20 + 40 * 1.20) / 1e6


def env(tmp_path, fake, *, batch_usd=10.0, run_usd=50.0, apps=("Alpha", "Beta"), n=1):
    cfg = load_config(write_config(tmp_path, run='coord_mode = "pixels"\nconcurrency = 2\nframe_concurrency = 2\nseed = 7\ntransport = "batch"\nfile_expiry_days = 7'))
    make_data(cfg.data, apps=apps, n=n)
    plan = b.load_plan(cfg.work / "batches.json", cfg.seed)
    batch = b.transition(b.plan_batch(plan, b.corpus_order(cfg.data, cfg.seed), 2), "running")
    client = BatchClient("sk-test", transport=fake.transport(), sleep=lambda s: None)
    meter = Meter(cfg.work / "ledger.jsonl", batch.id, Caps(batch_usd, run_usd), log=lambda *_: None)
    return cfg, batch, client, meter


def drive(cfg, batch, client, meter, logs, limit=30):
    outcomes = []
    for _ in range(limit):
        outcomes.append(rd.step(cfg, batch, client, meter, log=logs.append))
        if outcomes[-1] not in ("submitted", "pending"):
            return outcomes
    raise AssertionError(f"no terminal outcome after {limit} steps: {outcomes}")


def round_files(cfg, batch):
    return sorted(p.name for p in rd.batch_dir(cfg, batch).glob("round*.json"))


def test_happy_batch_runs_four_rounds_to_batch_done_at_half_price(tmp_path):
    fake = FakeBatchOpenAI(router=happy_router())
    cfg, batch, client, meter = env(tmp_path, fake)
    logs = []
    outcomes = drive(cfg, batch, client, meter, logs)
    assert outcomes == ["submitted"] * 4 + ["done"] and batch.state == "done" and batch.transport == "batch" and batch.finished_at
    assert round_files(cfg, batch) == ["round1.json", "round2.json", "round3.json", "round4.json"]
    assert [r for r in logs if r.startswith("ROUND_SUBMITTED")][0].startswith("ROUND_SUBMITTED batch=1 round=1 jobs=1 requests=2 projection=$")
    assert any(l.startswith("BATCH_DONE 1 frames=2 shipped=2 lowconf=0 abstained=0 unreadable=0 cost=$") for l in logs)
    rows = read_ledger(cfg.work / "ledger.jsonl")
    assert len(rows) == 12 == len(fake.requests) and all(r["transport"] == "batch" and r["job"].startswith("batch_") for r in rows)
    assert batch.cost_usd == pytest.approx(12 * SYNC_CALL_USD / 2) and batch.usd_per_frame == pytest.approx(batch.cost_usd / 2)
    assert Counter(r["kind"] for r in rows) == Counter(propose=2, describe=2, ground=6, verify=2)
    for app, name, index in batch.frames:
        rec = load_record(record_path(cfg.work, app, index))
        assert rec["status"] == "shipped" and (cfg.reports / "overlays" / "1" / f"{app}_{index:03d}_shipped.png").exists()
    stats = json.loads((cfg.reports / "qc_stats.batch1.json").read_text(encoding="utf-8"))
    assert stats["calls"] == 12 and stats["tokens"]["input"] == 12 * 2448 and stats["status"] == {"shipped": 2}
    vision = [f for f in fake.files.values() if f["purpose"] == "vision"]
    assert len(vision) == 2 and {f["name"] for f in vision} == {name for _, name, _ in batch.frames}         # one upload per frame
    assert all(f["form"]["expires_after[seconds]"] == str(7 * 86400) for f in vision)
    ground = [r for r in fake.requests if "Locate the element" in user_text_of(r)]
    assert all(r["input"][0]["content"][0]["file_id"].startswith("file-") for r in ground)      # the raw frame by reference
    describe = [r for r in fake.requests if DESCRIBE_MARK in user_text_of(r)]
    assert all("image_url" in r["input"][0]["content"][0] for r in describe)                    # a crop: inline
    files = json.loads((cfg.work / "batchapi" / "files.json").read_text(encoding="utf-8"))
    assert set(files) == {"Alpha/Alpha_001.png", "Beta/Beta_001.png"} and all(v["file_id"].startswith("file-") and v["expires_at"] for v in files.values())


def test_pending_rounds_report_progress_until_the_job_completes(tmp_path):
    fake = FakeBatchOpenAI(router=happy_router(), polls_before_done=3)
    cfg, batch, client, meter = env(tmp_path, fake)
    logs = []
    assert rd.step(cfg, batch, client, meter, log=logs.append) == "submitted"
    assert rd.step(cfg, batch, client, meter, log=logs.append) == "pending"
    assert logs[-1] == "ROUND_PENDING batch=1 round=1 jobs=0/1 requests=0/2"
    assert rd.summary(cfg, batch) == " round=1 jobs=0/1 requests=0/2"
    assert rd.step(cfg, batch, client, meter, log=logs.append) == "pending"
    assert rd.step(cfg, batch, client, meter, log=logs.append) == "submitted"                   # round 1 ingested, round 2 out
    assert any(l.startswith("ROUND_INGESTED batch=1 round=1 job=batch_1 status=completed replies=2 errors=0") for l in logs)
    assert rd.summary(cfg, batch) == " round=2 jobs=0/1 requests=0/2"


def test_a_repair_is_just_another_round(tmp_path):
    seen = []
    base = happy_router()

    def router(body):
        if DESCRIBE_MARK in user_text_of(body) and not seen:
            seen.append(1)
            return json.dumps(BAD_REPLIES["describe"][1])
        return base(body)
    fake = FakeBatchOpenAI(router=router)
    cfg, batch, client, meter = env(tmp_path, fake)
    outcomes = drive(cfg, batch, client, meter, [])
    assert outcomes[-1] == "done" and len(round_files(cfg, batch)) == 5 and len(read_ledger(cfg.work / "ledger.jsonl")) == 13
    repairs = [r for r in fake.requests if len(r["input"]) == 3]
    assert len(repairs) == 1 and repairs[0]["input"][1]["role"] == "assistant"


def test_an_expired_job_bills_only_what_finished_and_resubmits_the_rest(tmp_path):
    fake = FakeBatchOpenAI(router=happy_router(), expire_last=1)
    cfg, batch, client, meter = env(tmp_path, fake)
    logs = []
    outcomes = drive(cfg, batch, client, meter, logs)
    assert outcomes[-1] == "done" and any("status=expired replies=1" in l for l in logs)
    rows = read_ledger(cfg.work / "ledger.jsonl")
    assert len(rows) == 12 and len(fake.custom_ids) == 13 and len(set(fake.custom_ids)) == 12          # one request went twice, billed once


def test_a_request_erroring_twice_poisons_its_frame_and_the_rest_finishes(tmp_path):
    fake = FakeBatchOpenAI(router=happy_router(),
                           errors=lambda body: "invalid image" if DESCRIBE_MARK in user_text_of(body) and "Application: Alpha" in user_text_of(body) else None)
    cfg, batch, client, meter = env(tmp_path, fake)
    logs = []
    outcomes = drive(cfg, batch, client, meter, logs)
    assert outcomes[-1] == "incomplete" and batch.state == "running" and batch.stop_reason is None
    assert any(l.startswith("BATCH_INCOMPLETE 1 frames_done=1/2 cost=$") and l.endswith("poisoned=Alpha/Alpha_001.png") for l in logs)
    assert ba.load_poisoned(rd.batch_dir(cfg, batch)) and load_record(record_path(cfg.work, "Beta", 1))["status"] == "shipped"
    assert len(read_ledger(cfg.work / "ledger.jsonl")) == 7                                            # 2 propose + Beta's 1+3+1
    fake.errors = None
    assert ba.clear_poisoned(rd.batch_dir(cfg, batch), ba.ReplyCache(rd.batch_dir(cfg, batch) / "replies.jsonl")) == 1
    outcomes = drive(cfg, batch, client, meter, logs)
    assert outcomes[-1] == "done" and load_record(record_path(cfg.work, "Alpha", 1))["status"] == "shipped"


def test_budget_refusal_at_submission_uploads_nothing(tmp_path):
    fake = FakeBatchOpenAI(router=happy_router())
    cfg, batch, client, meter = env(tmp_path, fake, batch_usd=0.0001)
    logs = []
    assert rd.step(cfg, batch, client, meter, log=logs.append) == "refused"
    assert logs[-1].startswith("BUDGET_REFUSED round=1 projection=$0.0008 batch_cap=$0.00 run_total=$0.0000 run_cap=$50.00")
    assert fake.files == {} and fake.batches == {} and round_files(cfg, batch) == [] and not list(rd.batch_dir(cfg, batch).glob("*.jsonl"))
    assert batch.state == "running"


def test_projection_uses_the_ledgers_batch_priced_rate_per_kind(tmp_path):
    rows = [{"kind": "propose", "model": "gpt-5.6-luna", "input": 2000, "cached": 0, "output": 100, "transport": "sync"}]
    assert rd.projected_usd(Counter(propose=3, ground=2), rows, {"ground": "gpt-5.6-luna"}) == pytest.approx(
        3 * cost_usd(rows[0], "gpt-5.6-luna", "batch") + 2 * 0.016 / 19 * 0.5, abs=1e-8)              # rounded to 8 decimals


def test_a_step_killed_between_two_parts_resumes_the_same_round(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "PART_MAX_LINES", 1)
    fake = FakeBatchOpenAI(router=happy_router(), create_ok_before_failing=1, create_failures=3)
    cfg, batch, client, meter = env(tmp_path, fake)
    logs = []
    with pytest.raises(TransportError, match="unreachable"):
        rd.step(cfg, batch, client, meter, log=logs.append)
    rnd = rd.latest_round(rd.batch_dir(cfg, batch))
    assert rnd.parts == ["round1.part1.jsonl", "round1.part2.jsonl"] and len(rnd.jobs) == 1 and rnd.pending and len(fake.batches) == 1
    assert rd.step(cfg, batch, client, meter, log=logs.append) == "submitted" and len(fake.batches) == 2
    assert drive(cfg, batch, client, meter, logs)[-1] == "done"
    r1 = json.loads((rd.batch_dir(cfg, batch) / "round1.json").read_text(encoding="utf-8"))
    assert len(r1["parts"]) == 2 == len(r1["jobs"]) and all(j["ingested"] for j in r1["jobs"])
    assert len(read_ledger(cfg.work / "ledger.jsonl")) == 12


def test_a_failed_job_stops_the_batch_with_the_error_and_a_rerun_recollects(tmp_path):
    fake = FakeBatchOpenAI(router=happy_router(), fail="invalid json on line 1")
    cfg, batch, client, meter = env(tmp_path, fake)
    logs = []
    assert rd.step(cfg, batch, client, meter, log=logs.append) == "submitted"
    assert rd.step(cfg, batch, client, meter, log=logs.append) == "stopped" and batch.stop_reason == "transport"
    assert logs[-1].startswith("BATCH_STOPPED 1 reason=transport job=batch_1") and "invalid json on line 1" in logs[-1]
    fake.fail = None
    assert drive(cfg, batch, client, meter, logs)[-1] == "done" and batch.stop_reason is None
    assert round_files(cfg, batch) == ["round1.json", "round2.json", "round3.json", "round4.json", "round5.json"]


def test_cancel_open_cancels_the_running_jobs_only(tmp_path):
    fake = FakeBatchOpenAI(router=happy_router(), polls_before_done=5)
    cfg, batch, client, meter = env(tmp_path, fake)
    assert rd.cancel_open(cfg, batch, client) == []
    rd.step(cfg, batch, client, meter, log=lambda *_: None)
    assert rd.cancel_open(cfg, batch, client) == ["batch_1"] and fake.cancelled == ["batch_1"]
    assert rd.step(cfg, batch, client, meter, log=lambda *_: None) == "submitted"                    # cancelled: nothing answered, re-collected
    assert len(fake.batches) == 2
