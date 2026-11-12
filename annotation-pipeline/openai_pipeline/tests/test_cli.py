import json
import shutil
from pathlib import Path

import pytest

from annotator.grading import append_verdict
from oai import cli
from oai.batches import load_plan
from oai.client import OpenAIGenerator
from oai.config import load_config
from tests.conftest import FakeOpenAI, happy_router, make_data, make_frame, nosleep, write_config


def make_dataset(root, n=10):
    root.mkdir(parents=True, exist_ok=True)
    (root / "Annotations").mkdir(exist_ok=True)
    (root / "Images" / "X").mkdir(parents=True, exist_ok=True)
    make_frame().save(root / "Images" / "X" / "Screenshot_1.png")
    steps = [{"step_id": 1, "image_path": "Images/X/Screenshot_1.png", "instruction": f"Click step {i}" + (" miss" if i % 5 == 0 else ""),
              "actions": [{"type": "click", "target": "t", "bbox": [10, 10, 10, 5]}]} for i in range(1, n + 1)]
    (root / "Annotations" / "X_Annotation.json").write_text(json.dumps({"tasks": [{"task_overview": "t", "steps": steps}]}), encoding="utf-8")
    return root


def cli_env(tmp_path, monkeypatch, router=None, n=2, **cfg_over):
    cfg_path = write_config(tmp_path, **cfg_over)
    cfg = load_config(cfg_path)
    make_data(cfg.data, apps=("Alpha", "Beta"), n=n)
    make_dataset(cfg.benchmark)
    fake = FakeOpenAI(router=router or happy_router())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abcd")
    monkeypatch.setattr(cli, "_generator", lambda cfg, key: OpenAIGenerator(cfg.models, cfg.efforts, key, transport=fake.transport(),
                                                                            sleep=nosleep, concurrency=cfg.concurrency))
    return cfg_path, cfg, fake


def cmd(cfg_path, *args):
    return cli.main(["--config", str(cfg_path), *args])


def test_missing_config_is_refused_not_a_traceback(tmp_path, capsys):
    assert cli.main(["--config", str(tmp_path / "missing.toml"), "status"]) == 2
    assert "CONFIG_REFUSED" in capsys.readouterr().out


def test_key_is_required_for_calibrate_and_run_but_not_status(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, _ = cli_env(tmp_path, monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY")
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels") == 2 and "NO_API_KEY" in capsys.readouterr().out
    assert cmd(cfg_path, "status") == 0 and "NO_API_KEY" in capsys.readouterr().out


def test_calibrate_writes_a_marker_per_model_and_mode(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch)
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "10") == 0
    out = capsys.readouterr().out
    assert "CALIBRATION hits=8/10 accuracy=0.800 prompt_tokens=2448 cost=$" in out and "CALIBRATION_OK" in out
    assert (cfg.work / "CALIBRATION_OK.gpt-5.6-luna.pixels").exists() and (cfg.work / "calibration.gpt-5.6-luna.pixels.json").exists()
    assert not (cfg.work / "CALIBRATION_OK.gpt-5.6-luna.thousand").exists()
    assert "text" not in fake.requests[0]                                  # the probe is free text, no schema
    assert all(r["batch"] == "calibrate" for r in map(json.loads, (cfg.work / "ledger.jsonl").read_text().splitlines()))


def test_calibrate_fails_on_a_downscaled_probe(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch)
    fake.usage["input_tokens"] = 1500
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "10") == 3
    assert "RESOLUTION_FAIL prompt_tokens=1500 < 2200" in capsys.readouterr().out and not (cfg.work / "CALIBRATION_OK.gpt-5.6-luna.pixels").exists()


def test_calibrate_stops_on_the_run_cap_without_losing_the_paid_call(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch, caps="batch_usd = 0\nrun_usd = 0.0001")
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 2
    assert "CALIBRATION_STOPPED reason=budget" in capsys.readouterr().out


def test_calibrate_refuses_without_frames(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, _ = cli_env(tmp_path, monkeypatch)
    shutil.rmtree(cfg.data)
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels") == 2
    assert "DATA_INCOMPLETE no frames under" in capsys.readouterr().out


def test_plan_refuses_until_every_app_is_present_then_extends(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, _ = cli_env(tmp_path, monkeypatch, batches="sizes = [2, 2]\nexpected_apps = 30")
    assert cmd(cfg_path, "plan") == 2 and "DATA_INCOMPLETE apps=2 expected=30" in capsys.readouterr().out
    cfg_path, cfg, _ = cli_env(tmp_path / "b", monkeypatch)
    assert cmd(cfg_path, "plan") == 0 and "PLANNED batch=1 size=2 apps=2 remaining=2" in capsys.readouterr().out
    assert cmd(cfg_path, "plan", "--next", "1") == 0 and "PLANNED batch=2 size=1" in capsys.readouterr().out
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    assert [b["size"] for b in plan["batches"]] == [2, 1]


def test_run_refuses_without_marker_then_runs_and_status_reports(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch)
    assert cmd(cfg_path, "plan") == 0
    assert cmd(cfg_path, "run", "1") == 2 and "CALIBRATION_OK.gpt-5.6-luna.pixels is missing" in capsys.readouterr().out
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0
    capsys.readouterr()
    assert cmd(cfg_path, "run", "1") == 0
    out = capsys.readouterr().out
    assert "BATCH_START 1 frames=2 projection=$0.0320 cap=$2.00" in out and "BATCH_DONE 1 frames=2 shipped=2" in out
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    assert plan["batches"][0]["state"] == "done" and plan["batches"][0]["usd_per_frame"] > 0
    assert cmd(cfg_path, "run", "1") == 2 and "batch 1 is done" in capsys.readouterr().out
    assert cmd(cfg_path, "status") == 0
    out = capsys.readouterr().out
    assert "KEY ...abcd" in out and "LEDGER total=$" in out and "of run_usd=$50.00" in out and "PROJECTION corpus=4 frames" in out
    assert "  1  done" in out or "1 done" in out.replace("  ", " ")


def test_run_refuses_a_projection_over_the_caps(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, _ = cli_env(tmp_path, monkeypatch, caps="batch_usd = 0.01\nrun_usd = 50")
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0
    capsys.readouterr()
    assert cmd(cfg_path, "run", "1") == 2
    assert "BUDGET_REFUSED projection=$0.0320 batch_cap=$0.01" in capsys.readouterr().out


def test_run_refuses_to_resume_over_the_auto_cap_with_a_spend_hint(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, _ = cli_env(tmp_path, monkeypatch)
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0
    row = {"at": "2026-01-01T00:00:00", "batch": 1, "app": "Alpha", "frame": "Alpha_001.png", "kind": "propose",
           "model": cfg.models["ground"], "effort": cfg.efforts["ground"], "input": 0, "cached": 0, "output": 0,
           "reasoning": 0, "retries": 0, "usd": 3.0}
    with (cfg.work / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")
    capsys.readouterr()
    assert cmd(cfg_path, "run", "1") == 2
    out = capsys.readouterr().out
    assert "BUDGET_REFUSED" in out and "spent=$3.0000: set [caps].batch_usd above $3.03 to resume" in out


def test_run_refuses_to_resume_a_batch_started_under_a_different_config(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch)
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0 and cmd(cfg_path, "plan") == 0
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    from oai import batches as bt
    batch = bt.get_batch(plan, 1)
    batch.state, batch.prompt_hash, batch.config_hash = "running", "0000000000000000", "0000000000000000"
    bt.put_batch(plan, batch)
    bt.save_plan(cfg.work / "batches.json", plan)
    capsys.readouterr()
    assert cmd(cfg_path, "run", "1") == 2
    assert "RUN_REFUSED batch 1 ran under a different prompt/config" in capsys.readouterr().out


def test_run_keeps_a_batch_planned_meanwhile_in_another_shell(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, _ = cli_env(tmp_path, monkeypatch, n=3)
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0
    real = cli.run_batch

    async def run_then_plan(cfg_, batch, gen, meter, log=print):
        out = await real(cfg_, batch, gen, meter, log)
        assert cmd(cfg_path, "plan") == 0                     # another shell appends batch 2 while 1 is still running
        return out
    monkeypatch.setattr(cli, "run_batch", run_then_plan)
    assert cmd(cfg_path, "run", "1") == 0
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    assert [(b["id"], b["state"]) for b in plan["batches"]] == [(1, "done"), (2, "planned")]


def grading(verdict_for):
    """A stand-in for the keyboard tool: judges every sampled row with verdict_for(item)."""
    graded = []

    def fake_serve(sample_file, root, verdict_path, port, open_browser, only_local):
        items = json.loads(Path(sample_file).read_text(encoding="utf-8"))["items"]
        assert only_local and not open_browser
        for it in items:
            append_verdict(verdict_path, {"image": it["image"], "original_bbox_px": it["bbox_px"], "bbox_px": it["bbox_px"], "verdict": verdict_for(it)})
        graded.extend(items)
        return 0
    fake_serve.graded = graded
    return fake_serve


def test_end_to_end_plan_run_grade_approve_export(tmp_path, monkeypatch, capfd):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch, n=3)
    serve = grading(lambda it: "correct")
    monkeypatch.setattr(cli, "_serve", serve)
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "1") == 0
    assert cmd(cfg_path, "approve", "1") == 2 and "grade it first" in capfd.readouterr().out
    assert cmd(cfg_path, "grade", "1", "--no-open") == 0
    out = capfd.readouterr().out
    assert "GRADES batch=1 correct_rate=1.000 correct=2 loose=0 wrong=0 corrected=0 n=2 graded=2 tierC_graded=0 applied=2" in out
    assert "COST batch=$" in out and len(serve.graded) == 2
    assert (cfg.work / "review" / "sample_batch1.json").exists() and (cfg.work / "review" / "verdicts.jsonl").exists()
    assert cmd(cfg_path, "approve", "1") == 0 and "APPROVED batch=1 correct_rate=1.000 usd_per_frame=" in capfd.readouterr().out
    assert cmd(cfg_path, "export") == 0
    out = capfd.readouterr().out
    assert "IMAGES_OK" in out and "EXPORT_OK batches=1 frames=2 dirs=6" in out
    manifest = json.loads((cfg.export / "gui_grounding" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "openai" and manifest["batches"] == [1] and manifest["coord_mode"] == "pixels"
    assert manifest["dataset"] == "gui_grounding"
    assert manifest["frames"] == 2 and manifest["cost_usd"] > 0 and manifest["models"] == cfg.models and manifest["shape"] == "nested"
    assert manifest["runs"] == [{"batch": 1, "models": cfg.models, "effort": cfg.efforts, "coord_mode": "pixels"}]
    tasks = json.loads((cfg.export / "gui_grounding" / "gui_grounding_train.json").read_text(encoding="utf-8"))["tasks"]
    assert len(tasks) == 6 and tasks[0]["steps"][0]["image_path"].startswith(cfg.image_root + "/")
    assert json.loads((cfg.export / "gui_grounding_flat" / "manifest.json").read_text(encoding="utf-8"))["shape"] == "flat"
    assert not (cfg.export / "gui_grounding" / "gui_grounding_train_flat.json").exists()
    # the gate sequence: batch 2 runs only after 1 is approved; a stopped batch blocks the next without --after-stop
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "2") == 0
    assert cmd(cfg_path, "stop", "2", "--reason", "trying terra") == 0 and "STOPPED batch=2 reason=trying terra" in capfd.readouterr().out
    assert cmd(cfg_path, "plan") == 0
    assert cmd(cfg_path, "run", "3") == 2 and "--after-stop" in capfd.readouterr().out
    assert cmd(cfg_path, "run", "3", "--after-stop") == 0
    assert cmd(cfg_path, "grade", "3", "--no-open") == 0 and cmd(cfg_path, "approve", "3") == 0
    capfd.readouterr()
    assert cmd(cfg_path, "export") == 0 and "EXPORT_OK batches=1,3 frames=4" in capfd.readouterr().out
    manifest = json.loads((cfg.export / "gui_grounding" / "manifest.json").read_text(encoding="utf-8"))
    assert [r["batch"] for r in manifest["runs"]] == [1, 3]
    assert cmd(cfg_path, "plan") == 2 and "PLAN_REFUSED every frame" in capfd.readouterr().out
    assert cmd(cfg_path, "status") == 0
    out = capfd.readouterr().out
    assert "stopped: trying terra" in out and "PROJECTION corpus=6 frames" in out and "(batch 3)" in out


def test_approve_below_the_gate_needs_force(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_serve", grading(lambda it: "wrong" if it["app"] == "Alpha" else "correct"))
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0 and cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "1") == 0
    assert cmd(cfg_path, "grade", "1", "--no-open") == 0
    assert "correct_rate=0.500 correct=1 loose=0 wrong=1" in capsys.readouterr().out
    assert cmd(cfg_path, "approve", "1") == 3 and "APPROVE_REFUSED batch 1 correct_rate=0.500 < gate 0.75" in capsys.readouterr().out
    assert cmd(cfg_path, "approve", "1", "--force") == 0 and "FORCED" in capsys.readouterr().out
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    assert plan["batches"][0]["forced"] is True and plan["batches"][0]["state"] == "approved"
    assert cmd(cfg_path, "export") == 0
    manifest = json.loads((cfg.export / "gui_grounding" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames"] == 1                                           # the rejected row ships nothing


def test_grade_and_export_refusals(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, _ = cli_env(tmp_path, monkeypatch)
    assert cmd(cfg_path, "export") == 2 and "EXPORT_NONE" in capsys.readouterr().out
    assert cmd(cfg_path, "plan") == 0
    assert cmd(cfg_path, "grade", "1", "--no-open") == 2 and "GRADE_REFUSED batch 1 is planned" in capsys.readouterr().out
    assert cmd(cfg_path, "stop", "1", "--reason", "x") == 0 and cmd(cfg_path, "stop", "1", "--reason", "y") == 2
    assert "STOP_REFUSED" in capsys.readouterr().out


def test_batch_sample_caps_rows_and_adds_tier_c(tmp_path, monkeypatch):
    cfg = load_config(write_config(tmp_path))
    ab = [{"image": f"A/{i}.png", "app": "A", "frame": f"{i}.png", "frame_index": i, "tier": "A", "bbox_px": [0, 0, 1, 1]} for i in range(150)]
    c = [{"image": f"C/{i}.png", "app": "C", "frame": f"{i}.png", "frame_index": i, "tier": "C", "bbox_px": [0, 0, 1, 1]} for i in range(30)]
    monkeypatch.setattr(cli, "grade_rows", lambda work, tiers: ab if tiers == ("A", "B") else c)
    from oai.batches import Batch
    batch = Batch(id=9, size=180, frames=[[r["app"], r["frame"], r["frame_index"]] for r in ab + c])
    path = cli.batch_sample(cfg, batch, 1)
    data = json.loads(path.read_text(encoding="utf-8"))
    tiers = [it["tier"] for it in data["items"]]
    assert path.name == "sample_batch9.json" and data["batch"] == 9 and tiers.count("A") == 100 and tiers.count("C") == 20
    assert data["items"] == sorted(data["items"], key=lambda i: (i["app"], i["frame_index"]))
    assert json.loads(cli.batch_sample(cfg, batch, 1).read_text(encoding="utf-8")) == data      # seeded: same sample again


def test_approve_ungraded_skips_the_grading_step(tmp_path, monkeypatch, capfd):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch, n=3)
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "1") == 0
    capfd.readouterr()
    assert cmd(cfg_path, "approve", "1", "--ungraded") == 0
    out = capfd.readouterr().out
    assert "APPROVED batch=1 correct_rate=- usd_per_frame=" in out and "FORCED ungraded" in out
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    one = plan["batches"][0]
    assert one["state"] == "approved" and one["forced"] is True and one["correct_rate"] is None and one["graded_rows"] == 0
    assert not (cfg.work / "review" / "sample_batch1.json").exists()         # no grading artefacts were produced
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "2") == 0
    assert "rate=measured" in capfd.readouterr().out                          # batch 1's $/frame drives batch 2's projection
    assert cmd(cfg_path, "status") == 0
    line = next(l for l in capfd.readouterr().out.splitlines() if l.strip().startswith("1  approved"))
    assert line.rstrip().endswith("FORCED") and "  -  " in line
    assert cmd(cfg_path, "export") == 0 and "EXPORT_OK batches=1" in capfd.readouterr().out


def test_plan_can_leave_the_benchmark_apps_out(tmp_path, monkeypatch, capsys):
    cfg_path = write_config(tmp_path, batches="sizes = [3]\nexpected_apps = 2\nexclude_benchmark_apps = true")
    cfg = load_config(cfg_path)
    make_data(cfg.data, apps=("Alpha", "RadiAnt"), n=3)
    assert cmd(cfg_path, "plan") == 0
    assert "PLANNED batch=1 size=3 apps=1 remaining=0" in capsys.readouterr().out
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    assert {f[0] for f in plan["batches"][0]["frames"]} == {"Alpha"}


def test_plan_balanced_draws_from_every_app_including_benchmark_ones(tmp_path, monkeypatch, capsys):
    cfg_path = write_config(tmp_path, batches="sizes = [4]\nexpected_apps = 3\nbalanced = true")
    cfg = load_config(cfg_path)
    make_data(cfg.data, apps=("Alpha", "Beta", "RadiAnt"), n=3)
    assert cmd(cfg_path, "plan") == 0
    assert "PLANNED batch=1 size=4 apps=3 remaining=5" in capsys.readouterr().out
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    assert [f[0] for f in plan["batches"][0]["frames"]] == ["Alpha", "Beta", "RadiAnt", "Alpha"]


from oai.batchclient import BatchClient
from tests.conftest import FakeBatchOpenAI


def batch_cli_env(tmp_path, monkeypatch, **fake_kw):
    cfg_path = write_config(tmp_path, run='coord_mode = "pixels"\nconcurrency = 2\nframe_concurrency = 2\nseed = 7\ntransport = "batch"\npoll_seconds = 1')
    cfg = load_config(cfg_path)
    make_data(cfg.data, apps=("Alpha", "Beta"), n=2)
    make_dataset(cfg.benchmark)
    cfg.work.mkdir(parents=True, exist_ok=True)
    (cfg.work / "CALIBRATION_OK.gpt-5.6-luna.pixels").write_text("ok\n", encoding="utf-8")
    fake = FakeBatchOpenAI(router=happy_router(), **fake_kw)
    sleeps = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abcd")
    monkeypatch.setattr(cli, "_batch_client", lambda cfg, key: BatchClient(key, transport=fake.transport(), sleep=lambda s: None))
    monkeypatch.setattr(cli, "_sleep", sleeps.append)
    return cfg_path, cfg, fake, sleeps


def test_run_under_the_batch_transport_steps_once_and_wait_loops_to_done(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake, sleeps = batch_cli_env(tmp_path, monkeypatch, polls_before_done=2)
    assert cmd(cfg_path, "plan") == 0
    capsys.readouterr()
    assert cmd(cfg_path, "run", "1") == 1
    out = capsys.readouterr().out
    assert "BATCH_START 1 frames=2 projection=$0.0160 cap=$2.00 rate=estimate transport=batch" in out
    assert "JOB_CREATED batch=1 round=1 part=round1.part1.jsonl job=batch_1 requests=2" in out and "ROUND_SUBMITTED batch=1 round=1 jobs=1 requests=2" in out
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    assert plan["batches"][0]["state"] == "running" and plan["batches"][0]["transport"] == "batch"
    assert cmd(cfg_path, "status") == 0
    assert "  1  running        2" in capsys.readouterr().out and sleeps == []
    assert cmd(cfg_path, "run", "1") == 1 and "ROUND_PENDING batch=1 round=1 jobs=0/1" in capsys.readouterr().out
    assert cmd(cfg_path, "run", "1", "--wait") == 0
    out = capsys.readouterr().out
    assert "BATCH_DONE 1 frames=2 shipped=2" in out and sleeps and all(s == 1 for s in sleeps)
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    assert plan["batches"][0]["state"] == "done" and plan["batches"][0]["usd_per_frame"] == pytest.approx(6 * (2448 * 0.10 + 40 * 0.60) / 1e6)
    assert cmd(cfg_path, "status") == 0
    out = capsys.readouterr().out
    assert "  1  done" in out and "round=" not in out                                            # the summary is for running batches only
    assert cmd(cfg_path, "approve", "1", "--ungraded") == 0 and cmd(cfg_path, "plan") == 0
    capsys.readouterr()
    assert cmd(cfg_path, "run", "2") == 1
    assert "projection=$" in capsys.readouterr().out.split("BATCH_START 2")[1].split("\n")[0]      # measured batch rate carried over


def test_status_shows_the_open_round_of_a_running_batch(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake, _ = batch_cli_env(tmp_path, monkeypatch, polls_before_done=9)
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "1") == 1
    capsys.readouterr()
    assert cmd(cfg_path, "status") == 0
    assert "round=1 jobs=0/1 requests=0/2" in capsys.readouterr().out


def test_stop_cancels_the_open_jobs_under_the_batch_transport(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake, _ = batch_cli_env(tmp_path, monkeypatch, polls_before_done=9)
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "1") == 1
    capsys.readouterr()
    assert cmd(cfg_path, "stop", "1", "--reason", "enough") == 0
    out = capsys.readouterr().out
    assert "CANCELLED jobs=batch_1" in out and "STOPPED batch=1 reason=enough" in out and fake.cancelled == ["batch_1"]


def test_run_retry_poisoned_clears_the_set_aside_frames(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake, _ = batch_cli_env(tmp_path, monkeypatch)
    assert cmd(cfg_path, "plan") == 0
    bdir = cfg.work / "batchapi" / "batch1"
    bdir.mkdir(parents=True)
    (bdir / "poisoned.json").write_text('{"Alpha/Alpha_001.png": "describe failed 2 rounds: x"}\n', encoding="utf-8")
    assert cmd(cfg_path, "run", "1", "--retry-poisoned", "--wait") == 0
    assert "RETRY_POISONED batch=1 frames=1" in capsys.readouterr().out
    assert json.loads((bdir / "poisoned.json").read_text(encoding="utf-8")) == {}               # rewritten empty by the passes


def test_budget_refusal_under_the_batch_transport_exits_2(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake, _ = batch_cli_env(tmp_path, monkeypatch)
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8").replace("run_usd = 50", "run_usd = 0.0005"), encoding="utf-8")
    assert cmd(cfg_path, "plan") == 0
    capsys.readouterr()
    assert cmd(cfg_path, "run", "1") == 2 and "BUDGET_REFUSED" in capsys.readouterr().out and fake.batches == {}


def test_export_a_subset_of_approved_batches_under_its_own_name(tmp_path, monkeypatch, capfd):
    cfg_path, cfg, fake = cli_env(tmp_path, monkeypatch, n=3)
    assert cmd(cfg_path, "calibrate", "--coord-mode", "pixels", "--n", "5") == 0
    for i in (1, 2):
        assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", str(i)) == 0 and cmd(cfg_path, "approve", str(i), "--ungraded") == 0
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "3") == 0                # done, not approved
    capfd.readouterr()
    assert cmd(cfg_path, "export", "--batches", "2", "--name", "gui_grounding_b2") == 0
    out = capfd.readouterr().out
    assert "EXPORT_OK batches=2 frames=2 dirs=6" in out
    manifest = json.loads((cfg.export / "gui_grounding_b2_all" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["batches"] == [2] and manifest["frames"] == 2 and manifest["dataset"] == "gui_grounding_b2"
    assert (cfg.export / "gui_grounding_b2_all" / "gui_grounding_b2_train.json").exists()
    plan = load_plan(cfg.work / "batches.json", cfg.seed)
    wanted = {f[1] for f in plan["batches"][1]["frames"]}
    tasks = json.loads((cfg.export / "gui_grounding_b2" / "gui_grounding_b2_train.json").read_text(encoding="utf-8"))["tasks"]
    assert {t["steps"][0]["image_path"].rsplit("/", 1)[1] for t in tasks} == wanted            # batch 2's frames only
    assert not (cfg.export / "gui_grounding").exists()                                          # the default name untouched
    assert cmd(cfg_path, "export", "--batches", "2,3") == 2 and "EXPORT_REFUSED batch 3 is done, not approved" in capfd.readouterr().out
    assert cmd(cfg_path, "export", "--batches", "9") == 2 and "EXPORT_REFUSED no batch 9" in capfd.readouterr().out
    assert cmd(cfg_path, "export", "--name", "Bad Name") == 2 and "EXPORT_REFUSED --name must be a lowercase slug" in capfd.readouterr().out
    assert cmd(cfg_path, "export") == 0 and "EXPORT_OK batches=1,2 frames=4" in capfd.readouterr().out


def test_wait_survives_a_transport_blip_but_a_single_step_does_not(tmp_path, monkeypatch, capsys):
    cfg_path, cfg, fake, sleeps = batch_cli_env(tmp_path, monkeypatch, poll_failures=3)   # one full retry cycle fails
    assert cmd(cfg_path, "plan") == 0 and cmd(cfg_path, "run", "1") == 1
    capsys.readouterr()
    assert cmd(cfg_path, "run", "1", "--wait") == 0
    out = capsys.readouterr().out
    assert "TRANSPORT_RETRY 1/60: OpenAI unreachable after 3 attempts" in out and "BATCH_DONE 1" in out
    fake2_path, cfg2, fake2, _ = batch_cli_env(tmp_path / "two", monkeypatch, poll_failures=3)
    assert cmd(fake2_path, "plan") == 0 and cmd(fake2_path, "run", "1") == 1
    with pytest.raises(Exception, match="unreachable after 3 attempts"):
        cmd(fake2_path, "run", "1")                                                        # no --wait: the step fails as before
