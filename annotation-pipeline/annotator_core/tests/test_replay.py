import copy
import pytest

from annotator import replay
from annotator.records import load_record, record_path, save_record
from annotator.runner import RunConfig, run_app
from tests.conftest import FakeGenerator, happy_router, make_data, run


def finished_record(tmp_path):
    cfg = RunConfig(work=tmp_path / "work", data_root=make_data(tmp_path, 1), app="IrfanView")
    run(run_app(FakeGenerator(router=happy_router()), cfg, log=lambda *_: None))
    path = record_path(cfg.work, "IrfanView", 1)
    return path, load_record(path)


def test_unchanged_record_reports_zero_for_every_stage(tmp_path):
    _, rec = finished_record(tmp_path)
    for stage in replay.REPLAYABLE:
        assert replay.reparse(rec, stage) == 0
    assert len(rec["stages_done"]) == 6 and rec["status"] == "shipped"


def test_ground_reparse_restores_a_box_and_clears_later_blocks(tmp_path):
    _, rec = finished_record(tmp_path)
    c = rec["candidates"][0]
    c["ground"]["visual"].update(bbox_px=None, bbox_1000=None, box_sane=None, label="", error="no JSON value found")
    assert replay.reparse(rec, "ground") == 1
    assert c["ground"]["visual"]["bbox_px"] == [288, 162, 576, 216] and c["ground"]["visual"]["error"] is None
    assert c["consensus"] is None and c["verify"] is None and c["refine"] is None and c["final"] is None
    assert rec["stages_done"] == ["propose", "describe", "ground"] and rec["selected"] is None and rec["status"] is None
    assert c["ground"]["functional"]["bbox_px"] == [288, 162, 576, 216]     # untouched variants keep their values


def test_describe_reparse_revives_an_unreadable_candidate(tmp_path):
    _, rec = finished_record(tmp_path)
    c = rec["candidates"][0]
    c["describe"].update(bbox_px=None, bbox_1000=None, instructions=None, error="the box is not on the control")
    c.update(status="unreadable", status_stage="describe", status_reason="x", ground=None, consensus=None, verify=None, final=None)
    assert replay.reparse(rec, "describe") == 1
    assert c["status"] == "ok" and c["describe"]["error"] is None and c["describe"]["instructions"]["visual"].startswith("Click")
    assert abs(c["describe"]["bbox_px"][0] - 288) <= 1 and rec["stages_done"] == ["propose", "describe"]


def test_verify_reparse_recomputes_verified_and_clears_refine(tmp_path):
    _, rec = finished_record(tmp_path)
    c = rec["candidates"][0]
    c["verify"].update(verified=False, verdict="loose", readback_match=False)
    c["refine"] = {"outcome": "unchanged"}
    assert replay.reparse(rec, "verify") == 1
    assert c["verify"]["verified"] and c["verify"]["verdict"] == "correct" and c["verify"]["readback_match"] is True
    assert c["refine"] is None and c["final"] is None and c["consensus"] is not None
    assert rec["stages_done"] == ["propose", "describe", "ground", "verify"]


def test_propose_reparse_keeps_candidates_whose_seed_is_unchanged(tmp_path):
    _, rec = finished_record(tmp_path)
    rec["propose"]["intent"] = ""
    assert replay.reparse(rec, "propose") == 1
    assert rec["propose"]["intent"] == "open the File menu"
    assert rec["candidates"][0]["describe"] is not None            # same seed -> blocks survive
    assert rec["stages_done"] == ["propose"] and rec["selected"] is None


def test_propose_reparse_rebuilds_when_seeds_change(tmp_path):
    _, rec = finished_record(tmp_path)
    rec["candidates"][0]["seed_bbox_px"] = [0, 0, 10, 10]
    assert replay.reparse(rec, "propose") == 1
    assert rec["candidates"][0]["seed_bbox_px"] == [288, 162, 576, 216] and rec["candidates"][0]["describe"] is None


def test_refine_and_select_are_not_replayable(tmp_path):
    _, rec = finished_record(tmp_path)
    with pytest.raises(ValueError):
        replay.reparse(rec, "refine")
    with pytest.raises(ValueError):
        replay.reparse(rec, "select")


def test_reparse_restores_record_when_interpret_raises(tmp_path, monkeypatch):
    _, rec = finished_record(tmp_path)
    # Ensure we have at least 2 candidates
    if len(rec["candidates"]) < 2:
        rec["candidates"].append(copy.deepcopy(rec["candidates"][0]))
    # Corrupt ground blocks so reparse will attempt to restore them
    for cand in rec["candidates"]:
        if cand.get("ground"):
            for v in cand["ground"]:
                cand["ground"][v]["bbox_px"] = None
                cand["ground"][v]["bbox_1000"] = None
    before = copy.deepcopy(rec)
    calls = {"n": 0}
    real = replay.s3.interpret
    def boom(raw, *a, **k):
        calls["n"] += 1
        if calls["n"] > 3:  # After processing candidate 0's 3 variants
            raise RuntimeError("corrupt raw")
        return real(raw, *a, **k)
    monkeypatch.setattr(replay.s3, "interpret", boom)
    with pytest.raises(RuntimeError):
        replay.reparse(rec, "ground")
    assert rec == before


def test_reparse_uses_the_records_coordinate_mode(tmp_path):
    cfg = RunConfig(work=tmp_path / "work", data_root=make_data(tmp_path, 1), app="IrfanView", coord_mode="pixels")
    run(run_app(FakeGenerator(router=happy_router(mode="pixels")), cfg, log=lambda *_: None))
    path = record_path(cfg.work, "IrfanView", 1)
    rec = load_record(path)
    for stage in replay.REPLAYABLE:
        assert replay.reparse(rec, stage) == 0                     # idempotent under its own mode
    before = rec["candidates"][0]["ground"]["visual"]["bbox_px"]
    rec["provenance"]["coord_mode"] = "thousand"                   # lie about the space -> the raws re-read differently
    assert replay.reparse(rec, "ground") == 1
    assert rec["candidates"][0]["ground"]["visual"]["bbox_px"] != before
