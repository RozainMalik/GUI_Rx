import json

from annotator import records as r
from annotator.stages.propose import CandidateSeed

SEED = CandidateSeed(2, "Save", "toolbar button", "Save", "floppy disk", "plain", [10, 10, 50, 50], (19, 11, 96, 54))


def test_record_path_zero_pads_to_three():
    assert str(r.record_path("work", "IrfanView", 35)).replace("\\", "/") == "work/records/IrfanView/035.json"
    assert r.record_path("work", "A", 1234).name == "1234.json"


def test_new_record_shape():
    rec = r.new_record("IrfanView", "IrfanView_035.png", 35, (1920, 1080), {"prompt_hash": "h", "thinking_fallbacks": 0})
    assert rec["app"] == "IrfanView" and rec["frame_index"] == 35 and rec["image_size"] == [1920, 1080]
    assert rec["stages_done"] == [] and rec["candidates"] == [] and rec["selected"] is None
    assert rec["status"] is None and rec["review"] is None and rec["error"] is None
    assert rec["provenance"]["prompt_hash"] == "h"


def test_candidate_round_trips_through_seed():
    cand = r.candidate_from_seed(SEED)
    assert cand["rank"] == 2 and cand["element"]["icon"] == "floppy disk" and cand["status"] == "ok"
    assert cand["seed_bbox_px"] == [19, 11, 96, 54]
    for block in ("describe", "ground", "consensus", "verify", "refine", "final"):
        assert cand[block] is None
    assert r.seed_from_candidate(cand) == SEED


def test_save_is_atomic_and_round_trips(tmp_path):
    path = r.record_path(tmp_path, "A", 1)
    rec = r.new_record("A", "A_001.png", 1, (10, 10), {"thinking_fallbacks": 0})
    r.save_record(path, rec)
    assert path.exists() and not path.with_name("001.json.tmp").exists()
    assert path.read_text(encoding="utf-8").endswith("}\n") and "\r\n" not in path.read_bytes().decode()
    assert r.load_record(path) == rec
    assert r.load_record(tmp_path / "missing.json") is None


def test_done_markers_do_not_duplicate():
    rec = r.new_record("A", "A_001.png", 1, (10, 10), {})
    r.mark_done(rec, "propose")
    r.mark_done(rec, "propose")
    assert rec["stages_done"] == ["propose"] and r.is_done(rec, "propose") and not r.is_done(rec, "describe")


def test_reset_from_clears_later_stages_and_restores_status():
    rec = r.new_record("A", "A_001.png", 1, (10, 10), {})
    rec["propose"] = {"focus_region": {}}
    rec["stages_done"] = list(r.STAGES)
    ok = r.candidate_from_seed(SEED)
    ok.update(describe={"x": 1}, ground={"x": 1}, consensus={"x": 1}, verify={"x": 1}, refine={"x": 1}, final={"x": 1})
    bad = r.candidate_from_seed(SEED)
    bad.update(describe={"x": 1}, ground={"x": 1})
    r.set_candidate_status(bad, "unreadable", "verify", "missing keys")
    rec["candidates"] = [ok, bad]
    rec["selected"], rec["status"] = 0, "shipped"

    r.reset_from(rec, "ground")
    assert rec["stages_done"] == ["propose", "describe"]
    assert ok["describe"] == {"x": 1} and ok["ground"] is None and ok["consensus"] is None
    assert ok["verify"] is None and ok["refine"] is None and ok["final"] is None
    assert bad["status"] == "ok" and bad["status_stage"] is None and bad["status_reason"] is None
    assert rec["selected"] is None and rec["status"] is None

    r.set_candidate_status(bad, "dropped", "describe", "not_in_region")
    r.reset_from(rec, "verify")
    assert bad["status"] == "dropped"                     # set by an earlier stage: untouched

    r.reset_from(rec, "propose")
    assert rec["propose"] is None and rec["candidates"] == [] and rec["stages_done"] == []
