import json
import random

from annotator import grading
from annotator.export import write_export
from annotator.records import load_record, record_path, save_record
from annotator.runner import RunConfig, run_app
from tests.conftest import FakeGenerator, happy_router, make_data, run


def records_for(tmp_path, n=3):
    cfg = RunConfig(work=tmp_path / "work", data_root=make_data(tmp_path, n), app="IrfanView")
    run(run_app(FakeGenerator(router=happy_router()), cfg, log=lambda *_: None))
    return cfg


def test_build_sample_is_seeded_sorted_and_written(tmp_path):
    cfg = records_for(tmp_path, 5)
    a = grading.build_sample(cfg.work, 3, 20260912)
    b = grading.build_sample(cfg.work, 3, 20260912)
    assert a == b and len(a) == 3 and [i["frame_index"] for i in a] == sorted(i["frame_index"] for i in a)
    item = a[0]
    assert item["image"] == f"IrfanView/{item['frame']}" and item["bbox_px"] == [288, 162, 576, 216]
    assert item["tier"] == "A" and item["element"] == "Slideshow..." and set(item["instructions"]) == {"functional", "visual", "spatial"}
    written = json.loads(grading.sample_path(cfg.work, ("A", "B")).read_text())
    assert written["seed"] == 20260912 and len(written["items"]) == 3
    assert grading.build_sample(cfg.work, 3, 1, tiers=("C",)) == []


def test_build_sample_refreshes_existing_sample_keeps_frames(tmp_path):
    cfg = records_for(tmp_path, 3)
    first = grading.build_sample(cfg.work, 3, 20260912)
    assert len(first) == 3
    order = [(i["app"], i["frame"]) for i in first]

    p1 = record_path(cfg.work, "IrfanView", 1)
    r1 = load_record(p1)
    r1["candidates"][r1["selected"]]["final"]["bbox_px"] = [0, 0, 50, 50]
    save_record(p1, r1)

    p2 = record_path(cfg.work, "IrfanView", 2)
    r2 = load_record(p2)
    r2["candidates"][r2["selected"]]["final"]["tier"] = "C"        # no longer shipped in A/B
    save_record(p2, r2)

    logs = []
    refreshed = grading.build_sample(cfg.work, 3, 20260912, log=logs.append)
    kept_order = [pair for pair in order if pair[1] != "IrfanView_002.png"]
    assert [(i["app"], i["frame"]) for i in refreshed] == kept_order   # same order, minus the dropped one
    assert len(refreshed) == 2
    moved = next(i for i in refreshed if i["frame"] == "IrfanView_001.png")
    assert moved["bbox_px"] == [0, 0, 50, 50]
    assert any("dropped 1" in line for line in logs)
    written = json.loads(grading.sample_path(cfg.work, ("A", "B")).read_text())
    assert len(written["items"]) == 2                                 # refreshed file written to disk


def test_build_sample_resample_rebuilds(tmp_path):
    cfg = records_for(tmp_path, 3)
    original = grading.build_sample(cfg.work, 2, 5)
    assert len(original) == 2

    make_data(tmp_path, 4)                       # adds IrfanView_004.png to the existing data root
    run(run_app(FakeGenerator(router=happy_router()), RunConfig(work=cfg.work, data_root=cfg.data_root, app="IrfanView"),
                 log=lambda *_: None))

    refreshed = grading.build_sample(cfg.work, 2, 5)
    assert [(i["app"], i["frame"]) for i in refreshed] == [(i["app"], i["frame"]) for i in original]  # no new frames

    resampled = grading.build_sample(cfg.work, 2, 5, resample=True)
    rows = grading.grade_rows(cfg.work, ("A", "B"))
    expected = sorted(random.Random(5).sample(rows, min(2, len(rows))), key=lambda i: (i["app"], i["frame_index"]))
    assert [(i["app"], i["frame"]) for i in resampled] == [(i["app"], i["frame"]) for i in expected]


def test_verdicts_are_keyed_by_image_and_box_last_wins(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    grading.append_verdict(path, {"image": "A/A_001.png", "verdict": "wrong", "bbox_px": [1, 1, 5, 5], "original_bbox_px": [1, 1, 5, 5]})
    grading.append_verdict(path, {"image": "A/A_001.png", "verdict": "correct", "bbox_px": [1, 1, 5, 5], "original_bbox_px": [1, 1, 5, 5]})
    grading.append_verdict(path, {"image": "A/A_001.png", "verdict": "loose", "bbox_px": [9, 9, 20, 20], "original_bbox_px": [9, 9, 20, 20]})
    with path.open("a") as f:
        f.write("not json\n")
    v = grading.load_verdicts(path)
    assert len(v) == 2 and v[grading.verdict_key("A/A_001.png", [1, 1, 5, 5])]["verdict"] == "correct"
    assert all("at" in row for row in v.values())
    assert grading.load_verdicts(tmp_path / "missing.jsonl") == {}


def test_summarise_counts_only_sampled_items():
    items = [{"image": "A/1.png", "bbox_px": [1, 1, 5, 5]}, {"image": "A/2.png", "bbox_px": [1, 1, 5, 5]}]
    verdicts = {grading.verdict_key("A/1.png", [1, 1, 5, 5]): {"verdict": "correct"},
                grading.verdict_key("A/9.png", [1, 1, 5, 5]): {"verdict": "wrong"}}
    s = grading.summarise(verdicts, items)
    assert s == {"items": 2, "graded": 1, "correct": 1, "loose": 0, "wrong": 0, "corrected": 0, "correct_rate": 1.0}
    assert grading.summarise({}, items)["correct_rate"] is None


def test_apply_grades_writes_review_and_export_honours_it(tmp_path):
    cfg = records_for(tmp_path, 3)
    box = [288, 162, 576, 216]
    vp = grading.verdicts_path(cfg.work)
    grading.append_verdict(vp, {"image": "IrfanView/IrfanView_001.png", "verdict": "wrong", "bbox_px": box, "original_bbox_px": box})
    grading.append_verdict(vp, {"image": "IrfanView/IrfanView_002.png", "verdict": "corrected", "bbox_px": [0, 0, 192, 108], "original_bbox_px": box})
    grading.append_verdict(vp, {"image": "IrfanView/IrfanView_003.png", "verdict": "correct", "bbox_px": [1, 1, 2, 2], "original_bbox_px": [1, 1, 2, 2]})   # stale box: ignored
    assert grading.apply_grades(cfg.work) == 2
    assert grading.apply_grades(cfg.work) == 0                                  # idempotent
    r1 = load_record(record_path(cfg.work, "IrfanView", 1))
    assert r1["review"]["verdict"] == "rejected" and r1["review"]["bbox_px"] is None and r1["review"]["graded_at"]
    r2 = load_record(record_path(cfg.work, "IrfanView", 2))
    assert r2["review"]["verdict"] == "corrected" and r2["review"]["bbox_px"] == [0, 0, 192, 108]
    assert load_record(record_path(cfg.work, "IrfanView", 3))["review"] is None
    m = write_export(cfg.work, tmp_path / "out", "/r", log=lambda *_: None)
    assert m["frames"] == 2
    tasks = json.loads((tmp_path / "out" / "medspot_train.json").read_text())["tasks"]
    assert tasks[0]["steps"][0]["actions"][0]["bbox"] == [0.0, 0.0, 10.0, 10.0]
