import json

import pytest

from attn_ft.medspot_metrics import both_rules, tol_hit, tol_records
from medspot_strict.core import load_tasks, make_record

W, H = 1000, 500
BOX = [10, 20, 20, 20]          # percent xywh -> normalised (0.1, 0.2, 0.3, 0.4) -> px x 100..300, y 100..200


def _st(typ="click", box=BOX):
    return {"image_path": "Images/A/a.png", "instruction": "Click.", "actions": [{"type": typ, "bbox": box}]}


def _data(tmp_path, files):
    for name, tasks in files.items():
        (tmp_path / name).write_text(json.dumps({"tasks": tasks}))
    return load_tasks(tmp_path)


def _recs(tasks, points):
    """points: step key -> normalised point (or None)."""
    out = {}
    for t in tasks:
        for s in t.steps:
            r = make_record(s, points[s.key], "")
            r["image_size"] = [W, H]
            out[s.key] = r
    return out


def test_tolerance_boundary_is_14px_inclusive():
    box = (0.1, 0.2, 0.3, 0.4)
    assert tol_hit(((100 - 14) / W, 0.3), box, (W, H))          # exactly 14 px left of the box
    assert not tol_hit(((100 - 15) / W, 0.3), box, (W, H))
    assert tol_hit((0.3, (200 + 14) / H), box, (W, H))          # 14 px below
    assert not tol_hit(None, box, (W, H))


def test_tol_records_moves_only_tolerated_points_onto_the_box():
    rec = {"pred_point": [(100 - 10) / W, 0.3], "gt_bbox": [0.1, 0.2, 0.3, 0.4], "image_size": [W, H]}
    far = {"pred_point": [0.9, 0.9], "gt_bbox": [0.1, 0.2, 0.3, 0.4], "image_size": [W, H]}
    out = tol_records({"a": rec, "b": far})
    assert out["a"]["pred_point"] == pytest.approx([0.1, 0.3]) and out["b"]["pred_point"] == [0.9, 0.9]
    assert rec["pred_point"][0] == pytest.approx(0.09)          # input not mutated


def test_near_miss_counts_only_under_tol14px(tmp_path):
    tasks = _data(tmp_path, {"Orthanc_Annotation.json": [{"task_overview": "t", "steps": [_st(), _st()]}]})
    keys = [s.key for s in tasks[0].steps]
    m = both_rules(tasks, _recs(tasks, {keys[0]: (0.2, 0.3), keys[1]: ((100 - 10) / W, 0.3)}))
    assert m["strict"]["TCA (%)"] == 0.0 and m["strict"]["SHR (%)"] == 50.0
    assert m["tol14px"]["TCA (%)"] == 100.0 and m["tol14px"]["SHR (%)"] == 100.0


def test_drag_miss_fails_task_under_both_rules(tmp_path):
    tasks = _data(tmp_path, {"Orthanc_Annotation.json": [{"task_overview": "t", "steps": [_st(), _st("drag")]}]})
    keys = [s.key for s in tasks[0].steps]
    m = both_rules(tasks, _recs(tasks, {keys[0]: (0.2, 0.3), keys[1]: (0.9, 0.9)}))
    assert m["strict"]["evaluable_steps"] == 2
    assert m["strict"]["TCA (%)"] == 0.0 and m["tol14px"]["TCA (%)"] == 0.0
