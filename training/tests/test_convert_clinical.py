import json
import subprocess
import sys

import pytest

from scripts.convert_clinical import convert, rel_image, to_record


def _task(img, instr, bbox, root="/data/AnnotationPipeline/MedSPOT_Training_Data"):
    return {"task_overview": instr, "steps": [{"step_id": 1, "image_path": f"{root}/{img}", "instruction": instr,
            "actions": [{"type": "click", "target": "t", "bbox": bbox}]}]}


def test_percent_xywh_becomes_unit_xyxy_with_centre_click():
    r = to_record(_task("OHIF/OHIF_001.png", "Click Save.", [10, 20, 30, 40])["steps"][0])
    assert r["image"] == "OHIF/OHIF_001.png"
    human, gpt = r["conversations"]
    assert human == {"from": "human", "value": "<image> Click Save."}
    assert gpt["bbox_gt"] == pytest.approx([0.10, 0.20, 0.40, 0.60])
    assert gpt["value"] == "pyautogui.click(x=0.2500, y=0.4000)"
    assert gpt["recipient"] == "os" and gpt["end_turn"] is True


def test_edge_box_is_clamped():
    g = to_record(_task("a/a.png", "x", [90, 95, 10.01, 5.01])["steps"][0])["conversations"][1]
    x1, y1, x2, y2 = g["bbox_gt"]
    assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def test_degenerate_box_raises():
    with pytest.raises(ValueError):
        to_record(_task("a/a.png", "x", [50, 50, 0, 3])["steps"][0])


def test_rel_image_windows_and_posix_roots():
    assert rel_image("/home/u/AnnotationPipeline/MedSPOT_Training_Data/x/y.png") == "x/y.png"
    assert rel_image(r"C:\foo\MedSPOT_Training_Data\Mango\Mango_1.png") == "Mango/Mango_1.png"
    with pytest.raises(ValueError):
        rel_image("/somewhere/else/y.png")


def test_exclude_by_relative_path_ignores_root():
    tasks = [_task("A/1.png", "a", [1, 1, 2, 2], root="E:/x/MedSPOT_Training_Data"),
             _task("A/2.png", "b", [1, 1, 2, 2])]
    out = convert(tasks, exclude_images={"A/1.png"})
    assert [r["image"] for r in out] == ["A/2.png"]


def test_non_click_or_multi_action_step_raises():
    t = _task("A/1.png", "a", [1, 1, 2, 2])
    t["steps"][0]["actions"][0]["type"] = "type"
    with pytest.raises(ValueError):
        convert([t])


def test_cli_difference_and_expectations(tmp_path):
    set1 = {"tasks": [_task("A/1.png", "a", [1, 1, 2, 2])]}
    allt = {"tasks": set1["tasks"] + [_task("A/2.png", "b", [1, 1, 2, 2]), _task("A/2.png", "c", [3, 3, 2, 2])]}
    (tmp_path / "s1.json").write_text(json.dumps(set1))
    (tmp_path / "all.json").write_text(json.dumps(allt))
    out = tmp_path / "s2.json"
    cmd = [sys.executable, "scripts/convert_clinical.py", "--src", str(tmp_path / "all.json"),
           "--exclude_src", str(tmp_path / "s1.json"), "--out", str(out)]
    r = subprocess.run(cmd + ["--expect_images", "1", "--expect_records", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(json.load(open(out))) == 2
    r = subprocess.run(cmd + ["--expect_images", "5"], capture_output=True, text=True)
    assert r.returncode != 0 and "expected 5 images" in r.stderr
