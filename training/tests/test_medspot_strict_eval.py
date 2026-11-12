import json

from test_eval_qwen35 import _run, pointer_ckpt  # noqa: F401  (tiny exported pointer checkpoint + subprocess runner)


def _dataset(tmp_path):
    from PIL import Image
    data = tmp_path / "medspot"
    (data / "Images" / "App").mkdir(parents=True)
    (data / "Annotations").mkdir()
    Image.new("RGB", (640, 360), "white").save(data / "Images" / "App" / "a.png")
    st = lambda typ, box: {"image_path": "Images/App/a.png", "instruction": "Click.",
                           "actions": [{"type": typ, "bbox": box}] if box else []}
    everywhere = [0, 0, 100, 100]
    tasks = [{"task_overview": "t1", "steps": [st("click", everywhere), st("drag", everywhere)]},
             {"task_overview": "t2", "steps": [st("click", None), st("type", everywhere)]}]   # no-bbox step skipped
    for app in ("3DSlicer", "Orthanc"):
        json.dump({"tasks": tasks}, open(data / "Annotations" / f"{app}_Annotation.json", "w"))
    return data


def test_every_bbox_step_is_predicted_and_scored_under_both_rules(pointer_ckpt, tmp_path):
    data, out = _dataset(tmp_path), tmp_path / "out"
    args = ("eval/medspot_strict_qwen35.py", "--ckpt", pointer_ckpt, "--name", "tiny-1step", "--data", str(data),
            "--out_dir", str(out), "--max_pixels", str(200 * 32 * 32))
    _run(*args)
    recs = [json.loads(line) for line in open(out / "predictions.jsonl")]
    assert len(recs) == 6 and all(r["image_size"] == [640, 360] for r in recs)       # 3 bbox steps x 2 apps
    s = json.load(open(out / "summary.json"))
    assert s["evaluable_steps"] == 6 and s["total_tasks"] == 4 and s["complete_run"] is True
    assert s["TCA (%)"] == 100.0 and s["tol14px"]["TCA (%)"] == 100.0             # boxes cover the whole image
    assert set(s["SHR_by_action_type (%)"]) == {"click", "drag", "type"}
    assert "groups" not in s
    _run(*args)                                                                    # resume: nothing predicted twice
    assert sum(1 for _ in open(out / "predictions.jsonl")) == 6
