import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc"))
import verify_annotations as va  # noqa: E402

from tests.conftest import make_frame  # noqa: E402


def write_export(d, image_paths, bbox=(15.0, 15.0, 15.0, 5.0), flat=False, manifest_tasks=None):
    d.mkdir(parents=True, exist_ok=True)
    tasks = []
    for p in image_paths:
        step = {"step_id": 1, "image_path": p, "instruction": "x", "actions": [{"type": "click", "target": "t", "bbox": list(bbox)}]}
        tasks.append({"task_overview": "x", **step} if flat else {"task_overview": "x", "steps": [step]})
    (d / ("medspot_train_flat.json" if flat else "medspot_train.json")).write_text(json.dumps({"tasks": tasks}))
    (d / "manifest.json").write_text(json.dumps({"tasks": len(tasks) if manifest_tasks is None else manifest_tasks, "contaminated": False}))


def test_ok_when_every_image_exists(tmp_path, capsys):
    shot = tmp_path / "data" / "A" / "A_001.png"
    shot.parent.mkdir(parents=True)
    make_frame().save(shot)
    write_export(tmp_path / "exp", [str(shot).replace("\\", "/")] * 3)
    assert va.check(tmp_path / "exp") is True
    assert va.main([str(tmp_path / "exp")]) == 0 and "IMAGES_OK" in capsys.readouterr().out


def test_missing_image_bad_box_or_manifest_mismatch_fail(tmp_path, capsys):
    write_export(tmp_path / "e1", ["/nowhere/A/A_001.png"])
    assert va.check(tmp_path / "e1") is False and "MISSING" in capsys.readouterr().out
    shot = tmp_path / "A_001.png"
    make_frame().save(shot)
    write_export(tmp_path / "e2", [str(shot).replace("\\", "/")], bbox=(95.0, 10.0, 10.0, 5.0), flat=True)
    assert va.check(tmp_path / "e2") is False
    write_export(tmp_path / "e3", [str(shot).replace("\\", "/")], manifest_tasks=99)
    assert va.check(tmp_path / "e3") is False
    assert va.main([str(tmp_path / "e1"), str(tmp_path / "e3")]) == 1 and "IMAGES_FAIL" in capsys.readouterr().out
    assert va.check(tmp_path / "nothing_here") is False


def test_all_exports_empty_fails_even_though_each_check_passes(tmp_path, capsys):
    write_export(tmp_path / "e1", [])
    write_export(tmp_path / "e2", [])
    rc = va.main([str(tmp_path / "e1"), str(tmp_path / "e2")])
    out = capsys.readouterr().out
    assert rc == 1 and "IMAGES_FAIL no tasks in any export" in out


def test_one_empty_export_among_others_is_still_ok(tmp_path, capsys):
    shot = tmp_path / "A_001.png"
    make_frame().save(shot)
    write_export(tmp_path / "e1", [])
    write_export(tmp_path / "e2", [str(shot).replace("\\", "/")])
    rc = va.main([str(tmp_path / "e1"), str(tmp_path / "e2")])
    out = capsys.readouterr().out
    assert rc == 0 and "IMAGES_OK" in out


def test_any_dataset_name_is_found(tmp_path, capsys):
    shot = tmp_path / "data" / "A" / "A_001.png"
    shot.parent.mkdir(parents=True)
    make_frame().save(shot)
    write_export(tmp_path / "exp", [str(shot).replace("\\", "/")] * 3)
    (tmp_path / "exp" / "medspot_train.json").rename(tmp_path / "exp" / "gui_grounding_train.json")
    assert va.check(tmp_path / "exp") is True
    assert va.main([str(tmp_path / "exp")]) == 0 and "IMAGES_OK" in capsys.readouterr().out
