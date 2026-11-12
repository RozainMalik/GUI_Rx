import json

from annotator import export
from annotator.records import STAGES, new_record, record_path, save_record
from tests.conftest import INSTR

ROOT = "/data/AnnotationPipeline/MedSPOT_Training_Data"


def make_record(work, app, idx, tier="A", status="shipped", review=None, element_name="Slideshow..."):
    rec = new_record(app, f"{app}_{idx:03d}.png", idx, (1920, 1080), {"prompt_hash": "h", "thinking_fallbacks": 0})
    cand = {"rank": 1, "status": "ok",
            "element": {"element_name": element_name, "element_kind": "menu item", "text_on_screen": "Slideshow...", "icon": "", "state": "plain"},
            "final": {"tier": tier, "bbox_px": [288, 162, 576, 216], "instructions": dict(INSTR), "agreement": 1.0, "verified": True}}
    rec.update(candidates=[cand], selected=0, status=status, review=review, stages_done=list(STAGES))
    save_record(record_path(work, app, idx), rec)


def test_nested_export_three_tasks_per_frame_with_slash_paths(tmp_path):
    work = tmp_path / "work"
    make_record(work, "IrfanView", 1)
    make_record(work, "IrfanView", 2, tier="B")
    m = export.write_export(work, tmp_path / "out", ROOT, log=lambda *_: None)
    data = json.loads((tmp_path / "out" / "medspot_train.json").read_text(encoding="utf-8"))
    assert len(data["tasks"]) == 6 and m["frames"] == 2 and m["tasks"] == 6 and m["per_tier"] == {"A": 1, "B": 1}
    t = data["tasks"][0]
    assert t["task_overview"] == INSTR["functional"] and t["steps"][0]["step_id"] == 1
    assert t["steps"][0]["image_path"] == ROOT + "/IrfanView/IrfanView_001.png"
    assert t["steps"][0]["actions"][0] == {"type": "click", "target": "slideshow", "bbox": [15.0, 15.0, 15.0, 5.0]}
    assert [x["steps"][0]["instruction"] for x in data["tasks"][:3]] == [INSTR["functional"], INSTR["visual"], INSTR["spatial"]]
    assert m["bbox_format"] == "percent_xywh_0_100" and m["shape"] == "nested" and m["image_root"] == ROOT
    assert m["contaminated"] is False and m["benchmark_apps"] == [] and m["prompt_hash"] == "h"
    assert m["per_app"] == {"IrfanView": 2} and m["pipeline_version"] == "1.0.0"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["frames"] == 2


def test_flat_shape_drops_the_steps_wrapper(tmp_path):
    work = tmp_path / "work"
    make_record(work, "IrfanView", 1)
    export.write_export(work, tmp_path / "out", ROOT, flat=True, log=lambda *_: None)
    data = json.loads((tmp_path / "out" / "medspot_train_flat.json").read_text())
    t = data["tasks"][0]
    assert "steps" not in t and t["image_path"].endswith("/IrfanView_001.png") and t["actions"][0]["type"] == "click"


def test_benchmark_apps_excluded_unless_flagged(tmp_path):
    work = tmp_path / "work"
    make_record(work, "IrfanView", 1)
    make_record(work, "3D_Slicer", 1)
    m = export.write_export(work, tmp_path / "o1", ROOT, log=lambda *_: None)
    assert m["per_app"] == {"IrfanView": 1}
    logs = []
    m2 = export.write_export(work, tmp_path / "o2", ROOT, include_benchmark=True, log=logs.append)
    assert m2["per_app"] == {"3D_Slicer": 1, "IrfanView": 1} and m2["contaminated"] is True
    assert "3D_Slicer" in m2["benchmark_apps"] and any(line.startswith("WARNING") for line in logs)


def test_tier_c_set_and_review_gate(tmp_path):
    work = tmp_path / "work"
    make_record(work, "A", 1, tier="C", status="lowconf")
    make_record(work, "A", 2, tier="A", review={"verdict": "rejected"})
    make_record(work, "A", 3, tier="A", review={"verdict": "corrected", "bbox_px": [0, 0, 192, 108]})
    assert export.write_export(work, tmp_path / "c", ROOT, tiers=("C",), log=lambda *_: None)["frames"] == 1
    m = export.write_export(work, tmp_path / "ab", ROOT, log=lambda *_: None)
    assert m["frames"] == 1
    data = json.loads((tmp_path / "ab" / "medspot_train.json").read_text())
    assert data["tasks"][0]["steps"][0]["image_path"].endswith("A_003.png")
    assert data["tasks"][0]["steps"][0]["actions"][0]["bbox"] == [0.0, 0.0, 10.0, 10.0]     # the human's box ships


def test_snake_case_and_target_fallbacks(tmp_path):
    assert export.snake_case("Save As...") == "save_as" and export.snake_case("") == ""
    work = tmp_path / "work"
    make_record(work, "A", 1, element_name="")
    rows = export.shipped_rows(work)
    assert export.target_name(rows[0]) == "menu_item"


def test_windows_image_root_is_normalised(tmp_path):
    work = tmp_path / "work"
    make_record(work, "A", 1)
    export.write_export(work, tmp_path / "out", "C:\\data\\shots", log=lambda *_: None)
    data = json.loads((tmp_path / "out" / "medspot_train.json").read_text())
    assert data["tasks"][0]["steps"][0]["image_path"] == "C:/data/shots/A/A_001.png"


def test_empty_work_exports_nothing(tmp_path):
    m = export.write_export(tmp_path / "nowork", tmp_path / "out", ROOT, log=lambda *_: None)
    assert m["frames"] == 0 and m["prompt_hash"] == []


def test_dataset_name_drives_the_file_name_and_manifest(tmp_path):
    work = tmp_path / "work"
    make_record(work, "IrfanView", 1)
    m = export.write_export(work, tmp_path / "out", ROOT, dataset="gui_grounding", log=lambda *_: None)
    assert (tmp_path / "out" / "gui_grounding_train.json").exists() and not (tmp_path / "out" / "medspot_train.json").exists()
    assert m["dataset"] == "gui_grounding" and m["format"] == "medspot"          # the schema label stays; the dataset is named
    export.write_export(work, tmp_path / "flat", ROOT, dataset="gui_grounding", flat=True, log=lambda *_: None)
    assert (tmp_path / "flat" / "gui_grounding_train_flat.json").exists()
    m0 = export.write_export(work, tmp_path / "default", ROOT, log=lambda *_: None)
    assert (tmp_path / "default" / "medspot_train.json").exists() and m0["dataset"] == "medspot"
