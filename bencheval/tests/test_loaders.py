import json

from PIL import Image

from benchev.benchmarks import load


def _img(p, size=(200, 100)):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(p)


def test_screenspot_v2_layout(tmp_path):
    _img(tmp_path / "a.png")
    json.dump([{"img_filename": "a.png", "bbox": [20, 10, 40, 30], "instruction": "x", "split": "web",
                "data_type": "icon", "data_source": "gitlab"}], open(tmp_path / "subset.json", "w"))
    [s] = load("screenspot", tmp_path)
    assert s.sid == "0000_a" and s.gt == (0.1, 0.1, 0.3, 0.4) and s.tags["group"] == "web" and s.tags["ui_type"] == "icon"


def test_native_size_from_file(tmp_path):
    _img(tmp_path / "a.png", (400, 100))   # annotation has no img_size -> size read from the file
    json.dump([{"img_filename": "a.png", "bbox": [0, 0, 40, 10], "instruction": "x", "split": "web"}],
              open(tmp_path / "subset.json", "w"))
    assert load("screenspot", tmp_path)[0].size == (400, 100)


def test_sspro_layout(tmp_path):
    (tmp_path / "annotations").mkdir()
    _img(tmp_path / "images" / "b.png", (1000, 500))
    json.dump([{"id": "dev_1", "img_filename": "b.png", "bbox": [100, 50, 200, 100], "instruction": "x",
                "group": "Dev", "ui_type": "text", "img_size": [1000, 500]}],
              open(tmp_path / "annotations" / "all.json", "w"))
    [s] = load("sspro", tmp_path)
    assert s.sid == "dev_1" and s.gt == (0.1, 0.1, 0.2, 0.2) and s.tags == {"group": "Dev", "ui_type": "text"}


def test_osworld_all_three_kinds(tmp_path):
    rows = [{"id": "b", "image_path": "i.png", "image_size": [100, 100], "instruction": "x", "box_type": "bbox",
             "box_coordinates": [10, 10, 20, 20], "GUI_types": ["Button"]},
            {"id": "p", "image_path": "i.png", "image_size": [100, 100], "instruction": "x", "box_type": "polygon",
             "box_coordinates": [10, 10, 30, 10, 30, 30], "GUI_types": []},
            {"id": "r", "image_path": "i.png", "image_size": [100, 100], "instruction": "x", "box_type": "refusal",
             "box_coordinates": [0, 0, 0, 0], "GUI_types": []}]
    for f in ("OSWorld-G.json", "OSWorld-G_refined.json"):
        json.dump(rows, open(tmp_path / f, "w"))
    json.dump({"classified": {"text_matching": [{"id": "b"}], "element_recognition": [{"id": "b"}],
                              "refusal": [{"id": "r"}]}, "unclassified": []},
              open(tmp_path / "classification_result.json", "w"))
    _img(tmp_path / "images" / "i.png", (100, 100))
    s = {x.sid: x for x in load("osworld_g", tmp_path)}
    # upstream qwen25_vllm_osworld_g_jedi.py: every classification_result.json group a sample is listed in
    assert s["b"].gt == (0.1, 0.1, 0.3, 0.3) and s["b"].tags["category"] == ["text_matching", "element_recognition"]
    assert s["p"].gt_kind == "polygon" and s["p"].gt == ((0.1, 0.1), (0.3, 0.1), (0.3, 0.3))
    assert s["p"].tags["category"] == ["unclassified"]
    assert s["r"].gt_kind == "refusal" and s["r"].gt is None and s["r"].tags["category"] == ["refusal"]
    assert len(load("osworld_g_refined", tmp_path)) == 3


def test_ui_vision_and_mmbench(tmp_path):
    eg = tmp_path / "uiv" / "annotations" / "element_grounding"
    eg.mkdir(parents=True)
    for sp in ("basic", "functional", "spatial"):
        json.dump([{"image_path": "u.png", "image_size": [100, 50], "bbox": [10, 5, 20, 10],
                    "prompt_to_evaluate": "x", "element_type": "icon", "platform": "VSCode"}],
                  open(eg / f"element_grounding_{sp}.json", "w"))
    _img(tmp_path / "uiv" / "images" / "u.png", (100, 50))
    u = load("ui_vision", tmp_path / "uiv")
    assert [x.sid for x in u] == ["basic_0000", "functional_0000", "spatial_0000"] and u[0].gt == (0.1, 0.1, 0.2, 0.2)
    m = tmp_path / "mmb"
    _img(m / "images" / "sub" / "q.png", (100, 100))
    json.dump([{"index": 7, "image_path": "q.png", "image_size": [100, 100], "instruction": "x", "bbox": [.1, .2, .3, .4],
                "grounding_type": "basic", "data_type": "text", "platform": "windows", "app_name": "x"}],
              open(m / "L2_annotations.json", "w"))
    [q] = load("mmbench_gui", m)
    assert q.sid == "7" and q.image_path.endswith("q.png") and q.tags["group"] == "basic" and q.tags["app"] == "windows"


def test_medspot_uses_core_steps(tmp_path):
    ann = tmp_path / "Annotations"
    ann.mkdir()
    json.dump({"tasks": [{"task_overview": "t", "steps": [
        {"image_path": "Images/A/1.png", "instruction": "a", "actions": [{"type": "click", "bbox": [10, 10, 20, 20]}]},
        {"image_path": "Images/A/2.png", "instruction": "b", "actions": [{"type": "scroll"}]},
        {"image_path": "Images/A/3.png", "instruction": "c", "actions": [{"type": "drag", "bbox": [0, 0, 50, 50]}]}]}]},
        open(ann / "AppA.json", "w"))
    _img(tmp_path / "Images" / "A" / "1.png")
    _img(tmp_path / "Images" / "A" / "3.png")
    s = load("medspot", tmp_path)
    assert [x.sid for x in s] == ["AppA.json#0#1", "AppA.json#0#3"]
    assert s[0].gt == (0.1, 0.1, 0.3, 0.3) and s[1].tags["action_type"] == "drag"
