import json
from dataclasses import replace

from annotator import calibrate as cal
from tests.conftest import GROUND_MARK, FakeGenerator, make_frame, run


def make_dataset(tmp_path, n=5):
    root = tmp_path / "MedicalDataset"
    (root / "Annotations").mkdir(parents=True)
    (root / "Images" / "X").mkdir(parents=True)
    make_frame().save(root / "Images" / "X" / "Screenshot_1.png")
    steps = [{"step_id": 1, "image_path": "Images/X/Screenshot_1.png",
              "instruction": f"Click step {i}" + (" miss" if i % 5 == 0 else ""),
              "actions": [{"type": "click", "target": "t", "bbox": [10, 10, 10, 5]}]} for i in range(1, n + 1)]
    steps.append({"step_id": 2, "image_path": "Images/X/Screenshot_1.png", "instruction": "typed", "actions": [{"type": "type", "text": "x"}]})
    (root / "Annotations" / "X_Annotation.json").write_text(json.dumps({"tasks": [{"task_overview": "t", "steps": steps}]}), encoding="utf-8")
    return root


def router(call):
    if GROUND_MARK in call.user_text:
        box = [600, 600, 700, 650] if "miss" in call.user_text else [100, 100, 200, 150]
        return json.dumps([{"bbox_2d": box, "label": "x"}])
    return "screenshot"


def test_load_steps_is_deterministic_and_click_only(tmp_path):
    root = make_dataset(tmp_path, 10)
    a = cal.load_benchmark_steps(root, n=4, seed=1)
    b = cal.load_benchmark_steps(root, n=4, seed=1)
    assert a == b and len(a) == 4 and all(s["bbox_pct"] == [10, 10, 10, 5] for s in a) and a[0]["app"] == "X"
    assert len(cal.load_benchmark_steps(root, n=100)) == 10          # the typed step is not a click


def test_resolution_probe_aborts_before_grounding(tmp_path):
    root = make_dataset(tmp_path)
    gen = FakeGenerator(router=router, prompt_tokens=900)
    res = run(cal.calibrate(gen, root, make_frame()))
    assert not res.resolution_ok and not res.ok and res.total == 0 and res.prompt_tokens == 900
    assert len(gen.calls) == 1 and gen.calls[0].user_text == cal.CALIBRATE_USER and gen.calls[0].thinking is False


def test_calibrate_counts_centre_hits(tmp_path):
    root = make_dataset(tmp_path, 10)              # steps 5 and 10 miss
    res = run(cal.calibrate(FakeGenerator(router=router), root, make_frame(), n=10))
    assert res.resolution_ok and res.prompt_tokens == 2100
    assert res.total == 10 and res.hits == 8 and res.accuracy == 0.8 and res.ok
    d = res.details[0]
    assert d["gt_px"] == [192.0, 108.0, 384.0, 162.0] and d["image_size"] == [1920, 1080] and "hit" in d


def test_write_result_writes_marker_only_on_success(tmp_path):
    ok = cal.CalibrationResult(True, 2100, 50, 60, 50 / 60, True)
    bad = replace(ok, hits=40, accuracy=40 / 60, ok=False)
    cal.write_result(tmp_path / "w1", ok)
    assert (tmp_path / "w1" / cal.MARKER).exists()
    assert json.loads((tmp_path / "w1" / "calibration.json").read_text())["hits"] == 50
    cal.write_result(tmp_path / "w2", bad)
    assert (tmp_path / "w2" / "calibration.json").exists() and not (tmp_path / "w2" / cal.MARKER).exists()


def test_probe_threshold_is_a_parameter_and_the_probe_call_is_tagged(tmp_path):
    root = make_dataset(tmp_path)
    gen = FakeGenerator(router=router, prompt_tokens=2100)
    res = run(cal.calibrate(gen, root, make_frame(), min_tokens=2200))
    assert not res.resolution_ok and res.prompt_tokens == 2100 and gen.calls[0].kind == "probe"
    assert run(cal.check_resolution(FakeGenerator(["x"], prompt_tokens=2300), make_frame(), min_tokens=2200)) == (True, 2300)


def test_calibrate_passes_the_mode_to_grounding(tmp_path):
    root = make_dataset(tmp_path, 10)

    def px_router(call):
        if GROUND_MARK in call.user_text:
            box = [600, 600, 700, 650] if "miss" in call.user_text else [200, 110, 380, 160]
            return json.dumps({"boxes": [{"bbox_2d": box, "label": "x"}]})
        return "screenshot"
    gen = FakeGenerator(router=px_router)
    res = run(cal.calibrate(gen, root, make_frame(), n=10, mode="pixels"))
    assert res.ok and res.hits == 8 and res.coord_mode == "pixels"
    assert all("pixel coordinates (1920x1080 pixels)" in c.user_text for c in gen.calls if GROUND_MARK in c.user_text)
