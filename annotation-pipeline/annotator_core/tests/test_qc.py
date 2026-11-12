import json

from annotator import qc
from annotator.records import load_record, record_path
from annotator.runner import RunConfig, run_app
from tests.conftest import VERIFY_MARK, FakeGenerator, happy_router, make_data, make_frame, run


def records_for(tmp_path, router=None, n=2):
    cfg = RunConfig(work=tmp_path / "work", data_root=make_data(tmp_path, n), app="IrfanView")
    run(run_app(FakeGenerator(router=router or happy_router()), cfg, log=lambda *_: None))
    return cfg


def test_stats_over_happy_records(tmp_path):
    cfg = records_for(tmp_path)
    s = qc.qc_stats(cfg.work)
    o = s["overall"]
    assert o["frames"] == 2 and o["status"] == {"shipped": 2} and o["tier"] == {"A": 2} and o["verdict"] == {"correct": 2}
    assert o["agreement_histogram"] == {"1.0": 2} and o["refine_rate"] == 0.0 and o["calls_per_frame"] == 6.0
    assert o["abstain_rate"] == 0.0 and o["unreadable_rate"] == 0.0 and o["thinking_fallbacks"] == 0
    assert s["per_app"]["IrfanView"]["frames"] == 2


def test_stats_count_refine_outcomes(tmp_path):
    base = happy_router()

    def judge_says_no(call):
        if VERIFY_MARK in call.user_text:
            return json.dumps({"text_in_box": "Thumbnails", "element_in_box": "row", "verdict": "wrong_element", "reason": "r", "corrected_bbox": None})
        return base(call)
    cfg = records_for(tmp_path, judge_says_no, n=1)
    o = qc.qc_stats(cfg.work)["overall"]
    assert o["status"] == {"lowconf": 1} and o["tier"] == {"C": 1} and o["refine_rate"] == 1.0
    assert o["refine_outcome"] == {"unchanged": 1} and o["calls_per_frame"] == 13.0


def test_overlay_draws_final_box_and_text_strip(tmp_path):
    cfg = records_for(tmp_path, n=1)
    rec = load_record(record_path(cfg.work, "IrfanView", 1))
    img = qc.render_overlay(rec, make_frame())
    assert img.size == (1920, 1080 + qc.STRIP_H)
    assert img.getpixel((288, 162)) == qc.FINAL_COLOUR
    assert img.getpixel((10, 1080 + 4)) == (20, 20, 24)        # the strip background exists below the frame


def test_sample_is_seeded_and_write_qc_renders(tmp_path):
    cfg = records_for(tmp_path, n=4)
    a = [r["frame_index"] for r in qc.sample_records(cfg.work, 2, 20260912)]
    b = [r["frame_index"] for r in qc.sample_records(cfg.work, 2, 20260912)]
    assert a == b and len(a) == 2
    logs = []
    stats = qc.write_qc(cfg.work, tmp_path / "reports", cfg.data_root, n=2, seed=20260912, log=logs.append)
    assert stats["overall"]["frames"] == 4
    overlays = sorted((tmp_path / "reports" / "overlays").glob("*.png"))
    assert len(overlays) == 2 and overlays[0].name.startswith("IrfanView_") and overlays[0].name.endswith("_shipped.png")
    assert (tmp_path / "reports" / "qc_stats.json").exists() and any("overlays=2" in line for line in logs)


def test_review_image_draws_only_the_final_box_and_the_instructions(tmp_path):
    cfg = records_for(tmp_path, n=1)
    rec = load_record(record_path(cfg.work, "IrfanView", 1))
    img = qc.render_review(rec, make_frame())
    assert img.width == 1920 and img.height > 1080 + 150
    assert img.getpixel((288, 162)) == qc.REVIEW_COLOUR
    assert img.getpixel((10, 1080 + 4)) == qc.STRIP_BG
    assert qc.review_lines(rec)[1].startswith("F: ") and qc.review_lines(rec)[3].startswith("S: ")


def test_review_lines_explain_an_abstained_frame():
    rec = {"app": "IrfanView", "frame": "IrfanView_001.png", "frame_index": 1, "status": "abstained",
           "propose": {"rejected_reason": "no_candidates"}, "candidates": [], "selected": None}
    lines = qc.review_lines(rec)
    assert "abstained" in lines[0] and any("no_candidates" in line for line in lines)
    img = qc.render_review(rec, make_frame())
    assert img.getpixel((288, 162)) != qc.REVIEW_COLOUR


def test_batch_report_writes_one_png_per_app_under_its_batch(tmp_path):
    cfg = records_for(tmp_path, n=3)
    logs = []
    got = qc.write_batch_report(cfg.work, cfg.data_root, tmp_path / "qwen_batches", 1, 7, log=logs.append)
    pick = qc.batch_frame(cfg.data_root, "IrfanView", 1, 7)
    files = sorted((tmp_path / "qwen_batches" / "batch_001").glob("*.png"))
    assert [p.name for p in files] == [f"IrfanView_{pick.index:03d}_shipped.png"]
    assert got == {"frames": 1, "shipped": 1}
    assert any(line.startswith("BATCH_REPORT batch=1 frames=1 shipped=1") for line in logs)


def test_batch_report_marks_a_frame_with_no_record(tmp_path):
    root = make_data(tmp_path, 2)
    got = qc.write_batch_report(tmp_path / "work", root, tmp_path / "qwen_batches", 1, 7, log=lambda *_: None)
    [png] = (tmp_path / "qwen_batches" / "batch_001").glob("*.png")
    assert png.name.endswith("_missing.png") and got == {"frames": 1, "missing": 1}
