import json

import pytest

from scripts.build_groundcua import image_rel, image_url, load_instructions, sample, to_record, valid


def _row(bbox, platform="Spyder", rid="ab" * 32, instr="Click Save."):
    return {"id": rid, "platform": platform, "bbox": bbox, "text": "Save", "category": "Button", "instruction": instr}


def test_pixel_box_becomes_unit_box_with_centre_click():
    r = to_record(_row([192.0, 101.2, 384.0, 202.4]), 1920, 1012)
    assert r["image"] == "Spyder/" + "ab" * 32 + ".png"
    human, gpt = r["conversations"]
    assert human == {"from": "human", "value": "<image> Click Save."}
    assert gpt["bbox_gt"] == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert gpt["value"] == "pyautogui.click(x=0.1500, y=0.1500)"
    assert gpt["recipient"] == "os" and gpt["end_turn"] is True


def test_box_clamped_to_image():
    x1, y1, x2, y2 = to_record(_row([-5, 990, 100, 1030]), 1920, 1012)["conversations"][1]["bbox_gt"]
    assert x1 == 0.0 and y2 == 1.0 and x1 < x2 and y1 < y2


def test_box_fully_outside_raises():
    with pytest.raises(ValueError):
        to_record(_row([1950, 10, 2000, 40]), 1920, 1012)


def test_valid_drops_tiny_boxes_and_empty_instructions():
    assert valid(_row([10, 10, 12, 12]))
    assert not valid(_row([10, 10, 10.5, 30]))
    assert not valid(_row([10, 10, 30, 30], instr="  "))


def test_sample_is_seeded_sized_and_in_file_order():
    rows = [{"k": i} for i in range(1000)]
    a, b = sample(rows, 100, 0), sample(rows, 100, 0)
    assert a == b and len(a) == 100
    assert [r["k"] for r in a] == sorted(r["k"] for r in a)
    assert sample(rows, 100, 1) != a
    with pytest.raises(ValueError):
        sample(rows, 1001, 0)


def test_url_quotes_platform_and_path_keeps_it():
    r = _row([0, 0, 5, 5], platform="Code::Blocks")
    assert image_rel(r) == "Code::Blocks/" + "ab" * 32 + ".png"
    assert image_url(r).endswith("/images/Code%3A%3ABlocks/" + "ab" * 32 + ".png")
    assert " " not in image_url(_row([0, 0, 5, 5], platform="Arduino IDE"))


def test_load_instructions_reads_every_file_in_name_order(tmp_path):
    d = tmp_path / "groundcua_data"
    d.mkdir()
    (d / "direct_b.json").write_text(json.dumps([_row([0, 0, 5, 5], rid="b")]))
    (d / "direct_a.json").write_text(json.dumps([_row([0, 0, 5, 5], rid="a1"), _row([0, 0, 5, 5], rid="a2")]))
    assert [r["id"] for r in load_instructions(str(tmp_path))] == ["a1", "a2", "b"]


def test_point_only_and_instructionless_rows_are_invalid():
    """functional_instructions_extra / spatial_data have a point ('cordinate') but no box; remaining_instructions has
    boxes but no instruction. GUI-AIMA's patch labels need a box and the prompt needs an instruction."""
    assert not valid({"id": "x", "platform": "Affine", "cordinate": [169.4, 153.1], "instruction": "Select it"})
    assert not valid({"id": "x", "platform": "Affine", "bbox": [0, 0, 10, 10], "text": "Save", "category": "Button"})


def test_convert_drops_boxes_outside_the_image_but_fails_past_the_limit():
    """Real seed-0 sample: 20 of 250,000 boxes lie fully outside their screenshot (annotation errors) -> dropped.
    A systematic mismatch (more than max_drop of the records) must still stop the build."""
    from scripts.build_groundcua import convert
    rows = [_row([10, 10, 50, 50], rid=f"{i:02d}") for i in range(10)] + [_row([2000, 10, 2050, 40], rid="zz")]
    recs, dropped = convert(rows, lambda rel: (1920, 1012), max_drop=0.1)
    assert len(recs) == 10 and [d["id"] for d in dropped] == ["zz"]
    with pytest.raises(SystemExit):
        convert(rows, lambda rel: (1920, 1012), max_drop=0.05)
