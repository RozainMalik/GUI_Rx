import json

from annotator.imaging import make_crop
from annotator.prompts import SYSTEM_PLAIN
from annotator.stages import ground as s3
from tests.conftest import FakeGenerator, run


def box_reply(b, label="Slideshow..."):
    return json.dumps([{"bbox_2d": list(b), "label": label}])


def test_blind_grounding_maps_full_frame(frame):
    gen = FakeGenerator([box_reply([150, 150, 300, 200])])
    g = run(s3.ground_blind(gen, frame, "Start a slideshow of the loaded images."))
    assert g.bbox_px == [288, 162, 576, 216] and g.bbox_1000 == [150, 150, 300, 200] and g.box_sane is None
    assert g.label == "Slideshow..." and g.calls == 1 and g.error is None and not g.empty
    call = gen.calls[0]
    assert call.system == SYSTEM_PLAIN and call.thinking is False and call.max_tokens == 512
    assert 'refers\nto: "Start a slideshow of the loaded images."' in call.user_text
    assert call.images == (frame,)


def test_blind_reply_must_hold_exactly_one_box(frame):
    two = json.dumps([{"bbox_2d": [1, 1, 5, 5]}, {"bbox_2d": [2, 2, 6, 6]}])
    gen = FakeGenerator([two, box_reply([150, 150, 300, 200])])
    g = run(s3.ground_blind(gen, frame, "x"))
    assert g.calls == 2 and g.bbox_px == [288, 162, 576, 216]
    assert "exactly one object" in gen.calls[1].history[1][1]


def test_blind_empty_array_is_an_error(frame):
    g = run(s3.ground_blind(FakeGenerator(["[]", "[]"]), frame, "x"))
    assert g.error and g.bbox_px is None and not g.empty and g.calls == 2


def test_insane_box_is_kept_but_flagged(frame):
    g = run(s3.ground_blind(FakeGenerator([box_reply([0, 0, 1000, 1000])]), frame, "x"))
    assert g.bbox_px == [0, 0, 1920, 1080] and g.box_sane == "too_large"


def test_crop_grounding_maps_through_window_and_allows_empty(frame):
    crop = make_crop(frame, (100, 100, 612, 356))          # 512x256 -> shown at 1024x512
    gen = FakeGenerator([box_reply([0, 0, 500, 500]), "[]"])
    prev = 'A previous attempt boxed "Open" instead, which is wrong.'
    g = run(s3.ground_crop(gen, crop, "IrfanView", "Start it.", "Slideshow...", "", prev, frame.size))
    assert g.bbox_px == [100, 100, 356, 228] and g.box_sane is None and not g.empty
    text = gen.calls[0].user_text
    assert "magnified region of a screenshot of IrfanView" in text
    assert prev + ' The control you are looking for is labelled "Slideshow...".' in text
    assert gen.calls[0].images[0].size == (1024, 512) and gen.calls[0].thinking is False
    e = run(s3.ground_crop(gen, crop, "A", "i", "", "floppy disk", "prev", frame.size))
    assert e.empty and e.bbox_px is None and e.error is None and e.calls == 1
    assert "has no text label and carries a floppy disk icon." in gen.calls[1].user_text


def test_target_clause_variants():
    assert s3.target_clause("Save", "") == 'The control you are looking for is labelled "Save".'
    assert s3.target_clause("Save", "floppy disk") == 'The control you are looking for is labelled "Save" and carries a floppy disk icon.'
    assert s3.target_clause("", "") == "The control you are looking for has no text label."


def test_pixels_mode_blind_grounding_and_kinds(frame):
    gen = FakeGenerator([json.dumps({"boxes": [{"bbox_2d": [288, 162, 576, 216], "label": "x"}]})])
    g = run(s3.ground_blind(gen, frame, "Start a slideshow.", mode="pixels"))
    assert g.bbox_px == [288, 162, 576, 216] and g.bbox_1000 == [288, 162, 576, 216] and g.box_sane is None
    call = gen.calls[0]
    assert call.kind == "ground" and call.user_text.endswith("in pixel coordinates (1920x1080 pixels) of the image width and height.")
    assert "0-1000" not in call.user_text


def test_crop_grounding_kind_and_pixels_clause(frame):
    crop = make_crop(frame, (192, 108, 960, 432))
    gen = FakeGenerator(["[]"])
    g = run(s3.ground_crop(gen, crop, "App", "Click it.", "Save", "", "A previous attempt was wrong.", frame.size, mode="pixels"))
    assert g.empty and gen.calls[0].kind == "refine_ground"
    assert f"in pixel coordinates ({crop.image.size[0]}x{crop.image.size[1]} pixels) of this image." in gen.calls[0].user_text
