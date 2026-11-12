import json

from annotator.geometry import px_to_1000
from annotator.imaging import box_to_crop, make_crop, verify_window
from annotator.stages import verify as s4
from tests.conftest import FakeGenerator, run

BOX = (288, 162, 576, 216)
INSTR = {"functional": "Start a slideshow of the loaded images.",
         "visual": "Click the menu item labelled 'Slideshow...'.", "spatial": "Click the eighth row."}


def reply(verdict="correct", text="Slideshow...", corrected=None):
    return json.dumps({"text_in_box": text, "element_in_box": "menu row", "verdict": verdict,
                       "reason": "it is the row", "corrected_bbox": corrected})


def test_correct_with_matching_readback_is_verified(frame):
    gen = FakeGenerator([reply()])
    v = run(s4.verify(gen, frame, BOX, INSTR, "Slideshow..."))
    assert v.verified and v.verdict == "correct" and v.readback_match is True
    assert v.corrected_bbox_px is None and v.error is None and v.calls == 1
    assert v.window == list(verify_window(BOX, 1920, 1080)) == [0, 42, 864, 336]
    call = gen.calls[0]
    assert call.thinking is False and len(call.images) == 2
    assert call.images[0].size == (1920, 1080) and call.images[0].getpixel((288, 162)) == (255, 0, 0)
    _, zoom, crop = s4.images_for(frame, BOX)
    x, y = box_to_crop(BOX, crop)[:2]
    assert call.images[1].size == zoom.size and call.images[1].size[0] == 1024
    assert call.images[1].getpixel((x, y)) == (255, 0, 0)


def test_correct_but_readback_mismatch_is_not_verified(frame):
    v = run(s4.verify(FakeGenerator([reply(text="Thumbnails")]), frame, BOX, INSTR, "Slideshow..."))
    assert not v.verified and v.readback_match is False and v.verdict == "correct"


def test_no_label_means_readback_not_applicable(frame):
    v = run(s4.verify(FakeGenerator([reply(text="")]), frame, BOX, INSTR, ""))
    assert v.verified and v.readback_match is None


def test_wrong_element_with_corrected_box_maps_from_image_2(frame):
    window = verify_window(BOX, 1920, 1080)
    corrected_px = (300, 240, 560, 270)
    gen = FakeGenerator([reply("wrong_element", "Thumbnails", px_to_1000(corrected_px, window))])
    v = run(s4.verify(gen, frame, BOX, INSTR, "Slideshow..."))
    assert not v.verified and v.verdict == "wrong_element"
    assert all(abs(a - b) <= 1 for a, b in zip(v.corrected_bbox_px, corrected_px))
    assert v.corrected_box_sane is None


def test_corrected_box_ignored_when_verdict_correct(frame):
    v = run(s4.verify(FakeGenerator([reply("correct", corrected=[1, 1, 50, 50])]), frame, BOX, INSTR, "Slideshow..."))
    assert v.corrected_bbox_px is None and v.verified


def test_bad_verdict_enum_is_repaired(frame):
    gen = FakeGenerator([reply(verdict="right"), reply()])
    v = run(s4.verify(gen, frame, BOX, INSTR, "Slideshow..."))
    assert v.verified and v.calls == 2 and "verdict must be exactly one of" in gen.calls[1].history[1][1]


def test_missing_keys_repaired_then_unreadable(frame):
    gen = FakeGenerator(['{"verdict": "correct"}', '{"verdict": "correct"}'])
    v = run(s4.verify(gen, frame, BOX, INSTR, "Slideshow..."))
    assert v.error and "missing keys" in v.error and not v.verified and v.verdict is None and v.calls == 2


def test_prompt_carries_both_instructions(frame):
    gen = FakeGenerator([reply()])
    run(s4.verify(gen, frame, BOX, INSTR, "Slideshow..."))
    t = gen.calls[0].user_text
    assert INSTR["visual"] in t and INSTR["functional"] in t and "PROPOSED answer" in t


def test_pixels_mode_corrected_box_is_in_image_2_pixels(frame):
    box = (288, 162, 576, 216)
    crop = make_crop(frame, verify_window(box, *frame.size))
    space = crop.space("pixels")
    reply_ = json.dumps({"text_in_box": "Open...", "element_in_box": "row", "verdict": "wrong_element", "reason": "r",
                         "corrected_bbox": space.from_px((300, 170, 560, 210))})
    gen = FakeGenerator([reply_])
    v = run(s4.verify(gen, frame, box, INSTR, "Slideshow...", mode="pixels"))
    assert v.corrected_bbox_px == list(space.to_px(space.from_px((300, 170, 560, 210)))) and v.verdict == "wrong_element"
    call = gen.calls[0]
    assert call.kind == "verify" and f"in pixel coordinates ({crop.image.size[0]}x{crop.image.size[1]} pixels) of\n   IMAGE 2" in call.user_text
