import json

import pytest
from PIL import ImageDraw

from annotator.geometry import center, px_to_1000
from annotator.imaging import focus_windows
from annotator.stages import refine as s5
from annotator.stages.propose import CandidateSeed
from tests.conftest import REFINE_MARK, TARGET_PX, VERIFY_MARK, FakeGenerator, make_frame, run

W, H = 1920, 1080
D = (288, 162, 576, 216)
OFF = (288, 240, 576, 290)          # another row, inside every focus crop, maps exactly through all three
INSTR = {"functional": "Start a slideshow of the loaded images.",
         "visual": "Click the menu item labelled 'Slideshow...'.",
         "spatial": "Click the eighth row of the open File menu, just below 'Thumbnails'."}
SEED = CandidateSeed(1, "Slideshow...", "menu item", "Slideshow...", "", "highlighted", [150, 150, 300, 200], D)
DESCRIBE = {"bbox_px": list(D), "box_sane": None, "text_on_screen": "Slideshow...", "icon": "",
            "instructions": INSTR, "instruction_errors": []}
GROUND = {"functional": {"bbox_px": list(OFF), "box_sane": None}, "visual": {"bbox_px": list(OFF), "box_sane": None},
          "spatial": {"bbox_px": None, "box_sane": None, "error": "x"}}
VERIFY = {"verdict": "wrong_element", "text_in_box": "Thumbnails", "verified": False,
          "corrected_bbox_px": list(D), "corrected_box_sane": None}
CONSENSUS = {"bbox_px": list(OFF), "agreement": 0.5}
GOOD_VERIFY = json.dumps({"text_in_box": "Slideshow...", "element_in_box": "row", "verdict": "correct", "reason": "r", "corrected_bbox": None})


WINDOWS = focus_windows(center(D), W, H)            # (0,0,960,540), (96,0,768,378), (192,54,672,324)
TAGS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))


@pytest.fixture
def frame():
    """The synthetic frame with a coloured 24 px block at each focus window's origin, so a
    crop can be recognised from its top-left pixel. (All three crops upscale to 1024x576 -
    they share the frame's aspect ratio - so image size cannot tell them apart.)"""
    img = make_frame()
    d = ImageDraw.Draw(img)
    for (x, y, _, _), colour in zip(WINDOWS, TAGS):
        d.rectangle((x, y, x + 24, y + 24), fill=colour)
    return img


def window_of(call):
    px = call.images[0].getpixel((4, 4))
    return WINDOWS[max(range(3), key=lambda i: px[i])]


def make_router(frame, crop_target=D, empties=(), verify_reply=GOOD_VERIFY, rewrite_reply=None):
    """Route by prompt marker; recover which focus crop a call is on from its colour tag."""

    def route(call):
        t = call.user_text
        if "magnified region" in t:
            win = window_of(call)
            variant = "functional" if INSTR["functional"] in t else "visual"
            if variant in empties:
                return "[]"
            return json.dumps([{"bbox_2d": px_to_1000(crop_target, win), "label": "Slideshow..."}])
        if "red rectangle" in t:
            return verify_reply
        if "ZOOMED CROP" in t:
            return rewrite_reply
        raise AssertionError("unexpected call: " + t[:80])
    return route


def test_refine_votes_and_reverifies(frame):
    gen = FakeGenerator(router=make_router(frame, empties={"visual"}))
    r = run(s5.refine(gen, frame, "A", "menu", SEED, DESCRIBE, GROUND, VERIFY, CONSENSUS))
    assert r.focal_px == [432, 189] and len(r.windows) == 3 and len(r.crops) == 6
    assert sum(c["empty"] for c in r.crops) == 3
    assert set(r.pool) == {"crop_0.5_functional", "crop_0.35_functional", "crop_0.25_functional",
                           "verifier", "describe", "functional", "visual"}
    assert tuple(r.final_bbox_px) == D
    assert {"describe", "verifier"} <= set(r.cluster) and "functional" not in r.cluster
    assert r.reverify["verified"] and r.outcome == "fixed" and r.rewrite is None and r.instructions_changed == []
    assert r.calls == 7
    crop_calls = [c for c in gen.calls if "magnified region" in c.user_text]
    assert len(crop_calls) == 6 and all(not c.thinking for c in crop_calls)
    assert 'A previous attempt boxed "Thumbnails" instead, which is wrong.' in crop_calls[0].user_text
    assert all(c["window"] in r.windows for c in r.crops)


def test_previous_sentence_variants():
    assert s5.previous_sentence(None) == "A previous attempt boxed a different control, which is wrong."
    assert s5.previous_sentence({"verdict": None, "error": "x"}) == "A previous attempt boxed a different control, which is wrong."
    assert s5.previous_sentence({"verdict": "loose", "text_in_box": "x"}) == "A previous attempt drew a box that was too loose around it."
    assert s5.previous_sentence({"verdict": "partial", "text_in_box": "x"}) == "A previous attempt drew a box that was too loose around it."
    assert s5.previous_sentence({"verdict": "wrong_element", "text_in_box": "Open"}) == 'A previous attempt boxed "Open" instead, which is wrong.'
    assert s5.previous_sentence({"verdict": "wrong_element", "text_in_box": ""}) == "A previous attempt boxed a different control, which is wrong."
    assert s5.previous_sentence({"verdict": "correct", "text_in_box": "Open"}) == 'A previous attempt boxed "Open" instead, which is wrong.'


def test_all_crops_empty_and_no_other_votes_gives_no_box(frame):
    gen = FakeGenerator(router=make_router(frame, empties={"functional", "visual"}))
    r = run(s5.refine(gen, frame, "A", "menu", SEED, dict(DESCRIBE, box_sane="too_large"), {}, None, None))
    assert r.final_bbox_px is None and r.outcome == "no_box" and r.pool == {} and r.calls == 6 and r.reverify is None


def test_vote_tie_prefers_verifier_then_smaller_box():
    pool = {"a": (0, 0, 100, 100), "b": (2, 2, 102, 102), "verifier": (500, 500, 600, 600), "c": (502, 502, 602, 602)}
    box, cluster = s5.vote(pool)
    assert "verifier" in cluster and box == (500, 500, 600, 600)
    pool2 = {"a": (0, 0, 100, 100), "b": (2, 2, 102, 102), "c": (500, 500, 550, 550), "d": (502, 502, 552, 552)}
    assert s5.vote(pool2)[1] == ["c", "d"]


def test_rewrite_when_readback_label_differs(frame):
    verify_reply = json.dumps({"text_in_box": "Slide Show", "element_in_box": "row", "verdict": "correct", "reason": "r", "corrected_bbox": None})
    rewrite_reply = json.dumps({"element_present": True, "bbox": [300, 400, 700, 600], "text_on_screen": "Slide Show", "icon": "",
                                "neighbours": {"above": "Thumbnails", "below": "", "left": "", "right": ""}, "position": "8th row",
                                "instructions": dict(INSTR, visual="Click the menu item labelled 'Slide Show'.")})
    gen = FakeGenerator(router=make_router(frame, verify_reply=verify_reply, rewrite_reply=rewrite_reply))
    r = run(s5.refine(gen, frame, "A", "menu", SEED, DESCRIBE, GROUND, VERIFY, CONSENSUS))
    assert r.reverify["verified"] and r.rewrite is not None and r.instructions_changed == ["visual"] and r.calls == 8
    assert r.rewrite["instructions"]["visual"] == "Click the menu item labelled 'Slide Show'."
    rw_call = [c for c in gen.calls if "ZOOMED CROP" in c.user_text][0]
    assert "text_on_screen: Slide Show" in rw_call.user_text and rw_call.thinking


def test_outcome_unchanged_worsened_moved(frame):
    wrong = json.dumps({"text_in_box": "Thumbnails", "element_in_box": "row", "verdict": "wrong_element", "reason": "r", "corrected_bbox": None})
    partial = json.dumps({"text_in_box": "Slideshow...", "element_in_box": "row", "verdict": "partial", "reason": "r", "corrected_bbox": None})
    no_corr = dict(VERIFY, corrected_bbox_px=None)
    # crops replay the rejected consensus box and the judge still says no -> unchanged
    r = run(s5.refine(FakeGenerator(router=make_router(frame, crop_target=OFF, verify_reply=wrong)),
                      frame, "A", "menu", SEED, DESCRIBE, GROUND, no_corr, CONSENSUS))
    assert tuple(r.final_bbox_px) == OFF and r.outcome == "unchanged"
    # box moved and the verdict got worse (loose -> wrong_element) -> worsened
    r = run(s5.refine(FakeGenerator(router=make_router(frame, verify_reply=wrong)),
                      frame, "A", "menu", SEED, DESCRIBE, GROUND, dict(no_corr, verdict="loose"), CONSENSUS))
    assert tuple(r.final_bbox_px) == D and r.outcome == "worsened"
    # box moved, verdict not worse (wrong_element -> partial) -> moved
    r = run(s5.refine(FakeGenerator(router=make_router(frame, verify_reply=partial)),
                      frame, "A", "menu", SEED, DESCRIBE, GROUND, no_corr, CONSENSUS))
    assert r.outcome == "moved"


def test_mode_reaches_every_refine_call(frame):
    seed = CandidateSeed(1, "Slideshow...", "menu item", "Slideshow...", "", "highlighted", [150, 150, 300, 200], TARGET_PX)
    describe = {"bbox_px": list(TARGET_PX), "box_sane": None, "instructions": INSTR, "text_on_screen": "Slideshow...", "icon": ""}
    ground = {v: {"bbox_px": list(TARGET_PX), "box_sane": None} for v in ("functional", "visual", "spatial")}

    def route(call):
        if REFINE_MARK in call.user_text:
            return json.dumps({"boxes": []})
        if VERIFY_MARK in call.user_text:
            return json.dumps({"text_in_box": "Slideshow...", "element_in_box": "row", "verdict": "correct", "reason": "r", "corrected_bbox": None})
        raise AssertionError(call.user_text[:60])
    gen = FakeGenerator(router=route)
    r = run(s5.refine(gen, frame, "App", "menu", seed, describe, ground, None, None, mode="pixels"))
    kinds = [c.kind for c in gen.calls]
    assert kinds.count("refine_ground") == 6 and kinds.count("verify") == 1 and r.outcome == "fixed"
    assert r.final_bbox_px == list(TARGET_PX)
    assert all("0-1000" not in c.user_text and "pixel coordinates (" in c.user_text for c in gen.calls)
