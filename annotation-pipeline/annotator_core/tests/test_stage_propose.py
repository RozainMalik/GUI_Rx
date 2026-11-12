import json

from annotator.geometry import full_space
from annotator.prompts import COORDS_THOUSAND, SYSTEM_ANNOTATOR, render
from annotator.stages import propose as s1
from tests.conftest import FakeGenerator, run

CAND = {"rank": 1, "element_name": "Slideshow...", "element_kind": "menu item", "text_on_screen": "Slideshow...",
        "icon": "", "state": "highlighted", "bbox": [150, 150, 300, 200]}


def reply(region=(100, 100, 500, 400), cands=None, kind="menu"):
    cands = [CAND] if cands is None else cands
    return json.dumps({"intent": "open the File menu",
                       "focus_region": {"bbox": list(region), "kind": kind, "why": "open menu"},
                       "candidates": cands})


def test_valid_reply_maps_boxes_to_px(frame):
    gen = FakeGenerator([reply()])
    p = run(s1.propose(gen, frame, "IrfanView"))
    assert p.rejected_reason is None and p.calls == 1 and p.intent == "open the File menu"
    assert p.focus_region == {"bbox_1000": [100, 100, 500, 400], "bbox_px": [192, 108, 960, 432], "kind": "menu", "why": "open menu"}
    seed = p.candidates[0]
    assert seed.rank == 1 and seed.element_kind == "menu item" and seed.state == "highlighted"
    assert seed.bbox_1000 == [150, 150, 300, 200] and seed.bbox_px == (288, 162, 576, 216)
    call = gen.calls[0]
    assert call.thinking and call.system == render(SYSTEM_ANNOTATOR, coords=COORDS_THOUSAND)
    assert "Application: IrfanView" in call.user_text and "toolbar button, menu item" in call.user_text
    assert call.images == (frame,)


def test_region_covering_screen_is_repaired_then_rejected(frame):
    gen = FakeGenerator([reply(region=(0, 0, 1000, 1000)), reply(region=(0, 0, 900, 900))])
    p = run(s1.propose(gen, frame, "A"))
    assert p.rejected_reason and "60%" in p.rejected_reason
    assert p.calls == 2 and p.candidates == [] and p.focus_region is None
    assert "60%" in gen.calls[1].history[1][1]


def test_empty_candidates_is_a_valid_abstain(frame):
    p = run(s1.propose(FakeGenerator([reply(cands=[])]), frame, "A"))
    assert p.rejected_reason == "no_candidates" and p.focus_region is not None and p.calls == 1


def test_enum_miss_is_repaired(frame):
    bad = json.loads(reply())
    bad["candidates"][0]["element_kind"] = "widget"
    gen = FakeGenerator([json.dumps(bad), reply()])
    p = run(s1.propose(gen, frame, "A"))
    assert p.rejected_reason is None and p.calls == 2
    assert "element_kind" in gen.calls[1].history[1][1]


def test_candidate_outside_region_fails(frame):
    outside = [dict(CAND, bbox=[600, 600, 700, 650])]
    gen = FakeGenerator([reply(cands=outside), reply(cands=outside)])
    p = run(s1.propose(gen, frame, "A"))
    assert "outside the focus region" in p.rejected_reason


def test_more_than_three_candidates_fails(frame):
    four = [dict(CAND, rank=i) for i in range(1, 5)]
    gen = FakeGenerator([reply(cands=four), reply(cands=four)])
    p = run(s1.propose(gen, frame, "A"))
    assert "three" in p.rejected_reason


def test_ranks_follow_reply_order_and_missing_text_fields_default(frame):
    two = [dict(CAND, rank=7), {"element_name": "OK", "element_kind": "toolbar button", "state": "plain", "bbox": [160, 160, 200, 180]}]
    p = run(s1.propose(FakeGenerator([reply(cands=two)]), frame, "A"))
    assert [c.rank for c in p.candidates] == [1, 2]
    assert p.candidates[1].text_on_screen == "" and p.candidates[1].icon == ""


def test_pixels_mode_reads_boxes_as_pixels_of_the_frame(frame):
    reply = json.dumps({"intent": "i", "focus_region": {"bbox": [192, 108, 960, 432], "kind": "menu", "why": "w"},
                        "candidates": [{"rank": 1, "element_name": "Slideshow...", "element_kind": "menu item",
                                        "text_on_screen": "Slideshow...", "icon": "", "state": "plain", "bbox": [288, 162, 576, 216]}]})
    gen = FakeGenerator([reply])
    p = run(s1.propose(gen, frame, "App", mode="pixels"))
    assert p.focus_region["bbox_px"] == [192, 108, 960, 432] and p.focus_region["bbox_1000"] == [192, 108, 960, 432]
    assert p.candidates[0].bbox_px == (288, 162, 576, 216)
    call = gen.calls[0]
    assert call.kind == "propose" and "in pixel coordinates (1920x1080 pixels) of the width" in call.system
    assert "on a 0-1000 scale of the width" in s1.build_call(frame, "App").system


def test_pixels_mode_rejects_a_box_beyond_the_frame(frame):
    bad = json.dumps({"intent": "i", "focus_region": {"bbox": [0, 0, 2000, 400], "kind": "menu", "why": "w"}, "candidates": []})
    gen = FakeGenerator([bad, bad])
    p = run(s1.propose(gen, frame, "App", mode="pixels"))
    assert p.rejected_reason and "0-1920 across" in gen.calls[1].history[1][1]
