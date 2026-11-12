import json

from annotator.geometry import px_to_1000
from annotator.imaging import make_crop, region_window
from annotator.stages import describe as s2
from annotator.stages.propose import CandidateSeed
from tests.conftest import FRAME_H, FRAME_W, FakeGenerator, run

W, H = 1920, 1080
REGION_PX = (192, 108, 960, 432)
TARGET_PX = (288, 162, 576, 216)
SEED = CandidateSeed(1, "Slideshow...", "menu item", "Slideshow...", "", "highlighted", [150, 150, 300, 200], TARGET_PX)
GOOD_INSTR = {"functional": "Start a slideshow of the loaded images.",
              "visual": "Click the menu item labelled 'Slideshow...'.",
              "spatial": "Click the eighth row of the open File menu, just below 'Thumbnails'."}


def reply(box_px=TARGET_PX, pad=0.05, **over):
    window = region_window(REGION_PX, W, H, pad)
    obj = {"element_present": True, "bbox": px_to_1000(box_px, window), "text_on_screen": "Slideshow...", "icon": "",
           "neighbours": {"above": "Thumbnails", "below": "Batch Conversion", "left": "", "right": ""},
           "position": "8th of 24 rows in the File menu", "instructions": dict(GOOD_INSTR)}
    obj.update(over)
    return json.dumps(obj)


def close(a, b):
    return all(abs(x - y) <= 1 for x, y in zip(a, b))


def test_valid_reply(frame):
    gen = FakeGenerator([reply()])
    d = run(s2.describe(gen, frame, "IrfanView", "menu", REGION_PX, SEED))
    assert d.present and d.error is None and d.calls == 1 and not d.truncated
    assert d.window == [153, 91, 999, 449] and d.shown_size[0] == 1024 and d.scale > 1.2
    assert close(d.bbox_px, TARGET_PX) and d.box_sane is None
    assert d.instructions == GOOD_INSTR and d.instruction_errors == []
    assert d.neighbours["above"] == "Thumbnails" and d.position.startswith("8th")
    call = gen.calls[0]
    assert call.thinking and call.images[0].size[0] == 1024
    assert "ZOOMED CROP" in call.user_text and "element_name:   Slideshow..." in call.user_text and "shows the menu" in call.user_text


def test_element_absent(frame):
    d = run(s2.describe(FakeGenerator(['{"element_present": false}']), frame, "A", "menu", REGION_PX, SEED))
    assert d.present is False and d.bbox_px is None and d.error is None


def test_switching_control_is_repaired_then_unreadable(frame):
    other = (700, 300, 900, 340)       # inside the region, not touching the seed
    gen = FakeGenerator([reply(box_px=other), reply(box_px=other)])
    d = run(s2.describe(gen, frame, "A", "menu", REGION_PX, SEED))
    assert d.error and "do not switch" in d.error and d.calls == 2 and d.bbox_px is None


def test_renaming_control_is_rejected_then_repaired(frame):
    gen = FakeGenerator([reply(text_on_screen="Thumbnails"), reply()])
    d = run(s2.describe(gen, frame, "A", "menu", REGION_PX, SEED))
    assert d.error is None and d.calls == 2
    assert "different control" in gen.calls[1].history[1][1]


def test_label_correction_is_allowed(frame):
    d = run(s2.describe(FakeGenerator([reply(text_on_screen="Slideshow")]), frame, "A", "menu", REGION_PX, SEED))
    assert d.error is None and d.text_on_screen == "Slideshow"


def test_instruction_rules_failing_twice_keep_reply_with_errors(frame):
    bad = dict(GOOD_INSTR, functional="The slideshow starts here now.")
    gen = FakeGenerator([reply(instructions=bad), reply(instructions=bad)])
    d = run(s2.describe(gen, frame, "A", "menu", REGION_PX, SEED))
    assert d.error is None and d.present and d.bbox_px is not None
    assert any("must start with a verb" in e for e in d.instruction_errors)
    assert "must start with a verb" in gen.calls[1].history[1][1]


def test_edge_truncation_rebuilds_once_then_uses_seed_box(frame):
    w1 = region_window(REGION_PX, W, H, 0.05)
    w2 = region_window(REGION_PX, W, H, 0.10)
    cut1 = (w1[0] + 1, 162, 576, 216)          # 1 px from the crop's left edge
    cut2 = (w2[0] + 1, 162, 576, 216)
    gen = FakeGenerator([reply(box_px=cut1, pad=0.05), reply(box_px=cut2, pad=0.10)])
    d = run(s2.describe(gen, frame, "A", "menu", REGION_PX, SEED))
    assert d.calls == 2 and d.truncated and tuple(d.bbox_px) == TARGET_PX and d.bbox_1000 is None
    assert d.window == list(w2)
    assert gen.calls[1].images[0].tobytes() != gen.calls[0].images[0].tobytes()


def test_edge_truncation_recovers_on_the_rebuilt_crop(frame):
    w1 = region_window(REGION_PX, W, H, 0.05)
    gen = FakeGenerator([reply(box_px=(w1[0] + 1, 162, 576, 216), pad=0.05), reply(pad=0.10)])
    d = run(s2.describe(gen, frame, "A", "menu", REGION_PX, SEED))
    assert d.calls == 2 and not d.truncated and close(d.bbox_px, TARGET_PX)


def test_anchor_text_overrides_seed_label_in_prompt_and_check(frame):
    visual = dict(GOOD_INSTR, visual="Click the menu item labelled 'Slide show'.")
    gen = FakeGenerator([reply(text_on_screen="Slide show", instructions=visual)])
    d = run(s2.describe(gen, frame, "A", "menu", REGION_PX, SEED, anchor_text="Slide show"))
    assert "text_on_screen: Slide show" in gen.calls[0].user_text and d.error is None


def test_pixels_mode_maps_through_the_upscaled_crop(frame):
    from annotator.stages.propose import CandidateSeed

    seed = CandidateSeed(1, "Slideshow...", "menu item", "Slideshow...", "", "highlighted", [150, 150, 300, 200], TARGET_PX)
    crop = make_crop(frame, region_window(REGION_PX, FRAME_W, FRAME_H, 0.05))
    space = crop.space("pixels")
    reply = json.dumps({"element_present": True, "bbox": space.from_px(TARGET_PX), "text_on_screen": "Slideshow...", "icon": "",
                        "neighbours": {"above": "Thumbnails", "below": "Batch", "left": "", "right": ""},
                        "position": "8th of 24 rows", "instructions": GOOD_INSTR})
    gen = FakeGenerator([reply])
    d = run(s2.describe(gen, frame, "App", "menu", REGION_PX, seed, mode="pixels"))
    assert d.present and d.error is None and d.bbox_px == list(space.to_px(space.from_px(TARGET_PX)))
    assert d.shown_size == list(crop.image.size) and d.window == list(crop.window)
    call = gen.calls[0]
    assert call.kind == "describe" and f"({crop.image.size[0]}x{crop.image.size[1]} pixels)" in call.system
