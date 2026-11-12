"""Shared test doubles. No GPU, no network: the model is FakeGenerator."""
import asyncio

import pytest
from PIL import Image, ImageDraw

FRAME_W, FRAME_H = 1920, 1080


def make_frame(width=FRAME_W, height=FRAME_H):
    """A synthetic screenshot: grey chrome, a white 'menu' panel at (200,100)-(900,500)
    with dark 'rows' every 40 px, so crops and drawn boxes are visibly distinct."""
    img = Image.new("RGB", (width, height), (60, 60, 64))
    d = ImageDraw.Draw(img)
    d.rectangle((200, 100, 900, 500), fill=(245, 245, 245))
    for y in range(120, 500, 40):
        d.rectangle((210, y, 890, y + 24), fill=(200, 200, 210))
    return img


@pytest.fixture
def frame():
    return make_frame()


class FakeGenerator:
    """Scripted model. Either a list of replies consumed in call order, or a `router`
    callable(call) -> reply for tests whose calls run concurrently. A reply may be a str
    (becomes Reply(content=str)) or a Reply. Every Call received is kept in `.calls`."""

    def __init__(self, replies=None, router=None, prompt_tokens=2100):
        self.replies = list(replies or [])
        self.router = router
        self.calls = []
        self.prompt_tokens = prompt_tokens

    async def generate(self, call):
        from annotator.calling import Reply
        self.calls.append(call)
        if self.router is not None:
            reply = self.router(call)
        else:
            if not self.replies:
                raise AssertionError(f"FakeGenerator ran out of replies; last call:\n{call.user_text[:200]}")
            reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            reply = Reply(content=reply, prompt_tokens=self.prompt_tokens)
        return reply

    async def aclose(self):
        pass


def run(coro):
    """Synchronous entry for async code under test (no pytest-asyncio)."""
    return asyncio.run(coro)


import json
from pathlib import Path

INSTR = {"functional": "Start a slideshow of the loaded images.",
         "visual": "Click the menu item labelled 'Slideshow...'.",
         "spatial": "Click the eighth row of the open File menu, just below 'Thumbnails'."}
INSTR2 = {"functional": "Open the thumbnails window for the folder.",
          "visual": "Click the menu item labelled 'Thumbnails'.",
          "spatial": "Click the seventh row of the open File menu, just above 'Slideshow...'."}
REGION_1000, REGION_PX = [100, 100, 500, 400], (192, 108, 960, 432)
CAND_1000, TARGET_PX = [150, 150, 300, 200], (288, 162, 576, 216)
CAND2_1000, TARGET2_PX = [150, 100, 300, 140], (288, 108, 576, 151)

PROPOSE_MARK, DESCRIBE_MARK, GROUND_MARK, VERIFY_MARK, REFINE_MARK = (
    "Step 1 - Intent", "ZOOMED CROP", "Locate the element in this screenshot", "red rectangle", "magnified region")


def happy_router(two=False, mode="thousand"):
    """A router that walks one frame through the whole pipeline to tier A: one candidate
    ('Slideshow...'), every blind grounding agreeing, the judge saying correct. With two=True a
    second candidate ('Thumbnails') is proposed whose judge verdict is wrong_element.
    `mode` is the coordinate space the scripted model answers in."""
    from annotator.geometry import full_space
    from annotator.imaging import make_crop, region_window
    window = region_window(REGION_PX, FRAME_W, FRAME_H, 0.05)
    full = full_space(FRAME_W, FRAME_H, mode)
    crop_space = make_crop(make_frame(), window).space(mode)
    c1 = {"rank": 1, "element_name": "Slideshow...", "element_kind": "menu item", "text_on_screen": "Slideshow...",
          "icon": "", "state": "highlighted", "bbox": full.from_px(TARGET_PX)}
    c2 = {"rank": 2, "element_name": "Thumbnails", "element_kind": "menu item", "text_on_screen": "Thumbnails",
          "icon": "", "state": "plain", "bbox": full.from_px(TARGET2_PX)}

    def route(call):
        t = call.user_text
        second = "Thumbnails" in t.split("\n", 8)[7] if DESCRIBE_MARK in t else INSTR2["visual"] in t or INSTR2["functional"] in t
        if PROPOSE_MARK in t:
            return json.dumps({"intent": "open the File menu", "focus_region": {"bbox": full.from_px(REGION_PX), "kind": "menu", "why": "w"},
                               "candidates": [c1, c2] if two else [c1]})
        if DESCRIBE_MARK in t:
            box, label, instr = (TARGET2_PX, "Thumbnails", INSTR2) if second else (TARGET_PX, "Slideshow...", INSTR)
            return json.dumps({"element_present": True, "bbox": crop_space.from_px(box), "text_on_screen": label, "icon": "",
                               "neighbours": {"above": "Thumbnails" if not second else "Open...", "below": "Batch", "left": "", "right": ""},
                               "position": "8th of 24 rows", "instructions": instr})
        if GROUND_MARK in t:
            return json.dumps([{"bbox_2d": full.from_px(TARGET2_PX if second else TARGET_PX), "label": "x"}])
        if VERIFY_MARK in t:
            if second:
                return json.dumps({"text_in_box": "Open...", "element_in_box": "row", "verdict": "wrong_element", "reason": "r", "corrected_bbox": None})
            return json.dumps({"text_in_box": "Slideshow...", "element_in_box": "row", "verdict": "correct", "reason": "r", "corrected_bbox": None})
        if REFINE_MARK in t:
            return "[]"
        raise AssertionError("unexpected call: " + t[:80])
    return route


def make_data(tmp_path, n=2, app="IrfanView"):
    root = Path(tmp_path) / "data"
    folder = root / app
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        make_frame().save(folder / f"{app}_{i:03d}.png")
    return root
