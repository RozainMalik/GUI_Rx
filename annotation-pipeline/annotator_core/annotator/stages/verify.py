"""S4 - verify: the RegionFocus judge, hardened. Two images - the full frame with
the proposed box drawn, and a zoom of its neighbourhood with the same box - and a read-back of
the text inside the box that is checked in code. The prompt calls the box a proposal, never an
answer. Thinking OFF."""
from __future__ import annotations

from dataclasses import dataclass

from ..calling import Call, call_validated
from ..contract import parse_object
from ..geometry import CoordSpace
from ..imaging import box_to_crop, draw_box, make_crop, verify_window
from ..prompts import SYSTEM_PLAIN, VERIFY_USER, coords_clause, render
from ..validators import ValidationError, box_sanity, check_box, check_enum, readback_match

VERDICTS = ("correct", "loose", "partial", "wrong_element")
REQUIRED_KEYS = ("text_in_box", "element_in_box", "verdict", "reason")


@dataclass
class Verification:
    window: list                     # image 2's window in frame px
    shown_size: list
    text_in_box: str = ""
    element_in_box: str = ""
    verdict: str | None = None
    reason: str = ""
    readback_match: bool | None = None
    corrected_bbox_1000: list | None = None
    corrected_bbox_px: list | None = None
    corrected_box_sane: str | None = None
    raw: str = ""
    calls: int = 0
    verified: bool = False
    error: str | None = None


def interpret(raw, space: CoordSpace) -> dict:
    obj = parse_object(raw)
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValidationError(f"missing keys: {', '.join(missing)}; return all five keys in order")
    verdict = check_enum(obj["verdict"], VERDICTS, "verdict")
    corrected = obj.get("corrected_bbox")
    c1000 = cpx = None
    if verdict != "correct" and corrected is not None:
        c1000 = check_box(corrected, space.bounds())
        cpx = list(space.to_px(c1000))
    return {"text_in_box": str(obj["text_in_box"] or ""), "element_in_box": str(obj["element_in_box"] or ""),
            "verdict": verdict, "reason": str(obj["reason"] or ""),
            "corrected_bbox_1000": c1000, "corrected_bbox_px": cpx}


def images_for(frame, box_px):
    """(image 1, image 2, zoom crop): the frame and its zoom, each with the rectangle drawn.
    The zoom is drawn after upscaling so the line is 3 px in both images."""
    width, height = frame.size
    full = draw_box(frame, box_px)
    crop = make_crop(frame, verify_window(box_px, width, height))
    zoom = draw_box(crop.image, box_to_crop(box_px, crop))
    return full, zoom, crop


async def verify(gen, frame, box_px, instructions, text_on_screen, mode="thousand") -> Verification:
    width, height = frame.size
    box_px = tuple(box_px)
    full, zoom, crop = images_for(frame, box_px)
    space = crop.space(mode)
    text = render(VERIFY_USER, visual=instructions["visual"], functional=instructions["functional"], coords=coords_clause(space))
    call = Call(system=SYSTEM_PLAIN, user_text=text, images=(full, zoom), thinking=False, kind="verify")
    ex = await call_validated(gen, call, lambda raw: interpret(raw, space))
    out = Verification(window=list(crop.window), shown_size=list(crop.shown_size), raw=ex.raw, calls=ex.calls)
    if ex.value is None:
        out.error = ex.error
        return out
    v = ex.value
    out.text_in_box, out.element_in_box = v["text_in_box"], v["element_in_box"]
    out.verdict, out.reason = v["verdict"], v["reason"]
    out.corrected_bbox_1000, out.corrected_bbox_px = v["corrected_bbox_1000"], v["corrected_bbox_px"]
    if out.corrected_bbox_px:
        out.corrected_box_sane = box_sanity(tuple(out.corrected_bbox_px), width, height)
    out.readback_match = readback_match(text_on_screen, out.text_in_box)
    out.verified = out.verdict == "correct" and out.readback_match in (True, None)
    return out
