"""S2 - describe: on a zoomed crop of the focus region, box the anchored control tightly, read
its label, record its neighbours and write the three instructions. Thinking ON.
The candidate is the anchor: a reply that boxes or names a different control is a validation
error. Also runs as the S5 rewrite with the read-back label as the anchor."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..calling import Call, call_validated
from ..contract import parse_object
from ..geometry import intersects
from ..imaging import make_crop, region_window, touches_edge
from ..prompts import DESCRIBE_USER, SYSTEM_ANNOTATOR, coords_clause, render
from ..validators import (VARIANTS, SoftValidationError, ValidationError, box_sanity, check_box,
                          check_instructions, readback_match)
from .propose import CandidateSeed

PADS = (0.05, 0.10)          # the second is the 2x rebuild after an edge-truncated box
NEIGHBOUR_KEYS = ("above", "below", "left", "right")


@dataclass
class Description:
    present: bool
    window: list | None = None
    scale: float | None = None
    shown_size: list | None = None
    bbox_1000: list | None = None        # None when the S1 box was substituted (truncated)
    bbox_px: list | None = None
    box_sane: str | None = None
    text_on_screen: str = ""
    icon: str = ""
    neighbours: dict = field(default_factory=dict)
    position: str = ""
    instructions: dict | None = None
    instruction_errors: list = field(default_factory=list)
    raw: str = ""
    calls: int = 0
    thinking_chars: int = 0
    thinking_fallback: bool = False
    truncated: bool = False
    error: str | None = None             # set -> unreadable after the repair


def build_call(crop, app, region_kind, seed: CandidateSeed, anchor_text, mode="thousand") -> Call:
    text = render(DESCRIBE_USER, app=app, region_kind=region_kind, element_name=seed.element_name,
                  text_on_screen=anchor_text, icon=seed.icon)
    system = render(SYSTEM_ANNOTATOR, coords=coords_clause(crop.space(mode)))
    return Call(system=system, user_text=text, images=(crop.image,), thinking=True, kind="describe")


def interpret(raw, space, seed_box_px, anchor_text) -> dict:
    obj = parse_object(raw)
    if obj.get("element_present") is False:
        return {"present": False}
    b1000 = check_box(obj.get("bbox"), space.bounds())
    bpx = space.to_px(b1000)
    if not intersects(bpx, tuple(seed_box_px)):
        raise ValidationError("the box is not on the control you were given; box that control, do not switch to another")
    label = str(obj.get("text_on_screen") or "")
    if anchor_text and label and readback_match(anchor_text, label) is False:
        raise ValidationError(f"text_on_screen '{label}' names a different control from '{anchor_text}'; "
                              "only correct the label if it is the same control")
    raw_neighbours = obj.get("neighbours") if isinstance(obj.get("neighbours"), dict) else {}
    neighbours = {k: str(raw_neighbours.get(k) or "") for k in NEIGHBOUR_KEYS}
    instructions = obj.get("instructions")
    if not isinstance(instructions, dict) or any(
            not isinstance(instructions.get(v), str) or not instructions[v].strip() for v in VARIANTS):
        raise ValidationError("instructions must be an object with non-empty functional, visual and spatial strings")
    icon = str(obj.get("icon") or "")
    value = {"present": True, "bbox_1000": b1000, "bbox_px": list(bpx), "text_on_screen": label, "icon": icon,
             "neighbours": neighbours, "position": str(obj.get("position") or ""),
             "instructions": {v: instructions[v].strip() for v in VARIANTS}}
    errors = check_instructions(value["instructions"], label, icon, neighbours)
    if errors:
        raise SoftValidationError(value, errors)
    return value


async def describe(gen, frame, app, region_kind, region_box_px, seed: CandidateSeed, anchor_text=None, mode="thousand") -> Description:
    width, height = frame.size
    anchor = seed.text_on_screen if anchor_text is None else anchor_text
    calls = chars = 0
    fallback = False
    result = None
    for pad in PADS:
        window = region_window(region_box_px, width, height, pad)
        crop = make_crop(frame, window)
        space = crop.space(mode)
        ex = await call_validated(gen, build_call(crop, app, region_kind, seed, anchor, mode),
                                  lambda raw, space=space: interpret(raw, space, seed.bbox_px, anchor))
        calls += ex.calls
        chars += ex.thinking_chars
        fallback = fallback or ex.thinking_fallback
        common = dict(window=list(window), scale=crop.scale, shown_size=list(crop.shown_size), raw=ex.raw,
                      calls=calls, thinking_chars=chars, thinking_fallback=fallback)
        if ex.value is None:
            return Description(present=True, error=ex.error, **common)
        if not ex.value["present"]:
            return Description(present=False, **common)
        v = ex.value
        result = Description(present=True, bbox_1000=v["bbox_1000"], bbox_px=v["bbox_px"],
                             text_on_screen=v["text_on_screen"], icon=v["icon"], neighbours=v["neighbours"],
                             position=v["position"], instructions=v["instructions"],
                             instruction_errors=list(ex.soft_errors), **common)
        if not touches_edge(tuple(v["bbox_px"]), window, width, height):
            break
    else:
        # Still cut off after the 2x rebuild: keep the words, take the frame-level box from S1.
        result.bbox_px = list(seed.bbox_px)
        result.bbox_1000 = None
        result.truncated = True
    result.box_sane = box_sanity(tuple(result.bbox_px), width, height)
    return result
