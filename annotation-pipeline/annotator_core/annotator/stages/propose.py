"""S1 - propose: intent, one focus region, up to three ranked candidates. Full frame,
thinking ON. Validation failures that survive the repair abstain the frame."""
from __future__ import annotations

from dataclasses import dataclass

from ..calling import Call, call_validated
from ..contract import parse_object
from ..geometry import CoordSpace, area, full_space, inside, size
from ..prompts import ELEMENT_KINDS, PROPOSE_USER, REGION_KINDS, STATES, SYSTEM_ANNOTATOR, coords_clause, render
from ..validators import ValidationError, check_box, check_enum, check_nonempty_str

MAX_REGION_AREA_FRACTION = 0.60
MAX_CANDIDATES = 3
REGION_SLACK_FRACTION = 0.02


@dataclass
class CandidateSeed:
    rank: int
    element_name: str
    element_kind: str
    text_on_screen: str
    icon: str
    state: str
    bbox_1000: list
    bbox_px: tuple


@dataclass
class Proposal:
    intent: str | None
    focus_region: dict | None      # bbox_1000, bbox_px, kind, why
    candidates: list               # CandidateSeed, best first
    raw: str
    thinking_chars: int
    calls: int
    thinking_fallback: bool
    rejected_reason: str | None    # set -> the frame is abstained (final)


def build_call(frame, app, mode="thousand") -> Call:
    space = full_space(*frame.size, mode)
    text = render(PROPOSE_USER, app=app, kinds=", ".join(ELEMENT_KINDS))
    system = render(SYSTEM_ANNOTATOR, coords=coords_clause(space))
    return Call(system=system, user_text=text, images=(frame,), thinking=True, kind="propose")


def interpret(raw, space: CoordSpace) -> dict:
    """`space` is the full frame's: window (0, 0, W, H), shown size (W, H)."""
    obj = parse_object(raw)
    width, height = size(space.window)
    region = obj.get("focus_region")
    if not isinstance(region, dict):
        raise ValidationError("focus_region must be an object with bbox, kind and why")
    region_1000 = check_box(region.get("bbox"), space.bounds())
    region_px = space.to_px(region_1000)
    if area(region_px) > MAX_REGION_AREA_FRACTION * width * height:
        raise ValidationError("focus_region covers more than 60% of the screen; box only the area "
                              "being worked in, never the whole screen")
    kind = check_enum(region.get("kind"), REGION_KINDS, "focus_region.kind")
    cands = obj.get("candidates")
    if not isinstance(cands, list):
        raise ValidationError("candidates must be a list (empty if none can be identified)")
    if len(cands) > MAX_CANDIDATES:
        raise ValidationError("at most three candidates, best first")
    rw, rh = size(region_px)
    slack = round(max(rw, rh) * REGION_SLACK_FRACTION)
    seeds = []
    for i, c in enumerate(cands, start=1):
        if not isinstance(c, dict):
            raise ValidationError(f"candidate {i} must be an object")
        name = check_nonempty_str(c.get("element_name"), f"candidate {i}.element_name")
        ekind = check_enum(c.get("element_kind"), ELEMENT_KINDS, f"candidate {i}.element_kind")
        state = check_enum(c.get("state"), STATES, f"candidate {i}.state")
        b1000 = check_box(c.get("bbox"), space.bounds())
        bpx = space.to_px(b1000)
        if not inside(bpx, region_px, slack):
            raise ValidationError(f"candidate {i} box lies outside the focus region; every candidate must be inside it")
        seeds.append(CandidateSeed(i, name, ekind, str(c.get("text_on_screen") or ""),
                                   str(c.get("icon") or ""), state, b1000, bpx))
    return {"intent": str(obj.get("intent") or ""),
            "focus_region": {"bbox_1000": region_1000, "bbox_px": list(region_px), "kind": kind,
                             "why": str(region.get("why") or "")},
            "candidates": seeds}


async def propose(gen, frame, app, mode="thousand") -> Proposal:
    space = full_space(*frame.size, mode)
    ex = await call_validated(gen, build_call(frame, app, mode), lambda raw: interpret(raw, space))
    if ex.value is None:
        return Proposal(None, None, [], ex.raw, ex.thinking_chars, ex.calls, ex.thinking_fallback, ex.error)
    v = ex.value
    return Proposal(v["intent"], v["focus_region"], v["candidates"], ex.raw, ex.thinking_chars, ex.calls,
                    ex.thinking_fallback, None if v["candidates"] else "no_candidates")
