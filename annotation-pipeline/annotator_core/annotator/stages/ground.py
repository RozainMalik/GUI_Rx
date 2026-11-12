"""S3 - blind grounding of one instruction on the full frame, and the informed
per-crop grounding that S5 reuses. Thinking OFF and the plain system prompt
(the best setting in calibration)."""
from __future__ import annotations

from dataclasses import dataclass

from ..calling import Call, call_validated
from ..contract import parse_array
from ..geometry import CoordSpace, full_space
from ..prompts import (GROUND_USER, REFINE_GROUND_USER, REFINE_ICON_CLAUSE, REFINE_LABEL_CLAUSE,
                       REFINE_NO_LABEL_CLAUSE, SYSTEM_PLAIN, coords_clause, render)
from ..validators import ValidationError, box_sanity, check_box


@dataclass
class Grounding:
    bbox_1000: list | None
    bbox_px: list | None
    box_sane: str | None        # None = usable; else the sanity reason (box kept for audit)
    label: str
    raw: str
    calls: int
    empty: bool                 # the crop reply was [] - "not in this image" (S5 only)
    error: str | None           # unreadable after the repair


def interpret(raw, space: CoordSpace, allow_empty) -> dict:
    arr = parse_array(raw)
    if not arr:
        if allow_empty:
            return {"empty": True}
        raise ValidationError("output exactly one object with a bbox_2d")
    if len(arr) != 1 or not isinstance(arr[0], dict):
        raise ValidationError("output exactly one object with a bbox_2d, not several")
    b1000 = check_box(arr[0].get("bbox_2d"), space.bounds())
    return {"empty": False, "bbox_1000": b1000, "bbox_px": list(space.to_px(b1000)),
            "label": str(arr[0].get("label") or "")}


def _finish(ex, width, height) -> Grounding:
    if ex.value is None:
        return Grounding(None, None, None, "", ex.raw, ex.calls, False, ex.error)
    if ex.value.get("empty"):
        return Grounding(None, None, None, "", ex.raw, ex.calls, True, None)
    v = ex.value
    return Grounding(v["bbox_1000"], v["bbox_px"], box_sanity(tuple(v["bbox_px"]), width, height),
                     v["label"], ex.raw, ex.calls, False, None)


async def ground_blind(gen, frame, instruction, mode="thousand") -> Grounding:
    """Nothing about the candidate reaches this call - no box, no label, no element name."""
    width, height = frame.size
    space = full_space(width, height, mode)
    call = Call(system=SYSTEM_PLAIN, user_text=render(GROUND_USER, instruction=instruction, coords=coords_clause(space)),
                images=(frame,), thinking=False, kind="ground")
    ex = await call_validated(gen, call, lambda raw: interpret(raw, space, allow_empty=False))
    return _finish(ex, width, height)


def target_clause(text_on_screen, icon) -> str:
    icon_clause = render(REFINE_ICON_CLAUSE, icon=icon) if icon else ""
    if text_on_screen:
        return render(REFINE_LABEL_CLAUSE, label=text_on_screen, icon_clause=icon_clause)
    return render(REFINE_NO_LABEL_CLAUSE, icon_clause=icon_clause)


async def ground_crop(gen, crop, app, instruction, text_on_screen, icon, previous_sentence, frame_size, mode="thousand") -> Grounding:
    """S5's informed re-grounding on one RegionFocus crop; [] means 'not in this crop'."""
    width, height = frame_size
    space = crop.space(mode)
    text = render(REFINE_GROUND_USER, app=app, instruction=instruction, previous=previous_sentence,
                  target=target_clause(text_on_screen, icon), coords=coords_clause(space))
    call = Call(system=SYSTEM_PLAIN, user_text=text, images=(crop.image,), thinking=False, kind="refine_ground")
    ex = await call_validated(gen, call, lambda raw: interpret(raw, space, allow_empty=True))
    return _finish(ex, width, height)
