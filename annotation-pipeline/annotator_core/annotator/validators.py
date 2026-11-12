"""The mechanical checks shared by S2/S4/S5. Every rule returns or
raises a plain-English message, because the message is fed back to the model as the repair."""
from __future__ import annotations

import difflib
import re
import string

from .geometry import SCALE, area, inside, is_degenerate, size

VARIANTS = ("functional", "visual", "spatial")
DESCRIPTION_OPENERS = ("a ", "an ", "the ", "this ", "that ", "it ", "located", "opens", "opening",
                       "displays", "shows", "selects", "selecting", "clicks", "clicking", "allows",
                       "used to", "button", "icon", "menu item")
FORBIDDEN_WORDS = ("cursor", "hover", "highlight", "bounding box", "screenshot", "annotation")
ORDINAL_RE = re.compile(r"\b(\d+(st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", re.I)
RELATION_WORDS = ("above", "below", "beside", "next to", "under", "left", "right", "top", "bottom",
                  "corner", "middle", "centre", "center")
MIN_WORDS, MAX_WORDS = 4, 30
READBACK_RATIO = 0.8
MAX_AREA_FRACTION = 0.25
MIN_SIDE_PX = 10


class ValidationError(ValueError):
    """The reply parsed but breaks a rule; the message is what the model is told to fix."""


class SoftValidationError(ValidationError):
    """Structurally fine, but the instruction rules failed. If it fails twice the reply is
    kept with `errors` recorded (the box may still be good; the row goes to tier C)."""

    def __init__(self, value, errors):
        super().__init__("; ".join(errors))
        self.value = value
        self.errors = list(errors)


def _ws(s) -> str:
    return " ".join((s or "").split())


def normalise_label(s) -> str:
    s = (s or "").replace("…", "...").lower().replace("...", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    return _ws(s)


def readback_match(text_on_screen, text_in_box):
    """None when there is no label to check; True on normalised substring or a
    SequenceMatcher ratio >= 0.8; else False."""
    a = normalise_label(text_on_screen)
    if not a:
        return None
    b = normalise_label(text_in_box)
    if a in b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= READBACK_RATIO


def check_instructions(instructions, text_on_screen, icon, neighbours=None) -> list[str]:
    """Every instruction rule. Returns human-readable errors; empty means valid."""
    if not isinstance(instructions, dict):
        return ["instructions is not an object"]
    errors = []
    for variant in VARIANTS:
        text = instructions.get(variant)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{variant}: missing")
            continue
        low = _ws(text).lower()
        if any(low.startswith(o) for o in DESCRIPTION_OPENERS):
            errors.append(f"{variant}: must start with a verb, not a description ('{text.split()[0]}')")
        n = len(text.split())
        if not MIN_WORDS <= n <= MAX_WORDS:
            errors.append(f"{variant}: {n} words, need {MIN_WORDS}-{MAX_WORDS}")
        for w in FORBIDDEN_WORDS:
            if w in low:
                errors.append(f"{variant}: must not mention '{w}'")
        if not (icon or "").strip() and re.search(r"\bicons?\b", low):
            errors.append(f"{variant}: mentions an icon but the control has no icon")
    label = _ws(text_on_screen).lower()
    visual = _ws(instructions.get("visual") or "").lower()
    if label and visual and label not in visual:
        errors.append(f"visual: must quote the label '{text_on_screen}' verbatim")
    spatial = _ws(instructions.get("spatial") or "").lower()
    if spatial:
        labels = [_ws(x).lower() for x in (neighbours or {}).values() if isinstance(x, str) and _ws(x)]
        has_ordinal = bool(ORDINAL_RE.search(spatial))
        has_relation = any(re.search(r"\b" + re.escape(r) + r"\b", spatial) for r in RELATION_WORDS)
        has_neighbour = any(n in spatial for n in labels)
        if not (has_ordinal or has_relation or has_neighbour):
            errors.append("spatial: needs an ordinal, a relation word (above/below/left/...), or a neighbour's label")
    return errors


def check_box(value, bounds=(SCALE, SCALE)) -> list[int]:
    """[x1, y1, x2, y2] as ints inside `bounds` = (width, height) of the model's coordinate
    space: (1000, 1000) on the 0-1000 scale, the sent image's size in pixels mode."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValidationError("bbox must be [x1, y1, x2, y2]")
    try:
        box = [int(round(float(x))) for x in value]
    except (TypeError, ValueError):
        raise ValidationError("bbox values must be numbers") from None
    bw, bh = int(bounds[0]), int(bounds[1])
    if any(x < 0 for x in box) or box[0] > bw or box[2] > bw or box[1] > bh or box[3] > bh:
        if (bw, bh) == (SCALE, SCALE):
            raise ValidationError("bbox values must be integers on a 0-1000 scale")
        raise ValidationError(f"bbox values must be pixel coordinates inside the image: 0-{bw} across, 0-{bh} down")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValidationError("bbox is degenerate (x2 <= x1 or y2 <= y1)")
    return box


def check_box_1000(value) -> list[int]:
    return check_box(value, (SCALE, SCALE))


def box_sanity(box_px, width, height):
    """None when the box is usable; else why it is rejected. Rejects the box, not
    the candidate."""
    if is_degenerate(box_px):
        return "degenerate"
    if not inside(box_px, (0, 0, width, height)):
        return "outside_frame"
    w, h = size(box_px)
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return "too_small"
    if area(box_px) > MAX_AREA_FRACTION * width * height:
        return "too_large"
    return None


def check_enum(value, allowed, field) -> str:
    key = _ws(str(value or "")).lower()
    for a in allowed:
        if key == a.lower():
            return a
    raise ValidationError(f"{field} must be exactly one of: {', '.join(allowed)}")


def check_nonempty_str(value, field) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    return value.strip()
