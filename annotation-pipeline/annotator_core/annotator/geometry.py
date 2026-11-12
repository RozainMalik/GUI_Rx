"""Boxes: 0-1000 <-> pixels through a crop window, agreement by centre containment,
per-coordinate medians, MedSPOT percent xywh. Pure functions, no I/O."""
from __future__ import annotations

from dataclasses import dataclass

Box = tuple[int, int, int, int]      # x1, y1, x2, y2 in frame pixels; x2/y2 exclusive
SCALE = 1000
COORD_MODES = ("thousand", "pixels")


def is_degenerate(box) -> bool:
    x1, y1, x2, y2 = box
    return x2 <= x1 or y2 <= y1


def area(box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def size(box) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return x2 - x1, y2 - y1


def center(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def contains_point(box, pt) -> bool:
    x1, y1, x2, y2 = box
    x, y = pt
    return x1 <= x < x2 and y1 <= y < y2


def inside(inner, outer, slack=0) -> bool:
    return (inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack
            and inner[2] <= outer[2] + slack and inner[3] <= outer[3] + slack)


def intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def iou(a, b) -> float:
    """Reporting only - never a decision."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def agree(a, b) -> bool:
    """Two boxes agree when each one's centre lies inside the other."""
    return contains_point(a, center(b)) and contains_point(b, center(a))


def clamp_box(box, width, height) -> Box:
    x1, y1, x2, y2 = box
    return (min(max(int(x1), 0), width), min(max(int(y1), 0), height),
            min(max(int(x2), 0), width), min(max(int(y2), 0), height))


def box_1000_to_px(box_1000, window) -> Box:
    """A 0-1000 box on the image the model saw (= `window` of the frame) -> frame pixels.
    The upscale factor never enters: 0-1000 is resolution independent."""
    wx1, wy1, wx2, wy2 = window
    ww, wh = wx2 - wx1, wy2 - wy1
    x1, y1, x2, y2 = box_1000
    return (wx1 + round(x1 / SCALE * ww), wy1 + round(y1 / SCALE * wh),
            wx1 + round(x2 / SCALE * ww), wy1 + round(y2 / SCALE * wh))


def px_to_1000(box_px, window) -> list[int]:
    wx1, wy1, wx2, wy2 = window
    ww, wh = wx2 - wx1, wy2 - wy1
    x1, y1, x2, y2 = box_px
    return [round((x1 - wx1) / ww * SCALE), round((y1 - wy1) / wh * SCALE),
            round((x2 - wx1) / ww * SCALE), round((y2 - wy1) / wh * SCALE)]


@dataclass(frozen=True)
class CoordSpace:
    """The space a model's box is written in. `window` is the frame region the sent image
    showed, `shown_size` that image's pixel size. `thousand` is Qwen's 0-1000 scale of the
    image; `pixels` is OpenAI's convention, pixels of the image as sent. Either way frame
    pixels come back through the window alone - the upscale factor never enters."""
    window: tuple
    shown_size: tuple
    mode: str = "thousand"

    def __post_init__(self):
        if self.mode not in COORD_MODES:
            raise ValueError(f"coord mode must be one of {COORD_MODES}, not {self.mode!r}")

    def bounds(self) -> tuple[int, int]:
        """(width, height) a valid box must lie within - what validators.check_box is told."""
        if self.mode == "thousand":
            return SCALE, SCALE
        return int(self.shown_size[0]), int(self.shown_size[1])

    def to_px(self, box) -> Box:
        wx1, wy1, wx2, wy2 = self.window
        ww, wh = wx2 - wx1, wy2 - wy1
        bw, bh = self.bounds()
        x1, y1, x2, y2 = box
        return (wx1 + round(x1 / bw * ww), wy1 + round(y1 / bh * wh),
                wx1 + round(x2 / bw * ww), wy1 + round(y2 / bh * wh))

    def from_px(self, box_px) -> list[int]:
        wx1, wy1, wx2, wy2 = self.window
        ww, wh = wx2 - wx1, wy2 - wy1
        bw, bh = self.bounds()
        x1, y1, x2, y2 = box_px
        return [round((x1 - wx1) / ww * bw), round((y1 - wy1) / wh * bh),
                round((x2 - wx1) / ww * bw), round((y2 - wy1) / wh * bh)]


def full_space(width, height, mode="thousand") -> CoordSpace:
    """The whole frame sent at native resolution: window (0, 0, W, H), shown size (W, H)."""
    return CoordSpace((0, 0, width, height), (width, height), mode)


def median_box(boxes) -> Box:
    """Per-coordinate median. An even count takes the lower middle value, so the result is
    always an integer box and ties resolve toward the smaller coordinate deterministically."""
    boxes = list(boxes)
    if not boxes:
        raise ValueError("median of no boxes")
    out = []
    for i in range(4):
        vals = sorted(b[i] for b in boxes)
        out.append(int(vals[(len(vals) - 1) // 2]))
    return tuple(out)


def clusters(boxes: dict[str, Box]) -> list[list[str]]:
    """Connected components of the agreement graph, largest first. Components of equal
    size keep discovery order; members keep the dict's insertion order."""
    keys = list(boxes)
    seen, comps = set(), []
    for k in keys:
        if k in seen:
            continue
        comp, stack = [], [k]
        seen.add(k)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for other in keys:
                if other not in seen and agree(boxes[cur], boxes[other]):
                    seen.add(other)
                    stack.append(other)
        comps.append(sorted(comp, key=keys.index))
    comps.sort(key=lambda c: -len(c))        # sort is stable
    return comps


def px_to_pct_xywh(box, width, height) -> list[float]:
    """MedSPOT percent [x, y, w, h] on 0-100, two decimals - the inverse of the benchmark's
    percent_bbox_xywh_to_pixel_xyxy."""
    x1, y1, x2, y2 = box
    return [round(x1 / width * 100, 2), round(y1 / height * 100, 2),
            round((x2 - x1) / width * 100, 2), round((y2 - y1) / height * 100, 2)]


def pct_xywh_to_px_xyxy(b, width, height) -> list[float]:
    """Verbatim copy of the MedSPOT evaluation scripts' percent_bbox_xywh_to_pixel_xyxy. Used by calibrate and by tests."""
    x, y, w, h = b
    return [x / 100.0 * width, y / 100.0 * height, (x + w) / 100.0 * width, (y + h) / 100.0 * height]
