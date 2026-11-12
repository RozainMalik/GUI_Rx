"""Crops and the drawn verify rectangle. A crop is a window in frame px plus an upscale
factor; the model's 0-1000 maps back through the window alone."""
from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw

from .geometry import Box, CoordSpace, clamp_box, size

LONG_SIDE_TARGET = 1024
SCALE_MIN, SCALE_MAX = 1.0, 4.0
EDGE_TOLERANCE_PX = 2
RED = (255, 0, 0)
FOCUS_FRACTIONS = (0.5, 0.35, 0.25)


@dataclass(frozen=True)
class Crop:
    window: Box          # frame px the image covers
    scale: float         # LANCZOS upscale applied after cropping
    image: Image.Image

    @property
    def shown_size(self) -> tuple[int, int]:
        return self.image.size

    def space(self, mode="thousand") -> CoordSpace:
        return CoordSpace(self.window, self.shown_size, mode)


def pad_window(box, pad_x, pad_y, width, height) -> Box:
    """Grow a box by pad_x / pad_y on each side; floor the near edges, ceil the far ones, clamp."""
    x1, y1, x2, y2 = box
    return clamp_box((math.floor(x1 - pad_x), math.floor(y1 - pad_y),
                      math.ceil(x2 + pad_x), math.ceil(y2 + pad_y)), width, height)


def region_window(region_box, width, height, pad_fraction=0.05) -> Box:
    """S2 crop: the focus region padded by a fraction of its own size."""
    w, h = size(region_box)
    return pad_window(region_box, w * pad_fraction, h * pad_fraction, width, height)


def verify_window(box, width, height) -> Box:
    """S4 zoom: the box padded by max(w, 200) horizontally and max(2h, 120) vertically."""
    w, h = size(box)
    return pad_window(box, max(w, 200), max(2 * h, 120), width, height)


def focus_windows(focal, width, height, fractions=FOCUS_FRACTIONS) -> list[Box]:
    """S5 RegionFocus crops: each a fraction of the frame width at the frame's aspect ratio,
    centred on the focal point and slid - never shrunk - to stay inside the frame."""
    fx, fy = focal
    out = []
    for f in fractions:
        cw, ch = round(width * f), round(height * f)
        x1 = min(max(round(fx - cw / 2), 0), width - cw)
        y1 = min(max(round(fy - ch / 2), 0), height - ch)
        out.append((x1, y1, x1 + cw, y1 + ch))
    return out


def upscale_factor(window) -> float:
    w, h = size(window)
    return min(max(LONG_SIDE_TARGET / max(w, h), SCALE_MIN), SCALE_MAX)


def make_crop(frame, window) -> Crop:
    s = upscale_factor(window)
    w, h = size(window)
    img = frame.crop(window)
    if s != 1.0:
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return Crop(window=window, scale=s, image=img)


def box_to_crop(box_px, crop) -> Box:
    """Frame px -> pixel coordinates inside crop.image (for drawing on the zoom)."""
    wx1, wy1 = crop.window[0], crop.window[1]
    x1, y1, x2, y2 = box_px
    return (round((x1 - wx1) * crop.scale), round((y1 - wy1) * crop.scale),
            round((x2 - wx1) * crop.scale), round((y2 - wy1) * crop.scale))


def draw_box(image, box_px, width=3, color=RED) -> Image.Image:
    """The verify rectangle: pure red, 3 px, on a copy. x2/y2 exclusive."""
    out = image.copy()
    x1, y1, x2, y2 = box_px
    ImageDraw.Draw(out).rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=width)
    return out


def touches_edge(box_px, window, width, height, tol=EDGE_TOLERANCE_PX) -> bool:
    """True when the box lies within `tol` px of an *interior* edge of its crop window - a
    sign the control was cut off by the crop. An edge that is also the frame's edge does not
    count: a control at the screen border is not truncated."""
    x1, y1, x2, y2 = box_px
    wx1, wy1, wx2, wy2 = window
    return ((wx1 > 0 and x1 - wx1 <= tol) or (wy1 > 0 and y1 - wy1 <= tol)
            or (wx2 < width and wx2 - x2 <= tol) or (wy2 < height and wy2 - y2 <= tol))
