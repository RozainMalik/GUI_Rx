"""Geometry shared by drivers and scorers. Points/boxes are normalised to the native image unless named *_px."""
from __future__ import annotations


def point_in_box(p, b) -> bool:
    if p is None:
        return False
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def point_in_polygon(p, poly) -> bool:
    """Even-odd ray casting; points exactly on an edge may go either way (measure-zero)."""
    if p is None:
        return False
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def resize_to_area(w: int, h: int, target: int):
    """Port/upstream pre-resize: scale to `target` pixels with int() truncation; 0/None = keep."""
    if not target or w * h == target:
        return w, h
    s = (target / (w * h)) ** 0.5
    return int(w * s), int(h * s)


def area_crop_px(W: int, H: int, portion: int, ref_pixels: int, multiple: int) -> int:
    """= gui_aima.qwen35.inference.sspro_crop_px generalised to the adapter's token size."""
    return max(multiple, int((W * H * portion / ref_pixels) ** 0.5 / multiple) * multiple)


def crop_window(W: int, H: int, point, crop_px: int):
    """Identical to gui_aima.qwen35.inference.ground_two_step's window (shifted to stay inside the image)."""
    cx, cy = point[0] * W, point[1] * H
    half = crop_px / 2
    x0 = int(min(max(cx - half, 0), max(W - crop_px, 0)))
    y0 = int(min(max(cy - half, 0), max(H - crop_px, 0)))
    return x0, y0, min(x0 + crop_px, W), min(y0 + crop_px, H)
