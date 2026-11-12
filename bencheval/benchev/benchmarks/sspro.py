"""ScreenSpot-Pro (scripts/fetch_screenspot_pro.py layout): annotations/all.json + images/, bbox = [x1, y1, x2, y2] px."""
import json
from pathlib import Path

from benchev.benchmarks.screenspot import _size
from benchev.types import Sample


def load(root):
    root = Path(root)
    out = []
    for e in json.load(open(root / "annotations" / "all.json", encoding="utf-8")):
        p = root / "images" / e["img_filename"]
        w, h = _size(p)
        x1, y1, x2, y2 = e["bbox"]
        out.append(Sample(str(e["id"]), "sspro", str(p), e["instruction"], "box", (x1 / w, y1 / h, x2 / w, y2 / h),
                          (w, h), {"group": e["group"], "ui_type": e["ui_type"]}))
    return out
