"""SS-v1 / SS-v2 in the port layout: subset.json + images, bbox = [x, y, w, h] px. Ids = row index + stem.
The image size is always read from the file (annotation sizes are not trusted)."""
import json
from pathlib import Path

from PIL import Image

from benchev.types import Sample


def _size(p):
    with Image.open(p) as im:
        return im.size


def load(root):
    root = Path(root)
    out = []
    for i, e in enumerate(json.load(open(root / "subset.json", encoding="utf-8"))):
        p = root / e["img_filename"]
        w, h = _size(p)
        x, y, bw, bh = e["bbox"]
        out.append(Sample(f"{i:04d}_{Path(e['img_filename']).stem}", "screenspot", str(p), e["instruction"], "box",
                          (x / w, y / h, (x + bw) / w, (y + bh) / h), (w, h),
                          {"group": e["split"], "ui_type": e.get("data_type"), "app": e.get("data_source")}))
    return out
