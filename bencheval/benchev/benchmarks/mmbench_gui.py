"""MMBench-GUI L2 element grounding (HF OpenGVLab/MMBench-GUI): L2_annotations.json, bbox = normalised [x1, y1, x2, y2];
images from MMBench-GUI-OfflineImages.zip unpacked anywhere under images/ (resolved by basename).
group = grounding_type (basic / advanced), app = platform."""
import json
from pathlib import Path

from benchev.types import Sample


def load(root):
    root = Path(root)
    rows = json.load(open(root / "L2_annotations.json", encoding="utf-8"))
    imgs = {p.name: p for p in (root / "images").rglob("*") if p.is_file()}
    out = []
    for e in rows:
        w, h = e["image_size"]
        path = imgs.get(e["image_path"], root / "images" / e["image_path"])
        out.append(Sample(str(e["index"]), "mmbench_l2", str(path), e["instruction"], "box",
                          tuple(float(v) for v in e["bbox"]), (w, h),
                          {"group": e.get("grounding_type"), "ui_type": e.get("data_type"),
                           "app": e.get("platform"), "application": e.get("app_name")}))
    return out
