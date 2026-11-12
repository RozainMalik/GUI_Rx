"""UI-Vision element grounding (HF ServiceNow/ui-vision): annotations/element_grounding/element_grounding_<split>.json,
bbox = [x1, y1, x2, y2] px of image_size. group = split, app = platform."""
import json
from pathlib import Path

from benchev.types import Sample

SPLITS = ("basic", "functional", "spatial")


def load(root):
    root = Path(root)
    out = []
    for split in SPLITS:
        rows = json.load(open(root / "annotations" / "element_grounding" / f"element_grounding_{split}.json",
                              encoding="utf-8"))
        for i, e in enumerate(rows):
            w, h = e["image_size"]
            x1, y1, x2, y2 = e["bbox"]
            out.append(Sample(f"{split}_{i:04d}", "uivision", str(root / "images" / e["image_path"]),
                              e["prompt_to_evaluate"], "box", (x1 / w, y1 / h, x2 / w, y2 / h), (w, h),
                              {"group": split, "ui_type": e.get("element_type"), "app": e.get("platform"),
                               "category": e.get("category")}))
    return out
