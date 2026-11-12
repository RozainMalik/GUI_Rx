"""OSWorld-G (564): box_type bbox ([x, y, w, h] px) | polygon (flat x,y list px) | refusal ([0,0,0,0]).
category = list of every classification_result.json group the sample is listed in (text_matching, element_recognition,
layout_understanding, fine_grained_manipulation, refusal), else ["unclassified"] -- as upstream
OSWorld-G evaluation/qwen25_vllm_osworld_g_jedi.py, which counts a sample in each of its groups.
(buckets.json maps GUI types to groups but is not what the official per-group numbers use.)
raw_box is kept for upstream-compat scoring."""
import json
from pathlib import Path

from benchev.types import Sample


def load(root, refined=False):
    root = Path(root)
    data = json.load(open(root / ("OSWorld-G_refined.json" if refined else "OSWorld-G.json"), encoding="utf-8"))
    groups = {}
    for g, items in json.load(open(root / "classification_result.json", encoding="utf-8"))["classified"].items():
        for it in items:
            groups.setdefault(it["id"], []).append(g)
    out = []
    for e in data:
        w, h = e["image_size"]
        c = e["box_coordinates"]
        kind = e.get("box_type", "bbox")
        if kind == "refusal":
            gt, gk = None, "refusal"
        elif kind == "polygon":
            gt, gk = tuple((c[i] / w, c[i + 1] / h) for i in range(0, len(c), 2)), "polygon"
        else:
            x, y, bw, bh = c
            gt, gk = (x / w, y / h, (x + bw) / w, (y + bh) / h), "box"
        out.append(Sample(str(e["id"]), "oswg_refined" if refined else "oswg", str(root / "images" / e["image_path"]),
                          e["instruction"], gk, gt, (w, h),
                          {"category": groups.get(e["id"], ["unclassified"]), "raw_box": list(c)}))
    return out
