"""Smoke-test data mix: the N samples with the largest images (w*h) across a training yaml.

Peak GPU memory grows with the number of visual tokens, so the smoke run must see the biggest screenshots, not whatever the
length-grouped sampler happens to put first. Only image headers are read (PIL opens lazily).
Usage: python scripts/make_smoke_data.py --data configs/data_45k.yaml --top 160 --out data/smoke_largest
"""
import argparse
import heapq
import json
import os

import yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--top", type=int, default=160)
ap.add_argument("--out", required=True)
a = ap.parse_args()

datasets = yaml.safe_load(open(a.data))["datasets"]
cands = []  # (pixels, dataset index, row index, (w, h))
for di, d in enumerate(datasets):
    for ri, row in enumerate(json.load(open(d["json_path"]))):
        with Image.open(os.path.join(d["images_folder"], row["image"])) as im:
            w, h = im.size
        cands.append((w * h, di, ri, (w, h)))
top = heapq.nlargest(a.top, cands)

os.makedirs(a.out, exist_ok=True)
out = []
for di, d in enumerate(datasets):
    rows = json.load(open(d["json_path"]))
    picked = [rows[ri] for _, dj, ri, _ in top if dj == di]
    if picked:
        path = os.path.join(a.out, f"{di}_{os.path.basename(d['json_path'])}")
        json.dump(picked, open(path, "w"))
        out.append({"json_path": path, "images_folder": d["images_folder"], "sampling_strategy": "all"})
yaml.safe_dump({"datasets": out}, open(os.path.join(a.out, "smoke.yaml"), "w"))
w, h = top[0][3]
print(f"picked {len(top)} of {len(cands)} samples; largest {w}x{h}, smallest picked {top[-1][3][0]}x{top[-1][3][1]}")
