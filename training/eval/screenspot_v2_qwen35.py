"""ScreenSpot-v2 accuracy (click point inside the target box) for a trained GUI-AIMA Qwen3.5 checkpoint.

--ckpt is either a full checkpoint, or a LoRA adapter directory together with --base.
Data layout: the subset.json + images written by scripts/fetch_screenspot_v2.py (bbox = [x, y, w, h] px).
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from PIL import Image

from gui_aima.qwen35.inference import ground, ground_two_step, load_pointer_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base", default=None, help="base model when --ckpt is a LoRA adapter")
ap.add_argument("--data", default="data/screenspot_v2")
ap.add_argument("--max_pixels", type=int, default=5860400)
ap.add_argument("--two_step", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="runs/eval")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
model, proc = load_pointer_model(a.ckpt, a.base, a.device)

rows = json.load(open(os.path.join(a.data, "subset.json"), encoding="utf-8"))
rows = rows[: a.limit] if a.limit else rows
hit, tot = defaultdict(int), defaultdict(int)
fn = ground_two_step if a.two_step else ground
for r in rows:
    img = Image.open(os.path.join(a.data, r["img_filename"])).convert("RGB")
    x, y = fn(model, proc, img, r["instruction"], a.max_pixels)["point"]
    bx, by, bw, bh = r["bbox"]
    W, H = img.size
    ok = bx <= x * W <= bx + bw and by <= y * H <= by + bh
    for key in (r["split"], f'{r["split"]}_{r.get("data_type")}', "all"):
        hit[key] += ok
        tot[key] += 1
res = {k: hit[k] / tot[k] for k in sorted(tot)}
res["n"] = tot["all"]
os.makedirs(a.out, exist_ok=True)
name = os.path.basename(a.ckpt.rstrip("/")) + ("_2step" if a.two_step else "") + ".json"
json.dump(res, open(os.path.join(a.out, name), "w"), indent=1)
print(json.dumps(res, indent=1))
