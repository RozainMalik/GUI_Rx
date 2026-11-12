"""ScreenSpot-Pro accuracy (click point inside the target box) for a trained GUI-AIMA Qwen3.5 checkpoint.

--ckpt is either a full checkpoint, or a LoRA adapter directory together with --base.
Data layout: annotations/all.json + images/ written by scripts/fetch_screenspot_pro.py (bbox = [x1, y1, x2, y2] px).
Settings follow upstream eval/screenSpot_pro.py and eval/screenSpot_pro_two_stage.py:
  1-step: image resized to --resize_to_pixels (3200x1800 area), --max_pixels 14777616.
  2-step: original image at --max_pixels 5760000, then a square crop of sspro_crop_px(W, H) zoomed 3x.
Reports top-1 hit accuracy per group x ui_type, per group, per ui_type and overall (the paper's number).
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from PIL import Image

from gui_aima.qwen35.inference import ground, ground_two_step, load_pointer_model, sspro_crop_px

GROUPS = ["Dev", "Creative", "CAD", "Scientific", "Office", "OS"]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base", default=None, help="base model when --ckpt is a LoRA adapter")
ap.add_argument("--data", default="data/screenspot_pro")
ap.add_argument("--two_step", action="store_true")
ap.add_argument("--max_pixels", type=int, default=None, help="default: 14777616 (1-step) / 5760000 (2-step)")
ap.add_argument("--resize_to_pixels", type=int, default=3200 * 1800, help="1-step only; <=0 keeps the original size")
ap.add_argument("--zoom", type=float, default=3.0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="runs/eval")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
max_pixels = a.max_pixels or (5760000 if a.two_step else 14777616)
model, proc = load_pointer_model(a.ckpt, a.base, a.device)

rows = json.load(open(os.path.join(a.data, "annotations", "all.json"), encoding="utf-8"))
rows = rows[: a.limit] if a.limit else rows
hit, tot = defaultdict(int), defaultdict(int)
preds = []
for i, r in enumerate(rows):
    img = Image.open(os.path.join(a.data, "images", r["img_filename"])).convert("RGB")
    W, H = img.size
    if a.two_step:
        x, y = ground_two_step(model, proc, img, r["instruction"], max_pixels,
                               crop_px=sspro_crop_px(W, H), zoom=a.zoom)["point"]
    else:
        if a.resize_to_pixels > 0 and W * H != a.resize_to_pixels:
            s = (a.resize_to_pixels / (W * H)) ** 0.5
            img = img.resize((int(W * s), int(H * s)))
        x, y = ground(model, proc, img, r["instruction"], max_pixels)["point"]
    x1, y1, x2, y2 = r["bbox"]
    ok = x1 <= x * W <= x2 and y1 <= y * H <= y2
    for key in (f'{r["group"]}-{r["ui_type"]}', f'{r["group"]}-avg', f'All-{r["ui_type"]}', "All-avg"):
        hit[key] += ok
        tot[key] += 1
    preds.append({"id": r["id"], "point": [x, y], "hit": bool(ok)})
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(rows)} running acc {hit['All-avg'] / tot['All-avg']:.4f}", flush=True)

order = [f"{g}-{u}" for g in GROUPS for u in ("text", "icon", "avg")] + ["All-text", "All-icon", "All-avg"]
res = {k: hit[k] / tot[k] for k in order if tot[k]}
res["n"] = tot["All-avg"]
os.makedirs(a.out, exist_ok=True)
name = "sspro_" + os.path.basename(a.ckpt.rstrip("/")) + ("_2step" if a.two_step else "")
json.dump(res, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
json.dump(preds, open(os.path.join(a.out, name + "_preds.json"), "w"))
print(json.dumps(res, indent=1))
