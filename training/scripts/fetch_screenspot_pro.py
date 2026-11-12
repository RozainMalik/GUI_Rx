"""Download ScreenSpot-Pro (likaixin/ScreenSpot-Pro) annotations and the images of a (stratified) subset.

Writes <out>/annotations/all.json (bbox = [x1, y1, x2, y2] px) and <out>/images/<img_filename>, the layout
eval/screenspot_pro_qwen35.py and upstream eval/screenSpot_pro.py expect. Images are fetched one by one, so a
small subset needs little disk (the full set is 1581 samples of mostly 4K screenshots).
"""
import argparse
import json
import os
import random
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "likaixin/ScreenSpot-Pro"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/screenspot_pro")
ap.add_argument("--per_group", type=int, default=0, help="samples per group (Dev, CAD, ...); 0 = all")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)

rows = []
for f in sorted(p for p in list_repo_files(REPO, repo_type="dataset") if p.startswith("annotations/") and p.endswith(".json")):
    rows += json.load(open(hf_hub_download(REPO, f, repo_type="dataset"), encoding="utf-8"))
if a.per_group:
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    rows = [r for g in sorted(by_group) for r in rng.sample(by_group[g], min(a.per_group, len(by_group[g])))]

os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
for name in sorted({r["img_filename"] for r in rows}):
    dst = os.path.join(a.out, "images", name)
    if os.path.exists(dst):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src = hf_hub_download(REPO, f"images/{name}", repo_type="dataset")
    shutil.copyfile(src, dst)
    os.remove(os.path.realpath(src))  # keep one copy only
json.dump(rows, open(os.path.join(a.out, "annotations", "all.json"), "w", encoding="utf-8"), indent=1)
print(len(rows), "samples,", len({r["img_filename"] for r in rows}), "images")
