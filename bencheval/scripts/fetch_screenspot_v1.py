"""Download ScreenSpot v1 (all 1272 samples) into the port's SS-v2 layout: <out>/subset.json + <out>/<img_filename>.

Source: dataset repo KevinQHLin/ScreenSpot holds
metadata/screenspot_{mobile,desktop,web}.json (SeeClick format: img_filename, bbox = [x, y, w, h] px, instruction,
data_type, data_source) and 611 loose images under images/. rootsautomation/ScreenSpot only has parquet shards.
Acceptance: prints "1272 samples, ... 0 missing".
"""
import argparse
import json
import os
import shutil

from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/screenspot_v1")
ap.add_argument("--repo", default="KevinQHLin/ScreenSpot")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
subset = []
for split in ["mobile", "desktop", "web"]:
    path = hf_hub_download(a.repo, f"metadata/screenspot_{split}.json", repo_type="dataset")
    for r in json.load(open(path, encoding="utf-8")):
        r["split"] = split
        subset.append(r)
need = sorted({r["img_filename"] for r in subset})
for name in need:
    dst = os.path.join(a.out, name)
    if not os.path.exists(dst):
        shutil.copy(hf_hub_download(a.repo, f"images/{name}", repo_type="dataset"), dst)
json.dump(subset, open(os.path.join(a.out, "subset.json"), "w", encoding="utf-8"), indent=1)
missing = [n for n in need if not os.path.exists(os.path.join(a.out, n))]
print(len(subset), "samples,", len(need), "images,", len(missing), "missing")
