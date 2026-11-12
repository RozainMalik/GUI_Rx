"""Download ScreenSpot-v2 annotations and extract only the images of a stratified subset.

ScreenSpot-v2 `bbox` is [x, y, w, h] in original pixels. The 1.3 GB image zip is deleted after
extraction to save disk.
"""
import argparse
import json
import os
import random
import zipfile

from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/screenspot_v2")
ap.add_argument("--per_split", type=int, default=100, help="samples per split; 0 = all")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = random.Random(a.seed)
subset = []
for split in ["mobile", "desktop", "web"]:
    path = hf_hub_download("OS-Copilot/ScreenSpot-v2", f"screenspot_{split}_v2.json", repo_type="dataset")
    rows = json.load(open(path, encoding="utf-8"))
    picked = rows if a.per_split == 0 else rng.sample(rows, min(a.per_split, len(rows)))
    for r in picked:
        r["split"] = split
        subset.append(r)
zpath = hf_hub_download("OS-Copilot/ScreenSpot-v2", "screenspotv2_image.zip", repo_type="dataset")
need = {r["img_filename"] for r in subset}
with zipfile.ZipFile(zpath) as z:
    for name in z.namelist():
        if os.path.basename(name) in need:
            with open(os.path.join(a.out, os.path.basename(name)), "wb") as f:
                f.write(z.read(name))
os.remove(os.path.realpath(zpath))
json.dump(subset, open(os.path.join(a.out, "subset.json"), "w", encoding="utf-8"), indent=1)
missing = [n for n in need if not os.path.exists(os.path.join(a.out, n))]
print(len(subset), "samples,", len(need), "images,", len(missing), "missing")
