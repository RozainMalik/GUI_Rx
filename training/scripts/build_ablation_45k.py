"""Build the paper's 45k ablation set: first 10k GUIAct / AndroidControl / Wave-UI + first 15k UGround.

Annotations come from cckevinn/GUI-Actor-Data and smz8599/UGround-single (the sources listed in the upstream
README). Only the images these 45k samples use are taken out of the (40-250 GB) image zips, via HTTP range
requests (scripts/remote_zip.py), so the 45k set needs ~12 GB (~28.6k images) instead of ~400 GB.

Writes <out>/<name>.json (subset annotations), <out>/images/<name>/<file> and <out>/data_45k.yaml.
Re-running skips images that already exist.
"""
import argparse
import codecs
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_zip import RemoteZip, _headers  # noqa: E402

ACTOR = "https://huggingface.co/datasets/cckevinn/GUI-Actor-Data/resolve/main/"
UG_SINGLE = "https://huggingface.co/datasets/smz8599/UGround-single/resolve/main/uground_bbox_single.json"
# name -> (annotation url, first N samples, image zip parts, folder of the images inside the zip)
SOURCES = {
    "guiact": (ACTOR + "guiact_bbox.json", 10000, [ACTOR + "GUIAct_images.zip"], "GUIAct/web_imgs"),
    "androidcontrol": (ACTOR + "androidcontrol_bbox.json", 10000, [ACTOR + "AndroidControl_images.zip"],
                       "AndroidControl/tfrecord/images"),
    "wave_ui": (ACTOR + "wave_ui_bbox.json", 10000, [ACTOR + "Wave-UI_images.zip"], "Wave-UI/images_fixed"),
    "uground": (UG_SINGLE, 15000,
                [ACTOR + f"Uground_images_split.z0{i}" for i in range(1, 6)] + [ACTOR + "Uground_images_split.zip"],
                "Uground/images"),
}


SEP = re.compile(r"[\s,]*")  # whitespace and commas between records


def first_records(url, n):
    """Stream-parse the first n records of a remote JSON list without downloading the whole file."""
    dec, utf8 = json.JSONDecoder(), codecs.getincrementaldecoder("utf-8")()
    out, buf, pos, eof = [], "", None, False
    with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=120) as r:
        while len(out) < n and not eof:
            chunk = r.read(1 << 20)
            eof = not chunk
            buf += utf8.decode(chunk, final=eof)
            if pos is None:
                if "[" not in buf:
                    continue
                pos = buf.index("[") + 1
            while len(out) < n:
                pos = SEP.match(buf, pos).end()
                if pos >= len(buf) or buf[pos] == "]":
                    break
                try:
                    obj, pos = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break  # record continues in the next chunk
                out.append(obj)
            buf, pos = buf[pos:], 0
    return out


def image_names(rows):
    names = []
    for r in rows:
        imgs = r["image"] if isinstance(r["image"], list) else [r["image"]]
        names += imgs
    return sorted(set(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ablation_45k")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of each source's N (for smoke tests)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    yaml = ["datasets:"]
    for name in a.sources.split(","):
        ann_url, n, parts, inner = SOURCES[name]
        n = max(1, int(n * a.scale))
        ann_path = os.path.join(out, f"{name}.json")
        if os.path.exists(ann_path) and len(json.load(open(ann_path, encoding="utf-8"))) == n:
            rows = json.load(open(ann_path, encoding="utf-8"))
        else:
            rows = first_records(ann_url, n)
            json.dump(rows, open(ann_path, "w", encoding="utf-8"))
        img_dir = os.path.join(out, "images", name)
        os.makedirs(img_dir, exist_ok=True)
        need = [f for f in image_names(rows) if not os.path.exists(os.path.join(img_dir, f))]
        print(f"{name}: {len(rows)} samples, {len(image_names(rows))} images, {len(need)} to fetch", flush=True)
        if need:
            z = RemoteZip.from_urls(parts)
            if name == "uground":  # the split archive holds one stored zip, Uground_images.zip
                z = RemoteZip.nested(z, "Uground_images.zip")
            missing = [f for f in need if f"{inner}/{f}" not in z.members]
            if missing:
                raise SystemExit(f"{name}: {len(missing)} images not in the zip, e.g. {missing[:3]}")

            def fetch(f):
                data = z.read(f"{inner}/{f}")
                tmp = os.path.join(img_dir, f + ".part")
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, os.path.join(img_dir, f))
                return len(data)

            done = total = 0
            with ThreadPoolExecutor(a.workers) as ex:
                for size in ex.map(fetch, need):
                    done, total = done + 1, total + size
                    if done % 500 == 0 or done == len(need):
                        print(f"  {done}/{len(need)} images, {total / 1e9:.2f} GB", flush=True)
        yaml += [f"  - json_path: {ann_path}", f"    images_folder: {img_dir}/", '    sampling_strategy: "all"']
    with open(os.path.join(out, "data_45k.yaml"), "w") as f:
        f.write("\n".join(yaml) + "\n")
    print("wrote", os.path.join(out, "data_45k.yaml"))


if __name__ == "__main__":
    main()
