"""Stage B: a seeded random sample of GroundNext's instruction-tuning records
(ServiceNow/GroundCUA instruction_tuning.tar.gz) -> GUI-Actor records for the GUI-AIMA trainer.

1. download + unpack instruction_tuning.tar.gz into <out>/groundcua_raw/ (once);
2. keep records with a box of at least 1 px each way and a non-empty instruction (of the tarball's ~699k records,
   537k are boxed + instructed, in direct_* / functional_instructions; spatial_data and functional_instructions_extra
   are point-only and remaining_instructions has no instruction, so those drop out); sample N with the seed;
3. fetch the screenshots they use, images/<platform>/<id>.png, one HTTP GET each (skips existing files);
4. pixel boxes -> 0-1 with the real image size (clamped; a box with nothing inside the image is an error).
Writes <out>/groundcua.json and <out>/images/groundcua/<platform>/<id>.png.
Usage: python scripts/build_groundcua.py --out data/full509k [--n 250000] [--seed 0] [--plan]
"""
import argparse
import collections
import glob
import http.client
import json
import os
import random
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_zip import _headers, present  # noqa: E402

HF = "https://huggingface.co/datasets/ServiceNow/GroundCUA/resolve/main/"
MIN_PX, EPS = 1.0, 1e-4


def image_rel(row):
    return f"{row['platform']}/{row['id']}.png"


def image_url(row):
    return HF + "images/" + urllib.parse.quote(row["platform"], safe="") + f"/{row['id']}.png"


def valid(row):
    """A box (not a point-only 'cordinate' record) of at least 1 px each way, and a non-empty instruction."""
    if "bbox" not in row or not str(row.get("instruction", "")).strip():
        return False
    x1, y1, x2, y2 = row["bbox"]
    return x2 - x1 >= MIN_PX and y2 - y1 >= MIN_PX


def sample(rows, n, seed):
    if n > len(rows):
        raise ValueError(f"asked for {n} records, only {len(rows)} valid")
    return [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), n))]


def to_record(row, w, h):
    x1, y1, x2, y2 = row["bbox"]
    x1, y1, x2, y2 = max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h)
    if not (x2 - x1 > EPS and y2 - y1 > EPS):
        raise ValueError(f"box {row['bbox']} is outside {image_rel(row)} ({w}x{h})")
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return {"image": image_rel(row), "conversations": [
        {"from": "human", "value": f"<image> {row['instruction'].strip()}"},
        {"from": "gpt", "value": f"pyautogui.click(x={cx:.4f}, y={cy:.4f})", "recipient": "os", "end_turn": True,
         "bbox_gt": [round(v, 5) for v in (x1, y1, x2, y2)]}]}


def convert(rows, size_of, max_drop):
    """GUI-Actor records for rows; size_of(image_rel) -> (w, h). Rows whose box has nothing inside the image are
    annotation errors and are dropped; more than max_drop of them means a systematic mismatch, so stop."""
    recs, dropped = [], []
    for r in rows:
        try:
            recs.append(to_record(r, *size_of(image_rel(r))))
        except ValueError:
            dropped.append(r)
    if len(dropped) > max_drop * len(rows):
        raise SystemExit(f"{len(dropped)} of {len(rows)} boxes lie outside their image (limit {max_drop:.2%}), "
                         f"e.g. {[(image_rel(r), r['bbox']) for r in dropped[:3]]}")
    return recs, dropped


def load_instructions(raw_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "groundcua_data", "*.json"))):
        rows += json.load(open(p, encoding="utf-8"))
    return rows


def _download(url, dst, retries=6):
    """GET url -> dst via dst.part. A dropped connection or a body shorter than Content-Length is retried; urllib's
    read(amt) returns a short body silently, so the length check is what catches a truncated file."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".part", "wb") as fh:
                expect = r.headers.get("Content-Length")
                shutil.copyfileobj(r, fh, 1 << 20)
                fh.flush()
                os.fsync(fh.fileno())   # on disk before the rename, so an interrupted write cannot leave a 0-byte dst
            if expect is not None and os.path.getsize(dst + ".part") != int(expect):
                raise http.client.IncompleteRead(b"", int(expect) - os.path.getsize(dst + ".part"))
            os.replace(dst + ".part", dst)
            return
        except (OSError, http.client.HTTPException):
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt))


def fetch_images(rows, img_dir, workers):
    todo = {image_rel(r): image_url(r) for r in rows}
    need = [(rel, url) for rel, url in sorted(todo.items()) if not present(os.path.join(img_dir, rel))]
    print(f"{len(todo)} screenshots, {len(need)} to fetch", flush=True)

    def one(item):
        rel, url = item
        dst = os.path.join(img_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        _download(url, dst)

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, need), 1):
            if i % 2000 == 0 or i == len(need):
                print(f"  {i}/{len(need)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/full509k")
    ap.add_argument("--n", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", action="store_true", help="unpack, filter, sample and report; fetch nothing else")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_drop", type=float, default=0.001,
                    help="max fraction of sampled records whose box lies outside the image (seed 0: 20 / 250000)")
    a = ap.parse_args()
    raw = os.path.join(a.out, "groundcua_raw")
    os.makedirs(raw, exist_ok=True)
    tgz = os.path.join(raw, "instruction_tuning.tar.gz")
    if not glob.glob(os.path.join(raw, "groundcua_data", "*.json")):
        if not os.path.exists(tgz):
            _download(HF + "instruction_tuning.tar.gz", tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(raw)
    for p in sorted(glob.glob(os.path.join(raw, "groundcua_data", "*.json"))):
        print("instruction file:", os.path.basename(p), flush=True)
    rows = load_instructions(raw)
    ok = [r for r in rows if valid(r)]
    chosen = sample(ok, a.n, a.seed)
    per = collections.Counter(r["platform"] for r in chosen)
    print(f"{len(rows)} records, {len(ok)} valid, sampled {len(chosen)} (seed {a.seed}), "
          f"{len({image_rel(r) for r in chosen})} screenshots, {len(per)} platforms", flush=True)
    if a.plan:
        return
    img_dir = os.path.join(a.out, "images", "groundcua")
    fetch_images(chosen, img_dir, a.workers)
    sizes = {}

    def size_of(rel):
        if rel not in sizes:
            with Image.open(os.path.join(img_dir, rel)) as im:
                sizes[rel] = im.size
        return sizes[rel]

    recs, dropped = convert(chosen, size_of, a.max_drop)
    for r in dropped:
        print(f"dropped (box outside the image): {image_rel(r)} {r['bbox']} {size_of(image_rel(r))}")
    out = os.path.join(a.out, "groundcua.json")
    json.dump(recs, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{out}: {len(recs)} records ({len(dropped)} dropped), {len(sizes)} images")
    print(json.dumps(dict(sorted(per.items())), indent=0))


if __name__ == "__main__":
    main()
