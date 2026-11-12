"""Audit a training mix before a long run. Fails (exit 1) on: a record whose image is missing, an image that
does not decode (empty / truncated), a record with no usable bbox_gt, a training
image byte-identical (sha256) to any benchmark image, a --bench dir with no images, or a total record count other
than --expect_records. The leak check finds exact copies only, not re-encoded or resized ones. --drop_leaks rewrites each affected JSON without the leaked records and keeps the original as
<name>.orig.json; leaks then no longer count as a failure.
Usage: python scripts/audit_mix.py --data configs/stageA.yaml --bench data/screenspot_v2_full \
         --bench data/screenspot_pro --bench data/medspot [--drop_leaks] [--expect_records 254779]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decodes(p):
    """True if the image file is non-empty and fully decodes (PNG chunk CRCs checked by verify, pixels by load)."""
    try:
        if os.path.getsize(p) == 0:
            return False
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
        return True
    except Exception:
        return False


def inspect(p):
    return sha256_file(p), decodes(p)


def bench_files(d):
    return [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(IMG_EXT)]


def bench_hashes(dirs, workers=16):
    """sha256 of every benchmark image. Exact copies only: a re-encoded or resized screenshot is not caught. A dir with
    no images (typo, unmounted) would make the leak check vacuous, so it is an error."""
    files = []
    for d in dirs:
        fs = bench_files(d)
        print(f"{d}: {len(fs)} images")
        if not fs:
            print(f"AUDIT FAIL: no benchmark images in {d}")
            sys.exit(1)
        files += fs
    with ThreadPoolExecutor(workers) as ex:
        return set(ex.map(sha256_file, files))


def images_of(rec):
    return rec["image"] if isinstance(rec["image"], list) else [rec["image"]]


def boxes(rec):
    out = []
    for c in rec["conversations"]:
        b = c.get("bbox_gt")
        if b is None:
            continue
        out += b if b and isinstance(b[0], list) else [b]
    return out


def record_problem(rec):
    bs = boxes(rec)
    if not bs:
        return "no bbox_gt"
    for b in bs:
        if len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
            return "malformed box"
        if not (b[0] < b[2] and b[1] < b[3]):
            return "inverted box"
    return None


def audit(datasets, bench, workers=16):
    report = {"records": 0, "missing": 0, "corrupt": 0, "bad": 0, "leaked": 0, "out_of_range": 0, "per": []}
    for d in datasets:
        recs = json.load(open(d["json_path"], encoding="utf-8"))
        folder = d["images_folder"]
        paths = sorted({os.path.join(folder, i) for r in recs for i in images_of(r)})
        present = [p for p in paths if os.path.exists(p)]
        with ThreadPoolExecutor(workers) as ex:
            info = dict(zip(present, ex.map(inspect, present)))
        digest = {p: h for p, (h, _) in info.items()}
        corrupt = sorted(p for p, (_, ok) in info.items() if not ok)
        leaked_idx, missing, bad, oor = [], 0, 0, 0
        for i, r in enumerate(recs):
            ps = [os.path.join(folder, x) for x in images_of(r)]
            missing += sum(p not in digest for p in ps)
            bad += record_problem(r) is not None
            oor += any(v < -0.01 or v > 1.01 for b in boxes(r) if len(b) == 4 for v in b)
            if any(digest.get(p) in bench for p in ps):
                leaked_idx.append(i)
        report["per"].append({"json": d["json_path"], "records": len(recs), "images": len(paths), "missing": missing,
                              "corrupt": len(corrupt), "corrupt_files": corrupt[:20], "bad": bad, "out_of_range": oor, "leaked_idx": leaked_idx})
        for k, v in (("records", len(recs)), ("missing", missing), ("corrupt", len(corrupt)), ("bad", bad), ("leaked", len(leaked_idx)),
                     ("out_of_range", oor)):
            report[k] += v
    return report


def drop_leaks(per):
    for p in per:
        if not p["leaked_idx"]:
            continue
        src = p["json"]
        orig = src[:-5] + ".orig.json"
        if not os.path.exists(orig):
            shutil.copyfile(src, orig)
        drop = set(p["leaked_idx"])
        recs = json.load(open(src, encoding="utf-8"))
        json.dump([r for i, r in enumerate(recs) if i not in drop], open(src, "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"{src}: dropped {len(drop)} leaked records (original kept as {orig})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="training yaml (repeatable)")
    ap.add_argument("--bench", action="append", required=True, help="benchmark image dir (repeatable)")
    ap.add_argument("--expect_records", type=int, default=None)
    ap.add_argument("--drop_leaks", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    datasets = [d for y in a.data for d in yaml.safe_load(open(y))["datasets"]]
    datasets = list({os.path.normpath(d["json_path"]): d for d in datasets}.values())   # once each (--drop_leaks)
    bench = bench_hashes(a.bench, a.workers)
    print(f"benchmark images: {len(bench)} unique sha256")
    rep = audit(datasets, bench, a.workers)
    for p in rep["per"]:
        print(f"{p['json']}: {p['records']} records, {p['images']} images, missing {p['missing']}, corrupt {p['corrupt']}, "
              f"bad {p['bad']}, "
              f"out of [0,1] {p['out_of_range']}, leaked {len(p['leaked_idx'])}")
        for f in p["corrupt_files"]:
            print("CORRUPT", f)
    if a.drop_leaks and rep["leaked"]:
        drop_leaks(rep["per"])
        rep["records"] -= rep["leaked"]
        rep["leaked"] = 0
    print(f"records: {rep['records']}\nmissing images: {rep['missing']}\ncorrupt images: {rep['corrupt']}\nbad records: {rep['bad']}\n"
          f"leaked records: {rep['leaked']}\nout-of-range boxes (warning only): {rep['out_of_range']}")
    fail = rep["missing"] or rep["corrupt"] or rep["bad"] or rep["leaked"]
    if a.expect_records is not None and rep["records"] != a.expect_records:
        print(f"expected {a.expect_records} records, got {rep['records']}")
        fail = True
    print("AUDIT FAIL" if fail else "AUDIT OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
