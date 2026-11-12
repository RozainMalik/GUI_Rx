"""Build the GUI-AIMA lite mix, stage A: all of
GUIAct / AndroidControl / Wave-UI, the first 60k UGround-single records and the 60k GTA1 no-web records.

Zip sources: only the images the records use are read out of the remote zips (scripts/remote_zip.py).
GTA1: the images are an uncompressed tar split into parts; the parts are streamed in order and only the needed
members are written. --plan writes the annotation JSONs and prints records / images / bytes per source, no images.
Writes <out>/<name>.json and <out>/images/<name>/<image>. Re-running skips images that already exist.
Usage: python scripts/build_mix.py --out data/full509k [--sources guiact,uground] [--plan]
"""
import argparse
import http.client
import io
import json
import os
import pathlib
import shutil
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ablation_45k import ACTOR, UG_SINGLE, first_records, image_names  # noqa: E402
from remote_zip import RemoteZip, _headers, present  # noqa: E402

GTA = "https://huggingface.co/datasets/smz8599/GTA_data_no_web/resolve/main/"
UG_PARTS = [ACTOR + f"Uground_images_split.z0{i}" for i in range(1, 6)] + [ACTOR + "Uground_images_split.zip"]
SOURCES = {
    "guiact": dict(ann=ACTOR + "guiact_bbox.json", take="all", n=42545,
                   zip=[ACTOR + "GUIAct_images.zip"], inner="GUIAct/web_imgs"),
    "androidcontrol": dict(ann=ACTOR + "androidcontrol_bbox.json", take="all", n=39701,
                           zip=[ACTOR + "AndroidControl_images.zip"], inner="AndroidControl/tfrecord/images"),
    "wave_ui": dict(ann=ACTOR + "wave_ui_bbox.json", take="all", n=52529,
                    zip=[ACTOR + "Wave-UI_images.zip"], inner="Wave-UI/images_fixed"),
    "uground": dict(ann=UG_SINGLE, take="first:60000", n=60000,
                    zip=UG_PARTS, nested="Uground_images.zip", inner="Uground/images"),
    # upstream GUI-AIMA data/gta/gta_data_wo_web_output_60k.json, sha256 46c48bb8309bfb02ac49a8a871259e5e191fb90cbb5a04accbd0839faec55111
    "gta": dict(ann="data/full509k/gta_data_wo_web_output_60k.json", take="all", n=60004,
                tar=[GTA + f"gta_data_part_a{c}" for c in "abcdefg"], inner="gta_data/image"),
}


def select(rows, take):
    if take == "all":
        return rows
    kind, _, n = take.partition(":")
    if kind != "first":
        raise ValueError(f"unsupported take {take!r}")
    return rows[:int(n)]


def load_rows(ann, take):
    if not ann.startswith("http"):
        return select(json.load(open(ann, encoding="utf-8")), take)
    if take.startswith("first:"):
        return first_records(ann, int(take.split(":")[1]))
    with urllib.request.urlopen(urllib.request.Request(ann, headers=_headers()), timeout=600) as r:
        return json.load(r)


def zip_plan(members, inner, names):
    """(names with no <inner>/<name> member, compressed bytes of the ones present)."""
    missing, total = [], 0
    for n in names:
        m = members.get(f"{inner}/{n}")
        if m is None:
            missing.append(n)
        else:
            total += m[1]
    return missing, total


class ConcatStream(io.RawIOBase):
    """The URLs read one after another as a single stream (the parts of a split tar). A part whose connection drops
    or ends short of its Content-Length is reopened at the current offset with a Range request (RETRIES in a row)."""

    RETRIES = 8

    def __init__(self, urls):
        self.urls, self.cur, self.pos, self.size = list(urls), None, 0, None

    def readable(self):
        return True

    def _open(self):
        hdr = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        self.cur = urllib.request.urlopen(urllib.request.Request(self.urls[0], headers=_headers(hdr)), timeout=600)
        if self.pos and getattr(self.cur, "status", 206) != 206:
            raise OSError(f"{self.urls[0]}: range request not honoured (status {self.cur.status})")
        if self.size is None and self.cur.headers.get("Content-Length") is not None:
            self.size = int(self.cur.headers["Content-Length"])

    def _drop(self):
        if self.cur is not None:
            try:
                self.cur.close()
            except Exception:
                pass
        self.cur = None

    def readinto(self, b):
        fails = 0
        while self.urls:
            try:
                if self.cur is None:
                    self._open()
                n = self.cur.readinto(b)
                if not n and self.size is not None and self.pos < self.size:
                    raise http.client.IncompleteRead(b"", self.size - self.pos)
            except (OSError, http.client.HTTPException):
                self._drop()
                fails += 1
                if fails > self.RETRIES:
                    raise
                time.sleep(min(60, 2 ** fails))
                continue
            if n:
                self.pos += n
                return n
            self._drop()
            self.urls.pop(0)
            self.pos, self.size = 0, None
        return 0


def tar_urls(s, parts_dir):
    """The source's tar part URLs, or file:// URLs of the same parts already downloaded into parts_dir."""
    if not parts_dir:
        return s["tar"]
    paths = [os.path.abspath(os.path.join(parts_dir, u.rsplit("/", 1)[1])) for u in s["tar"]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing tar parts in {parts_dir}: {missing}")
    return [pathlib.Path(p).as_uri() for p in paths]


def _write(dst, src):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".part", "wb") as fh:
        if isinstance(src, bytes):
            fh.write(src)
        else:
            shutil.copyfileobj(src, fh, 1 << 20)
        fh.flush()
        os.fsync(fh.fileno())   # on disk before the rename, so an interrupted write cannot leave a 0-byte dst
    os.replace(dst + ".part", dst)


def extract_tar_stream(fileobj, inner, wanted, img_dir):
    """Write members <inner>/<name>, name in wanted, of a streamed tar to <img_dir>/<name>. Returns the names written."""
    prefix, got = inner.rstrip("/") + "/", set()
    with tarfile.open(fileobj=fileobj, mode="r|") as tf:
        for m in tf:
            name = m.name[len(prefix):] if m.name.startswith(prefix) else None
            if m.isfile() and name in wanted:
                _write(os.path.join(img_dir, name), tf.extractfile(m))
                got.add(name)
    return got


def fetch_zip(z, inner, need, img_dir, workers):
    def one(f):
        data = z.read(f"{inner}/{f}")
        _write(os.path.join(img_dir, f), data)
        return len(data)

    done = total = 0
    with ThreadPoolExecutor(workers) as ex:
        for size in ex.map(one, need):
            done, total = done + 1, total + size
            if done % 2000 == 0 or done == len(need):
                print(f"  {done}/{len(need)} images, {total / 1e9:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/full509k")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--gta_json", default=None, help="override SOURCES['gta']['ann']")
    ap.add_argument("--gta_parts_dir", default=None, help="read the GTA tar parts from here (downloaded beforehand)")
    ap.add_argument("--plan", action="store_true", help="annotations + sizes only, no images")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in a.sources.split(","):
        s = dict(SOURCES[name])
        if name == "gta" and a.gta_json:
            s["ann"] = a.gta_json
        ann_path = os.path.join(a.out, f"{name}.json")
        if os.path.exists(ann_path):
            rows = json.load(open(ann_path, encoding="utf-8"))
        else:
            rows = load_rows(s["ann"], s["take"])
        if len(rows) != s["n"]:
            raise SystemExit(f"{name}: expected {s['n']} records, got {len(rows)}")
        if not os.path.exists(ann_path):
            json.dump(rows, open(ann_path, "w", encoding="utf-8"))
        img_dir = os.path.join(a.out, "images", name)
        names = image_names(rows)
        need = [f for f in names if not present(os.path.join(img_dir, f))]
        if "zip" in s:
            z = RemoteZip.from_urls(s["zip"])
            if "nested" in s:
                z = RemoteZip.nested(z, s["nested"])
            missing, nbytes = zip_plan(z.members, s["inner"], need)
            print(f"{name}: {len(rows)} records, {len(names)} images, {len(need)} to fetch, {nbytes / 1e9:.1f} GB",
                  flush=True)
            if missing:
                raise SystemExit(f"{name}: {len(missing)} images not in the zip, e.g. {missing[:3]}")
            if need and not a.plan:
                fetch_zip(z, s["inner"], need, img_dir, a.workers)
        else:
            print(f"{name}: {len(rows)} records, {len(names)} images, {len(need)} to fetch, "
                  f"streams all {len(s['tar'])} tar parts", flush=True)
            if need and not a.plan:
                got = extract_tar_stream(io.BufferedReader(ConcatStream(tar_urls(s, a.gta_parts_dir)), 1 << 20), s["inner"],
                                         set(need),
                                         img_dir)
                lost = sorted(set(need) - got)
                if lost:
                    raise SystemExit(f"{name}: {len(lost)} images not in the tar, e.g. {lost[:3]}")
    print("plan only, no images written" if a.plan else "done")


if __name__ == "__main__":
    main()
