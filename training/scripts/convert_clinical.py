"""MedSPOT-format training export (tasks -> steps -> click actions, bbox = percent [x, y, w, h]) -> GUI-Actor records
the GUI-AIMA trainer reads (bbox_gt = 0-1 [x1, y1, x2, y2], click at the box centre). Image paths become relative to
the screenshot root, so the same records work with any copy of the images (images_folder in the yaml).
--exclude_src drops every record whose image is in another export.
Usage: python scripts/convert_clinical.py --src export.json --out data/clinical_train/all.json
"""
import argparse
import collections
import json
import sys

ROOT_MARK = "MedSPOT_Training_Data/"
EPS = 1e-4


def rel_image(path):
    p = path.replace("\\", "/")
    i = p.find(ROOT_MARK)
    if i < 0:
        raise ValueError(f"image path is not under {ROOT_MARK}: {path}")
    return p[i + len(ROOT_MARK):]


def to_record(step):
    clicks = [a for a in step["actions"] if a.get("type") == "click" and "bbox" in a]
    if len(step["actions"]) != 1 or len(clicks) != 1:
        raise ValueError(f"expected exactly one click action: {step['image_path']} {step['actions']}")
    x, y, w, h = clicks[0]["bbox"]
    x1, y1 = max(0.0, x / 100), max(0.0, y / 100)
    x2, y2 = min(1.0, (x + w) / 100), min(1.0, (y + h) / 100)
    if not (x2 - x1 > EPS and y2 - y1 > EPS):
        raise ValueError(f"degenerate box {clicks[0]['bbox']} in {step['image_path']}")
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return {"image": rel_image(step["image_path"]), "conversations": [
        {"from": "human", "value": f"<image> {step['instruction'].strip()}"},
        {"from": "gpt", "value": f"pyautogui.click(x={cx:.4f}, y={cy:.4f})", "recipient": "os", "end_turn": True,
         "bbox_gt": [round(v, 5) for v in (x1, y1, x2, y2)]}]}


def convert(tasks, exclude_images=frozenset()):
    out = []
    for t in tasks:
        for s in t["steps"]:
            rec = to_record(s)
            if rec["image"] not in exclude_images:
                out.append(rec)
    return out


def _images(path):
    return {rel_image(s["image_path"]) for t in json.load(open(path, encoding="utf-8"))["tasks"] for s in t["steps"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--exclude_src", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect_images", type=int, default=None)
    ap.add_argument("--expect_records", type=int, default=None)
    a = ap.parse_args()
    excl = _images(a.exclude_src) if a.exclude_src else set()
    recs = convert(json.load(open(a.src, encoding="utf-8"))["tasks"], excl)
    imgs = {r["image"] for r in recs}
    if a.expect_images is not None and len(imgs) != a.expect_images:
        sys.exit(f"expected {a.expect_images} images, got {len(imgs)}")
    if a.expect_records is not None and len(recs) != a.expect_records:
        sys.exit(f"expected {a.expect_records} records, got {len(recs)}")
    json.dump(recs, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    per_app = collections.Counter(i.split("/")[0] for i in imgs)
    print(f"{a.out}: {len(recs)} records, {len(imgs)} images, excluded {len(excl)} images")
    print(json.dumps(dict(sorted(per_app.items())), indent=0))


if __name__ == "__main__":
    main()
