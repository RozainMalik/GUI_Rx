"""The only proof that the exported images exist: stat every distinct image_path in
each export directory, check every bbox is percent xywh inside 0-100, compare the task count
with the manifest, and print IMAGES_OK or IMAGES_FAIL. Run it where the images live, after `export`."""
import json
import sys
from pathlib import Path


def check(export_dir, totals=None) -> bool:
    """totals, when given a list, gets this directory's task count appended - so main() can
    sum task counts across every directory without re-parsing the exports."""
    d = Path(export_dir)
    files = sorted(d.glob("*_train.json")) + sorted(d.glob("*_train_flat.json"))   # <dataset>_train[_flat].json
    if not files:
        print(f"{d}: no export file")
        if totals is not None:
            totals.append(0)
        return False
    paths, bad_boxes, tasks = set(), 0, 0
    for f in files:
        for task in json.loads(f.read_text(encoding="utf-8"))["tasks"]:
            tasks += 1
            for step in task.get("steps") or [task]:
                paths.add(step["image_path"])
                for action in step["actions"]:
                    x, y, w, h = action["bbox"]
                    if not (0 <= x <= 100 and 0 <= y <= 100 and w > 0 and h > 0 and x + w <= 100.01 and y + h <= 100.01):
                        bad_boxes += 1
    missing = sorted(p for p in paths if not Path(p).is_file())
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
    expected = manifest.get("tasks", tasks)
    print(f"{d}: tasks={tasks} manifest_tasks={expected} images={len(paths)} missing={len(missing)} "
          f"bad_boxes={bad_boxes} contaminated={manifest.get('contaminated')}")
    for p in missing[:20]:
        print("  MISSING", p)
    if totals is not None:
        totals.append(tasks)
    return not missing and not bad_boxes and tasks == expected


def main(argv) -> int:
    totals = []
    results = [check(d, totals) for d in argv]   # a list, so every directory is reported
    ok = bool(results) and all(results)
    if ok and sum(totals) == 0:
        # every check passed vacuously - e.g. every frame abstained, or APPS named an empty
        # folder - so nothing was actually exported; a single empty directory among others
        # (medspot_lowconf with no tier-C rows) is fine, only the grand total matters.
        print("IMAGES_FAIL no tasks in any export")
        return 1
    print("IMAGES_OK" if ok else "IMAGES_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
