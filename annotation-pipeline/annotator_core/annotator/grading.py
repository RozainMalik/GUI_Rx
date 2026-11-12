"""Human grading: a seeded sample of shipped rows for tools/review_server.py, the
append-only verdict log, and apply-grades, which writes `review` into the records. Verdicts are
keyed by image AND box, so a frame whose box changed after a replay is queued and graded again."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from .export import iter_records
from .records import save_record

VERDICTS = ("correct", "loose", "wrong", "corrected")
REVIEW_VERDICT = {"correct": "correct", "loose": "loose", "wrong": "rejected", "corrected": "corrected"}


def sample_path(work, tiers) -> Path:
    return Path(work) / "review" / f"sample_{''.join(tiers)}.json"


def verdicts_path(work) -> Path:
    return Path(work) / "review" / "verdicts.jsonl"


def grade_rows(work, tiers=("A", "B")) -> list[dict]:
    """Every record whose selected candidate is in `tiers`, with the pipeline's box (never the
    reviewer's) - rejected ones included, so a replayed box is re-judged."""
    rows = []
    for rec in iter_records(work):
        if rec.get("selected") is None:
            continue
        cand = rec["candidates"][rec["selected"]]
        final = cand.get("final")
        if not final or final["tier"] not in tiers:
            continue
        rows.append({"image": f"{rec['app']}/{rec['frame']}", "app": rec["app"], "frame": rec["frame"],
                     "frame_index": rec["frame_index"], "tier": final["tier"], "bbox_px": list(final["bbox_px"]),
                     "element": cand["element"]["element_name"], "element_kind": cand["element"]["element_kind"],
                     "instructions": dict(final["instructions"])})
    return rows


def build_sample(work, n, seed, tiers=("A", "B"), resample=False, log=print) -> list[dict]:
    """The seeded sample for `grade`. Re-grading is meant to be re-grading the SAME rows: once a sample file exists, a plain call refreshes it in place - same (app,
    frame) set, same order, current bbox_px/tier/element/element_kind/instructions - rather than
    re-sampling the population, which shifts under a prompt fix + replay. A frame no longer
    shipped in `tiers` is dropped (never replaced - no new frames are added). `resample=True`
    (the old behaviour) ignores the existing file and rebuilds from scratch."""
    path = sample_path(work, tiers)
    if not resample and path.exists():
        return _refresh_sample(work, path, tiers, log)
    rows = grade_rows(work, tiers)
    items = random.Random(seed).sample(rows, min(n, len(rows)))
    items.sort(key=lambda i: (i["app"], i["frame_index"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n": n, "seed": seed, "tiers": list(tiers), "items": items}, indent=1) + "\n", encoding="utf-8")
    return items


def _refresh_sample(work, path, tiers, log=print) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    current = {(r["app"], r["frame"]): r for r in grade_rows(work, tiers)}
    items, dropped = [], 0
    for it in data.get("items", []):
        row = current.get((it["app"], it["frame"]))
        if row is None:
            dropped += 1
            continue
        items.append(dict(it, bbox_px=row["bbox_px"], tier=row["tier"], element=row["element"],
                           element_kind=row["element_kind"], instructions=row["instructions"]))
    if dropped:
        log(f"grade: dropped {dropped} sample row(s) no longer shipped in tiers {''.join(tiers)}")
    path.write_text(json.dumps({**data, "items": items}, indent=1) + "\n", encoding="utf-8")
    return items


def verdict_key(image, bbox_px) -> str:
    return f"{image}|{','.join(str(int(v)) for v in bbox_px)}"


def load_verdicts(path) -> dict:
    """key -> last verdict row. Last wins, so re-judging simply appends."""
    out = {}
    path = Path(path)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if v.get("verdict") in VERDICTS and v.get("image") and v.get("original_bbox_px"):
            out[verdict_key(v["image"], v["original_bbox_px"])] = v
    return out


def append_verdict(path, verdict: dict) -> None:
    """Written and flushed before the reviewer's view advances."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verdict = dict(verdict, at=verdict.get("at") or time.strftime("%Y-%m-%dT%H:%M:%S"))
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(verdict, ensure_ascii=False) + "\n")
        fh.flush()


def summarise(verdicts: dict, items: list) -> dict:
    counts = {v: 0 for v in VERDICTS}
    graded = 0
    for it in items:
        v = verdicts.get(verdict_key(it["image"], it["bbox_px"]))
        if v:
            graded += 1
            counts[v["verdict"]] += 1
    return {"items": len(items), "graded": graded, **counts,
            "correct_rate": round(counts["correct"] / graded, 3) if graded else None}


def apply_grades(work, verdicts_file=None) -> int:
    """Write `review` into every record whose selected box has a verdict. Returns how many changed."""
    verdicts = load_verdicts(verdicts_file or verdicts_path(work))
    applied = 0
    for rec in iter_records(work):
        path = rec.pop("_path")
        if rec.get("selected") is None:
            continue
        final = rec["candidates"][rec["selected"]].get("final")
        if not final:
            continue
        v = verdicts.get(verdict_key(f"{rec['app']}/{rec['frame']}", final["bbox_px"]))
        if not v:
            continue
        review = {"verdict": REVIEW_VERDICT[v["verdict"]],
                  "bbox_px": list(v["bbox_px"]) if v["verdict"] == "corrected" and v.get("bbox_px") else None,
                  "original_bbox_px": list(v["original_bbox_px"]), "graded_at": v.get("at")}
        if rec.get("review") == review:
            continue
        rec["review"] = review
        save_record(path, rec)
        applied += 1
    return applied
