"""Overlays and qc_stats.json. The fixed seed makes pilot and full-run renders
comparable frame for frame: the final box in red, every pool box in a thin distinct colour,
and the tier / verdict / read-back / agreement / instructions in a strip under the frame."""
from __future__ import annotations

import json
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .export import iter_records
from .frames import batch_frame, list_apps
from .records import load_record, record_path
from .selection import candidate_calls

POOL_COLOURS = {"describe": (0, 160, 255), "functional": (0, 200, 90), "visual": (255, 170, 0),
                "spatial": (200, 0, 255), "verifier": (0, 220, 220)}
CROP_COLOUR = (150, 150, 150)
FINAL_COLOUR = (255, 0, 0)
STRIP_BG = (20, 20, 24)
STRIP_H = 170
REVIEW_COLOUR = (0, 230, 60)
REVIEW_FONT = 28
REVIEW_WRAP = 118


def sample_records(work, n, seed, statuses=("shipped", "lowconf")) -> list[dict]:
    recs = [r for r in iter_records(work) if r.get("status") in statuses]
    return random.Random(seed).sample(recs, min(n, len(recs)))


def _font(size=16):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                     # Pillow < 10.1
        return ImageFont.load_default()


def _pool(cand) -> dict:
    pool = {}
    d = cand.get("describe") or {}
    if d.get("bbox_px"):
        pool["describe"] = d["bbox_px"]
    for v, gv in (cand.get("ground") or {}).items():
        if gv and gv.get("bbox_px"):
            pool[v] = gv["bbox_px"]
    ver = cand.get("verify") or {}
    if ver.get("corrected_bbox_px"):
        pool["verifier"] = ver["corrected_bbox_px"]
    for c in (cand.get("refine") or {}).get("crops") or []:
        if c.get("bbox_px"):
            pool[f"crop_{c['fraction']}_{c['variant']}"] = c["bbox_px"]
    return pool


def render_overlay(rec, frame) -> Image.Image:
    cand = rec["candidates"][rec["selected"]]
    final = cand["final"]
    canvas = Image.new("RGB", (frame.width, frame.height + STRIP_H), STRIP_BG)
    canvas.paste(frame, (0, 0))
    d = ImageDraw.Draw(canvas)
    pool = _pool(cand)
    for name, b in pool.items():
        d.rectangle((b[0], b[1], b[2] - 1, b[3] - 1), outline=POOL_COLOURS.get(name, CROP_COLOUR), width=1)
    fb = final["bbox_px"]
    d.rectangle((fb[0], fb[1], fb[2] - 1, fb[3] - 1), outline=FINAL_COLOUR, width=3)
    ver = (cand.get("refine") or {}).get("reverify") or cand.get("verify") or {}
    lines = [f"{rec['app']}/{rec['frame']}   tier {final['tier']}   verdict {final.get('verdict')}   "
             f"agreement {final['agreement']}   read-back {ver.get('text_in_box', '')!r}   status {rec['status']}",
             "F: " + final["instructions"]["functional"], "V: " + final["instructions"]["visual"],
             "S: " + final["instructions"]["spatial"], "pool: " + ", ".join(pool)]
    font = _font()
    y = frame.height + 8
    for line in lines:
        d.text((10, y), line[:220], fill=(235, 235, 235), font=font)
        y += 30
    return canvas


def _bucket():
    return {"frames": 0, "status": Counter(), "tier": Counter(), "verdict": Counter(), "agreement": Counter(),
            "refined": 0, "refine_outcome": Counter(), "calls": 0, "candidates": 0, "unreadable_candidates": 0,
            "thinking_fallbacks": 0}


def _add(b, rec):
    b["frames"] += 1
    b["status"][rec.get("status") or "incomplete"] += 1
    b["thinking_fallbacks"] += (rec.get("provenance") or {}).get("thinking_fallbacks", 0)
    b["calls"] += (rec.get("propose") or {}).get("calls", 0)
    for c in rec.get("candidates", []):
        b["candidates"] += 1
        b["calls"] += candidate_calls(c)
        if c.get("status") == "unreadable":
            b["unreadable_candidates"] += 1
        if c.get("refine"):
            b["refined"] += 1
            b["refine_outcome"][c["refine"].get("outcome", "?")] += 1
    sel = rec.get("selected")
    if sel is not None:
        f = rec["candidates"][sel]["final"]
        b["tier"][f["tier"]] += 1
        b["verdict"][str(f.get("verdict"))] += 1
        b["agreement"][str(f["agreement"])] += 1


def _finish(b) -> dict:
    n = b["frames"] or 1
    return {"frames": b["frames"], "status": dict(b["status"]), "tier": dict(b["tier"]), "verdict": dict(b["verdict"]),
            "agreement_histogram": dict(sorted(b["agreement"].items())),
            "refine_rate": round(b["refined"] / max(b["candidates"], 1), 3), "refine_outcome": dict(b["refine_outcome"]),
            "calls_per_frame": round(b["calls"] / n, 2),
            "abstain_rate": round(b["status"].get("abstained", 0) / n, 3),
            "unreadable_rate": round(b["status"].get("unreadable", 0) / n, 3),
            "unreadable_candidate_rate": round(b["unreadable_candidates"] / max(b["candidates"], 1), 3),
            "thinking_fallbacks": b["thinking_fallbacks"]}


def qc_stats(work) -> dict:
    overall, per_app = _bucket(), defaultdict(_bucket)
    for rec in iter_records(work):
        _add(overall, rec)
        _add(per_app[rec["app"]], rec)
    return {"overall": _finish(overall), "per_app": {a: _finish(b) for a, b in sorted(per_app.items())}}


def write_qc(work, out_dir, data_root, n=100, seed=20260912, log=print) -> dict:
    out = Path(out_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    stats = qc_stats(work)
    (out / "qc_stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    rendered = 0
    for rec in sample_records(work, n, seed):
        src = Path(data_root) / rec["app"] / rec["frame"]
        if not src.exists():
            continue
        render_overlay(rec, Image.open(src).convert("RGB")).save(
            out / "overlays" / f"{rec['app']}_{rec['frame_index']:03d}_{rec['status']}.png")
        rendered += 1
    log(f"QC overlays={rendered} stats={out / 'qc_stats.json'}")
    return stats


def review_lines(rec) -> list[str]:
    """Header, then F/V/S for a selected candidate - or why nothing was selected."""
    sel = rec.get("selected")
    head = f"{rec['app']} / {rec['frame']}   status {rec.get('status') or 'incomplete'}"
    if sel is None:
        why = (rec.get("propose") or {}).get("rejected_reason")
        why = why or "; ".join(f"candidate {i}: {c.get('status')} at {c.get('status_stage')} ({c.get('status_reason')})"
                               for i, c in enumerate(rec.get("candidates", []))) or "no candidates"
        return [head, "no annotation: " + str(why)]
    final = rec["candidates"][sel]["final"]
    head += f"   tier {final['tier']}   verdict {final.get('verdict')}   agreement {final['agreement']}"
    ins = final["instructions"]
    return [head, "F: " + ins["functional"], "V: " + ins["visual"], "S: " + ins["spatial"]]


def render_review(rec, frame) -> Image.Image:
    """The batch review image: the frame with only the final box, and the instructions large underneath."""
    rows = [row for line in review_lines(rec) for row in textwrap.wrap(line, REVIEW_WRAP) or [""]]
    step = REVIEW_FONT + 12
    canvas = Image.new("RGB", (frame.width, frame.height + 16 + step * len(rows)), STRIP_BG)
    canvas.paste(frame, (0, 0))
    d = ImageDraw.Draw(canvas)
    if rec.get("selected") is not None:
        fb = rec["candidates"][rec["selected"]]["final"]["bbox_px"]
        d.rectangle((fb[0], fb[1], fb[2] - 1, fb[3] - 1), outline=REVIEW_COLOUR, width=4)
    font = _font(REVIEW_FONT)
    y = frame.height + 10
    for row in rows:
        d.text((12, y), row, fill=(240, 240, 240), font=font)
        y += step
    return canvas


def write_batch_report(work, data_root, out_dir, batch, seed, apps=None, log=print) -> dict:
    """One review PNG per app for batch k, under <out_dir>/batch_<k>/; a frame the batch picked but
    that has no record yet is still drawn, marked missing."""
    out = Path(out_dir) / f"batch_{batch:03d}"
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for app in apps or list_apps(data_root):
        fr = batch_frame(data_root, app, batch, seed)
        if fr is None:
            continue
        path = record_path(work, app, fr.index)
        rec = load_record(path) if Path(path).exists() else {
            "app": app, "frame": fr.name, "frame_index": fr.index, "status": "missing", "selected": None,
            "candidates": [], "propose": {"rejected_reason": "not annotated yet"}}
        status = rec.get("status") or "incomplete"
        render_review(rec, Image.open(fr.path).convert("RGB")).save(out / f"{app}_{fr.index:03d}_{status}.png")
        counts["frames"] += 1
        counts[status] += 1
    log(f"BATCH_REPORT batch={batch} frames={counts['frames']} "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "frames") + f" dir={out}")
    return dict(counts)
