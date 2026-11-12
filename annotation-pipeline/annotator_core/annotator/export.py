"""Records -> MedSPOT tasks/steps/actions. image_path is
posixpath-joined from --image-root so Git Bash on Windows can never rewrite it. A record with
review.verdict == "rejected" ships nothing - the one human gate; "corrected" ships the human's box."""
from __future__ import annotations

import json
import posixpath
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import PIPELINE_VERSION
from .frames import BENCHMARK_APPS, is_benchmark_app
from .geometry import px_to_pct_xywh
from .validators import VARIANTS


@dataclass
class ShippedRow:
    app: str
    frame: str
    frame_index: int
    tier: str
    bbox_px: list
    image_size: list
    instructions: dict
    element_name: str
    element_kind: str
    prompt_hash: str
    record_path: str


def iter_records(work):
    root = Path(work) / "records"
    if not root.exists():
        return
    for app_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(app_dir.glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            rec["_path"] = str(path)
            yield rec


def shipped_rows(work, tiers=("A", "B"), include_benchmark=False) -> list[ShippedRow]:
    rows = []
    for rec in iter_records(work):
        if rec.get("selected") is None:
            continue
        review = rec.get("review") or {}
        if review.get("verdict") == "rejected":
            continue
        if not include_benchmark and is_benchmark_app(rec["app"]):
            continue
        cand = rec["candidates"][rec["selected"]]
        final = cand.get("final")
        if not final or final["tier"] not in tiers:
            continue
        corrected = review.get("verdict") == "corrected" and review.get("bbox_px")
        box = review["bbox_px"] if corrected else final["bbox_px"]
        rows.append(ShippedRow(rec["app"], rec["frame"], rec["frame_index"], final["tier"], list(box),
                               list(rec["image_size"]), dict(final["instructions"]),
                               cand["element"]["element_name"], cand["element"]["element_kind"],
                               (rec.get("provenance") or {}).get("prompt_hash", ""), rec["_path"]))
    return rows


def snake_case(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def target_name(row: ShippedRow) -> str:
    return snake_case(row.element_name) or snake_case(row.element_kind) or "target"


def to_tasks(rows, image_root, flat=False) -> list[dict]:
    """Three tasks per frame - one per phrasing - sharing the box; one step each, step_id 1."""
    tasks = []
    for row in rows:
        for variant in VARIANTS:
            instr = row.instructions[variant]
            step = {"step_id": 1, "image_path": posixpath.join(image_root, row.app, row.frame), "instruction": instr,
                    "actions": [{"type": "click", "target": target_name(row),
                                 "bbox": px_to_pct_xywh(tuple(row.bbox_px), *row.image_size)}]}
            tasks.append({"task_overview": instr, **step} if flat else {"task_overview": instr, "steps": [step]})
    return tasks


def manifest(rows, image_root, flat, include_benchmark, tiers, dataset="medspot") -> dict:
    per_app, per_tier = {}, {}
    for r in rows:
        per_app[r.app] = per_app.get(r.app, 0) + 1
        per_tier[r.tier] = per_tier.get(r.tier, 0) + 1
    hashes = sorted({r.prompt_hash for r in rows})
    return {"dataset": dataset, "format": "medspot", "shape": "flat" if flat else "nested", "bbox_format": "percent_xywh_0_100",
            "image_root": image_root, "tiers": list(tiers), "frames": len(rows), "tasks": 3 * len(rows),
            "per_app": dict(sorted(per_app.items())), "per_tier": per_tier, "pipeline_version": PIPELINE_VERSION,
            "prompt_hash": hashes[0] if len(hashes) == 1 else hashes, "contaminated": bool(include_benchmark),
            "benchmark_apps": list(BENCHMARK_APPS) if include_benchmark else [],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def write_export(work, out_dir, image_root, tiers=("A", "B"), include_benchmark=False, flat=False, log=print,
                 dataset="medspot") -> dict:
    """`dataset` names the set being built (file stem and manifest["dataset"]); "format" stays the
    MedSPOT task/step/action schema whatever the set is called."""
    image_root = str(image_root).replace("\\", "/")
    rows = shipped_rows(work, tiers, include_benchmark)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{dataset}_train_flat.json" if flat else f"{dataset}_train.json"
    (out / name).write_text(json.dumps({"tasks": to_tasks(rows, image_root, flat)}, indent=1, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    m = manifest(rows, image_root, flat, include_benchmark, tiers, dataset)
    (out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    if include_benchmark:
        log(f"WARNING: {out} includes the benchmark apps ({', '.join(BENCHMARK_APPS)}) - manifest says "
            f"contaminated: true; never evaluate on MedSPOT with a model trained on it")
    log(f"EXPORTED {out / name} frames={len(rows)} tasks={3 * len(rows)} tiers={''.join(tiers)}")
    return m
