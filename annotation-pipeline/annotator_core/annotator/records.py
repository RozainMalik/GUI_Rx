"""One JSON record per frame, written atomically after every stage, and the
stage-level reset that lets a prompt fix replay only the stages after it."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .stages.propose import CandidateSeed

STAGES = ("propose", "describe", "ground", "verify", "refine", "select")
# Which candidate blocks each stage writes (propose writes the frame-level blocks instead).
CANDIDATE_BLOCKS = {"describe": ("describe",), "ground": ("ground",), "verify": ("consensus", "verify"),
                    "refine": ("refine",), "select": ("final",)}


def record_path(work, app, frame_index) -> Path:
    return Path(work) / "records" / app / f"{frame_index:03d}.json"


def new_record(app, frame_name, frame_index, image_size, provenance) -> dict:
    return {"app": app, "frame": frame_name, "frame_index": frame_index, "image_size": list(image_size),
            "stages_done": [], "propose": None, "candidates": [], "selected": None, "status": None,
            "review": None, "provenance": dict(provenance), "error": None}


def candidate_from_seed(seed: CandidateSeed) -> dict:
    return {"rank": seed.rank,
            "element": {"element_name": seed.element_name, "element_kind": seed.element_kind,
                        "text_on_screen": seed.text_on_screen, "icon": seed.icon, "state": seed.state},
            "seed_bbox_1000": list(seed.bbox_1000), "seed_bbox_px": list(seed.bbox_px),
            "status": "ok", "status_stage": None, "status_reason": None,
            "describe": None, "ground": None, "consensus": None, "verify": None, "refine": None, "final": None}


def seed_from_candidate(cand: dict) -> CandidateSeed:
    e = cand["element"]
    return CandidateSeed(cand["rank"], e["element_name"], e["element_kind"], e["text_on_screen"], e["icon"],
                         e["state"], list(cand["seed_bbox_1000"]), tuple(cand["seed_bbox_px"]))


def load_record(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_record(path, record) -> None:
    """Write a sibling temp file and os.replace() it, so a kill mid-write leaves the previous
    record intact and never a partial file at the real path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def is_done(record, stage) -> bool:
    return stage in record["stages_done"]


def mark_done(record, stage) -> None:
    if stage not in record["stages_done"]:
        record["stages_done"].append(stage)


def set_candidate_status(cand, status, stage, reason) -> None:
    cand["status"], cand["status_stage"], cand["status_reason"] = status, stage, reason


def reset_from(record, stage) -> dict:
    """Clear `stage` and every later one. A candidate status set by a cleared stage reverts to
    ok; one set by an earlier stage (e.g. dropped at describe) stays."""
    idx = STAGES.index(stage)
    cleared = STAGES[idx:]
    record["stages_done"] = [s for s in record["stages_done"] if s not in cleared]
    if stage == "propose":
        record["propose"] = None
        record["candidates"] = []
    for cand in record["candidates"]:
        for s in cleared:
            for block in CANDIDATE_BLOCKS.get(s, ()):
                cand[block] = None
        if cand.get("status_stage") in cleared:
            set_candidate_status(cand, "ok", None, None)
    record["selected"], record["status"], record["error"] = None, None, None
    return record
