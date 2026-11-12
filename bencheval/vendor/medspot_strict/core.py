"""Strict MedSPOT grounding scorer.

Scoring rule (the only one in this package):
  - Every step whose action carries a bbox is a grounding target, whatever the
    action type (click, double_click, drag, hover, type, ...). The first such
    action in the step is the ground truth. Steps with no bbox are skipped.
  - A step is a hit iff the predicted point lies inside the GT box
    (x1 <= px <= x2 and y1 <= py <= y2, normalised 0-1 space).
    No pixel tolerance, no patch overlap, no top-k, no clamping.
  - Every step of every task is predicted (no early stop).
  - TCA: a task is complete iff all its evaluable steps are hits.

No torch import here, so this module runs (and is unit-tested) on any Python >= 3.8.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SMALL_TARGET_AREA = 4e-4
MEDIUM_TARGET_AREA = 3e-3


@dataclass(frozen=True)
class Step:
    key: str            # "<source_file>#<task_idx>#<step_idx>", unique and stable
    source_file: str
    task_idx: int       # index of the task inside its annotation file
    step_idx: int       # 1-based position in the task's full steps list
    image_path: str     # relative to dataset_root
    instruction: str
    action_type: str
    gt_bbox: tuple      # normalised (x1, y1, x2, y2)


@dataclass
class Task:
    source_file: str
    task_idx: int
    overview: str
    steps: list = field(default_factory=list)   # evaluable Steps only


def percent_xywh_to_norm_xyxy(b):
    x, y, w, h = (float(v) for v in b)
    return (x / 100.0, y / 100.0, (x + w) / 100.0, (y + h) / 100.0)


def bbox_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def difficulty_tag(area):
    if area < SMALL_TARGET_AREA:
        return "small"
    if area < MEDIUM_TARGET_AREA:
        return "medium"
    return "large"


def find_gt_action(step):
    """First action in the step that has a bbox, of any type; None if none."""
    return next((a for a in step.get("actions", []) if a.get("bbox")), None)


def load_tasks(data_dir, limit_tasks=0):
    """All tasks from every *.json in data_dir (sorted), each with its evaluable steps.

    limit_tasks > 0 keeps only the first N tasks of each annotation file (smoke runs).
    """
    tasks = []
    for jf in sorted(Path(data_dir).glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            raw_tasks = json.load(f).get("tasks", [])
        if limit_tasks > 0:
            raw_tasks = raw_tasks[:limit_tasks]
        for ti, t in enumerate(raw_tasks):
            task = Task(jf.name, ti, t.get("task_overview", ""))
            for si, s in enumerate(t.get("steps", []), start=1):
                a = find_gt_action(s)
                if a is None:
                    continue
                task.steps.append(Step(
                    key=f"{jf.name}#{ti}#{si}",
                    source_file=jf.name,
                    task_idx=ti,
                    step_idx=si,
                    image_path=s["image_path"],
                    instruction=s.get("instruction", ""),
                    action_type=a.get("type", ""),
                    gt_bbox=percent_xywh_to_norm_xyxy(a["bbox"]),
                ))
            tasks.append(task)
    return tasks


def is_hit(point, bbox):
    """Strict point-in-box. point is normalised (px, py) or None."""
    if point is None:
        return False
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


class PredictionLog:
    """Append-only JSONL of per-step predictions; the basis for resuming a run.

    One line per finished step. A torn last line (crash mid-write) is ignored,
    so that step is simply predicted again.
    """

    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        records = {}
        if not self.path.exists():
            return records
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[rec["key"]] = rec
        return records

    def append(self, rec):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torn = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
        with open(self.path, "a", encoding="utf-8") as f:
            if torn:                 # isolate a torn line so this record parses
                f.write("\n")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())


def make_record(step, pred_point, raw_output):
    return {
        "key": step.key,
        "source_file": step.source_file,
        "task_idx": step.task_idx,
        "step_idx": step.step_idx,
        "action_type": step.action_type,
        "instruction": step.instruction,
        "image_path": step.image_path,
        "gt_bbox": list(step.gt_bbox),
        "pred_point": list(pred_point) if pred_point is not None else None,
        "raw_output": raw_output,
    }


def _pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def compute_metrics(tasks, records):
    """Score tasks against logged predictions. Hits are recomputed from
    pred_point and the GT box, never read back from the log."""
    scored = [t for t in tasks if t.steps]
    n_steps = sum(len(t.steps) for t in scored)

    missing = 0
    hits = no_pred = completed = step1_hits = prefix_hits = 0
    diff_tot, diff_hit = defaultdict(int), defaultdict(int)
    act_tot, act_hit = defaultdict(int), defaultdict(int)
    app_tot, app_done = defaultdict(int), defaultdict(int)

    for t in scored:
        step_hits = []
        for s in t.steps:
            rec = records.get(s.key)
            if rec is None:
                missing += 1
                step_hits.append(False)
                continue
            pred = rec.get("pred_point")
            h = is_hit(tuple(pred) if pred is not None else None, s.gt_bbox)
            step_hits.append(h)
            hits += h
            no_pred += pred is None
            d = difficulty_tag(bbox_area(s.gt_bbox))
            diff_tot[d] += 1
            diff_hit[d] += h
            act_tot[s.action_type] += 1
            act_hit[s.action_type] += h

        done = all(step_hits)
        completed += done
        step1_hits += step_hits[0]
        for h in step_hits:          # hits before the first miss (legacy early-stop SHR)
            if not h:
                break
            prefix_hits += 1
        app = t.source_file.rsplit(".", 1)[0]
        app_tot[app] += 1
        app_done[app] += done

    return {
        "complete_run": missing == 0,
        "missing_steps": missing,
        "total_tasks": len(scored),
        "tasks_without_evaluable_steps": len(tasks) - len(scored),
        "evaluable_steps": n_steps,
        "TCA (%)": _pct(completed, len(scored)),
        "completed_tasks": completed,
        "SHR (%)": _pct(hits, n_steps),
        "hit_steps": hits,
        "step1_accuracy (%)": _pct(step1_hits, len(scored)),
        "prefix_SHR (%)": _pct(prefix_hits, n_steps),
        "no_prediction_steps": no_pred,
        "SHR_by_difficulty (%)": {d: _pct(diff_hit[d], diff_tot[d]) for d in sorted(diff_tot)},
        "SHR_by_action_type (%)": {a: _pct(act_hit[a], act_tot[a]) for a in sorted(act_tot)},
        "steps_by_action_type": dict(sorted(act_tot.items())),
        "TCA_by_app (%)": {a: _pct(app_done[a], app_tot[a]) for a in sorted(app_tot)},
    }
