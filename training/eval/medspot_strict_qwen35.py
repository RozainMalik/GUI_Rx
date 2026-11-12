"""MedSPOT for a GUI-AIMA Qwen3.5 checkpoint, scored exactly like every other MedSPOT model.

Protocol = medspot_strict.core (the MedSPOT strict scorer, unchanged): every step whose action has a bbox is a
target whatever its type, every step is predicted (no early stop), top-1 point. Two hit rules on the same predictions:
strict (point inside the box; the top-level metrics, same keys as the other models' summary.json) and tol14px (the
legacy +-14 px patch rule, under "tol14px").
Resumable: finished steps live in <out_dir>/predictions.jsonl and are skipped on rerun.
Usage: python eval/medspot_strict_qwen35.py --ckpt runs/.../final_bf16 --name my-model-1step --out_dir DIR [--two_step]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from attn_ft.medspot_metrics import both_rules
from gui_aima.qwen35.inference import ground, ground_two_step, load_pointer_model
from medspot_strict.core import PredictionLog, is_hit, load_tasks, make_record

RULE = ("strict point-in-box (no pixel tolerance, top-1 only); every step with a bbox action "
        "is scored regardless of action type; all steps predicted (no early stop)")
TOL_RULE = "+-14 px: 28x28 px patch centred on the point overlaps the GT box (original pixels), same steps"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--data", default="data/medspot")
ap.add_argument("--two_step", action="store_true")
ap.add_argument("--max_pixels", type=int, default=5860400)
ap.add_argument("--limit_tasks", type=int, default=0, help="first N tasks per app (smoke)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
tasks = load_tasks(Path(a.data, "Annotations"), a.limit_tasks)
steps = [s for t in tasks for s in t.steps]
missing = [s.image_path for s in steps if not Path(a.data, s.image_path).exists()]
if missing:
    sys.exit(f"{len(missing)} images missing, e.g. {missing[0]}")
log = PredictionLog(out / "predictions.jsonl")
todo = [s for s in steps if s.key not in log.load()]
print(f"[{a.name}] {len(tasks)} tasks, {len(steps)} evaluable steps, {len(todo)} to run -> {out}", flush=True)
if todo:
    model, proc = load_pointer_model(a.ckpt, None, a.device)
    fn = ground_two_step if a.two_step else ground
    t0 = time.time()
    for i, s in enumerate(todo, start=1):
        img = Image.open(Path(a.data, s.image_path)).convert("RGB")
        p = fn(model, proc, img, s.instruction, a.max_pixels)["point"]
        p = None if p is None else (float(p[0]), float(p[1]))
        rec = make_record(s, p, "")
        rec["image_size"] = list(img.size)
        log.append(rec)
        print(f"  [{i}/{len(todo)}] {s.key} {s.action_type:12s} {'HIT ' if is_hit(p, s.gt_bbox) else 'miss'} "
              f"pred={p} ({i / max(time.time() - t0, 1e-9) * 60:.1f} steps/min)", flush=True)

rules = both_rules(tasks, log.load())
summary = {"model": a.name, "hf_id": "", "weights": a.ckpt, "rule": RULE, "limit_tasks": a.limit_tasks,
           "legacy_rule": TOL_RULE + " (under tol14px)", "legacy_TCA": rules["tol14px"]["TCA (%)"],
           "deviation": "", "max_pixels": a.max_pixels, "two_step": a.two_step,
           **rules["strict"], "tol14px": rules["tol14px"]}
json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), indent=2)
for rule in ("strict", "tol14px"):
    m = rules[rule]
    print(f"{rule:8s} TCA {m['TCA (%)']:.2f}  SHR {m['SHR (%)']:.2f}  S1A {m['step1_accuracy (%)']:.2f}  "
          f"prefix-SHR {m['prefix_SHR (%)']:.2f}  ({m['total_tasks']} tasks / {m['evaluable_steps']} steps)")
sys.exit(0 if summary["complete_run"] else 3)
