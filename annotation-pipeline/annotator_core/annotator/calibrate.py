"""The two gates that run before any training frame is spent: the resolution
probe (one native frame must cost >= 1800 prompt tokens, else RESOLUTION_FAIL) and 60 benchmark
ground-truth steps grounded blind at >= 80% centre hits. Writes work/calibration.json always
and work/CALIBRATION_OK only on success - the positive marker a job script gates on."""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from .calling import Call
from .geometry import center, contains_point, pct_xywh_to_px_xyxy
from .prompts import CALIBRATE_USER, SYSTEM_PLAIN
from .stages.ground import ground_blind

MIN_PROMPT_TOKENS = 1800
REQUIRED_ACCURACY = 0.80
DEFAULT_STEPS = 60
DEFAULT_SEED = 20260912
MARKER = "CALIBRATION_OK"


@dataclass
class CalibrationResult:
    resolution_ok: bool
    prompt_tokens: int
    hits: int
    total: int
    accuracy: float
    ok: bool
    details: list = field(default_factory=list)
    coord_mode: str = "thousand"


def load_benchmark_steps(dataset_root, n=DEFAULT_STEPS, seed=DEFAULT_SEED) -> list[dict]:
    """Every click step of the MedSPOT benchmark annotations, sampled deterministically."""
    root = Path(dataset_root)
    steps = []
    for ann in sorted((root / "Annotations").glob("*.json")):
        data = json.loads(ann.read_text(encoding="utf-8"))
        for task in data.get("tasks", []):
            for step in task.get("steps", []):
                for action in step.get("actions", []):
                    if action.get("type") == "click" and action.get("bbox"):
                        steps.append({"app": ann.stem.replace("_Annotation", ""),
                                      "image_path": str(root / step["image_path"]),
                                      "instruction": step["instruction"], "bbox_pct": action["bbox"]})
                        break
    return random.Random(seed).sample(steps, min(n, len(steps)))


async def check_resolution(gen, frame, min_tokens=MIN_PROMPT_TOKENS):
    """(ok, prompt_tokens): a 1920x1080 frame at native resolution costs ~2,040 visual tokens on
    Qwen and 2,448 on OpenAI; a silently downscaled one far less. `min_tokens` is the floor."""
    reply = await gen.generate(Call(system=SYSTEM_PLAIN, user_text=CALIBRATE_USER, images=(frame,), thinking=False, kind="probe"))
    return reply.prompt_tokens >= min_tokens, reply.prompt_tokens


async def calibrate(gen, dataset_root, probe_frame, n=DEFAULT_STEPS, seed=DEFAULT_SEED,
                    min_tokens=MIN_PROMPT_TOKENS, mode="thousand") -> CalibrationResult:
    res_ok, tokens = await check_resolution(gen, probe_frame, min_tokens)
    if not res_ok:
        return CalibrationResult(False, tokens, 0, 0, 0.0, False, coord_mode=mode)
    steps = load_benchmark_steps(dataset_root, n, seed)

    async def one(step):
        img = Image.open(step["image_path"]).convert("RGB")
        w, h = img.size
        gt = pct_xywh_to_px_xyxy(step["bbox_pct"], w, h)
        g = await ground_blind(gen, img, step["instruction"], mode)
        hit = g.bbox_px is not None and contains_point(gt, center(tuple(g.bbox_px)))
        return {**step, "image_size": [w, h], "gt_px": gt, "pred_px": g.bbox_px, "hit": bool(hit), "error": g.error}

    details = await asyncio.gather(*(one(s) for s in steps))
    hits, total = sum(d["hit"] for d in details), len(details)
    acc = hits / total if total else 0.0
    return CalibrationResult(True, tokens, hits, total, acc, acc >= REQUIRED_ACCURACY, list(details), mode)


def write_result(work, result: CalibrationResult) -> Path:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "calibration.json").write_text(json.dumps(asdict(result), indent=1), encoding="utf-8")
    marker = work / MARKER
    if result.ok:
        marker.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} accuracy={result.accuracy:.3f} "
                          f"prompt_tokens={result.prompt_tokens}\n", encoding="utf-8")
    return marker
