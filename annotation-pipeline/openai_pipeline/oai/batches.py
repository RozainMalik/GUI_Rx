"""The batch plan and its state machine: work/batches.json, the seeded corpus
order spread across apps, and every refusal - plain functions over the plan dict. A plan is
extended, never reshuffled: each frame lands in exactly one batch and stays there."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from annotator.frames import list_apps, list_frames

STATES = ("planned", "running", "done", "graded", "approved", "stopped")
TRANSITIONS = {"planned": ("running", "stopped"), "running": ("running", "done", "stopped"),
               "done": ("graded", "stopped"), "graded": ("graded", "approved", "stopped"),
               "approved": (), "stopped": ("running",)}


class PlanError(RuntimeError):
    """A refusal: printed as a marker by the CLI, exit 2."""


@dataclass
class Batch:
    id: int
    size: int
    frames: list                        # [[app, frame name, index], ...]
    state: str = "planned"
    cost_usd: float = 0.0
    usd_per_frame: float | None = None
    correct_rate: float | None = None
    graded_rows: int = 0
    config_hash: str = ""
    prompt_hash: str = ""
    models: dict = field(default_factory=dict)
    coord_mode: str = ""
    efforts: dict = field(default_factory=dict)
    stop_reason: str | None = None
    forced: bool = False
    transport: str = "sync"             # the transport of the last `run` of this batch (sync | batch)
    planned_at: str | None = None
    finished_at: str | None = None


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def app_orders(data_root, seed, skip=None) -> dict[str, list[list]]:
    """Each app's frames permuted with the seed, apps in name order. `skip(app)` drops whole apps;
    the rest keep their permutation, so switching it on between batches never reshuffles anything."""
    orders = {}
    for app in list_apps(data_root):
        if skip and skip(app):
            continue
        frames = [[fr.app, fr.name, fr.index] for fr in list_frames(data_root, app)]
        random.Random(f"{seed}:{app}").shuffle(frames)
        orders[app] = frames
    return orders


def interleave(orders: dict) -> list[list]:
    """The apps visited round-robin by position, so any prefix is a draw spread over every app."""
    out = []
    for i in range(max((len(f) for f in orders.values()), default=0)):
        for frames in orders.values():
            if i < len(frames):
                out.append(frames[i])
    return out


def corpus_order(data_root, seed, skip=None) -> list[list]:
    """Every frame once, round-robin over apps by absolute position."""
    return interleave(app_orders(data_root, seed, skip))


def load_plan(path, seed) -> dict:
    path = Path(path)
    if not path.exists():
        return {"version": 1, "seed": seed, "batches": []}
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("seed") != seed:
        raise PlanError(f"{path} was planned with seed {plan.get('seed')} but config says {seed}; the order is never reshuffled")
    return plan


def save_plan(path, plan) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def batches(plan) -> list[Batch]:
    return [Batch(**b) for b in plan["batches"]]


def get_batch(plan, batch_id) -> Batch:
    for b in plan["batches"]:
        if b["id"] == batch_id:
            return Batch(**b)
    raise PlanError(f"no batch {batch_id}; run `plan` first")


def put_batch(plan, batch: Batch) -> None:
    for i, b in enumerate(plan["batches"]):
        if b["id"] == batch.id:
            plan["batches"][i] = asdict(batch)
            return
    plan["batches"].append(asdict(batch))


def used_frames(plan) -> set:
    return {(app, name) for b in plan["batches"] for app, name, _ in b["frames"]}


def next_size(plan, sizes) -> int:
    n = len(plan["batches"])
    return int(sizes[n] if n < len(sizes) else sizes[-1])


def plan_batch(plan, order, size) -> Batch:
    used = used_frames(plan)
    free = [f for f in order if (f[0], f[1]) not in used]
    if not free:
        raise PlanError("every frame is already in a batch")
    frames = [list(f) for f in free[:size]]
    batch = Batch(id=len(plan["batches"]) + 1, size=len(frames), frames=frames, planned_at=now())
    plan["batches"].append(asdict(batch))
    return batch


def plan_batch_balanced(plan, orders: dict, size: int) -> Batch:
    """The next batch as an equal share of every app's *remaining* frames: each
    app's unplanned frames in their seeded order, interleaved round-robin, first `size` taken."""
    used = used_frames(plan)
    free = {app: [f for f in frames if (f[0], f[1]) not in used] for app, frames in orders.items()}
    pool = interleave(free)
    if not pool:
        raise PlanError("every frame is already in a batch")
    frames = [list(f) for f in pool[:size]]
    batch = Batch(id=len(plan["batches"]) + 1, size=len(frames), frames=frames, planned_at=now())
    plan["batches"].append(asdict(batch))
    return batch


def transition(batch: Batch, new_state) -> Batch:
    if new_state not in TRANSITIONS[batch.state]:
        raise PlanError(f"batch {batch.id} is {batch.state}; cannot become {new_state}")
    batch.state = new_state
    return batch


def check_runnable(plan, batch_id, after_stop=False) -> Batch:
    batch = get_batch(plan, batch_id)
    if "running" not in TRANSITIONS[batch.state]:
        raise PlanError(f"batch {batch_id} is {batch.state}; cannot become running")
    if batch_id > 1:
        prev = get_batch(plan, batch_id - 1)
        if prev.state == "stopped" and not after_stop:
            raise PlanError(f"batch {prev.id} was stopped ({prev.stop_reason}); pass --after-stop to run past it")
        if prev.state not in ("approved", "stopped"):
            raise PlanError(f"batch {prev.id} is {prev.state}, not approved")
    return batch


def check_approvable(batch: Batch, gate, force=False, ungraded=False) -> None:
    if ungraded:                                   # the caller accepts the batch on cost alone (no correct rate)
        if batch.state != "done":
            raise PlanError(f"batch {batch.id} is {batch.state}; run it to completion first")
        return
    if batch.state != "graded":
        raise PlanError(f"batch {batch.id} is {batch.state}; grade it first")
    if batch.correct_rate is None or not batch.graded_rows:
        raise PlanError(f"batch {batch.id} has no graded rows")
    if batch.correct_rate < gate and not force:
        raise PlanError(f"batch {batch.id} correct_rate={batch.correct_rate:.3f} < gate {gate:.2f}; pass --force to approve anyway")


def approved(plan) -> list[Batch]:
    return [b for b in batches(plan) if b.state == "approved"]


def last_measured(plan) -> Batch | None:
    measured = [b for b in approved(plan) if b.usd_per_frame]
    return measured[-1] if measured else None
