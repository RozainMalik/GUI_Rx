"""One batch through the shared runner: its frames in `frame_concurrency` flights,
every call priced by the meter, the shared circuit breaker, budget/anomaly stops that leave the
frame in flight resumable, then overlays, batch stats and the BATCH_* marker. `run_app` is not
used because a batch spans apps; this is its per-frame loop over an explicit frame list."""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from annotator.calling import TransportError
from annotator.frames import Frame
from annotator.prompts import prompt_hash
from annotator.qc import render_overlay
from annotator.records import STAGES, is_done, load_record, record_path
from annotator.runner import SERVER_DOWN_AFTER, RunConfig, process_frame

from .batches import Batch, now, transition
from .config import Config, config_hash
from .meter import AnomalyStop, BudgetExceeded, Meter, Metered

STATUSES = ("shipped", "lowconf", "abstained", "unreadable")
RATE_LIMIT_HINT_FRACTION = 0.2


def run_config(cfg: Config, batch: Batch) -> RunConfig:
    return RunConfig(work=cfg.work, data_root=cfg.data, app="*", concurrency=cfg.concurrency,
                     frame_concurrency=cfg.frame_concurrency, early_stop=cfg.early_stop, run_id=f"batch{batch.id}",
                     model=cfg.models["ground"], coord_mode=cfg.coord_mode, models=dict(cfg.models),
                     provenance_extra={"provider": "openai", "effort": dict(cfg.efforts), "batch": batch.id})


def batch_frames(cfg: Config, batch: Batch) -> list[Frame]:
    return [Frame(app, name, index, Path(cfg.data) / app / name) for app, name, index in batch.frames]


def _complete(rec) -> bool:
    return bool(rec) and all(is_done(rec, s) for s in STAGES)


def remaining(cfg: Config, batch: Batch) -> list[Frame]:
    return [fr for fr in batch_frames(cfg, batch) if not _complete(load_record(record_path(cfg.work, fr.app, fr.index)))]


def batch_records(cfg: Config, batch: Batch) -> list[dict]:
    recs = (load_record(record_path(cfg.work, app, index)) for app, _, index in batch.frames)
    return [r for r in recs if r]


async def run_batch(cfg: Config, batch: Batch, gen, meter: Meter, log=print) -> Batch:
    transition(batch, "running")
    frames = batch_frames(cfg, batch)
    todo = remaining(cfg, batch)
    rcfg = run_config(cfg, batch)
    sem = asyncio.Semaphore(cfg.frame_concurrency)
    counts = Counter()
    consecutive = 0
    server_down = False
    stopped = None

    async def one(fr):
        nonlocal consecutive, server_down, stopped
        async with sem:
            if server_down or stopped:
                counts["skipped"] += 1
                return
            try:
                rec = await process_frame(Metered(gen, meter, fr.app, fr.name), rcfg, fr)
                meter.frame_done(fr.app, fr.name)
                counts[rec["status"] or "incomplete"] += 1
                consecutive = 0
            except (BudgetExceeded, AnomalyStop) as e:
                stopped = meter.stop_reason or "budget"
                counts["stopped"] += 1
                log(f"{fr.app}/{fr.name}: {e}")
            except TransportError as e:
                counts["transport_error"] += 1
                consecutive += 1
                log(f"{fr.app}/{fr.name}: transport failure, stage left undone: {e}")
                if consecutive >= SERVER_DOWN_AFTER:
                    server_down = True
            except Exception as e:                    # a bug in one frame must not end the batch
                counts["error"] += 1
                log(f"{fr.app}/{fr.name}: {type(e).__name__}: {e}")

    await asyncio.gather(*(one(fr) for fr in todo))

    records = batch_records(cfg, batch)
    complete = [r for r in records if _complete(r)]
    write_reports(cfg, batch, records, meter, gen)
    batch.cost_usd = meter.batch_total
    batch.usd_per_frame = round(batch.cost_usd / len(complete), 8) if complete else None
    batch.prompt_hash, batch.config_hash, batch.models = prompt_hash(cfg.coord_mode), config_hash(cfg), dict(cfg.models)
    batch.coord_mode, batch.efforts = cfg.coord_mode, dict(cfg.efforts)
    calls = getattr(gen, "calls", 0)
    if calls and getattr(gen, "rate_limited", 0) > RATE_LIMIT_HINT_FRACTION * calls:
        log("HINT: 429 on > 20% of calls - lower [run].concurrency")
    progress = f"frames_done={len(complete)}/{len(frames)} cost=${batch.cost_usd:.4f}"
    if stopped:
        batch.stop_reason = stopped
        log(f"BATCH_STOPPED {batch.id} reason={stopped} {progress}")
        if stopped == "budget":
            log(f"HINT: batch {batch.id} spent ${meter.batch_total:.4f} against batch cap ${meter.caps.batch_usd:.2f} - "
                f"set [caps].batch_usd above the spend to resume")
        return batch
    if server_down:
        batch.stop_reason = "transport"
        log(f"SERVER_DOWN after {SERVER_DOWN_AFTER} consecutive transport failures; {counts['skipped']} frames skipped")
        log(f"BATCH_STOPPED {batch.id} reason=transport {progress}")
        return batch
    if len(complete) < len(frames):
        log(f"BATCH_INCOMPLETE {batch.id} {progress} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return batch
    batch.finished_at = now()
    batch.stop_reason = None
    transition(batch, "done")
    by_status = Counter(r["status"] for r in complete)
    log(f"BATCH_DONE {batch.id} frames={len(frames)} " + " ".join(f"{s}={by_status.get(s, 0)}" for s in STATUSES)
        + f" cost=${batch.cost_usd:.4f} usd_per_frame={batch.usd_per_frame:.6f}")
    return batch


def write_reports(cfg: Config, batch: Batch, records, meter: Meter, gen) -> dict:
    """reports/overlays/<batch>/ for every record with a selected candidate, and the batch's stats."""
    out = Path(cfg.reports) / "overlays" / str(batch.id)
    out.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for rec in records:
        src = Path(cfg.data) / rec["app"] / rec["frame"]
        if rec.get("selected") is None or not src.exists():
            continue
        render_overlay(rec, Image.open(src).convert("RGB")).save(out / f"{rec['app']}_{rec['frame_index']:03d}_{rec['status']}.png")
        rendered += 1
    stats = {"batch": batch.id, "frames": len(records), "complete": sum(_complete(r) for r in records),
             "status": dict(Counter(r["status"] for r in records if r.get("status"))), "overlays": rendered,
             "calls": meter.calls, "refusals": getattr(gen, "refusals", 0), "rate_limited": getattr(gen, "rate_limited", 0),
             "tokens": dict(meter.tokens), "cost_usd": round(meter.batch_total, 6), "anomalous": [list(k) for k in meter.anomalous]}
    (Path(cfg.reports) / f"qc_stats.batch{batch.id}.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    return stats
