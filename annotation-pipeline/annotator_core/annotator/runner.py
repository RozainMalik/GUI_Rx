"""Walk an app's frames: stage-level resume, per-candidate fan-out, asyncio concurrency and
atomic record writes. Every model call goes through one Throttled generator so at
most `concurrency` requests are in flight - the server's --max-num-seqs."""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from . import PIPELINE_VERSION
from .calling import TransportError
from .frames import list_frames, spread
from .prompts import prompt_hash
from .records import (STAGES, candidate_from_seed, is_done, load_record, mark_done, new_record, record_path,
                      save_record, seed_from_candidate, set_candidate_status)
from .selection import choose, consensus, finalize, frame_status, independent_boxes, provisional_tier
from .stages import describe as s2
from .stages import ground as s3
from .stages import propose as s1
from .stages import refine as s5
from .stages import verify as s4
from .validators import VARIANTS


@dataclass
class RunConfig:
    work: Path
    data_root: Path
    app: str
    concurrency: int = 8            # requests in flight (server --max-num-seqs)
    frame_concurrency: int = 8      # frames in flight
    early_stop: bool = False        # skip refine for a frame once a candidate is provisionally tier A
    allow_mixed: bool = False
    limit: int | None = None        # evenly spread subset (the pilot)
    indices: list | None = None
    run_id: str = ""
    model: str = "Qwen3.5-9B"
    coord_mode: str = "thousand"    # the space every model box is read in (geometry.COORD_MODES)
    generators: dict | None = None  # Call.kind -> Generator for kinds that use another model; the rest use the default
    models: dict | None = None      # kind -> model id, written to provenance when a run mixes models
    provenance_extra: dict | None = None   # a provider's own provenance keys (openai: provider, effort, batch)


class Routed:
    """One generator per Call.kind; a kind not in the table goes to the default. Only the default
    is bounded by run_app's Throttled - a per-kind generator bounds itself."""

    def __init__(self, default, by_kind):
        self.default, self.by_kind = default, dict(by_kind)

    async def generate(self, call):
        return await self.by_kind.get(call.kind, self.default).generate(call)


class Throttled:
    def __init__(self, gen, n):
        self.gen = gen
        self.sem = asyncio.Semaphore(n)
        self.calls = 0

    async def generate(self, call):
        async with self.sem:
            self.calls += 1
            return await self.gen.generate(call)


SERVER_DOWN_AFTER = 5      # consecutive transport failures before we stop dispatching


class MixedPromptError(RuntimeError):
    """A record was produced with different prompts; adding stages would mix them."""


def provenance(cfg) -> dict:
    out = {"model": cfg.model, "pipeline_version": PIPELINE_VERSION, "prompt_hash": prompt_hash(cfg.coord_mode),
           "run_id": cfg.run_id, "thinking_fallbacks": 0, "coord_mode": cfg.coord_mode}
    if cfg.models:
        out["models"] = dict(cfg.models)
    out.update(cfg.provenance_extra or {})
    return out


def select_frames(cfg) -> list:
    frames = list_frames(cfg.data_root, cfg.app)
    if cfg.indices:
        wanted = set(cfg.indices)
        frames = [f for f in frames if f.index in wanted]
    if cfg.limit:
        frames = spread(frames, cfg.limit)
    return frames


def check_prompt_hash(cfg, frames) -> None:
    current = prompt_hash(cfg.coord_mode)
    mixed = 0
    for fr in frames:
        rec = load_record(record_path(cfg.work, fr.app, fr.index))
        if rec and rec["provenance"].get("prompt_hash") != current and not all(is_done(rec, s) for s in STAGES):
            mixed += 1
    if mixed and not cfg.allow_mixed:
        raise MixedPromptError(f"{mixed} unfinished record(s) of {cfg.app} were produced with different prompts; "
                               f"run `reset` on them or pass --allow-mixed")


def _ok(cands):
    return [c for c in cands if c["status"] == "ok"]


async def process_frame(gen, cfg, frame) -> dict:
    """Run every stage not yet in stages_done, saving after each. Within a stage, only the
    candidates without that stage's block are run, so a parse-only replay resumes cleanly."""
    if cfg.generators:
        gen = Routed(gen, cfg.generators)
    path = record_path(cfg.work, frame.app, frame.index)
    rec = load_record(path) or new_record(frame.app, frame.name, frame.index, (0, 0), provenance(cfg))
    if all(is_done(rec, s) for s in STAGES):
        return rec
    image = Image.open(frame.path).convert("RGB")
    width, height = image.size
    rec["image_size"] = [width, height]
    fallbacks = 0

    if not is_done(rec, "propose"):
        p = await s1.propose(gen, image, frame.app, cfg.coord_mode)
        fallbacks += int(p.thinking_fallback)
        rec["propose"] = {"intent": p.intent, "focus_region": p.focus_region, "raw": p.raw,
                          "thinking_chars": p.thinking_chars, "calls": p.calls,
                          "thinking_fallback": p.thinking_fallback, "rejected_reason": p.rejected_reason}
        rec["candidates"] = [candidate_from_seed(s) for s in p.candidates]
        if p.rejected_reason:                       # final: not retried on rerun
            rec["stages_done"] = list(STAGES)
            rec["status"] = "abstained"
            rec["provenance"]["thinking_fallbacks"] += fallbacks
            save_record(path, rec)
            return rec
        mark_done(rec, "propose")
        save_record(path, rec)

    region = rec["propose"]["focus_region"]
    region_box, region_kind = tuple(region["bbox_px"]), region["kind"]

    if not is_done(rec, "describe"):
        todo = [c for c in _ok(rec["candidates"]) if c["describe"] is None]
        results = await asyncio.gather(*(
            s2.describe(gen, image, frame.app, region_kind, region_box, seed_from_candidate(c), mode=cfg.coord_mode) for c in todo))
        for c, d in zip(todo, results):
            fallbacks += int(d.thinking_fallback)
            c["describe"] = asdict(d)
            if d.error:
                set_candidate_status(c, "unreadable", "describe", d.error)
            elif not d.present:
                set_candidate_status(c, "dropped", "describe", "not_in_region")
        mark_done(rec, "describe")
        save_record(path, rec)

    if not is_done(rec, "ground"):
        todo = [c for c in _ok(rec["candidates"]) if c["ground"] is None]

        async def ground_all(c):
            gs = await asyncio.gather(*(s3.ground_blind(gen, image, c["describe"]["instructions"][v], cfg.coord_mode) for v in VARIANTS))
            return {v: asdict(g) for v, g in zip(VARIANTS, gs)}
        results = await asyncio.gather(*(ground_all(c) for c in todo))
        for c, g in zip(todo, results):
            c["ground"] = g
        mark_done(rec, "ground")
        save_record(path, rec)

    if not is_done(rec, "verify"):
        todo = [c for c in _ok(rec["candidates"]) if c["verify"] is None and c["consensus"] is None]

        async def verify_one(c):
            cons = consensus(independent_boxes(c))
            if cons is None:                        # no sane box yet: the candidate goes to refine
                return None, None
            box, cluster = cons
            v = await s4.verify(gen, image, box, c["describe"]["instructions"], c["describe"]["text_on_screen"], cfg.coord_mode)
            return {"bbox_px": list(box), "agreement": len(cluster) / 4, "cluster": cluster}, asdict(v)
        results = await asyncio.gather(*(verify_one(c) for c in todo))
        for c, (cons, v) in zip(todo, results):
            c["consensus"], c["verify"] = cons, v
        mark_done(rec, "verify")
        save_record(path, rec)

    if not is_done(rec, "refine"):
        tiers = {id(c): provisional_tier(c) for c in _ok(rec["candidates"])}
        skip_all = cfg.early_stop and "A" in tiers.values()
        todo = [c for c in _ok(rec["candidates"]) if c["refine"] is None and tiers[id(c)] != "A" and not skip_all]
        results = await asyncio.gather(*(
            s5.refine(gen, image, frame.app, region_kind, seed_from_candidate(c), c["describe"], c["ground"],
                      c["verify"], c["consensus"], cfg.coord_mode) for c in todo))
        for c, r in zip(todo, results):
            c["refine"] = asdict(r)
            if r.rewrite:
                fallbacks += int(r.rewrite.get("thinking_fallback", False))
            if r.final_bbox_px is None and c["consensus"] is None:
                set_candidate_status(c, "unreadable", "refine", "no_box")
        mark_done(rec, "refine")
        save_record(path, rec)

    if not is_done(rec, "select"):
        for c in rec["candidates"]:
            c["final"] = finalize(c, width, height) if c["status"] == "ok" else None
            if c["status"] == "ok" and c["final"] is None:
                set_candidate_status(c, "unreadable", "select", "no_box")
        rec["selected"] = choose(rec["candidates"])
        rec["status"] = frame_status(rec)
        rec["provenance"]["thinking_fallbacks"] += fallbacks
        mark_done(rec, "select")
        save_record(path, rec)
    return rec


async def run_app(gen, cfg, log=print) -> dict:
    """Every frame of one app, `frame_concurrency` at a time, `concurrency` requests in flight.
    Returns counts by outcome (shipped/lowconf/abstained/transport_error/error/skipped/frames/
    calls) and prints the RECORDS_WRITTEN marker job scripts gate on. A circuit breaker trips
    after SERVER_DOWN_AFTER consecutive transport failures (the counter resets whenever a frame
    completes its try body): every later frame is then counted `skipped` without dispatching a
    single call, so a dead server does not burn hours of walltime for nothing."""
    frames = select_frames(cfg)
    check_prompt_hash(cfg, frames)
    throttled = Throttled(gen, cfg.concurrency)
    frame_sem = asyncio.Semaphore(cfg.frame_concurrency)
    counts = Counter()
    consecutive_failures = 0
    server_down = False

    async def one(fr):
        nonlocal consecutive_failures, server_down
        async with frame_sem:
            if server_down:
                counts["skipped"] += 1
                return
            try:
                rec = await process_frame(throttled, cfg, fr)
                counts[rec["status"] or "incomplete"] += 1
                consecutive_failures = 0
            except TransportError as e:
                counts["transport_error"] += 1
                consecutive_failures += 1
                log(f"{fr.app}/{fr.name}: transport failure, stage left undone: {e}")
                if consecutive_failures >= SERVER_DOWN_AFTER:
                    server_down = True
            except Exception as e:                    # a bug in one frame must not end a 3-day run
                counts["error"] += 1
                log(f"{fr.app}/{fr.name}: {type(e).__name__}: {e}")

    await asyncio.gather(*(one(fr) for fr in frames))
    counts["frames"] = len(frames)
    counts["calls"] = throttled.calls
    if server_down:
        log(f"SERVER_DOWN after {SERVER_DOWN_AFTER} consecutive transport failures; {counts['skipped']} frames skipped")
    log(f"RECORDS_WRITTEN {cfg.app} frames={len(frames)} "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k not in ("frames",)))
    return dict(counts)
