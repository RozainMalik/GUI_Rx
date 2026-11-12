"""Rounds of the batch transport: a pass's part files become Batch
jobs, jobs are polled and their output ingested once into the reply cache and the ledger, the caps
are checked before anything is submitted, and `step` is the one thing `oai run` calls. Every state
change is written before the next API call, so a killed step resumes where it was."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from annotator.prompts import prompt_hash

from .batchapi import FRAME_REF_PREFIX, PassResult, ReplyCache, load_poisoned, run_pass, save_poisoned
from .batches import Batch, now, transition
from .client import parse_output, usage_of
from .config import Config, config_hash
from .meter import Meter, rate_per_kind, read_ledger
from .pricing import estimate_usd_per_call
from .run import STATUSES, _complete, batch_records, write_reports

TERMINAL = {"completed", "expired", "failed", "cancelled"}
FRAME_REF = re.compile(r'"' + re.escape(FRAME_REF_PREFIX) + r'((?:[^"\\]|\\.)*)"')
REUPLOAD_MARGIN = 86400            # re-upload a frame whose file expires within a day


def batch_dir(cfg: Config, batch: Batch) -> Path:
    return Path(cfg.work) / "batchapi" / f"batch{batch.id}"


class FrameFiles:
    """work/batchapi/files.json: each raw frame uploaded once (purpose vision, expiring), by app/name."""

    def __init__(self, path, client, data_root, expiry_days):
        self.path, self.client, self.data_root = Path(path), client, Path(data_root)
        self.expiry_seconds = int(expiry_days) * 86400
        self.entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def file_id(self, app, name) -> str:
        key = f"{app}/{name}"
        entry = self.entries.get(key)
        if entry and entry["expires_at"] - time.time() > REUPLOAD_MARGIN:
            return entry["file_id"]
        data = (self.data_root / app / name).read_bytes()
        obj = self.client.upload(name, data, "vision", expires_seconds=self.expiry_seconds)
        self.entries[key] = {"file_id": obj["id"], "sha256": hashlib.sha256(data).hexdigest(),
                             "expires_at": int(obj.get("expires_at") or time.time() + self.expiry_seconds)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return obj["id"]


def resolve_refs(text: str, files: FrameFiles) -> str:
    """Swap every "@frame:app/name" placeholder for the uploaded file's id (uploading on first use)."""
    def sub(m):
        app, name = json.loads('"' + m.group(1) + '"').split("/", 1)
        return json.dumps(files.file_id(app, name))
    return FRAME_REF.sub(sub, text)


@dataclass
class Round:
    number: int
    batch_id: int
    parts: list                # part file names, in order
    jobs: list                 # [{part, input_file_id, id, status, requests, completed, ingested}, ...] - one per submitted part
    index: dict                # custom_id -> {key, app, frame, kind, model, effort}
    path: Path

    def save(self) -> None:
        payload = {"round": self.number, "batch": self.batch_id, "parts": self.parts, "jobs": self.jobs, "index": self.index}
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(self.path)

    @property
    def pending(self) -> bool:
        return len(self.jobs) < len(self.parts) or any(not j["ingested"] for j in self.jobs)


def latest_round(bdir) -> Round | None:
    files = sorted(Path(bdir).glob("round*.json"), key=lambda p: int(re.search(r"round(\d+)\.json$", p.name).group(1)))
    if not files:
        return None
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    return Round(d["round"], d["batch"], d["parts"], d["jobs"], d["index"], files[-1])


def new_round(bdir, batch_id, number, result: PassResult) -> Round:
    rnd = Round(number, batch_id, [Path(p).name for p in result.parts], [], result.index, Path(bdir) / f"round{number}.json")
    rnd.save()
    return rnd


def submit_missing(rnd: Round, client, files: FrameFiles, log=print) -> int:
    """Upload and create a job for every part without one; the round file is rewritten after each."""
    created = 0
    for part in rnd.parts[len(rnd.jobs):]:
        text = (rnd.path.parent / part).read_text(encoding="utf-8")
        payload = resolve_refs(text, files).encode("utf-8")
        fid = client.upload(part, payload, "batch")["id"]
        job = client.create_batch(fid, {"batch": str(rnd.batch_id), "round": str(rnd.number), "part": part})
        rnd.jobs.append({"part": part, "input_file_id": fid, "id": job["id"], "status": job.get("status", "validating"),
                         "requests": text.count("\n"), "completed": 0, "ingested": False})
        rnd.save()
        created += 1
        log(f"JOB_CREATED batch={rnd.batch_id} round={rnd.number} part={part} job={job['id']} requests={text.count(chr(10))}")
    return created


def apply_line(line: dict, rnd: Round, cache: ReplyCache, meter: Meter, job_id: str) -> str | None:
    """One output/error line -> the cache and (for a reply) one ledger row. Idempotent per key."""
    entry = rnd.index.get(line.get("custom_id"))
    if entry is None:
        return None
    key = entry["key"]
    resp = line.get("response") or {}
    body = resp.get("body") if isinstance(resp.get("body"), dict) else {}
    if line.get("error") or resp.get("status_code") != 200:
        err = line.get("error") or body.get("error") or {}
        msg = str(err.get("message") if isinstance(err, dict) else err) or f"status {resp.get('status_code')}"
        return "error" if cache.add_error(key, msg[:300], job_id) else None
    if key in cache.replies:
        return None
    text, refused = parse_output(body)
    usage = usage_of(body, entry["model"], entry["effort"], 0)
    cache.add_reply(key, text, refused, usage)
    meter.record(app=entry["app"], frame=entry["frame"], kind=entry["kind"], usage=usage, transport="batch", job=job_id)
    return "reply"


def poll(rnd: Round, client, cache: ReplyCache, meter: Meter, log=print) -> str:
    """Poll every job not yet ingested; ingest the terminal ones. 'failed' wins over 'pending'."""
    failed = None
    for job in rnd.jobs:
        if job["ingested"]:
            continue
        obj = client.get_batch(job["id"])
        counts = obj.get("request_counts") or {}
        job["status"], job["completed"] = obj["status"], int(counts.get("completed") or 0)
        if obj["status"] not in TERMINAL:
            rnd.save()
            continue
        if obj["status"] == "failed":
            job["error"] = json.dumps(obj.get("errors"))[:500]
            failed = job
        else:
            replies = errors = 0
            for fid in (obj.get("output_file_id"), obj.get("error_file_id")):
                if not fid:
                    continue
                for raw in client.download(fid).decode("utf-8").splitlines():
                    if raw.strip():
                        kind = apply_line(json.loads(raw), rnd, cache, meter, job["id"])
                        replies += kind == "reply"
                        errors += kind == "error"
            log(f"ROUND_INGESTED batch={rnd.batch_id} round={rnd.number} job={job['id']} status={obj['status']} replies={replies} errors={errors}")
        job["ingested"] = True
        rnd.save()
    if failed:
        return "failed"
    return "pending" if any(not j["ingested"] for j in rnd.jobs) else "ingested"


def projected_usd(kinds: Counter, rows, models: dict) -> float:
    rates = rate_per_kind(rows, "batch")
    return round(sum(n * rates.get(kind, estimate_usd_per_call(models.get(kind) or models["ground"], "batch")) for kind, n in kinds.items()), 8)


def finalise(cfg: Config, batch: Batch, meter: Meter, poisoned: dict, log=print) -> str:
    """Reports and the batch's fields exactly as run_batch sets them; BATCH_DONE or BATCH_INCOMPLETE."""
    records = batch_records(cfg, batch)
    complete = [r for r in records if _complete(r)]
    rows = [r for r in read_ledger(meter.path) if r.get("batch") == batch.id]
    meter.calls = len(rows)
    meter.tokens = {k: sum(int(r.get(k) or 0) for r in rows) for k in meter.tokens}
    write_reports(cfg, batch, records, meter, None)
    batch.cost_usd = meter.batch_total
    batch.usd_per_frame = round(batch.cost_usd / len(complete), 8) if complete else None
    batch.prompt_hash, batch.config_hash, batch.models = prompt_hash(cfg.coord_mode), config_hash(cfg), dict(cfg.models)
    batch.coord_mode, batch.efforts, batch.transport = cfg.coord_mode, dict(cfg.efforts), "batch"
    progress = f"frames_done={len(complete)}/{len(batch.frames)} cost=${batch.cost_usd:.4f}"
    if len(complete) < len(batch.frames):
        batch.stop_reason = None
        log(f"BATCH_INCOMPLETE {batch.id} {progress} poisoned={','.join(sorted(poisoned)) or '-'}")
        return "incomplete"
    batch.finished_at = now()
    batch.stop_reason = None
    transition(batch, "done")
    by_status = Counter(r["status"] for r in complete)
    log(f"BATCH_DONE {batch.id} frames={len(batch.frames)} " + " ".join(f"{s}={by_status.get(s, 0)}" for s in STATUSES)
        + f" cost=${batch.cost_usd:.4f} usd_per_frame={batch.usd_per_frame:.6f}")
    return "done"


def step(cfg: Config, batch: Batch, client, meter: Meter, log=print) -> str:
    """One step: finish submitting or poll the open round; when every job is ingested, run a pass and
    submit the next round, or finalise when the pass collects nothing."""
    bdir = batch_dir(cfg, batch)
    bdir.mkdir(parents=True, exist_ok=True)
    cache = ReplyCache(bdir / "replies.jsonl")
    files = FrameFiles(Path(cfg.work) / "batchapi" / "files.json", client, cfg.data, cfg.file_expiry_days)
    rnd = latest_round(bdir)
    if rnd is not None and rnd.pending:
        if submit_missing(rnd, client, files, log):
            log(f"ROUND_SUBMITTED batch={batch.id} round={rnd.number} jobs={len(rnd.jobs)} requests={len(rnd.index)}")
            return "submitted"
        outcome = poll(rnd, client, cache, meter, log)
        if outcome == "failed":
            job = next(j for j in rnd.jobs if j.get("error"))
            batch.stop_reason = "transport"
            log(f"BATCH_STOPPED {batch.id} reason=transport job={job['id']}: {job['error']}")
            return "stopped"
        if outcome == "pending":
            done = sum(j["ingested"] for j in rnd.jobs)
            completed = sum(int(j.get("completed") or 0) for j in rnd.jobs)
            log(f"ROUND_PENDING batch={batch.id} round={rnd.number} jobs={done}/{len(rnd.parts)} requests={completed}/{len(rnd.index)}")
            return "pending"
    poisoned = load_poisoned(bdir)
    number = rnd.number + 1 if rnd else 1
    result = asyncio.run(run_pass(cfg, batch, cache, poisoned, bdir / f"round{number}", log))
    save_poisoned(bdir, poisoned)
    if result.count == 0:
        return finalise(cfg, batch, meter, poisoned, log)
    projection = projected_usd(result.kinds, read_ledger(meter.path), cfg.models)
    if meter.batch_total + projection > meter.caps.batch_usd or meter.run_total + projection > meter.caps.run_usd:
        result.discard()
        log(f"BUDGET_REFUSED round={number} projection=${projection:.4f} batch_cap=${meter.caps.batch_usd:.2f} "
            f"run_total=${meter.run_total:.4f} run_cap=${meter.caps.run_usd:.2f}")
        return "refused"
    rnd = new_round(bdir, batch.id, number, result)
    submit_missing(rnd, client, files, log)
    log(f"ROUND_SUBMITTED batch={batch.id} round={number} jobs={len(rnd.jobs)} requests={result.count} projection=${projection:.4f}")
    return "submitted"


def summary(cfg: Config, batch: Batch) -> str:
    """For `status`: ' round=N jobs=done/total requests=completed/total', or '' without a round."""
    rnd = latest_round(batch_dir(cfg, batch))
    if rnd is None:
        return ""
    done = sum(j["ingested"] for j in rnd.jobs)
    completed = sum(int(j.get("completed") or 0) for j in rnd.jobs)
    return f" round={rnd.number} jobs={done}/{len(rnd.parts)} requests={completed}/{len(rnd.index)}"


def cancel_open(cfg: Config, batch: Batch, client) -> list[str]:
    """`stop`: cancel the open round's jobs that are still running; returns their ids."""
    rnd = latest_round(batch_dir(cfg, batch))
    if rnd is None:
        return []
    cancelled = []
    for job in rnd.jobs:
        if job["ingested"]:
            continue
        status = client.get_batch(job["id"])["status"]
        if status not in TERMINAL and status != "cancelling":
            client.cancel_batch(job["id"])
            cancelled.append(job["id"])
    return cancelled
