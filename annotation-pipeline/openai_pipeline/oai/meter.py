"""The ledger and the only component allowed to halt a run: one JSONL line per
API call, running totals for the batch and for everything ever spent, the two caps, and the
anomaly guard. Both stops are TransportErrors on purpose - the shared runner leaves the stage
undone, so the frame in flight resumes cleanly on the next `run`."""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from annotator.calling import TransportError

from .pricing import cost_usd

ANOMALY_FACTOR = 5.0
ANOMALY_MIN_FRAMES = 4        # completed frames before a median means anything
ANOMALY_STOP_AT = 2           # the first runaway frame is logged and finishes; the second stops the batch


class BudgetExceeded(TransportError):
    """A cap was crossed. Raised before the next call, never mid-call."""


class AnomalyStop(TransportError):
    """Two frames cost more than 5x the running median - a runaway reasoning loop, most likely."""


@dataclass(frozen=True)
class Caps:
    batch_usd: float
    run_usd: float


def read_ledger(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def ledger_total(rows, batch=None) -> float:
    return round(sum(float(r.get("usd") or 0) for r in rows if batch is None or r.get("batch") == batch), 8)


def reprice(rows) -> float:
    """The ledger at today's PRICES - what `status` shows when a price changed after the fact."""
    return round(sum(cost_usd(r, r["model"], r.get("transport", "sync")) for r in rows), 8)


def rate_per_kind(rows, transport: str = "batch") -> dict[str, float]:
    """Mean cost of one call of each kind, every row priced at `transport` whatever it ran under -
    the per-request rate a round's projection multiplies out."""
    totals, counts = {}, {}
    for r in rows:
        kind = r.get("kind")
        if not kind or "model" not in r:
            continue
        totals[kind] = totals.get(kind, 0.0) + cost_usd(r, r["model"], transport)
        counts[kind] = counts.get(kind, 0) + 1
    return {k: totals[k] / counts[k] for k in totals}


class Meter:
    def __init__(self, path, batch_id, caps: Caps, log=print):
        self.path, self.batch_id, self.caps, self.log = Path(path), batch_id, caps, log
        prior = read_ledger(self.path)
        self.run_total = ledger_total(prior)
        self.batch_total = ledger_total(prior, batch_id)     # resuming a batch continues its total
        self.open_frames: dict[tuple, float] = {}
        self.done_frames: dict[tuple, float] = {}
        self.anomalous: set = set()
        self.stop_reason: str | None = None
        self.calls = 0
        self.tokens = {"input": 0, "cached": 0, "output": 0, "reasoning": 0}

    def check(self) -> None:
        """Called before every call. Raises once a cap is crossed or the anomaly rule tripped."""
        if self.stop_reason == "anomaly":
            raise AnomalyStop(f"{len(self.anomalous)} frames cost > {ANOMALY_FACTOR:g}x the batch median")
        if self.batch_total > self.caps.batch_usd:
            self.stop_reason = "budget"
            raise BudgetExceeded(f"batch cap ${self.caps.batch_usd:.2f} crossed: ${self.batch_total:.4f}")
        if self.run_total > self.caps.run_usd:
            self.stop_reason = "budget"
            raise BudgetExceeded(f"run cap ${self.caps.run_usd:.2f} crossed: ${self.run_total:.4f}")

    def record(self, *, app, frame, kind, usage: dict, transport: str = "sync", job: str | None = None) -> float:
        usd = cost_usd(usage, usage["model"], transport)
        row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "batch": self.batch_id, "app": app, "frame": frame, "kind": kind,
               "model": usage["model"], "effort": usage.get("effort"), "input": int(usage.get("input") or 0),
               "cached": int(usage.get("cached") or 0), "output": int(usage.get("output") or 0),
               "reasoning": int(usage.get("reasoning") or 0), "retries": int(usage.get("retries") or 0), "usd": usd,
               "transport": transport}
        if job:
            row["job"] = job
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")
        self.batch_total = round(self.batch_total + usd, 8)
        self.run_total = round(self.run_total + usd, 8)
        self.calls += 1
        for k in self.tokens:
            self.tokens[k] += row[k]
        key = (app, frame)
        self.open_frames[key] = self.open_frames.get(key, 0.0) + usd
        self._watch(key)
        return usd

    def _watch(self, key) -> None:
        if len(self.done_frames) < ANOMALY_MIN_FRAMES or key in self.anomalous:
            return
        median = statistics.median(self.done_frames.values())
        if median > 0 and self.open_frames[key] > ANOMALY_FACTOR * median:
            self.anomalous.add(key)
            self.log(f"ANOMALY {key[0]}/{key[1]} cost=${self.open_frames[key]:.4f} > {ANOMALY_FACTOR:g}x median ${median:.4f}")
            if len(self.anomalous) >= ANOMALY_STOP_AT:
                self.stop_reason = "anomaly"

    def frame_done(self, app, frame) -> None:
        self.done_frames[(app, frame)] = self.open_frames.pop((app, frame), 0.0)


class Metered:
    """One frame's view of the meter: the caps are checked before every call, every reply is
    priced and attributed to this frame."""

    def __init__(self, gen, meter: Meter, app, frame):
        self.gen, self.meter, self.app, self.frame = gen, meter, app, frame

    async def generate(self, call):
        self.meter.check()
        reply = await self.gen.generate(call)
        if reply.usage:
            self.meter.record(app=self.app, frame=self.frame, kind=call.kind, usage=reply.usage)
        return reply
