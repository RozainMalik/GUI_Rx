"""results/registry.jsonl: one row per (model, bench, split, mode, protocol, provenance). Source of truth for all numbers."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROVENANCES = {"harness", "imported", "paper"}
STATUSES = {"complete", "partial"}
PARITIES = {"verified", "unverified", "n/a"}


class DuplicateRow(ValueError):
    pass


@dataclass
class Row:
    model_id: str
    bench: str
    split: str
    mode: str
    protocol_id: str
    metrics: dict
    n: int
    provenance: str
    source: str
    harness_commit: str | None = None
    date: str = ""
    parity: str = "n/a"
    notes: str = ""
    status: str = "complete"

    def __post_init__(self):
        if self.provenance not in PROVENANCES:
            raise ValueError(f"provenance {self.provenance!r} not in {PROVENANCES}")
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")
        if self.parity not in PARITIES:
            raise ValueError(f"parity {self.parity!r} not in {PARITIES}")
        if self.mode not in ("1step", "2step"):
            raise ValueError(f"mode {self.mode!r}")

    def key(self):
        return (self.model_id, self.bench, self.split, self.mode, self.protocol_id, self.provenance)


def load(path) -> list[Row]:
    p = Path(path)
    if not p.exists():
        return []
    return [Row(**json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write(path, rows):
    """Temp file + rename, so a crash mid-write never truncates the registry."""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def append(path, rows, supersede: bool = False, keep_verified: bool = False) -> None:
    """keep_verified: a superseding row whose metrics and n equal a verified old row's keeps parity='verified'
    (a re-ingest would otherwise silently drop the mark)."""
    path = Path(path)
    existing = load(path)
    seen = {}
    for r in rows:
        if r.key() in seen:
            raise DuplicateRow(f"duplicate key inside batch: {r.key()}")
        seen[r.key()] = r
    clash = [r for r in existing if r.key() in seen]
    if clash and not supersede:
        raise DuplicateRow(f"{len(clash)} existing rows, e.g. {clash[0].key()}; pass supersede=True to replace")
    if clash and keep_verified:
        for o in clash:
            new = seen[o.key()]
            if o.parity == "verified" and (new.metrics, new.n) == (o.metrics, o.n):
                new.parity = "verified"
    if clash:
        with open(path.with_name("registry_superseded.jsonl"), "a", encoding="utf-8") as f:
            for r in clash:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    keep = [r for r in existing if r.key() not in seen]
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, keep + list(rows))
