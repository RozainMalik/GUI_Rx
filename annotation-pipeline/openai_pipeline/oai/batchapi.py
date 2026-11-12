"""The deferred generator and one pass of the batch transport. Instead of
calling the model, `generate` looks the call up in the batch's reply cache: a hit is replayed for
free; a miss is recorded for the next round and the frame stops here (`Deferred`); a call that
errored in two rounds sets the frame aside (`Poisoned`). `process_frame` runs unchanged - it saves
the record per stage, and every sibling call of a stage registers before the frame unwinds
(asyncio schedules the children of a gather FIFO before the first exception reaches the parent).
A frame's requests are buffered and written to the round's part files only when it stops on
Deferred, so a poisoned or crashed frame never spends anything."""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from annotator.calling import Call, Reply
from annotator.frames import Frame
from annotator.runner import process_frame

from .client import DEFAULT_EFFORT, data_url_part, request_body
from .config import Config
from .run import remaining, run_config

POISON_AFTER = 2
PART_MAX_BYTES = 190_000_000       # the API takes 200 MB; run_pass passes [run].part_mb, smaller because uploads are uplink-bound
PART_MAX_LINES = 50_000
FRAME_REF_PREFIX = "@frame:"       # placeholder file_id; rounds.resolve_refs swaps in the uploaded file's id


class Deferred(Exception):
    """This call was recorded for the next round; the frame stops here."""


class Poisoned(Exception):
    """This call has errored in POISON_AFTER rounds; the frame is set aside."""


def image_digest(im) -> str:
    return hashlib.sha256(f"{im.mode}:{im.size}".encode() + im.tobytes()).hexdigest()


def request_key(call: Call, model: str, effort: str, digests: list[str]) -> str:
    """Everything that reaches the API except the image transport."""
    h = hashlib.sha256()
    for part in (call.kind, model, effort, str(call.thinking), call.system, call.user_text, json.dumps(list(call.history)), *digests):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def custom_id_of(key: str) -> str:
    return key[:32]


def frame_ref(app: str, name: str) -> dict:
    return {"type": "input_image", "file_id": f"{FRAME_REF_PREFIX}{app}/{name}", "detail": "high"}


class ReplyCache:
    """work/batchapi/batch<id>/replies.jsonl: append-only; {key, text, refused, usage} lines are
    replies (first wins), {key, error, job} lines are counted once per (key, job)."""

    def __init__(self, path):
        self.path = Path(path)
        self.replies: dict[str, tuple[str, bool]] = {}
        self.errors: Counter = Counter()
        self.last_error: dict[str, str] = {}
        self.seen: set = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    def _apply(self, row) -> bool:
        key = row["key"]
        if "error" in row:
            if (key, row.get("job")) in self.seen:
                return False
            self.seen.add((key, row.get("job")))
            self.errors[key] += 1
            self.last_error[key] = row["error"]
            return True
        if key in self.replies:
            return False
        self.replies[key] = (row.get("text") or "", bool(row.get("refused")))
        return True

    def _append(self, row) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")

    def add_reply(self, key, text, refused, usage) -> None:
        row = {"key": key, "text": text, "refused": bool(refused), "usage": usage}
        if self._apply(row):
            self._append(row)

    def add_error(self, key, message, job) -> bool:
        row = {"key": key, "error": message, "job": job}
        if not self._apply(row):
            return False
        self._append(row)
        return True

    def reset_errors(self) -> None:
        """Drop every error line (run --retry-poisoned): the keys are collected afresh."""
        self.errors, self.last_error, self.seen = Counter(), {}, set()
        if self.path.exists():
            kept = [json.dumps({"key": k, "text": t, "refused": r, "usage": None}) for k, (t, r) in self.replies.items()]
            self.path.write_text("".join(line + "\n" for line in kept), encoding="utf-8", newline="\n")


class Collector:
    """One round's requests: buffered per frame, written as JSONL parts that roll over at the
    API's limits; the index maps custom_id back to the frame and kind."""

    def __init__(self, prefix: Path, *, max_bytes=None, max_lines=None):
        self.prefix = Path(prefix)
        self.max_bytes = max_bytes or PART_MAX_BYTES
        self.max_lines = max_lines or PART_MAX_LINES
        self.keys: set = set()
        self.pending: dict[tuple, list] = {}
        self.parts: list[Path] = []
        self.index: dict = {}
        self.kinds: Counter = Counter()
        self.count = 0
        self._fh = None
        self._bytes = self._lines = 0

    def add(self, key, body, *, app, frame, kind, model, effort) -> None:
        if key in self.keys:
            return
        self.keys.add(key)
        cid = custom_id_of(key)
        line = json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": body}) + "\n"
        entry = {"key": key, "app": app, "frame": frame, "kind": kind, "model": model, "effort": effort}
        self.pending.setdefault((app, frame), []).append((cid, line.encode("utf-8"), entry))

    def flush(self, app, frame) -> None:
        for cid, data, entry in self.pending.pop((app, frame), []):
            if self._fh is None or self._bytes + len(data) > self.max_bytes or self._lines >= self.max_lines:
                self._open_part()
            self._fh.write(data)
            self._bytes += len(data)
            self._lines += 1
            self.index[cid] = entry
            self.kinds[entry["kind"]] += 1
            self.count += 1

    def drop(self, app, frame) -> None:
        for _, _, entry in self.pending.pop((app, frame), []):
            self.keys.discard(entry["key"])

    def _open_part(self) -> None:
        self.close()
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        path = self.prefix.with_name(f"{self.prefix.name}.part{len(self.parts) + 1}.jsonl")
        self._fh = path.open("wb")
        self.parts.append(path)
        self._bytes = self._lines = 0

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def discard(self) -> None:
        self.close()
        for p in self.parts:
            p.unlink(missing_ok=True)


class DeferredGenerator:
    """One frame's generator for one pass."""

    def __init__(self, frame: Frame, cache: ReplyCache, collector: Collector, models: dict, efforts: dict):
        self.frame, self.cache, self.collector, self.models, self.efforts = frame, cache, collector, dict(models), dict(efforts)
        self._frame_digest = None

    @property
    def frame_digest(self) -> str:
        if self._frame_digest is None:
            self._frame_digest = image_digest(Image.open(self.frame.path).convert("RGB"))
        return self._frame_digest

    def _image_part(self, digests):
        it = iter(digests)

        def part(im):
            if next(it) == self.frame_digest:
                return frame_ref(self.frame.app, self.frame.name)
            return data_url_part(im)
        return part

    async def generate(self, call: Call) -> Reply:
        model = self.models.get(call.kind) or self.models["ground"]
        effort = self.efforts.get(call.kind, DEFAULT_EFFORT)
        digests = [image_digest(im) for im in call.images]
        key = request_key(call, model, effort, digests)
        hit = self.cache.replies.get(key)
        if hit is not None:
            text, _refused = hit
            return Reply(content=text, usage=None)
        if self.cache.errors[key] >= POISON_AFTER:
            raise Poisoned(f"{call.kind} failed {self.cache.errors[key]} rounds: {self.cache.last_error.get(key)}")
        if key not in self.collector.keys:
            body = request_body(call, model, effort, self._image_part(digests))
            self.collector.add(key, body, app=self.frame.app, frame=self.frame.name, kind=call.kind, model=model, effort=effort)
        raise Deferred(key)


def load_poisoned(bdir) -> dict:
    path = Path(bdir) / "poisoned.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_poisoned(bdir, poisoned: dict) -> None:
    path = Path(bdir) / "poisoned.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poisoned, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def clear_poisoned(bdir, cache: ReplyCache) -> int:
    """run --retry-poisoned: forget the set-aside frames and the errors that set them aside."""
    n = len(load_poisoned(bdir))
    (Path(bdir) / "poisoned.json").unlink(missing_ok=True)
    cache.reset_errors()
    return n


@dataclass
class PassResult:
    count: int
    parts: list = field(default_factory=list)
    index: dict = field(default_factory=dict)
    kinds: Counter = field(default_factory=Counter)
    poisoned_now: list = field(default_factory=list)

    def discard(self) -> None:
        for p in self.parts:
            Path(p).unlink(missing_ok=True)


async def run_pass(cfg: Config, batch, cache: ReplyCache, poisoned: dict, prefix: Path, log=print) -> PassResult:
    """Every frame of the batch that is neither complete nor poisoned through process_frame with a
    DeferredGenerator, frame_concurrency at a time. Local work only."""
    rcfg = run_config(cfg, batch)
    todo = [fr for fr in remaining(cfg, batch) if f"{fr.app}/{fr.name}" not in poisoned]
    collector = Collector(prefix, max_bytes=cfg.part_mb * 1_000_000)
    sem = asyncio.Semaphore(cfg.frame_concurrency)
    newly = []

    async def one(fr):
        async with sem:
            gen = DeferredGenerator(fr, cache, collector, cfg.models, cfg.efforts)
            try:
                await process_frame(gen, rcfg, fr)
                collector.drop(fr.app, fr.name)                      # completed from the cache alone: nothing to send
            except Deferred:
                collector.flush(fr.app, fr.name)
            except Poisoned as e:
                collector.drop(fr.app, fr.name)
                poisoned[f"{fr.app}/{fr.name}"] = str(e)
                newly.append(f"{fr.app}/{fr.name}")
                log(f"{fr.app}/{fr.name}: POISONED {e}")
            except Exception as e:                                   # a bug in one frame must not end the pass
                collector.drop(fr.app, fr.name)
                log(f"{fr.app}/{fr.name}: {type(e).__name__}: {e}")

    await asyncio.gather(*(one(fr) for fr in todo))
    for _ in range(3):                                               # let any straggling sibling task finish its step
        await asyncio.sleep(0)
    collector.close()
    return PassResult(collector.count, list(collector.parts), dict(collector.index), Counter(collector.kinds), newly)
