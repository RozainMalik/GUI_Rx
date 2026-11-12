"""config.toml -> Config. Relative paths resolve against the config file's folder;
shared_repo defaults to the checkout `annotator` was installed from, so no machine-specific path is needed."""
from __future__ import annotations

import hashlib
import re
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import annotator
from annotator.geometry import COORD_MODES

from .pricing import PRICES_AS_OF, TRANSPORTS, UnknownModel, price_of

KINDS = ("propose", "describe", "ground", "verify", "refine_ground")
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_SIZES = (25, 50, 100, 250, 500)


class ConfigError(ValueError):
    """config.toml is missing something or holds a value that would make a run meaningless."""


@dataclass(frozen=True)
class Config:
    models: dict
    efforts: dict
    coord_mode: str
    concurrency: int
    frame_concurrency: int
    seed: int
    early_stop: bool
    transport: str                       # sync | batch - how `run` reaches the model
    poll_seconds: int                    # `run --wait` polls open jobs this often
    file_expiry_days: int                # uploaded frame PNGs expire server-side after this
    part_mb: int                         # a round's JSONL rolls into another job at this size (API limit 200; uplink-bound)
    sizes: tuple
    gate: float
    expected_apps: int
    exclude_benchmark_apps: bool         # plan draws from the 24 non-MedSPOT apps only
    balanced: bool                       # plan: equal share of every app's remaining frames per batch
    batch_usd: float
    run_usd: float
    data: Path
    benchmark: Path
    work: Path
    export: Path
    reports: Path
    shared_repo: Path
    image_root: str
    remote: str
    remote_data: str
    remote_benchmark: str
    dataset: str                         # name of the training set being built (export file stem)
    path: Path


def _section(raw, name) -> dict:
    s = raw.get(name)
    if s is None:
        return {}
    if not isinstance(s, dict):
        raise ConfigError(f"[{name}] must be a table")
    return s


def load_config(path) -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    models, efforts = _section(raw, "models"), _section(raw, "effort")
    export = _section(raw, "export")
    for k in KINDS:
        if not isinstance(models.get(k), str) or not models[k].strip():
            raise ConfigError(f"[models].{k} missing")
        try:
            price_of(models[k])
        except UnknownModel:
            raise ConfigError(f"[models].{k} = {models[k]!r} is not in the price table (prices as of {PRICES_AS_OF})")
        if efforts.get(k) not in EFFORTS:
            raise ConfigError(f"[effort].{k} must be one of {', '.join(EFFORTS)}")
    run, batches, caps, paths, remote = (_section(raw, n) for n in ("run", "batches", "caps", "paths", "remote"))
    mode = run.get("coord_mode", "pixels")
    if mode not in COORD_MODES:
        raise ConfigError(f"[run].coord_mode must be one of {', '.join(COORD_MODES)}")
    transport = str(run.get("transport", "sync"))
    if transport not in TRANSPORTS:
        raise ConfigError(f"[run].transport must be one of {', '.join(TRANSPORTS)}")
    poll_seconds = int(run.get("poll_seconds", 60))
    if poll_seconds < 1:
        raise ConfigError("[run].poll_seconds must be >= 1")
    file_expiry_days = int(run.get("file_expiry_days", 30))
    if not 1 <= file_expiry_days <= 30:
        raise ConfigError("[run].file_expiry_days must be between 1 and 30")
    part_mb = int(run.get("part_mb", 48))
    if not 1 <= part_mb <= 190:
        raise ConfigError("[run].part_mb must be between 1 and 190")
    sizes = tuple(batches.get("sizes", DEFAULT_SIZES))
    if not sizes or any(not isinstance(s, int) or isinstance(s, bool) or s <= 0 for s in sizes):
        raise ConfigError("[batches].sizes must be positive integers")
    gate = float(batches.get("gate", 0.75))
    if not 0 <= gate <= 1:
        raise ConfigError("[batches].gate must be between 0 and 1")
    batch_usd, run_usd = float(caps.get("batch_usd", 0)), float(caps.get("run_usd", 50))
    if batch_usd < 0:
        raise ConfigError("[caps].batch_usd must be >= 0")
    if run_usd < 0:
        raise ConfigError("[caps].run_usd must be >= 0")
    dataset = str(export.get("dataset", "gui_grounding"))
    if not re.fullmatch(r"[a-z0-9_]+", dataset):
        raise ConfigError("[export].dataset must be a lowercase slug ([a-z0-9_]+)")

    def local(key, default):
        p = Path(paths.get(key, default))
        return (p if p.is_absolute() else base / p).resolve()

    shared = local("shared_repo", "") if paths.get("shared_repo") else Path(annotator.__file__).resolve().parents[1]
    return Config(models={k: models[k] for k in KINDS}, efforts={k: efforts[k] for k in KINDS}, coord_mode=mode,
                  concurrency=int(run.get("concurrency", 4)), frame_concurrency=int(run.get("frame_concurrency", 4)),
                  seed=int(run.get("seed", 20260916)), early_stop=bool(run.get("early_stop", False)),
                  transport=transport, poll_seconds=poll_seconds, file_expiry_days=file_expiry_days, part_mb=part_mb,
                  sizes=sizes, gate=gate, expected_apps=int(batches.get("expected_apps", 30)),
                  exclude_benchmark_apps=bool(batches.get("exclude_benchmark_apps", False)),
                  balanced=bool(batches.get("balanced", False)),
                  batch_usd=batch_usd, run_usd=run_usd,
                  data=local("data", "data"), benchmark=local("benchmark", "benchmark"), work=local("work", "work"),
                  export=local("export", "export"), reports=local("reports", "reports"), shared_repo=shared,
                  image_root=str(paths.get("image_root", "data")).replace("\\", "/"),
                  remote=str(remote.get("login", "user@host")),
                  remote_data=str(remote.get("data", "AnnotationPipeline/MedSPOT_Training_Data")),
                  remote_benchmark=str(remote.get("benchmark", "MedicalDataset")), dataset=dataset, path=path)


def config_hash(cfg: Config) -> str:
    """What a batch ran under - the knobs that change the model's answers."""
    payload = {"models": cfg.models, "efforts": cfg.efforts, "coord_mode": cfg.coord_mode, "early_stop": cfg.early_stop}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
