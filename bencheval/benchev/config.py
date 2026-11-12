"""YAML configs: configs/benchmarks/<bench>.yaml, configs/protocols/<id>.yaml, configs/models/<id>.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from benchev import ROOT

CFG = ROOT / "configs"


@dataclass(frozen=True)
class Protocol:
    id: str
    bench: str
    n_expected: int
    max_pixels: dict
    resize_to_pixels: dict
    crop: dict
    zoom: float
    source: str
    flags: list = field(default_factory=list)


@dataclass(frozen=True)
class BenchCfg:
    name: str
    data_subdir: str
    loader: str
    canonical: str
    split_keys: list
    scorer: str


@dataclass(frozen=True)
class ModelCfg:
    id: str
    kind: str          # ours | published
    adapter: str       # gui_aima_qwen35 | gui_aima_upstream | gui_actor | text_coords
    ckpt: str | None
    base: str | None
    status: str        # trained | published
    notes: str = ""
    recipe: str | None = None          # benchev.adapters.recipes.RECIPES key (prompt + parser + coordinate space)
    max_pixels: int | None = None      # the authors' image limit; None = the protocol's
    env: str | None = None             # venv name the run scripts map to a python
    flags: list[str] = field(default_factory=list)   # shown in registry notes / tables, e.g. no-published-reference
    parity: dict | None = None         # {bench, published, ours, delta, status: pass|miss}, filled after the parity run
    params: str | None = None          # parameter count shown in tables, e.g. "7B"


def _y(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def load_protocol(pid: str) -> Protocol:
    return Protocol(**_y(CFG / "protocols" / f"{pid}.yaml"))


def load_bench(name: str) -> BenchCfg:
    return BenchCfg(**_y(CFG / "benchmarks" / f"{name}.yaml"))


def map_path(p: str | None) -> str | None:
    """Rewrite a checkpoint path prefix: BENCHEV_PATH_MAP="old_prefix=new_prefix[;...]" (first match wins)."""
    import os
    if not p:
        return p
    for rule in filter(None, os.environ.get("BENCHEV_PATH_MAP", "").split(";")):
        old, new = rule.split("=", 1)
        if p.startswith(old):
            return new + p[len(old):]
    return p


def load_model(mid: str) -> ModelCfg:
    d = _y(CFG / "models" / f"{mid}.yaml")
    d["ckpt"], d["base"] = map_path(d.get("ckpt")), map_path(d.get("base"))
    return ModelCfg(**d)


# Display / suite order of the benchmarks; tests check it names exactly the configs/benchmarks/*.yaml files.
BENCH_ORDER = ["ssv1", "ssv2", "sspro", "oswg", "oswg_refined", "uivision", "mmbench_l2", "medspot"]


def all_benches() -> list[str]:
    return sorted(p.stem for p in (CFG / "benchmarks").glob("*.yaml"))


def all_models() -> list[ModelCfg]:
    return [load_model(p.stem) for p in sorted((CFG / "models").glob("*.yaml"))]
