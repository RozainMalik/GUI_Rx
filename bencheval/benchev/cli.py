"""python -m benchev.cli {run, ingest, report}."""
import argparse
import os
import sys
from pathlib import Path

from benchev import ROOT
from benchev.config import BENCH_ORDER

REG = ROOT / "results" / "registry.jsonl"
CORE = ["ssv2", "sspro", "medspot"]
SUITE = BENCH_ORDER


def expand_benches(b):
    return CORE if b == "core" else SUITE if b == "suite" else [b]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchev", description="run | ingest | report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate a model on a bench (or core / suite)")
    r.add_argument("--model", required=True)
    r.add_argument("--bench", required=True)
    r.add_argument("--mode", choices=["1step", "2step", "zf", "both"], default="both")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--largest", action="store_true", help="with --limit: the N samples with the biggest images")
    r.add_argument("--data", default=os.environ.get("BENCHEV_DATA", str(ROOT / "data")))
    r.add_argument("--out", default=str(ROOT / "results" / "raw"))
    r.add_argument("--device", default="cuda")
    r.add_argument("--ingest", action="store_true")
    i = sub.add_parser("ingest", help="add a finished run dir to the registry")
    i.add_argument("run_dir")
    i.add_argument("--supersede", action="store_true")
    rp = sub.add_parser("report", help="regenerate results/LEADERBOARD.md")
    rp.add_argument("--medspot-table", action="store_true", help="also write results/MEDSPOT_TABLE.md")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):   # Windows cp1252 console cannot print Δ/✓
        sys.stdout.reconfigure(errors="replace")

    from benchev import registry
    if a.cmd == "run":
        from benchev.ingest import rows_from_run
        from benchev.run import run, run_dirname
        for b in expand_benches(a.bench):
            for mode in (["1step", "2step"] if a.mode == "both" else [a.mode]):
                s = run(a.model, b, mode, a.data, a.out, device=a.device, limit=a.limit, largest=a.largest)
                print(b, mode, s["status"], s["n_scored"], "/", s["n_expected"], flush=True)
                if a.ingest and not a.limit:     # a --limit test run never replaces registry rows
                    d = run_dirname(mode, s["protocol_id"], s["protocol_id"])
                    registry.append(REG, rows_from_run(Path(a.out) / a.model / b / d), supersede=True)
    elif a.cmd == "ingest":
        from benchev.ingest import rows_from_run
        registry.append(REG, rows_from_run(a.run_dir), supersede=a.supersede)
    elif a.cmd == "report":
        from benchev.report import write
        write(REG, ROOT / "results" / "LEADERBOARD.md")
        if a.medspot_table:
            from benchev.report import write_medspot_table
            write_medspot_table(REG, ROOT / "results" / "MEDSPOT_TABLE.md")


if __name__ == "__main__":
    main()
