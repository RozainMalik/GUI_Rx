"""Did a smoke (limit) run pass?  python scripts/smoke_check.py RESULTS MODEL BENCH MODE LIMIT
Passes (exit 0) if its predictions.jsonl has at least one point and at least half of the predictions have one.
Standard library only (runs under any venv).

Zoom + fusion (MODE zf) is also a parity gate, against the same model's existing 1-step run found under one of the
results roots in $ZF_REF (os.pathsep-separated, ":" on Linux; missing reference = fail):
  * pass 1 is the 1-step pass: first_point within ZF_TOL_PX (default 4) px of the 1-step click on >= 90% of samples;
  * fusion at λ = 0 is the zoom map alone: fuse["0"] within ZF_TOL_PX of zoom_point on >= 90% of samples
    (GUI-AIMA's readout vs the vendored fast_readout)."""
import json
import os
import re
import sys
from pathlib import Path

results, model, bench, mode, lim = sys.argv[1:6]
base = Path(results) / model / bench
dirs = sorted(d for d in (base.iterdir() if base.is_dir() else [])
              if re.fullmatch(rf"{re.escape(mode)}(@.*)?_(limit|largest){lim}", d.name))
recs = []
if dirs and (dirs[0] / "predictions.jsonl").exists():
    recs = [json.loads(l) for l in (dirs[0] / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
n = sum(r.get("point") is not None for r in recs)
print(f"{n}/{len(recs)} points in {dirs[0] if dirs else base}")
ok = n > 0 and 2 * n >= len(recs)

if ok and mode == "zf":
    tol = float(os.environ.get("ZF_TOL_PX", "4"))
    ref = None
    for root in filter(None, os.environ.get("ZF_REF", "").split(os.pathsep)):
        p = Path(root) / model / bench / "1step" / "predictions.jsonl"
        if p.exists():
            ref = {}
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    ref[r["sid"]] = r
            break
    if ref is None:
        print(f"zf parity: no 1-step reference for {model} {bench} under ZF_REF={os.environ.get('ZF_REF', '')!r}")
        ok = False
    else:
        def dpx(a, b, size):
            return ((a[0] - b[0]) * size[0]) ** 2 + ((a[1] - b[1]) * size[1]) ** 2

        both = [r for r in recs if r.get("first_point") and ref.get(r["sid"], {}).get("point")]
        same1 = sum(dpx(r["first_point"], ref[r["sid"]]["point"], r["image_size"]) <= tol ** 2 for r in both)
        fz = [r for r in recs if r.get("fuse") and r.get("zoom_point")]
        same0 = sum(dpx(r["fuse"]["0"], r["zoom_point"], r["image_size"]) <= tol ** 2 for r in fz)
        print(f"zf parity: pass 1 = 1-step click {same1}/{len(both)}; fuse lambda=0 = zoom click {same0}/{len(fz)} "
              f"(tol {tol:g} px)")
        ok = bool(both) and bool(fz) and 10 * same1 >= 9 * len(both) and 10 * same0 >= 9 * len(fz)
sys.exit(0 if ok else 1)
