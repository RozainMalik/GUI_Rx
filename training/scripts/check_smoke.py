"""Checks a smoke run: steps done, finite loss, memory under the GPU budget, only allowed weights changed."""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--max_reserved_gib", type=float, default=44.0)
a = ap.parse_args()
s = json.load(open(os.path.join(a.run_dir, "run_summary.json")))
audit = json.load(open(os.path.join(a.run_dir, "param_audit.json")))
checks = [
    ("reached max_steps", s["global_step"] == s["max_steps"], f'{s["global_step"]}/{s["max_steps"]}'),
    ("finite loss", s["last_loss"] is not None and math.isfinite(s["last_loss"]), s["last_loss"]),
    ("peak reserved memory", s["peak_reserved_gib"] is not None and s["peak_reserved_gib"] <= a.max_reserved_gib,
     f'{s["peak_reserved_gib"]} GiB (limit {a.max_reserved_gib})'),
    ("only allowed params changed", not audit["unexpected"], audit["unexpected"][:5]),
    ("pointer rows changed", any(n.endswith("embed_tokens.rows") for n in audit["changed"]), ""),
]
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
print(f'trainable params: {s["trainable_params"]:,}')
sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
