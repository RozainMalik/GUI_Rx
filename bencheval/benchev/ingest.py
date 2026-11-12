"""Turn a finished run directory into registry rows (one per split; MedSPOT: one per hit rule)."""
import datetime as dt
import json
from pathlib import Path

from benchev.registry import Row

MEDSPOT_KEYS = ("TCA (%)", "SHR (%)", "step1_accuracy (%)", "prefix_SHR (%)")


def _notes(run_json):
    m = run_json.get("model", {})
    parts = [f"slurm {run_json.get('slurm_job')}"]
    if m.get("max_pixels"):
        parts.append(f"max_pixels={m['max_pixels']} (model)")
    if m.get("flags"):
        parts.append(",".join(m["flags"]))
    return "; ".join(parts)


def rows_from_run(run_dir):
    d = Path(run_dir)
    s = json.loads((d / "summary.json").read_text())
    r = json.loads((d / "run.json").read_text())
    base = dict(model_id=s["model_id"], bench=s["bench"], mode=s["mode"], protocol_id=s["protocol_id"],
                provenance="harness", source=str(d), harness_commit=r.get("harness_commit"),
                date=r["finished"][:10] if r.get("finished") else dt.date.today().isoformat(), status=s["status"],
                notes=_notes(r))
    m = s["metrics"]
    if s["bench"] == "medspot":
        return [Row(split=f"rule={rule}", metrics={k: m[rule][k] for k in MEDSPOT_KEYS},
                    n=m[rule]["evaluable_steps"], **base) for rule in ("strict", "tol14px")]
    rows = [Row(split="all", metrics={k: v for k, v in m.items() if k.startswith("acc")}, n=m["n"], **base)]
    rows += [Row(split=k, metrics={"acc": v}, n=m["split_n"][k], **base) for k, v in m["split"].items()]
    return rows
