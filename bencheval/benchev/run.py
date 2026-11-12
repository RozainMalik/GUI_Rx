"""Resumable runner: predictions.jsonl (append, fsync, torn-line tolerant) -> scorer -> summary.json + run.json."""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from PIL import Image

from benchev import ROOT
from benchev.benchmarks import load as load_samples
from benchev.config import load_bench, load_model, load_protocol
from benchev.inference import predict_sample
from benchev.scorers import score


def _load_preds(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["sid"]] = r
    return out


def _torn(path):
    """True if the file's last byte is not a newline (reads one byte, not the whole file)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) != b"\n"


def _append(path, rec, torn=False):
    with open(path, "a", encoding="utf-8") as f:
        f.write(("\n" if torn else "") + json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _commit():
    """git HEAD (+ "-dirty" if tracked files differ from it), or ROOT/.harness_commit for a copy without .git."""
    try:
        p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        c = p.stdout.strip() if p.returncode == 0 else ""   # outside a repo git echoes "HEAD" with rc != 0
        if c and subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD"], capture_output=True).returncode:
            c += "-dirty"
    except OSError:
        c = ""
    f = Path(ROOT) / ".harness_commit"
    return c or (f.read_text().strip() if f.exists() else None)


def run_dirname(mode, protocol_id, canonical, limit=0, largest=False):
    """1step | 1step@<protocol> for a non-canonical protocol | ..._limit<N> (first N) | ..._largest<N> (N biggest images),
    so test runs never touch full ones."""
    tail = (f"_largest{limit}" if largest else f"_limit{limit}") if limit else ""
    return mode + ("" if protocol_id == canonical else f"@{protocol_id}") + tail


def _recipe_sha(mcfg):
    """Hash of the resolved recipe (prompt text included), so an edited prompt under the same name is caught."""
    if not getattr(mcfg, "recipe", None):
        return None
    import hashlib
    from benchev.adapters.recipes import RECIPES
    return hashlib.sha256(repr(RECIPES[mcfg.recipe]).encode()).hexdigest()[:16]


def _check_resume(run_json, mcfg, protocol_id):
    """Refuse to reuse predictions made with another checkpoint, adapter, recipe (name or text), image limit or protocol.
    run.json is written before the first prediction, so a crashed run is guarded too."""
    if not run_json.exists():
        return
    old = json.loads(run_json.read_text())
    for k in ("ckpt", "adapter", "base", "recipe", "max_pixels"):
        if old.get("model", {}).get(k) != getattr(mcfg, k, None):
            raise SystemExit(f"{run_json.parent}: stored predictions used {k}={old['model'].get(k)!r}, "
                             f"config now says {getattr(mcfg, k, None)!r}; move the directory away to re-run")
    if old.get("recipe_sha") != _recipe_sha(mcfg):
        raise SystemExit(f"{run_json.parent}: stored predictions used another recipe text (recipe_sha "
                         f"{old.get('recipe_sha')} vs {_recipe_sha(mcfg)}); move the directory away to re-run")
    if old.get("protocol_id") != protocol_id:
        raise SystemExit(f"{run_json.parent}: stored predictions used protocol {old.get('protocol_id')}")


def run(model_id, bench, mode, data_root, out_root, adapter=None, device="cuda", limit=0, protocol_id=None, largest=False):
    """largest: with limit, take the `limit` samples with the biggest images (a smoke that is also a memory check)."""
    bcfg, mcfg = load_bench(bench), load_model(model_id)
    if mode == "zf" and bcfg.scorer == "medspot":
        raise NotImplementedError("zf on MedSPOT needs task-level cross-fitting; not supported in this harness")
    proto = load_protocol(protocol_id or bcfg.canonical)
    data_dir = Path(data_root) / bcfg.data_subdir
    samples = load_samples(bcfg.loader, data_dir)
    if limit and largest:
        def area(s):
            with Image.open(s.image_path) as im:
                return im.width * im.height
        samples = sorted(samples, key=area, reverse=True)
    samples = samples[:limit] if limit else samples
    out = Path(out_root) / model_id / bench / run_dirname(mode, proto.id, bcfg.canonical, limit, largest)
    out.mkdir(parents=True, exist_ok=True)
    _check_resume(out / "run.json", mcfg, proto.id)
    pred_path = out / "predictions.jsonl"
    done = _load_preds(pred_path)
    todo = [s for s in samples if s.sid not in done]
    t0 = time.time()
    rj = out / "run.json"
    if todo and not rj.exists():   # provisional record, completed after the loop
        rj.write_text(json.dumps(_run_record(mcfg, proto, limit, None, None), indent=1), encoding="utf-8")
    if todo:
        if adapter is None:
            from benchev.adapters import get_adapter
            adapter = get_adapter(mcfg.adapter)
            adapter.load(mcfg, device)
        torn = _torn(pred_path)
        for i, s in enumerate(todo, 1):
            img = Image.open(s.image_path).convert("RGB")
            o = predict_sample(adapter, img, s.instruction, proto, mode, getattr(mcfg, "max_pixels", None))
            W, H = img.size
            p = o["point"]
            rec = {"sid": s.sid, "point": None if p is None else list(p),
                   "point_px": None if p is None else [p[0] * W, p[1] * H], "image_size": [W, H],
                   "first_point": None if o["first_point"] is None else list(o["first_point"]),
                   "crop_box": None if o["crop_box"] is None else list(o["crop_box"]),
                   "input_size": list(o["input_size"]), "raw": o["raw"]}
            if mode == "zf":
                rec["zoom_point"] = None if o["zoom_point"] is None else list(o["zoom_point"])
                rec["fuse"] = o["fuse"]
            _append(pred_path, rec, torn)
            torn = False
            if i % 50 == 0:
                print(f"[{model_id} {bench} {mode}] {i}/{len(todo)}", flush=True)
    done = _load_preds(pred_path)
    ids = {s.sid for s in samples}
    preds = {k: (None if r["point"] is None else tuple(r["point"])) for k, r in done.items() if k in ids}
    if bcfg.scorer == "medspot":
        metrics = _score_medspot(data_dir, done, samples)
    elif mode == "zf":
        metrics = _score_zf(bcfg, samples, {k: r for k, r in done.items() if k in ids})
    else:
        metrics = score(bcfg.scorer, samples, preds, bcfg.split_keys)
    n_scored = len(preds)
    summary = {"model_id": model_id, "bench": bench, "mode": mode, "protocol_id": proto.id,
               "n_expected": proto.n_expected, "n_scored": n_scored,
               "status": "complete" if n_scored == proto.n_expected else "partial", "metrics": metrics}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    now = dt.datetime.now().isoformat(timespec="seconds")
    if not todo and rj.exists() and json.loads(rj.read_text()).get("finished"):
        # re-score only: keep the record of the job that made the predictions, log this pass under "rescored"
        run_json = json.loads(rj.read_text())
        run_json.setdefault("rescored", []).append({"harness_commit": _commit(), "at": now,
                                                    "slurm_job": os.environ.get("SLURM_JOB_ID")})
    else:
        run_json = _run_record(mcfg, proto, limit, now, round(time.time() - t0, 1))
    rj.write_text(json.dumps(run_json, indent=1), encoding="utf-8")
    return summary


def _run_record(mcfg, proto, limit, finished, seconds):
    return {"harness_commit": _commit(), "model": mcfg.__dict__, "recipe_sha": _recipe_sha(mcfg), "protocol_id": proto.id,
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
            "finished": finished, "seconds": seconds, "limit": limit}


def _score_medspot(root, done, samples):
    """Core records built from our predictions; the vendored scorer recomputes hits from pred_point + gt_bbox."""
    from medspot_strict.core import load_tasks
    from benchev.scorers import medspot as ms
    keep = {s.sid for s in samples}
    tasks = [t for t in load_tasks(Path(root) / "Annotations") if any(st.key in keep for st in t.steps)]
    gt = {st.key: list(st.gt_bbox) for t in tasks for st in t.steps}
    records = {k: {"key": k, "pred_point": r["point"], "gt_bbox": gt[k], "image_size": r["image_size"]}
               for k, r in done.items() if k in keep}
    return ms.score(tasks, records)


def _score_zf(bcfg, samples, recs):
    """Zoom + fusion: the headline metrics are fuse_cv (λ cross-fitted per half, benchev.fusion.cross_fit);
    acc_l<λ> are the fixed-λ accuracies, acc_zoom the plain zoom click (λ = 0), acc_first the 1-step click.
    A sample with no fused point (no first point / no map) scores its first point (or a miss) in every variant."""
    from benchev.fusion import LAMBDAS, cross_fit

    def pt(r, key):
        if r.get("fuse"):
            return tuple(r["fuse"][key])
        return None if r.get("point") is None else tuple(r["point"])

    def preds_for(key, subset=None):
        return {k: pt(r, key) for k, r in recs.items() if subset is None or k in subset}

    def acc(key, subset=None):
        keep = set(subset) if subset is not None else None
        ss = [s for s in samples if s.sid in recs and (keep is None or s.sid in keep)]
        return score(bcfg.scorer, ss, preds_for(key, keep), bcfg.split_keys)["acc"]

    pick, chosen = cross_fit(list(recs), lambda lam, sub: acc(str(lam), sub))
    cv = {k: pt(recs[k], str(pick[k])) for k in recs}
    metrics = score(bcfg.scorer, [s for s in samples if s.sid in recs], cv, bcfg.split_keys)
    for lam in LAMBDAS:
        metrics[f"acc_l{lam}"] = acc(str(lam))
    metrics["acc_zoom"] = acc("0")
    first = {k: (None if r.get("first_point") is None else tuple(r["first_point"])) for k, r in recs.items()}
    metrics["acc_first"] = score(bcfg.scorer, [s for s in samples if s.sid in recs], first, bcfg.split_keys)["acc"]
    metrics["fuse_cv_lambdas"] = chosen
    return metrics
