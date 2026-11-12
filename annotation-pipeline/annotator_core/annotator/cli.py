"""annotator: calibrate | annotate | reset | export | qc | batch-report | grade | apply-grades.

Every command prints a positive marker on success (CALIBRATION_OK, RECORDS_WRITTEN, ANNOTATE_DONE,
EXPORTED, ...) - job scripts gate on those, never on exit code 0. `annotate` prints
ANNOTATE_INCOMPLETE instead of ANNOTATE_DONE, and exits 1, when any frame ended in transport_error
or error."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from .calibrate import MARKER, MIN_PROMPT_TOKENS, calibrate, write_result
from .export import write_export
from .frames import BATCH_SEED, apps_by_size, batch_frame, list_frames
from .grading import apply_grades, build_sample, load_verdicts, sample_path, summarise, verdicts_path
from .qc import write_batch_report, write_qc
from .records import STAGES, load_record, reset_from, save_record
from .replay import reparse
from .runner import MixedPromptError, RunConfig, Throttled, run_app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # so `tools` imports from a checkout


def _generator(args):
    """The one seam tests replace: the real vLLM transport."""
    from .clients import VLLMGenerator
    return VLLMGenerator(base_url=args.server, model=args.model)


def _serve(sample_file, root, verdict_path, port, open_browser, only_local):
    from tools.review_server import serve
    return serve(sample_file, root, verdict_path, port, open_browser, only_local)


def _server_args(p):
    p.add_argument("--server", default="http://localhost:8600/v1")
    p.add_argument("--model", default="qwen35_9b")
    p.add_argument("--concurrency", type=int, default=8)


def _tiers(name):
    return ("A", "B") if name == "AB" else ("C",)


def cmd_calibrate(args) -> int:
    probe = Image.open(args.probe).convert("RGB")
    gen = _generator(args)

    async def go():
        try:
            return await calibrate(Throttled(gen, args.concurrency), args.dataset, probe, args.n, args.seed)
        finally:
            await gen.aclose()
    res = asyncio.run(go())
    write_result(args.work, res)
    if not res.resolution_ok:
        print(f"RESOLUTION_FAIL prompt_tokens={res.prompt_tokens} < {MIN_PROMPT_TOKENS}: the frame was downscaled "
              f"- check --mm-processor-kwargs max_pixels on the server")
        return 2
    print(f"CALIBRATION hits={res.hits}/{res.total} accuracy={res.accuracy:.3f} prompt_tokens={res.prompt_tokens}")
    if not res.ok:
        print("CALIBRATION_FAIL")
        return 3
    print("CALIBRATION_OK")
    return 0


def cmd_annotate(args) -> int:
    work = Path(args.work)
    if not args.skip_calibration_check and not (work / MARKER).exists():
        print(f"refusing to annotate: {work / MARKER} is missing - run `calibrate` first "
              f"(--skip-calibration-check is for local dry runs only)")
        return 2
    apps = [args.app] if args.app else args.apps.split(",") if args.apps else apps_by_size(args.data_root)
    indices = [int(x) for x in args.frames.split(",")] if args.frames else None
    per_app = {}
    if args.batch is not None:
        if args.limit or args.frames:
            print("refusing: --batch picks its own frames - drop --limit / --frames")
            return 2
        for app in apps:
            fr = batch_frame(args.data_root, app, args.batch, args.batch_seed)
            if fr is None:
                print(f"BATCH_SKIP {app} frames={len(list_frames(args.data_root, app))}")
            else:
                per_app[app] = [fr.index]
        apps = [a for a in apps if a in per_app]
        print(f"BATCH {args.batch} seed={args.batch_seed} apps={len(apps)} frames={len(apps)}")
    gen = _generator(args)

    async def go():
        totals = Counter()
        try:
            for app in apps:
                cfg = RunConfig(work=work, data_root=Path(args.data_root), app=app, concurrency=args.concurrency,
                                early_stop=args.early_stop, allow_mixed=args.allow_mixed, limit=args.limit,
                                indices=per_app.get(app, indices), run_id=args.run_id, model=args.model)
                totals.update(await run_app(gen, cfg))
        finally:
            await gen.aclose()
        return totals
    try:
        totals = asyncio.run(go())
    except MixedPromptError as e:
        print(f"MIXED_PROMPTS {e}")
        return 4
    tail = " ".join(f"{k}={v}" for k, v in sorted(totals.items()))
    if totals.get("transport_error") or totals.get("error"):
        print(f"ANNOTATE_INCOMPLETE {tail}")
        return 1
    print(f"ANNOTATE_DONE {tail}")
    return 0


def cmd_reset(args) -> int:
    paths = sorted((Path(args.work) / "records" / args.app).glob("*.json"))
    if args.frames:
        wanted = {int(x) for x in args.frames.split(",")}
        paths = [p for p in paths if int(p.stem) in wanted]
    changed = 0
    for p in paths:
        rec = load_record(p)
        if args.parse_only:
            try:
                changed += reparse(rec, args.stage)
            except ValueError as e:
                print(f"RESET_FAIL {e}")
                return 2
        else:
            reset_from(rec, args.stage)
        save_record(p, rec)
    print(f"RESET {args.app} records={len(paths)} stage={args.stage}" + (f" parse_only changed={changed}" if args.parse_only else ""))
    return 0


def cmd_export(args) -> int:
    write_export(args.work, args.out, args.image_root, _tiers(args.tier), args.include_benchmark, args.flat)
    return 0


def cmd_qc(args) -> int:
    write_qc(args.work, args.out, args.data_root, args.n, args.seed)
    return 0


def cmd_batch_report(args) -> int:
    write_batch_report(args.work, args.data_root, args.out, args.batch, args.batch_seed)
    return 0


def cmd_grade(args) -> int:
    tiers = _tiers(args.tier)
    items = build_sample(args.work, args.n, args.seed, tiers, resample=args.resample)
    print(f"sample of {len(items)} rows -> {sample_path(args.work, tiers)}")
    return _serve(sample_path(args.work, tiers), args.root, verdicts_path(args.work), args.port, not args.no_open, not args.all_apps)


def cmd_apply_grades(args) -> int:
    applied = apply_grades(args.work, args.verdicts)
    print(f"APPLIED_GRADES {applied}")
    verdicts = load_verdicts(args.verdicts or verdicts_path(args.work))
    for tiers in (("A", "B"), ("C",)):
        sp = sample_path(args.work, tiers)
        if sp.exists():
            s = summarise(verdicts, json.loads(sp.read_text(encoding="utf-8"))["items"])
            print(f"GRADES {''.join(tiers)} " + " ".join(f"{k}={v}" for k, v in s.items()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotator", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="resolution probe + 60 benchmark steps; writes work/CALIBRATION_OK")
    p.add_argument("--dataset", required=True, help="~/MedicalDataset (Annotations/ + Images/)")
    p.add_argument("--probe", required=True, help="one 1920x1080 training frame for the token-count probe")
    p.add_argument("--work", default="work")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260912)
    _server_args(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("annotate", help="run every stage over an app's frames, resumable")
    p.add_argument("--data-root", required=True)
    p.add_argument("--work", default="work")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--app")
    g.add_argument("--apps", help="comma-separated app folder names")
    g.add_argument("--all", action="store_true", help="every app, ascending by frame count")
    p.add_argument("--limit", type=int, help="N frames spread evenly across each app (the pilot)")
    p.add_argument("--frames", help="comma-separated frame indices")
    p.add_argument("--batch", type=int, help="review batch k: one seeded frame per app, never repeated across batches")
    p.add_argument("--batch-seed", type=int, default=BATCH_SEED)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--allow-mixed", action="store_true")
    p.add_argument("--run-id", default="")
    p.add_argument("--skip-calibration-check", action="store_true")
    _server_args(p)
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("reset", help="clear a stage and every later one (or --parse-only: re-interpret stored raws)")
    p.add_argument("--work", default="work")
    p.add_argument("--app", required=True)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--frames")
    p.add_argument("--parse-only", action="store_true")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("export", help="records -> MedSPOT json + manifest")
    p.add_argument("--work", default="work")
    p.add_argument("--out", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--tier", choices=("AB", "C"), default="AB")
    p.add_argument("--include-benchmark", action="store_true")
    p.add_argument("--flat", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("qc", help="overlays + qc_stats.json")
    p.add_argument("--work", default="work")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="reports")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260912)
    p.set_defaults(func=cmd_qc)

    p = sub.add_parser("batch-report", help="one review PNG per app for a batch: final box + instructions")
    p.add_argument("--work", default="work")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="reports/qwen_batches")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--batch-seed", type=int, default=BATCH_SEED)
    p.set_defaults(func=cmd_batch_report)

    p = sub.add_parser("grade", help="build the seeded grading sample and serve the review tool")
    p.add_argument("--work", default="work")
    p.add_argument("--root", required=True, help="local folder with <App>/<frame>.png")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--tier", choices=("AB", "C"), default="AB")
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--all-apps", action="store_true")
    p.add_argument("--resample", action="store_true",
                    help="rebuild the sample from the current population instead of refreshing the existing one")
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("apply-grades", help="write review verdicts into the records")
    p.add_argument("--work", default="work")
    p.add_argument("--verdicts")
    p.set_defaults(func=cmd_apply_grades)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
