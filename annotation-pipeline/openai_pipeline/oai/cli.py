"""oai: init-data | calibrate | plan | run | grade | approve | stop | status | export.

Every command prints a positive marker on success - DATA_OK, CALIBRATION_OK, PLANNED, BATCH_DONE
(under the batch transport also ROUND_SUBMITTED / ROUND_PENDING per step), GRADES, APPROVED,
STOPPED, EXPORT_OK - and a named refusal otherwise. Exit codes: 0 ok, 1 the
batch is not finished, 2 refusal or missing precondition, 3 a gate failed. The key comes from
OPENAI_API_KEY and never leaves this process."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from annotator.calibrate import calibrate
from annotator.calling import TransportError
from annotator.export import write_export
from annotator.frames import is_benchmark_app, list_apps, list_frames
from annotator.grading import apply_grades, grade_rows, load_verdicts, summarise, verdicts_path
from annotator.prompts import prompt_hash
from annotator.records import record_path

from . import batches as bt
from . import rounds
from .batchapi import ReplyCache, clear_poisoned
from .config import Config, ConfigError, config_hash, load_config
from .meter import BudgetExceeded, Caps, Meter, Metered, ledger_total, read_ledger, reprice
from .pricing import BATCH_DISCOUNT, PRICES_AS_OF, estimate_usd_per_frame
from .run import remaining, run_batch

MIN_TOKENS = 2200                 # a native 1920x1080 frame is 2,448 input tokens; a downscaled one far less
CALIBRATE_STEPS = 60
CALIBRATE_SEED = 20260912
CALIBRATE_CAP_USD = 1.0


def _generator(cfg: Config, key: str):
    """The one seam tests replace: the real transport."""
    from .client import OpenAIGenerator
    return OpenAIGenerator(cfg.models, cfg.efforts, key, concurrency=cfg.concurrency)


def _batch_client(cfg: Config, key: str):
    """The batch transport's seam: the Files and Batches endpoints."""
    from .batchclient import BatchClient
    return BatchClient(key)


def frame_rate(cfg: Config, measured) -> float:
    """$/frame for a projection under cfg.transport: the last measured batch's rate, converted when
    it was measured under the other transport, else the built-in estimate."""
    rate = measured.usd_per_frame if measured else estimate_usd_per_frame(cfg.models["ground"])
    if measured and measured.transport != cfg.transport:
        return rate * BATCH_DISCOUNT if cfg.transport == "batch" else rate / BATCH_DISCOUNT
    if not measured and cfg.transport == "batch":
        return rate * BATCH_DISCOUNT
    return rate


_sleep = time.sleep                  # `run --wait` polls with this; tests replace it


def _serve(sample_file, root, verdict_path, port, open_browser, only_local) -> int:
    from tools.review_server import serve            # the shared keyboard grader; sys.path is set in cmd_grade
    return serve(sample_file, root, verdict_path, port, open_browser, only_local)


def key_suffix(key) -> str:
    return f"KEY ...{key[-4:]}" if key else "NO_API_KEY"


def _key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("NO_API_KEY: set OPENAI_API_KEY in this shell session (never in a file)")
    return key


def marker_path(cfg: Config, mode=None) -> Path:
    return cfg.work / f"CALIBRATION_OK.{cfg.models['ground']}.{mode or cfg.coord_mode}"


def _corpus(cfg: Config) -> int:
    return sum(len(list_frames(cfg.data, a)) for a in list_apps(cfg.data)) if Path(cfg.data).is_dir() else 0


def _first_frame(cfg: Config):
    """The first frame under data, or None when init-data has not run yet."""
    for app in (list_apps(cfg.data) if Path(cfg.data).is_dir() else []):
        frames = list_frames(cfg.data, app)
        if frames:
            return frames[0].path
    return None


def cmd_init_data(args, cfg) -> int:
    from .data import init_data
    counts = init_data(cfg)
    return 0 if counts["apps"] >= cfg.expected_apps else 2


def cmd_calibrate(args, cfg) -> int:
    key = _key()
    if not key:
        return 2
    mode = args.coord_mode or cfg.coord_mode
    probe_path = args.probe or _first_frame(cfg)
    if not probe_path:
        print(f"DATA_INCOMPLETE no frames under {cfg.data}: run init-data first")
        return 2
    probe = Image.open(probe_path).convert("RGB")
    gen = _generator(cfg, key)
    meter = Meter(cfg.work / "ledger.jsonl", "calibrate", Caps(CALIBRATE_CAP_USD, cfg.run_usd))
    metered = Metered(gen, meter, "calibrate", mode)

    async def go():
        try:
            return await calibrate(metered, cfg.benchmark, probe, args.n, args.seed, min_tokens=MIN_TOKENS, mode=mode)
        finally:
            await gen.aclose()
    try:
        res = asyncio.run(go())
    except BudgetExceeded as e:
        print(f"CALIBRATION_STOPPED reason=budget {e}")
        return 2
    cfg.work.mkdir(parents=True, exist_ok=True)
    model = cfg.models["ground"]
    (cfg.work / f"calibration.{model}.{mode}.json").write_text(json.dumps({**asdict(res), "model": model, "cost_usd": meter.batch_total}, indent=1), encoding="utf-8")
    if not res.resolution_ok:
        print(f"RESOLUTION_FAIL prompt_tokens={res.prompt_tokens} < {MIN_TOKENS}: the frame was not sent at native resolution")
        return 3
    print(f"CALIBRATION hits={res.hits}/{res.total} accuracy={res.accuracy:.3f} prompt_tokens={res.prompt_tokens} cost=${meter.batch_total:.4f} mode={mode} model={model}")
    if not res.ok:
        print("CALIBRATION_FAIL")
        return 3
    marker_path(cfg, mode).write_text(f"{bt.now()} accuracy={res.accuracy:.3f} prompt_tokens={res.prompt_tokens}\n", encoding="utf-8")
    print("CALIBRATION_OK")
    return 0


def cmd_plan(args, cfg) -> int:
    apps = list_apps(cfg.data) if Path(cfg.data).is_dir() else []
    if len(apps) < cfg.expected_apps:
        print(f"DATA_INCOMPLETE apps={len(apps)} expected={cfg.expected_apps}: run init-data first")
        return 2
    path = cfg.work / "batches.json"
    try:
        plan = bt.load_plan(path, cfg.seed)
        orders = bt.app_orders(cfg.data, cfg.seed, skip=is_benchmark_app if cfg.exclude_benchmark_apps else None)
        size = args.next or bt.next_size(plan, cfg.sizes)
        batch = bt.plan_batch_balanced(plan, orders, size) if cfg.balanced else bt.plan_batch(plan, bt.interleave(orders), size)
    except bt.PlanError as e:
        print(f"PLAN_REFUSED {e}")
        return 2
    bt.save_plan(path, plan)
    remaining_frames = sum(len(v) for v in orders.values()) - len(bt.used_frames(plan))
    print(f"PLANNED batch={batch.id} size={batch.size} apps={len({f[0] for f in batch.frames})} remaining={remaining_frames}")
    return 0


def cmd_run(args, cfg) -> int:
    key = _key()
    if not key:
        return 2
    marker = marker_path(cfg)
    if not marker.exists():
        print(f"refusing to run: {marker} is missing - run `calibrate --coord-mode {cfg.coord_mode}` first")
        return 2
    path = cfg.work / "batches.json"
    try:
        plan = bt.load_plan(path, cfg.seed)
        batch = bt.check_runnable(plan, args.batch, args.after_stop)
    except bt.PlanError as e:
        print(f"RUN_REFUSED {e}")
        return 2
    if batch.prompt_hash and (batch.prompt_hash, batch.config_hash) != (prompt_hash(cfg.coord_mode), config_hash(cfg)):
        print(f"RUN_REFUSED batch {batch.id} ran under a different prompt/config (coord_mode, models, effort or early_stop changed); "
              f"finish or stop it under the config it started with")
        return 2
    todo = remaining(cfg, batch)
    measured = bt.last_measured(plan)
    rate = frame_rate(cfg, measured)
    projection = len(todo) * rate
    batch_cap = cfg.batch_usd or max(2.0, 1.5 * len(batch.frames) * rate)
    rows = read_ledger(cfg.work / "ledger.jsonl")
    run_total = ledger_total(rows)
    spent = ledger_total(rows, batch.id)
    if spent + projection > batch_cap or run_total + projection > cfg.run_usd:
        hint = f" spent=${spent:.4f}: set [caps].batch_usd above ${spent + projection:.2f} to resume" if spent else ""
        print(f"BUDGET_REFUSED projection=${projection:.4f} batch_cap=${batch_cap:.2f} run_total=${run_total:.4f} run_cap=${cfg.run_usd:.2f}{hint}")
        return 2
    batch.transport = cfg.transport
    bt.put_batch(plan, bt.transition(batch, "running"))
    bt.save_plan(path, plan)
    print(f"BATCH_START {batch.id} frames={len(todo)} projection=${projection:.4f} cap=${batch_cap:.2f} "
          f"rate={'measured' if measured else 'estimate'} transport={cfg.transport}")
    meter = Meter(cfg.work / "ledger.jsonl", batch.id, Caps(batch_cap, cfg.run_usd))
    if cfg.transport == "batch":
        return _run_batch_transport(cfg, path, batch, key, meter, args)
    gen = _generator(cfg, key)

    async def go():
        try:
            return await run_batch(cfg, batch, gen, meter)
        finally:
            await gen.aclose()
    result = asyncio.run(go())
    plan = bt.load_plan(path, cfg.seed)                # a `plan`/`stop` in another shell meanwhile must not be discarded
    bt.put_batch(plan, result)                         # this batch's row is owned by this run; a concurrent `stop` of it is overwritten
    bt.save_plan(path, plan)
    return 0 if result.state == "done" else 1


EXIT_BY_OUTCOME = {"done": 0, "incomplete": 1, "pending": 1, "submitted": 1, "stopped": 1, "refused": 2}
WAIT_MAX_TRANSPORT_FAILURES = 60      # consecutive unreachable steps --wait tolerates (an hour at poll_seconds=60)


def _run_batch_transport(cfg: Config, path, batch, key, meter, args) -> int:
    """One step per invocation; --wait repeats while a round is open. A step that
    cannot reach the API ends a single run, but under --wait it is just retried on the next poll:
    every state change is on disk before the next call, and a DNS blip must not kill a night's loop."""
    client = _batch_client(cfg, key)
    if args.retry_poisoned:
        bdir = rounds.batch_dir(cfg, batch)
        n = clear_poisoned(bdir, ReplyCache(bdir / "replies.jsonl"))
        print(f"RETRY_POISONED batch={batch.id} frames={n}")
    failures = 0
    try:
        while True:
            try:
                outcome = rounds.step(cfg, batch, client, meter)
            except TransportError as e:
                failures += 1
                if not args.wait or failures > WAIT_MAX_TRANSPORT_FAILURES:
                    raise
                print(f"TRANSPORT_RETRY {failures}/{WAIT_MAX_TRANSPORT_FAILURES}: {e}")
                _sleep(cfg.poll_seconds)
                continue
            failures = 0
            plan = bt.load_plan(path, cfg.seed)
            bt.put_batch(plan, batch)
            bt.save_plan(path, plan)
            if args.wait and outcome in ("pending", "submitted"):
                _sleep(cfg.poll_seconds)
                continue
            return EXIT_BY_OUTCOME[outcome]
    finally:
        client.close()


def cmd_status(args, cfg) -> int:
    print(key_suffix(os.environ.get("OPENAI_API_KEY", "").strip()))
    rows = read_ledger(cfg.work / "ledger.jsonl")
    total = ledger_total(rows)
    print(f"LEDGER total=${total:.4f} of run_usd=${cfg.run_usd:.2f} calls={len(rows)} repriced=${reprice(rows):.4f} (prices as of {PRICES_AS_OF})")
    plan = bt.load_plan(cfg.work / "batches.json", cfg.seed) if (cfg.work / "batches.json").exists() else {"batches": []}
    print(f"{'id':>3}  {'state':<9} {'frames':>6} {'cost':>9} {'$/frame':>9} {'correct':>8} {'graded':>6}  models")
    for b in bt.batches(plan):
        rate = f"{b.usd_per_frame:.4f}" if b.usd_per_frame else "-"
        correct = f"{b.correct_rate:.3f}" if b.correct_rate is not None else "-"
        extra = rounds.summary(cfg, b) if b.state == "running" else ""
        print(f"{b.id:>3}  {b.state:<9} {b.size:>6} {b.cost_usd:>9.4f} {rate:>9} {correct:>8} {b.graded_rows:>6}  "
              f"{','.join(sorted(set(b.models.values()))) or '-'}{' FORCED' if b.forced else ''}{' stopped: ' + b.stop_reason if b.stop_reason else ''}{extra}")
    corpus = _corpus(cfg)
    measured = bt.last_measured(plan)
    rate = frame_rate(cfg, measured)
    print(f"PROJECTION corpus={corpus} frames at ${rate:.4f}/frame {cfg.transport} ({'batch ' + str(measured.id) if measured else 'estimate'}) "
          f"= ${corpus * rate:.2f}")
    return 0


GRADE_MAX_ROWS = 100
GRADE_TIER_C_ROWS = 20
EXPORTS = (("", ("A", "B"), False), ("_all", ("A", "B"), True), ("_lowconf", ("C",), True))   # suffixes on cfg.dataset


def batch_sample(cfg: Config, batch, seed) -> Path:
    """The batch's shipped rows (all of them up to 100, else a seeded 100) plus up to 20 tier-C
    rows, in the sample format tools/review_server.py reads."""
    wanted = {(app, name) for app, name, _ in batch.frames}
    ab = [r for r in grade_rows(cfg.work, ("A", "B")) if (r["app"], r["frame"]) in wanted]
    c = [r for r in grade_rows(cfg.work, ("C",)) if (r["app"], r["frame"]) in wanted]
    rng = random.Random(seed)
    if len(ab) > GRADE_MAX_ROWS:
        ab = rng.sample(ab, GRADE_MAX_ROWS)
    if len(c) > GRADE_TIER_C_ROWS:
        c = rng.sample(c, GRADE_TIER_C_ROWS)
    items = sorted(ab + c, key=lambda i: (i["app"], i["frame_index"]))
    path = cfg.work / "review" / f"sample_batch{batch.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n": GRADE_MAX_ROWS, "seed": seed, "tiers": ["A", "B", "C"], "batch": batch.id, "items": items}, indent=1) + "\n",
                    encoding="utf-8")
    return path


def _plan_and_batch(cfg: Config, batch_id):
    plan = bt.load_plan(cfg.work / "batches.json", cfg.seed)
    return plan, bt.get_batch(plan, batch_id)


def cmd_grade(args, cfg) -> int:
    try:
        plan, batch = _plan_and_batch(cfg, args.batch)
        if batch.state not in ("done", "graded"):
            raise bt.PlanError(f"batch {batch.id} is {batch.state}; run it to completion first")
    except bt.PlanError as e:
        print(f"GRADE_REFUSED {e}")
        return 2
    sample = batch_sample(cfg, batch, args.seed)
    items = json.loads(sample.read_text(encoding="utf-8"))["items"]
    if not items:
        print(f"GRADES_NONE batch={batch.id}: no shipped rows to grade")
        return 3
    print(f"sample of {len(items)} rows -> {sample}")
    sys.path.insert(0, str(cfg.shared_repo))                         # tools.review_server lives in the shared checkout
    _serve(sample, cfg.data, verdicts_path(cfg.work), args.port, not args.no_open, True)
    applied = apply_grades(cfg.work)
    verdicts = load_verdicts(verdicts_path(cfg.work))
    ab = [i for i in items if i["tier"] in ("A", "B")]
    s, sc = summarise(verdicts, ab), summarise(verdicts, [i for i in items if i["tier"] == "C"])
    if not s["graded"]:
        print(f"GRADES_NONE batch={batch.id} graded=0 of {len(ab)}")
        return 3
    batch.correct_rate, batch.graded_rows = s["correct_rate"], s["graded"]
    bt.put_batch(plan, bt.transition(batch, "graded"))
    bt.save_plan(cfg.work / "batches.json", plan)
    print(f"GRADES batch={batch.id} correct_rate={s['correct_rate']:.3f} correct={s['correct']} loose={s['loose']} wrong={s['wrong']} "
          f"corrected={s['corrected']} n={len(ab)} graded={s['graded']} tierC_graded={sc['graded']} applied={applied}")
    print(f"COST batch=${batch.cost_usd:.4f} usd_per_frame={batch.usd_per_frame} total=${ledger_total(read_ledger(cfg.work / 'ledger.jsonl')):.4f}")
    return 0


def cmd_approve(args, cfg) -> int:
    try:
        plan, batch = _plan_and_batch(cfg, args.batch)
        bt.check_approvable(batch, cfg.gate, args.force, args.ungraded)
    except bt.PlanError as e:
        print(f"APPROVE_REFUSED {e}")
        return 3 if "gate" in str(e) else 2
    if args.ungraded:                              # done -> graded (0 rows) -> approved; the ladder itself is unchanged
        batch.forced = True
        bt.transition(batch, "graded")
    else:
        batch.forced = bool(args.force and batch.correct_rate < cfg.gate)
    bt.put_batch(plan, bt.transition(batch, "approved"))
    bt.save_plan(cfg.work / "batches.json", plan)
    rate = f"{batch.correct_rate:.3f}" if batch.correct_rate is not None else "-"
    print(f"APPROVED batch={batch.id} correct_rate={rate} usd_per_frame={batch.usd_per_frame}"
          f"{' FORCED' if batch.forced else ''}{' ungraded' if args.ungraded else ''}")
    return 0


def cmd_stop(args, cfg) -> int:
    try:
        plan, batch = _plan_and_batch(cfg, args.batch)
        batch.stop_reason = args.reason
        bt.put_batch(plan, bt.transition(batch, "stopped"))
    except bt.PlanError as e:
        print(f"STOP_REFUSED {e}")
        return 2
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if cfg.transport == "batch" and key:
        client = _batch_client(cfg, key)
        try:
            cancelled = rounds.cancel_open(cfg, batch, client)
        finally:
            client.close()
        if cancelled:
            print(f"CANCELLED jobs={','.join(cancelled)}")
    bt.save_plan(cfg.work / "batches.json", plan)
    print(f"STOPPED batch={batch.id} reason={args.reason}")
    return 0


def stage_records(cfg: Config, batches) -> Path:
    """Copies of the given batches' records under work/export_stage, so the shared write_export
    (which walks a whole work dir) sees only approved frames."""
    stage = cfg.work / "export_stage"
    shutil.rmtree(stage, ignore_errors=True)
    for b in batches:
        for app, name, index in b.frames:
            src = record_path(cfg.work, app, index)
            if src.exists():
                dst = record_path(stage, app, index)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    return stage


def cmd_export(args, cfg) -> int:
    """Every approved batch under [export].dataset - or, with --batches, an approved subset under
    --name (an increment to train on from an earlier checkpoint)."""
    plan = bt.load_plan(cfg.work / "batches.json", cfg.seed) if (cfg.work / "batches.json").exists() else {"batches": []}
    approved = bt.approved(plan)
    name = args.name or cfg.dataset
    if not re.fullmatch(r"[a-z0-9_]+", name):
        print("EXPORT_REFUSED --name must be a lowercase slug ([a-z0-9_]+)")
        return 2
    if args.batches:
        wanted = [int(x) for x in args.batches.split(",") if x.strip()]
        try:
            chosen = [bt.get_batch(plan, i) for i in wanted]
        except bt.PlanError as e:
            print(f"EXPORT_REFUSED {e}")
            return 2
        for b in chosen:
            if b.state != "approved":
                print(f"EXPORT_REFUSED batch {b.id} is {b.state}, not approved")
                return 2
        approved = chosen
    if not approved:
        print("EXPORT_NONE no approved batches")
        return 2
    stage = stage_records(cfg, approved)
    runs = [{"batch": b.id, "models": b.models, "effort": b.efforts, "coord_mode": b.coord_mode} for b in approved]
    same = lambda key: runs[0][key] if all(r[key] == runs[0][key] for r in runs) else None   # noqa: E731
    extra = {"provider": "openai", "batches": [b.id for b in approved], "runs": runs,
             "models": same("models"), "effort": same("effort"), "coord_mode": same("coord_mode"),
             "cost_usd": round(sum(b.cost_usd for b in approved), 4)}
    dirs = []
    for suffix, tiers, include_benchmark in EXPORTS:
        for flat in (False, True):
            out = cfg.export / (name + suffix + ("_flat" if flat else ""))
            write_export(stage, out, cfg.image_root, tiers, include_benchmark, flat, dataset=name)
            m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            (out / "manifest.json").write_text(json.dumps({**m, **extra}, indent=1) + "\n", encoding="utf-8")
            dirs.append(str(out))
    verify = cfg.shared_repo / "hpc" / "verify_annotations.py"
    rc = subprocess.run([sys.executable, str(verify), *dirs]).returncode
    if rc != 0:
        print("EXPORT_FAIL verify_annotations reported a problem (see IMAGES_FAIL above)")
        return 3
    print(f"EXPORT_OK batches={','.join(str(b.id) for b in approved)} frames={sum(b.size for b in approved)} dirs={len(dirs)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-data", help="pull the missing app folders and the benchmark set; make the data junction")
    p.set_defaults(func=cmd_init_data)

    p = sub.add_parser("calibrate", help="resolution probe + benchmark steps for one coordinate mode; writes CALIBRATION_OK.<model>.<mode>")
    p.add_argument("--coord-mode", choices=("pixels", "thousand"))
    p.add_argument("--probe", help="one native frame for the token-count probe (default: the first frame under data)")
    p.add_argument("--n", type=int, default=CALIBRATE_STEPS)
    p.add_argument("--seed", type=int, default=CALIBRATE_SEED)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("plan", help="append the next batch to work/batches.json")
    p.add_argument("--next", type=int, help="size of the next batch instead of the configured sequence")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="run one planned batch (resumable); requires the previous batch approved. "
                                    "Under [run].transport = \"batch\" one step per call: --wait polls until the batch is done")
    p.add_argument("batch", type=int)
    p.add_argument("--after-stop", action="store_true", help="run even though the previous batch was stopped")
    p.add_argument("--wait", action="store_true", help="batch transport: keep stepping every [run].poll_seconds until done")
    p.add_argument("--retry-poisoned", action="store_true", help="batch transport: forget the frames set aside after repeated errors")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="batches, spend, projection")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("grade", help="grade a finished batch in the keyboard tool; applies the verdicts and prints GRADES")
    p.add_argument("batch", type=int)
    p.add_argument("--seed", type=int, default=CALIBRATE_SEED)
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("approve", help="release the next batch; refuses below the gate unless --force, "
                                        "or skips grading entirely with --ungraded (recorded as FORCED)")
    p.add_argument("batch", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--ungraded", action="store_true", help="approve a finished batch without grading it")
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("stop", help="mark a batch stopped; later batches need --after-stop")
    p.add_argument("batch", type=int)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("export", help="MedSPOT exports over every approved batch, then verify_annotations")
    p.add_argument("--batches", help="comma-separated approved batch ids to export instead of all of them (e.g. 8,9)")
    p.add_argument("--name", help="dataset name for the output dirs and file stems (default [export].dataset)")
    p.set_defaults(func=cmd_export)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as e:
        print(f"CONFIG_REFUSED {e}")
        return 2
    return args.func(args, cfg)
