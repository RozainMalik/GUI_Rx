# openai_pipeline — the OpenAI variant of the annotation pipeline

Runs the shared five-stage GUI-grounding pipeline (`annotator`, in `../annotator_core`) against OpenAI
models over the Responses API, on this machine, in **graded batches with hard cost caps**.

## Setup (once)

See the top-level `README.md` for install and data layout. `$PY` below is the Python of that venv.
`init-data` only pulls app folders from a remote machine over ssh; you can fill `data/` by hand instead.
Every command takes `--config` (default `config.toml`) and prints a positive marker on success.

## The gate sequence

1. `oai calibrate --coord-mode pixels` and `--coord-mode thousand` (≈ $0.05 each) → `CALIBRATION_OK`; put the
   better mode in `config.toml [run].coord_mode`. `run` refuses without `work/CALIBRATION_OK.<ground model>.<mode>`.
2. `oai plan` → `PLANNED batch=1 size=25`. `oai run 1` → `BATCH_DONE … cost=$… usd_per_frame=…` (≈ $0.25–0.40 on
   luna). Look at `reports/overlays/1/`.
3. `oai grade 1` → the keyboard tool (C correct, L loose, W wrong, drag + Enter to correct); ctrl-c when done →
   `GRADES batch=1 correct_rate=…`. `oai approve 1` (or `oai stop 1 --reason "…"`). Below 0.75 only `--force` approves.
   `oai approve 1 --ungraded` releases a finished batch on cost alone, skipping the grading step; it is recorded as
   `FORCED` with `correct_rate=-`, so `status` shows which batches nobody looked at.
4. Repeat: `plan` → `run 2` (50) → `grade` → `approve` … sizes 25, 50, 100, 250, 500, then 500s. Change
   `[models]` / `[effort]` only between batches; each batch records the config and prompt hash it ran under.
5. `oai status` any time: batches, $ and $/frame, correct-rate trend, ledger total against `[caps].run_usd`, the
   projected corpus cost. `oai export` after any approval → `export/<dataset>`, `<dataset>_all`, `<dataset>_lowconf`
   (+ `_flat` twins), each holding `<dataset>_train*.json` + `manifest.json`, and `EXPORT_OK`. `[export].dataset` names
   the set (default `gui_grounding`); `[batches].exclude_benchmark_apps = true` makes `plan` draw only from the 24
   apps outside the MedSPOT evaluation set, so nothing is spent on frames the training export would drop anyway;
   `[batches].balanced = true` makes every batch an equal share of each app's *remaining* frames.
6. Commit `work/review/` and `work/batches.json` after grading (or after `approve --ungraded`); records, ledger
   and markers stay untracked.

## Batch transport

`[run].transport = "batch"` sends every call through the Batch API at half price. `oai run <id>` then does
**one step** and exits — submit a round, or poll the open one and, once every job is ingested, run the next
pass and submit the next round — so it can be re-run from anywhere, any time; `oai run <id> --wait` keeps
stepping every `[run].poll_seconds` until `BATCH_DONE`. A 500-frame batch takes 5–8 rounds (one per stage,
plus repairs); each round is one Batch job (more if a round's file would pass 190 MB) with a 24 h window.

    $PY -m oai run 8 --wait                 # finish batch 8's last frames through the batch transport
    $PY -m oai approve 8 --ungraded
    $PY -m oai plan                         # PLANNED batch=9 size=500 apps=23 ... ([batches].balanced = true)
    $PY -m oai run 9 --wait                 # or `run 9` once an hour; both resume
    $PY -m oai approve 9 --ungraded && $PY -m oai export

Markers: `ROUND_SUBMITTED batch=9 round=3 jobs=1 requests=1625 projection=$…` (exit 1), `ROUND_PENDING … jobs=0/1
requests=812/1625` (exit 1), `ROUND_INGESTED …`, `BUDGET_REFUSED round=…` (exit 2; raise a cap and re-run — nothing was
submitted), `BATCH_STOPPED … reason=transport job=…: <the API's error>` (exit 1; fix, re-run: the round is collected
afresh), `BATCH_INCOMPLETE … poisoned=<app>/<frame>,…` (frames whose request errored in two rounds are set aside;
`run <id> --retry-poisoned` clears them). `stop <id>` also cancels the open jobs. State lives in `work/batchapi/`
(untracked): `files.json` (frame uploads, 30-day expiry), `batch<id>/round<N>.json` + `.part<k>.jsonl`, `replies.jsonl`,
`poisoned.json`. Ledger rows carry `"transport": "batch"` and the job id; `status` prices them at the batch rate.

## What stops a run

- `BUDGET_REFUSED` before a batch starts: projection (frames × last approved $/frame, or $0.016 for batch 1) over
  `[caps].batch_usd` (0 = max($2, 1.5 × projection)) or over what is left of `run_usd`.
- `BATCH_STOPPED reason=budget|anomaly|transport` mid-batch: the frame in flight stays resumable. After
  `reason=transport` or `anomaly`, `oai run <id>` continues it under the same cap; after `reason=budget` the batch
  has spent its cap - set `[caps].batch_usd` above the spend first (the auto cap never grows). Anomaly = two frames
  costing > 5× the batch median.
- Calls of the stage in flight when a stop hits are paid and discarded (the shared runner saves per stage), then
  repeated on resume - at most one stage per frame in flight, so the ledger can run a few cents ahead of the records.
- Any 4xx other than 429 stops at once with the body (a bad schema or image recurs on every frame).

## Tests

    $PY -m pytest -q          # no network, no key: FakeOpenAI scripts the Responses API behind httpx.MockTransport
