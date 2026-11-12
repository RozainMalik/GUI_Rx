# Clinical GUI-grounding annotation pipeline

This pipeline turns screenshots of medical-imaging software into click-grounding training data in the MedSPOT
task/step/action format.
Each output row is an instruction, the target UI element and its bounding box. The models are OpenAI
models called through the Responses API. The pipeline runs in graded batches, and every batch has a hard
cost cap.

This is the code that produced the released clinical training set: 11 batches, 3,447 frames and 10,341
instruction-box pairs (tiers A+B), about $34 of API cost in total. The dataset itself is released separately.

## Layout

| Folder | What |
|---|---|
| `annotator_core/` | The shared five-stage pipeline (`annotator` package): propose → describe → ground → verify → refine. It also holds the selection/tiering logic, the MedSPOT exporter, the QC overlays, the keyboard grading tool (`tools/review_server.py`) and the export checker (`hpc/verify_annotations.py`). |
| `openai_pipeline/` | The OpenAI driver (`oai` package): batch planning, cost meter and caps, the sync and Batch API transports, calibration, grading, approval and export. `config.toml` holds every setting. |

## Setup

Python 3.11 or newer.

    cd medspot-annotation-pipeline
    python -m venv .venv
    .venv\Scripts\activate                   # Windows;  source .venv/bin/activate on Linux/macOS
    pip install -e annotator_core -e "openai_pipeline[test]"

The `-e` (editable) install is required. `oai` finds the grading tool and the export checker in the
`annotator_core` checkout it was installed from.

**Check the install (no data, no API key, no cost):**

    cd annotator_core  && python -m pytest -q     # 227 passed
    cd ../openai_pipeline && python -m pytest -q  # 134 passed, incl. an end-to-end plan -> run -> grade -> approve -> export
                                                  # against a scripted fake OpenAI

## Data (not included)

Run every `oai` command from `openai_pipeline/`. Paths in `config.toml` resolve against that folder.

    openai_pipeline/data/<App>/<App>_NNN.png              # the screenshots, one folder per app (30 apps, ~7,900 frames)
    openai_pipeline/benchmark/Annotations/*.json          # the MedSPOT benchmark; only `calibrate` uses it
    openai_pipeline/benchmark/Images/...

To use the data where it already sits, point `[paths].data` and `[paths].image_root` at that folder
instead, and `[paths].benchmark` at the benchmark folder. `plan` refuses until `data/` holds
`[batches].expected_apps` app folders (30). Lower that value to work on a subset. You don't need
`oai init-data` when you fill `data/` by hand. It only pulls folders from a remote machine over ssh (`[remote]` in `config.toml`).

## Running a batch

    set OPENAI_API_KEY=sk-...                # PowerShell: $env:OPENAI_API_KEY="sk-..."  (only for the session; never in a file)
    python -m oai calibrate --coord-mode pixels   # ~$0.05: probes resolution + 60 benchmark steps -> CALIBRATION_OK
    python -m oai plan                       # PLANNED batch=1 size=25 ...
    python -m oai run 1                      # BATCH_DONE ... cost=$... ; overlays in reports/overlays/1/
    python -m oai grade 1                    # browser grader: C correct, L loose, W wrong, drag + Enter to correct
    python -m oai approve 1                  # gate: correct_rate >= 0.75 (or --force / --ungraded)
    python -m oai export                     # export/gui_grounding[_all|_lowconf][_flat]/ + EXPORT_OK
    python -m oai status                     # any time: batches, spend, projection

Then repeat `plan` → `run N` → `grade N` → `approve N`. Batch sizes grow 25, 50, 100, 250, 500, then
1000. `run` refuses until the previous batch is approved, and it refuses when the projected cost would
pass `[caps].batch_usd` or `[caps].run_usd`. Change models or effort only between batches. Each batch
records the config and prompt hash it ran under. `openai_pipeline/README.md` documents every marker,
stop reason, and the Batch API transport.

## Outputs

- `export/gui_grounding_all/gui_grounding_train.json`: the training set, tiers A+B, all apps.
  `gui_grounding/` leaves out the 6 apps that appear in the MedSPOT benchmark, and `_lowconf/` holds
  tier C. Each has a `_flat` twin. `bbox` is `[x, y, w, h]` as a percentage (0-100) of the image.
- `reports/overlays/<batch>/<App>_<NNN>_<status>.png`: every chosen box drawn on its frame, for a
  quick visual check.
- `work/records/<App>/<NNN>.json`: the full per-frame record (each stage's output, candidates,
  selection, provenance).
- `work/ledger.jsonl`: every API call with its tokens and cost.

## Settings used for the released dataset

gpt-5.6-luna for every stage. Reasoning effort was low for propose, describe and ground, and none for
verify and refine. Pixel coordinates, sync transport, ~$0.011 per frame. Batches 8-10 used the Batch API
at ~$0.0055 per frame. The model prices in `openai_pipeline/oai/pricing.py` are as of 2026-09-16, so
check them before relying on the cost caps.
