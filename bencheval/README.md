# bencheval — one harness for every model and benchmark

Evaluates our checkpoints and the published baselines under one set of per-benchmark protocols, with resumable runs
and a results registry.

| Path | What |
|---|---|
| `benchev/` | The package: configs, benchmark loaders, scorers, model adapters, the 1-step / 2-step / zoom + fusion drivers (`inference.py`, `fusion.py`), the resumable runner (`run.py`), registry and report |
| `configs/benchmarks/`, `configs/protocols/` | One file per benchmark and its protocol: image limits, pre-resize, zoom crop and factor |
| `configs/models/` | One file per model: adapter, prompt recipe, checkpoint, image limit |
| `vendor/` | The Qwen3.5 GUI-AIMA port, the MedSPOT scorer, and the official GUI-AIMA / GUI-Actor code (`vendor/VENDOR.md`) |
| `scripts/` | `fetch_screenspot_v1.py`, `smoke_check.py` |
| `tests/` | CPU tests (`python -m pytest`); GPU tests are marked `gpu` and skipped by default |

## Benchmarks

`--data` (default `bencheval/data`, or `$BENCHEV_DATA`) holds one folder per benchmark:

| Bench id | Folder | Samples | Notes |
|---|---|---|---|
| `ssv1` | `screenspot_v1/` | 1,272 | `subset.json` + images (`scripts/fetch_screenspot_v1.py`) |
| `ssv2` | `screenspot_v2_full/` | 1,272 | `subset.json` + images (`../training/scripts/fetch_screenspot_v2.py --per_split 0`) |
| `sspro` | `screenspot_pro/` | 1,581 | `annotations/all.json` + `images/` (`../training/scripts/fetch_screenspot_pro.py`) |
| `oswg`, `oswg_refined` | `osworld_g/` | 564 | `OSWorld-G.json` / `OSWorld-G_refined.json`, `classification_result.json`, `images/` |
| `uivision` | `ui_vision/` | 5,479 | `annotations/element_grounding/element_grounding_<split>.json` + images |
| `mmbench_l2` | `mmbench_gui/` | 3,594 | `L2_annotations.json` + `MMBench-GUI-OfflineImages.zip` unpacked under `images/` |
| `medspot` | `medspot/` | 216 tasks / 603 steps | `Annotations/*.json` + images; scored with the strict MedSPOT scorer (TCA, step accuracy) and the ±14 px rule |

## Models

`configs/models/ours-{2b,4b,9b}-ab.yaml` are the general models (stages A + B) and `ours-*-ab-clin.yaml` the models
after the clinical stage. Their `ckpt` points to `checkpoints/<id>`: put the exported `final_bf16` directories
there, edit `ckpt`, or remap prefixes with `BENCHEV_PATH_MAP="checkpoints/=/my/path/"`. Published models point to
their Hugging Face repositories; the pinned revision is in each file's `notes`. `qwen35-*-base` are the zero-shot
base models answering with text coordinates.

Environments: our models, the Qwen3.5 base models and every `env: hf` model run in the training environment
(`../training/scripts/setup_env.sh`). `gui_aima_upstream` and `gui_actor` need the upstream GUI-AIMA environment
(torch 2.6, transformers 4.51.3, flash-attn 2.7.3, qwen-vl-utils 0.0.8). `deepseek-vl2-small` needs the official
`deepseek_vl2` package (transformers 4.38); `cogagent-9b` and `seeclick` run through `trust_remote_code` in their
model cards' environments.

## Running

    python -m benchev.cli run --model ours-4b-ab-clin --bench medspot --mode both     # 1-step and 2-step (zoom)
    python -m benchev.cli run --model ours-4b-ab-clin --bench ssv2 --mode zf          # zoom + fusion
    python -m benchev.cli run --model ours-4b-ab-clin --bench core --mode 1step --limit 20 --largest   # smoke on the 20 largest images
    python -m benchev.cli run --model gui-aima-3b --bench sspro --mode 2step --ingest
    python -m benchev.cli report                                                       # results/LEADERBOARD.md

`--bench core` is SS-v2, SS-Pro and MedSPOT; `--bench suite` is all eight. Predictions are appended to
`results/raw/<model>/<bench>/<mode>/predictions.jsonl`, so an interrupted run resumes where it stopped; a run refuses
to reuse predictions made with a different checkpoint, adapter, prompt, image limit or protocol. `--ingest` adds the
finished run to `results/registry.jsonl`, from which `report` builds the tables.

**Zoom + fusion (`zf`).** Pass 1 is the 1-step pass, pass 2 grounds the protocol's crop of the native image around the
first click, zoomed; the full-view map is resampled onto the crop grid and the click is read from
`p_full^λ · p_zoom`. The headline number (`acc`) is `fuse_cv`, where λ ∈ {0.25, 0.5, 1, 2} is chosen by two-fold
cross-fitting and never on the half it is scored on; `acc_l<λ>`, `acc_zoom` and `acc_first` are reported alongside.
`zf` needs a model that returns its grounding map (our checkpoints). On MedSPOT, where the task-level metric needs
task-level cross-fitting, `zf` is not supported by this harness.
