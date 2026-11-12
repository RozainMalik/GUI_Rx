# training — attention-anchored grounding on Qwen3.5

The model reads the click from its own attention, following GUI-AIMA: an anchor token appended after the instruction
attends over the visual tokens, and its attention rows, weighted per head by the visual-sink query token, form a
grounding map over image patches. Training supervises that map with a KL loss against a Gaussian patch target, plus the
usual next-token loss. On Qwen3.5 only the full-attention layers have attention maps (one layer in four; the rest are
Gated DeltaNet), so only their attention blocks and the embedding rows of the five new tokens are trained:
88.1M (2B), 293.6M (4B) and 469.8M (9B) parameters. Everything else stays frozen.

## Layout

| Path | What |
|---|---|
| `train_attn.py` | Training entry point (Hugging Face `Trainer`); resumes from the newest checkpoint, exports `final_bf16/` |
| `src/gui_aima/` | GUI-AIMA data pipeline, grounding head and point readout; `qwen35/` holds the Qwen3.5 model, attention-row recomputation, trainer and inference |
| `src/attn_ft/` | Which weights train (`freeze.py`), the trainable pointer-token rows (`pointer_rows.py`), callbacks and export |
| `src/medspot_strict/` | The MedSPOT strict scorer |
| `configs/` | Data mixes: `stageA.yaml`, `stageB.yaml`, `clinical.yaml`, and the 45k ablation mix `data_45k.yaml` |
| `config.env` | Run settings (model size, learning rate, batch, pixels); every value can be overridden from the environment |
| `scripts/` | Environment setup, data builders, the stage runner, smoke checks |
| `eval/` | Single-model evaluation: ScreenSpot-v2, ScreenSpot-Pro, MedSPOT (strict), 1-step and 2-step |
| `tests/` | CPU tests on a tiny random Qwen3.5 (`python -m pytest tests`, with `PYTHONPATH=src:.`) |

## Setup

    bash scripts/setup_env.sh          # .venv with torch 2.14 (cu126), transformers 5.17, flash-linear-attention, causal-conv1d

Put the Qwen3.5 base checkpoints under `models/` (`models/Qwen3.5-2B`, `-4B`, `-9B`) or set `MODEL_ROOT`.

## Data

| Stage | Records | Build |
|---|---|---|
| A: GUI-AIMA lite mix (GUIAct, AndroidControl, Wave-UI, UGround, GTA1) | 254,779 | `python scripts/build_mix.py --out data/full509k` |
| B: GroundCUA, seeded 250k sample (20 out-of-image boxes dropped) | 249,980 | `python scripts/build_groundcua.py --out data/full509k --n 250000 --seed 0` |
| Clinical: the released clinical set, tiers A + B | 10,341 (3,447 screenshots) | `python scripts/convert_clinical.py --src <export.json> --out data/clinical_train/all.json`, screenshots under `data/clinical_train/images/` |
| 45k ablation mix | 45,000 | `python scripts/build_ablation_45k.py --out data/ablation_45k` |

The builders download only the images the records use (HTTP range requests into the remote archives) and can be
re-run: finished images are skipped. Before a long run, check a mix with
`python scripts/audit_mix.py --data configs/stageA.yaml --bench data/screenspot_v2_full --bench data/screenspot_pro --bench data/medspot`;
it fails on missing or truncated images, unusable boxes and byte-identical copies of benchmark images.

## Training

    MODEL_SIZE=4b bash scripts/run_stages.sh smoke      # 20 steps on the largest images: parameter audit + memory check
    MODEL_SIZE=4b bash scripts/run_stages.sh all        # stage A -> stage B -> clinical stage

| | General stages A and B | Clinical stage |
|---|---|---|
| Initialisation | Qwen3.5 base (A), A's `final_bf16` (B) | B's `final_bf16`, fresh optimiser and schedule |
| Optimiser | AdamW (fused), fp32 master weights, bf16 autocast | AdamW 8-bit, fp32 master weights, bf16 autocast |
| LR / schedule | 5e-6, cosine, 3% warm-up, no weight decay | same |
| Batch | 1 x 8 gradient accumulation | same |
| Epochs | 1 per stage | 2 |
| Max pixels per image | 5,860,400 | same |

Outputs go to `runs/<size>/{A_lite,B_groundcua,clinical}/`; each stage ends with `final_bf16/` (and an `EXPORT_DONE`
marker). A killed run resumes from its newest checkpoint when the same command is run again. For a custom run, call
`scripts/train.sh` with `MODEL`, `DATA` and `OUT` set; extra arguments are passed to `train_attn.py`.

## Evaluation

For the paper's numbers use `../bencheval`. The scripts here evaluate one checkpoint directly:

    bash scripts/eval.sh runs/4b/clinical/final_bf16 runs/4b/clinical/eval        # ScreenSpot-v2 + ScreenSpot-Pro, 1- and 2-step
    python eval/medspot_strict_qwen35.py --ckpt runs/4b/clinical/final_bf16 --name 4b-2step --out_dir out/medspot --two_step

`scripts/fetch_screenspot_v2.py --per_split 0 --out data/screenspot_v2_full` and `scripts/fetch_screenspot_pro.py`
download the two ScreenSpot benchmarks; MedSPOT goes in `data/medspot/` (`Annotations/`, `Images/`).
