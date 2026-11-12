# Clinical GUI grounding: data pipeline, model training and evaluation

Code for the paper's three parts:

| Folder | What | Start at |
|---|---|---|
| `annotation-pipeline/` | The pipeline that labelled the clinical training set: five model stages (propose → describe → ground → verify → refine), consensus selection and tiers, human grading, MedSPOT-format export | `annotation-pipeline/README.md` |
| `training/` | GUI-AIMA-style attention-anchored grounding on Qwen3.5 (2B / 4B / 9B): data builders for the 509k general mix, the three training stages (A, B, clinical), single-model evaluation scripts | `training/README.md` |
| `bencheval/` | One harness for every benchmark and model in the paper: ScreenSpot v1/v2/Pro, OSWorld-G, UI-Vision, MMBench-GUI L2, MedSPOT; 1-step, 2-step (zoom) and zoom + fusion inference; adapters for the published baselines | `bencheval/README.md` |

## Pipeline at a glance

1. **Label clinical screenshots** with `annotation-pipeline/` → a MedSPOT-format export (tiers A + B).
2. **Build the general data** with `training/scripts/build_mix.py` (stage A) and `training/scripts/build_groundcua.py` (stage B),
   and convert the clinical export with `training/scripts/convert_clinical.py`.
3. **Train** with `MODEL_SIZE=2b|4b|9b bash training/scripts/run_stages.sh all` (stage A → stage B → clinical stage).
4. **Evaluate** the exported `final_bf16` checkpoints with `bencheval` (`--mode 1step | 2step | zf`).

MedSPOT is used for evaluation only: no MedSPOT screenshot or task is part of any training set.

## Requirements

- Training and Qwen3.5 inference: Python 3.11, CUDA 12.x, one GPU (peak memory 9.4 / 18.5 / 30.3 GiB for 2B / 4B / 9B).
  `training/scripts/setup_env.sh` builds the environment.
- Published baselines in `bencheval` need their own environments (see `bencheval/README.md`).
- Annotation: Python 3.11 and an OpenAI API key; no GPU.

## Licence

Apache-2.0 (`LICENSE`). Third-party code and its licences are listed in `NOTICE`.
