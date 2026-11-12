#!/usr/bin/env bash
# The three training stages of the paper, for one model size (MODEL_SIZE in config.env):
#   smoke     20 steps on the 160 largest stage-A screenshots: parameter audit, memory budget, export
#   A         general stage A (GUI-AIMA lite mix, 1 epoch) from the Qwen3.5 base model
#   B         general stage B (GroundCUA 250k, 1 epoch) from A's final_bf16
#   clinical  clinical stage (2 epochs, 8-bit AdamW, fresh schedule) from B's final_bf16
#   all       A, B, clinical in order
# Every stage resumes from its newest checkpoint; a finished stage (final_bf16/EXPORT_DONE) is skipped.
# Usage: MODEL_SIZE=4b bash scripts/run_stages.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
. ./config.env
export PYTHONPATH=src:. PYTHONUNBUFFERED=1
A_OUT=$RUN_ROOT/A_lite; B_OUT=$RUN_ROOT/B_groundcua; C_OUT=$RUN_ROOT/clinical
exported() { [ -f "$1/EXPORT_DONE" ]; }

stage() {   # stage <init> <data yaml> <out> [extra train.sh args]
  local init=$1 data=$2 out=$3; shift 3
  if exported "$out/final_bf16"; then echo "done: $out/final_bf16"; return 0; fi
  [ -f "$init/config.json" ] || { echo "init checkpoint $init is missing" >&2; exit 1; }
  MODEL=$init DATA=$data OUT=$out bash scripts/train.sh "$@"
}

case ${1:?smoke|A|B|clinical|all} in
  smoke)
    out=$RUN_ROOT/smoke; rm -rf "$out"
    sm=data/smoke_largest
    [ -f $sm/smoke.yaml ] || "$PYTHON" scripts/make_smoke_data.py --data configs/stageA.yaml --top 160 --out $sm
    MODEL=$MODEL DATA=$sm/smoke.yaml OUT=$out bash scripts/train.sh --max_steps 20 --logging_steps 1 --save_steps 20 \
      --audit_params True --export_final False
    "$PYTHON" scripts/check_smoke.py --run_dir "$out" --max_reserved_gib "$SMOKE_MAX_RESERVED_GIB"
    "$PYTHON" -m attn_ft.export "$out/checkpoint-20" "$out/exp" && rm -rf "$out/exp"
    echo "smoke OK" ;;
  A) stage "$MODEL" configs/stageA.yaml "$A_OUT" ;;
  B) exported "$A_OUT/final_bf16" || { echo "run stage A first" >&2; exit 1; }
     stage "$A_OUT/final_bf16" configs/stageB.yaml "$B_OUT" ;;
  clinical)
     exported "$B_OUT/final_bf16" || { echo "run stage B first" >&2; exit 1; }
     export OPTIM=adamw_bnb_8bit SAVE_STEPS=250
     stage "$B_OUT/final_bf16" configs/clinical.yaml "$C_OUT" --num_train_epochs 2 ;;
  all) for s in A B clinical; do bash "$0" $s; done ;;
  *) echo "usage: run_stages.sh smoke|A|B|clinical|all" >&2; exit 2 ;;
esac
