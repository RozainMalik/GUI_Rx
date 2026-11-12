#!/usr/bin/env bash
# Full evaluation of one exported checkpoint: ScreenSpot-v2 (1272) and ScreenSpot-Pro (1581), 1-step and 2-step.
# Usage: bash scripts/eval.sh [ckpt_dir] [out_dir]   (defaults: $OUT/final_bf16, $OUT/eval)
set -euo pipefail
cd "$(dirname "$0")/.."
. ./config.env
CKPT=${1:-$OUT/final_bf16}; EVAL_OUT=${2:-$OUT/eval}
export PYTHONPATH=src:. PYTHONUNBUFFERED=1
for two in "" "--two_step"; do
  "$PYTHON" eval/screenspot_v2_qwen35.py --ckpt "$CKPT" --data data/screenspot_v2_full --out "$EVAL_OUT" $two
  "$PYTHON" eval/screenspot_pro_qwen35.py --ckpt "$CKPT" --data data/screenspot_pro --out "$EVAL_OUT" $two
done
ls -la "$EVAL_OUT"
