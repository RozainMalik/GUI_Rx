#!/usr/bin/env bash
# Attention-only training at the paper recipe. Usage: bash scripts/train.sh [extra HF args]  (extra args come last = override)
# Re-running with the same OUT resumes from its newest checkpoint-*.
set -euo pipefail
cd "$(dirname "$0")/.."
. ./config.env
export PYTHONPATH=src:. PYTHONUNBUFFERED=1
exec "$PYTHON" train_attn.py \
  --model_type qwen35 --model_name_or_path "$MODEL" --data_path "$DATA" --image_folder "" --output_dir "$OUT" \
  --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps "$GA" \
  --learning_rate "$LR" --weight_decay 0. --warmup_steps 0.03 --lr_scheduler_type cosine --optim "$OPTIM" \
  --bf16 True --tf32 True --frozen_dtype bfloat16 --expect_trainable_range "$EXPECT_TRAINABLE" --gradient_checkpointing True --model_max_length 24576 \
  --max_pixels "$MAX_PIXELS" --image_patch_size 16 --weighting query_1 --number_of_points 1 \
  --pointer_loss_weight 1.0 --lm_loss_weight 1.0 \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 2 --logging_steps 10 \
  --dataloader_num_workers 8 --group_by_modality_length True --empty_cache_every_n_steps -1 --report_to none "$@"
