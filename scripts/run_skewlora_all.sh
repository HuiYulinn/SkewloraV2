#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL=${MODEL:-roberta-base}
ROOT=${ROOT:-"$PROJECT_ROOT/outputs/roberta_glue"}
SKEW_ALPHA=${SKEW_ALPHA:-3.0}
SKEW_STD=${SKEW_STD:-0.02}
COMMON_ARGS=${COMMON_ARGS:-"--max_length 128 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --learning_rate 2e-4 --num_train_epochs 3 --warmup_ratio 0.06 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --evaluation_strategy epoch --save_strategy epoch"}

for TASK in sst2 mrpc rte stsb; do
  python "$PROJECT_ROOT/train_glue_peft.py" \
    --task_name "$TASK" \
    --model_name_or_path "$MODEL" \
    --method skewlora \
    --skew_alpha "$SKEW_ALPHA" \
    --skew_std "$SKEW_STD" \
    --output_dir "$ROOT/skewlora/$TASK" \
    $COMMON_ARGS
done