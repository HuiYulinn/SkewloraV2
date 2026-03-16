#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL=${MODEL:-roberta-base}
ROOT=${ROOT:-"$PROJECT_ROOT/outputs/rte_waware_compare"}
SEEDS=${SEEDS:-"42 43 44"}
COMMON_ARGS=${COMMON_ARGS:-"--task_name rte --model_name_or_path $MODEL --max_length 128 --lora_r 8 --lora_alpha 16 --learning_rate 2e-4 --num_train_epochs 10 --warmup_ratio 0.1 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --evaluation_strategy epoch --save_strategy epoch"}

for SEED in $SEEDS; do
  python "$PROJECT_ROOT/train_glue_peft.py" \
    --method skewlora \
    --output_dir "$ROOT/skewlora_fixed/seed${SEED}" \
    --seed "$SEED" \
    --skew_alpha 3.0 \
    --skew_std 0.02 \
    $COMMON_ARGS

done

for SEED in $SEEDS; do
  python "$PROJECT_ROOT/train_glue_peft.py" \
    --method skewlora \
    --waware_init \
    --output_dir "$ROOT/skewlora_waware/seed${SEED}" \
    --seed "$SEED" \
    --skew_alpha 3.0 \
    --skew_std 0.02 \
    --waware_stats_mode mean_var_skew \
    --waware_c1 1.0 \
    --waware_c2 0.5 \
    --waware_alpha_max 8.0 \
    --waware_std_min 0.005 \
    --waware_std_max 0.05 \
    $COMMON_ARGS

done

echo "[RTE-WAWARE] done"