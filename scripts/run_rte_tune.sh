#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK=${TASK:-rte}
MODEL=${MODEL:-roberta-base}
ROOT=${ROOT:-"$PROJECT_ROOT/outputs/rte_tune"}
METHODS=${METHODS:-"lora skewlora"}
SEEDS=${SEEDS:-"42 43 44"}
LR_LIST=${LR_LIST:-"1e-4 2e-4 3e-4"}
EPOCH_LIST=${EPOCH_LIST:-"5 10"}
DROPOUT_LIST=${DROPOUT_LIST:-"0.05 0.1"}

COMMON_ARGS=${COMMON_ARGS:-"--max_length 128 --lora_r 8 --lora_alpha 16 --warmup_ratio 0.1 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --evaluation_strategy epoch --save_strategy epoch"}

mkdir -p "$ROOT"

echo "[RTE-TUNE] root=$ROOT"
for METHOD in $METHODS; do
  for LR in $LR_LIST; do
    for EP in $EPOCH_LIST; do
      for DO in $DROPOUT_LIST; do
        TAG="lr${LR}_ep${EP}_do${DO}"
        for SEED in $SEEDS; do
          OUT="$ROOT/$METHOD/$TAG/seed${SEED}"
          EXTRA=""
          if [ "$METHOD" = "skewlora" ]; then
            EXTRA="--skew_alpha 3.0 --skew_std 0.02"
          fi
          echo "[RUN] method=$METHOD lr=$LR ep=$EP do=$DO seed=$SEED"
          python "$PROJECT_ROOT/train_glue_peft.py" \
            --task_name "$TASK" \
            --model_name_or_path "$MODEL" \
            --method "$METHOD" \
            --output_dir "$OUT" \
            --learning_rate "$LR" \
            --num_train_epochs "$EP" \
            --lora_dropout "$DO" \
            --seed "$SEED" \
            $COMMON_ARGS \
            $EXTRA
        done
      done
    done
  done
done

echo "[RTE-TUNE] done"