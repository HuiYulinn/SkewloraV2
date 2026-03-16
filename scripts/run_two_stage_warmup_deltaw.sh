#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL=${MODEL:-roberta-base}
ROOT=${ROOT:-"$PROJECT_ROOT/outputs/warmup_two_stage_deltaw"}
SEEDS=${SEEDS:-"42 43 44"}
TARGET_PROFILE=${TARGET_PROFILE:-qkvo}
WARMUP_STEPS=${WARMUP_STEPS:-300}

COMMON_ARGS=${COMMON_ARGS:-"--task_name rte --model_name_or_path $MODEL --target_profile $TARGET_PROFILE --max_length 128 --lora_r 8 --lora_alpha 16 --learning_rate 2e-4 --num_train_epochs 10 --warmup_ratio 0.1 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --evaluation_strategy epoch --save_strategy epoch"}

mkdir -p "$ROOT"

# Step A: LoRA warmup runs
for SEED in $SEEDS; do
  python "$PROJECT_ROOT/train_glue_peft.py" \
    --method lora \
    --max_steps "$WARMUP_STEPS" \
    --output_dir "$ROOT/lora_warmup/default/seed${SEED}" \
    --seed "$SEED" \
    $COMMON_ARGS

done

# Step B: build DeltaW stats from warmup runs
RUN_DIRS=""
for SEED in $SEEDS; do
  RUN_DIRS="$RUN_DIRS $ROOT/lora_warmup/default/seed${SEED}"
done

python "$PROJECT_ROOT/analyze_deltaw_stats.py" \
  --model_name_or_path "$MODEL" \
  --target_profile "$TARGET_PROFILE" \
  --run_dirs $RUN_DIRS \
  --init_skew_source eff_skew \
  --init_std_source eff_std \
  --output_json "$ROOT/deltaw_stats_mean.json" \
  --output_csv "$ROOT/deltaw_stats_mean.csv"

# Step C1: SkewLoRA fixed (full training)
for SEED in $SEEDS; do
  python "$PROJECT_ROOT/train_glue_peft.py" \
    --method skewlora \
    --skew_alpha 3.0 \
    --skew_std 0.02 \
    --output_dir "$ROOT/skewlora_fixed/default/seed${SEED}" \
    --seed "$SEED" \
    $COMMON_ARGS

done

# Step C2: eff_same
for SEED in $SEEDS; do
  python "$PROJECT_ROOT/train_glue_peft.py" \
    --method skewlora \
    --init_from_deltaw \
    --deltaw_stats_json "$ROOT/deltaw_stats_mean.json" \
    --init_skew_direction same \
    --init_skew_source eff_skew \
    --init_std_source eff_std \
    --skew_alpha 3.0 \
    --skew_std 0.02 \
    --waware_c1 1.0 \
    --waware_c2 0.5 \
    --waware_alpha_max 8.0 \
    --waware_std_min 0.005 \
    --waware_std_max 0.05 \
    --output_dir "$ROOT/skewlora_eff_same/default/seed${SEED}" \
    --seed "$SEED" \
    $COMMON_ARGS

done

# Step C3: eff_opposite
for SEED in $SEEDS; do
  python "$PROJECT_ROOT/train_glue_peft.py" \
    --method skewlora \
    --init_from_deltaw \
    --deltaw_stats_json "$ROOT/deltaw_stats_mean.json" \
    --init_skew_direction opposite \
    --init_skew_source eff_skew \
    --init_std_source eff_std \
    --skew_alpha 3.0 \
    --skew_std 0.02 \
    --waware_c1 1.0 \
    --waware_c2 0.5 \
    --waware_alpha_max 8.0 \
    --waware_std_min 0.005 \
    --waware_std_max 0.05 \
    --output_dir "$ROOT/skewlora_eff_opposite/default/seed${SEED}" \
    --seed "$SEED" \
    $COMMON_ARGS

done

# Step C4: eff_damped
for SEED in $SEEDS; do
  python "$PROJECT_ROOT/train_glue_peft.py" \
    --method skewlora \
    --init_from_deltaw \
    --deltaw_stats_json "$ROOT/deltaw_stats_mean.json" \
    --init_skew_direction damped \
    --init_damped_ratio 0.5 \
    --init_skew_source eff_skew \
    --init_std_source eff_std \
    --skew_alpha 3.0 \
    --skew_std 0.02 \
    --waware_c1 1.0 \
    --waware_c2 0.5 \
    --waware_alpha_max 8.0 \
    --waware_std_min 0.005 \
    --waware_std_max 0.05 \
    --output_dir "$ROOT/skewlora_eff_damped/default/seed${SEED}" \
    --seed "$SEED" \
    $COMMON_ARGS

done

# Step D: summarize
python "$PROJECT_ROOT/summarize_multiseed.py" \
  --root "$ROOT" \
  --output_csv "$ROOT/two_stage_summary.csv" \
  --best_csv "$ROOT/two_stage_best.csv" \
  --scale100

echo "[TWO-STAGE-WARMUP-DELTAW] done"