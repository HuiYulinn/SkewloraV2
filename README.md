# SkewLoRA

A compact experimental codebase for running **LoRA** and **SkewLoRA** on GLUE tasks with Hugging Face Transformers + PEFT.

## What is included

- `train_glue_peft.py`: main training entry for GLUE sequence-classification tasks
- `skewlora/initialization.py`: skew-normal initialization for LoRA A and zero initialization for LoRA B
- `plot_early_convergence.py`: plot LoRA vs SkewLoRA epoch curves
- `summarize_results.py`: build a CSV summary table from finished runs
- `scripts/run_lora_all.sh`: run LoRA on SST-2 / MRPC / RTE / STS-B
- `scripts/run_skewlora_all.sh`: run SkewLoRA on SST-2 / MRPC / RTE / STS-B
- `scripts/run_ablation_alpha.sh`: run skew-alpha ablations on one task

## Recommended experiment setting

- Model: `roberta-base`
- Tasks: `sst2`, `mrpc`, `rte`, `stsb`
- Methods to run yourself: `lora`, `skewlora`
- Suggested paper baselines to cite separately: LoRA, AdaLoRA, LoRA+, DoRA, PiSSA

## Installation

```bash
conda create -n skewlora python=3.10 -y
conda activate skewlora
pip install -r requirements.txt
```

## Quick start

### 1) Run paper baseline LoRA yourself

```bash
bash scripts/run_lora_all.sh
```

### 2) Run SkewLoRA

```bash
bash scripts/run_skewlora_all.sh
```

### 3) Plot early convergence for one task

```bash
python plot_early_convergence.py \
  --lora_csv outputs/roberta_glue/lora/sst2/epoch_history.csv \
  --skew_csv outputs/roberta_glue/skewlora/sst2/epoch_history.csv \
  --ylabel Accuracy \
  --title "SST-2 Early Convergence" \
  --output outputs/figures/sst2_early_convergence.png
```

### 4) Build a CSV summary table

```bash
python summarize_results.py \
  --root outputs/roberta_glue \
  --output_csv outputs/summary_roberta_glue.csv
```

## Single-command examples

### LoRA on SST-2

```bash
python train_glue_peft.py \
  --task_name sst2 \
  --model_name_or_path roberta-base \
  --method lora \
  --output_dir outputs/roberta_glue/lora/sst2 \
  --max_length 128 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --warmup_ratio 0.06 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy epoch \
  --save_strategy epoch
```

### SkewLoRA on SST-2

```bash
python train_glue_peft.py \
  --task_name sst2 \
  --model_name_or_path roberta-base \
  --method skewlora \
  --skew_alpha 3.0 \
  --skew_std 0.02 \
  --output_dir outputs/roberta_glue/skewlora/sst2 \
  --max_length 128 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --warmup_ratio 0.06 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy epoch \
  --save_strategy epoch
```

## Output files

Each experiment directory contains:

- `eval_metrics.json`: final validation metrics
- `experiment_summary.json`: compact summary including main metric
- `epoch_history.csv`: epoch-level metric history for plotting
- `epoch_history.json`: same history in JSON form
- `trainer_state_log_history.json`: raw trainer log history

## Notes

1. This repository is intentionally compact and focused on **GLUE + RoBERTa-base** because that is the easiest setting for aligning with prior LoRA-family baselines.
2. The paper-level baseline numbers in your manuscript should be checked against the original tables before submission.
3. If you later want to extend to DeBERTa-v3-base, the same code can be reused, but you should re-check target module names and paper alignment carefully.
