#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import evaluate
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from skewlora.initialization import (
    apply_deltaw_stats_initialization,
    apply_skew_lora_initialization,
    apply_waware_skew_lora_initialization,
)
from skewlora.logging_utils import EpochHistoryLogger, save_final_metrics
from skewlora.target_modules import infer_target_modules


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASK_TO_MAIN_METRIC = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
}


@dataclass
class ExperimentConfig:
    task_name: str
    model_name_or_path: str
    output_dir: str
    method: str
    target_profile: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    skew_alpha: float
    skew_std: float
    waware_init: bool
    waware_stats_mode: str
    waware_c1: float
    waware_c2: float
    waware_alpha_max: float
    waware_std_min: float
    waware_std_max: float
    init_from_deltaw: bool
    deltaw_stats_json: str
    init_skew_direction: str
    init_damped_ratio: float
    init_skew_source: str
    init_std_source: str
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    num_train_epochs: float
    max_steps: int
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    seed: int
    fp16: bool
    gradient_accumulation_steps: int
    save_strategy: str
    evaluation_strategy: str


class EpochMetricsCallback(TrainerCallback):
    def __init__(self, logger: EpochHistoryLogger, main_metric_name: str):
        self.logger = logger
        self.main_metric_name = main_metric_name
        self.last_train_loss: Optional[float] = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.last_train_loss = float(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        metric_key = f"eval_{self.main_metric_name}"
        metric_value = metrics.get(metric_key)
        if metric_value is None:
            return
        self.logger.add(
            epoch=float(state.epoch or 0.0),
            train_loss=self.last_train_loss,
            eval_metric_name=self.main_metric_name,
            eval_metric_value=float(metric_value),
        )


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Train LoRA/SkewLoRA on GLUE tasks.")
    parser.add_argument("--task_name", type=str, required=True, choices=sorted(TASK_TO_KEYS.keys()))
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="skewlora", choices=["lora", "skewlora"])
    parser.add_argument("--target_profile", type=str, default="qkvo", choices=["qv", "qkv", "qkvo"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--skew_alpha", type=float, default=3.0)
    parser.add_argument("--skew_std", type=float, default=0.02)
    parser.add_argument("--waware_init", action="store_true")
    parser.add_argument("--waware_stats_mode", type=str, default="mean_var_skew")
    parser.add_argument("--waware_c1", type=float, default=1.0)
    parser.add_argument("--waware_c2", type=float, default=0.5)
    parser.add_argument("--waware_alpha_max", type=float, default=8.0)
    parser.add_argument("--waware_std_min", type=float, default=0.005)
    parser.add_argument("--waware_std_max", type=float, default=0.05)
    parser.add_argument("--init_from_deltaw", action="store_true")
    parser.add_argument("--deltaw_stats_json", type=str, default="")
    parser.add_argument("--init_skew_direction", type=str, default="same", choices=["same", "opposite", "damped"])
    parser.add_argument("--init_damped_ratio", type=float, default=0.5)
    parser.add_argument("--init_skew_source", type=str, default="eff_skew", choices=["eff_skew", "delta_skew"])
    parser.add_argument("--init_std_source", type=str, default="eff_std", choices=["eff_std", "delta_std"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    args = parser.parse_args()
    return ExperimentConfig(**vars(args))


def build_compute_metrics(task_name: str):
    metric = evaluate.load("glue", task_name)

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions, labels = eval_pred
        if task_name == "stsb":
            preds = np.squeeze(predictions)
        else:
            preds = np.argmax(predictions, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["combined_score"] = float(np.mean(list(result.values())))
        return result

    return compute_metrics


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    raw_datasets = load_dataset("glue", cfg.task_name)
    is_regression = cfg.task_name == "stsb"
    label_list = None if is_regression else raw_datasets["train"].features["label"].names
    num_labels = 1 if is_regression else len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    sentence1_key, sentence2_key = TASK_TO_KEYS[cfg.task_name]

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=False, max_length=cfg.max_length, truncation=True)
        result["label"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(preprocess_function, batched=True, desc="Tokenizing")
    train_dataset = processed_datasets["train"]
    eval_key = "validation_matched" if cfg.task_name == "mnli" else "validation"
    eval_dataset = processed_datasets[eval_key]

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        num_labels=num_labels,
    )

    target_modules = infer_target_modules(cfg.model_name_or_path, cfg.target_profile)
    peft_kwargs = dict(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    if "roberta" in cfg.model_name_or_path.lower():
        peft_kwargs["modules_to_save"] = ["classifier"]

    peft_config = LoraConfig(**peft_kwargs)
    model = get_peft_model(model, peft_config)

    init_report = None
    if cfg.method == "skewlora":
        if cfg.init_from_deltaw:
            if not cfg.deltaw_stats_json:
                raise ValueError("--init_from_deltaw requires --deltaw_stats_json")
            with open(cfg.deltaw_stats_json, "r", encoding="utf-8") as f:
                deltaw_stats = json.load(f)
            init_report = apply_deltaw_stats_initialization(
                model=model,
                deltaw_stats=deltaw_stats,
                target_modules=target_modules,
                base_alpha=cfg.skew_alpha,
                base_std=cfg.skew_std,
                c1=cfg.waware_c1,
                c2=cfg.waware_c2,
                alpha_max=cfg.waware_alpha_max,
                std_min=cfg.waware_std_min,
                std_max=cfg.waware_std_max,
                skew_direction=cfg.init_skew_direction,
                damped_ratio=cfg.init_damped_ratio,
                skew_source=cfg.init_skew_source,
                std_source=cfg.init_std_source,
                verbose=True,
            )
            save_final_metrics(init_report, cfg.output_dir, filename="deltaw_init_applied.json")
        elif cfg.waware_init:
            init_report = apply_waware_skew_lora_initialization(
                model=model,
                target_modules=target_modules,
                base_alpha=cfg.skew_alpha,
                base_std=cfg.skew_std,
                stats_mode=cfg.waware_stats_mode,
                c1=cfg.waware_c1,
                c2=cfg.waware_c2,
                alpha_max=cfg.waware_alpha_max,
                std_min=cfg.waware_std_min,
                std_max=cfg.waware_std_max,
                verbose=True,
            )
            save_final_metrics(init_report, cfg.output_dir, filename="waware_init_stats.json")
        else:
            apply_skew_lora_initialization(model, alpha=cfg.skew_alpha, std=cfg.skew_std, verbose=True)
    else:
        print("[LoRA] Using PEFT default initialization.")

    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics(cfg.task_name)
    main_metric = TASK_TO_MAIN_METRIC[cfg.task_name]

    history_logger = EpochHistoryLogger()
    callbacks = [EpochMetricsCallback(history_logger, main_metric)]

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=False,
        fp16=cfg.fp16,
        report_to="none",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        seed=cfg.seed,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)

    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    train_metrics = trainer.state.log_history

    history_logger.save(cfg.output_dir)
    save_final_metrics(eval_metrics, cfg.output_dir, filename="eval_metrics.json")

    experiment_summary = {
        "task_name": cfg.task_name,
        "model_name_or_path": cfg.model_name_or_path,
        "method": cfg.method,
        "main_metric": main_metric,
        "main_metric_value": eval_metrics.get(f"eval_{main_metric}"),
        "target_modules": target_modules,
        "target_profile": cfg.target_profile,
        "waware_init": cfg.waware_init,
        "init_from_deltaw": cfg.init_from_deltaw,
        "config": cfg.__dict__,
    }
    if init_report is not None:
        experiment_summary["init_touched"] = init_report.get("touched")
    save_final_metrics(experiment_summary, cfg.output_dir, filename="experiment_summary.json")

    with open(os.path.join(cfg.output_dir, "trainer_state_log_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=2)

    print("[Done] Evaluation metrics:")
    print(json.dumps(eval_metrics, indent=2))


if __name__ == "__main__":
    main()