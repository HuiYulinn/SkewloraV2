#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification


TASK_TO_TARGET_MODULES = {
    "roberta": ["query", "value"],
    "bert": ["query", "value"],
    "deberta": ["query_proj", "value_proj"],
    "deberta-v2": ["query_proj", "value_proj"],
}


def infer_target_modules(model_name_or_path: str):
    lower = model_name_or_path.lower()
    for key, value in TASK_TO_TARGET_MODULES.items():
        if key in lower:
            return value
    return ["query", "value"]


def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    x = t.detach().float().reshape(-1)
    mu = float(x.mean().item())
    sigma = float(x.std(unbiased=False).item())
    z = (x - mu) / max(sigma, 1e-8)
    skew = float((z.pow(3).mean()).item())
    return {
        "numel": int(x.numel()),
        "mean": mu,
        "std": sigma,
        "skew": skew,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen W distribution on LoRA target layers.")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    target_modules = infer_target_modules(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    rows: List[Dict[str, object]] = []
    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        if not any(tok in module_name for tok in target_modules):
            continue
        base = None
        if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            base = module.base_layer.weight
        elif hasattr(module, "get_base_layer"):
            try:
                b = module.get_base_layer()
                if hasattr(b, "weight"):
                    base = b.weight
            except Exception:
                base = None
        if base is None:
            continue

        stats = tensor_stats(base)
        row = {
            "module_name": module_name,
            "module_class": module.__class__.__name__,
            "target_modules": ",".join(target_modules),
            **stats,
        }
        rows.append(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "target_modules": target_modules,
        "num_layers": len(rows),
        "layers": rows,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["module_name", "module_class", "target_modules", "numel", "mean", "std", "skew", "min", "max"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved W-distribution JSON: {args.output_json}")
    print(f"Saved W-distribution CSV: {args.output_csv}")


if __name__ == "__main__":
    main()