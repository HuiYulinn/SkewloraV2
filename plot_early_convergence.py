#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_history(path: Path) -> Tuple[List[float], List[float]]:
    epochs, values = [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("eval_metric_value") in (None, ""):
                continue
            epochs.append(float(row["epoch"]))
            values.append(float(row["eval_metric_value"]))
    return epochs, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LoRA vs SkewLoRA early convergence curves.")
    parser.add_argument("--lora_csv", type=Path, required=True)
    parser.add_argument("--skew_csv", type=Path, required=True)
    parser.add_argument("--ylabel", type=str, default="Validation Metric")
    parser.add_argument("--title", type=str, default="Early Convergence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lora_epochs, lora_values = read_history(args.lora_csv)
    skew_epochs, skew_values = read_history(args.skew_csv)

    plt.figure(figsize=(6, 4))
    plt.plot(lora_epochs, lora_values, marker="o", label="LoRA")
    plt.plot(skew_epochs, skew_values, marker="o", label="SkewLoRA")
    plt.xlabel("Epoch")
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
