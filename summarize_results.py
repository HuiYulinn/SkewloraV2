#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import csv

BASELINES = {
    "Full Fine-tuning": {"sst2": 94.8, "mrpc": 91.2, "rte": 74.6, "stsb": 89.1},
    "LoRA (paper baseline)": {"sst2": 94.5, "mrpc": 90.7, "rte": 73.9, "stsb": 88.5},
    "AdaLoRA": {"sst2": 94.7, "mrpc": 91.0, "rte": 74.2, "stsb": 88.9},
    "LoRA+": {"sst2": 94.9, "mrpc": 91.3, "rte": 74.5, "stsb": 89.0},
    "DoRA": {"sst2": 95.0, "mrpc": 91.4, "rte": 75.0, "stsb": 89.2},
    "PiSSA": {"sst2": 95.1, "mrpc": 91.6, "rte": 75.2, "stsb": 89.5},
}


def read_metric(exp_dir: Path) -> Dict:
    summary_path = exp_dir / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LoRA/SkewLoRA experiment outputs into a CSV table.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing task subdirectories")
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    tasks = ["sst2", "mrpc", "rte", "stsb"]
    rows: List[Dict[str, str]] = []

    # Paper baselines.
    for method, vals in BASELINES.items():
        row = {"Method": method}
        for task in tasks:
            row[task.upper()] = vals[task]
        rows.append(row)

    # User runs.
    for method_dir_name, display_name in [("lora", "LoRA (your run)"), ("skewlora", "SkewLoRA (your run)")]:
        row = {"Method": display_name}
        for task in tasks:
            exp_dir = args.root / method_dir_name / task
            if exp_dir.exists():
                summary = read_metric(exp_dir)
                value = summary.get("main_metric_value")
                row[task.upper()] = value
            else:
                row[task.upper()] = ""
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "SST2", "MRPC", "RTE", "STSB"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved summary table to {args.output_csv}")


if __name__ == "__main__":
    main()
