#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize multi-seed tuning outputs.")
    p.add_argument("--root", type=Path, required=True, help="Root, e.g. outputs/rte_tune")
    p.add_argument("--output_csv", type=Path, required=True, help="Detailed per-config summary")
    p.add_argument("--best_csv", type=Path, default=None, help="Best config per method")
    p.add_argument("--scale100", action="store_true", help="Multiply scores by 100")
    return p.parse_args()


def load_summary(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def fmt(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.4f}"


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, str]] = []
    best_by_method: Dict[str, Dict[str, str]] = {}

    if not args.root.exists():
        raise FileNotFoundError(f"Root not found: {args.root}")

    for method_dir in sorted([d for d in args.root.iterdir() if d.is_dir()]):
        method = method_dir.name
        method_best_val = -math.inf
        method_best_row: Optional[Dict[str, str]] = None

        for cfg_dir in sorted([d for d in method_dir.iterdir() if d.is_dir()]):
            values: List[float] = []
            main_metric_name: Optional[str] = None

            for seed_dir in sorted([d for d in cfg_dir.iterdir() if d.is_dir()]):
                summary = load_summary(seed_dir / "experiment_summary.json")
                if not summary:
                    continue
                val = safe_float(summary.get("main_metric_value"))
                if val is None:
                    continue
                main_metric_name = summary.get("main_metric", main_metric_name)
                values.append(val)

            if not values:
                continue

            m = mean(values)
            s = stdev(values) if len(values) > 1 else 0.0
            best_seed_val = max(values)
            worst_seed_val = min(values)

            scale = 100.0 if args.scale100 else 1.0
            row = {
                "method": method,
                "config": cfg_dir.name,
                "metric": main_metric_name or "",
                "num_seeds": str(len(values)),
                "mean": fmt(m * scale),
                "std": fmt(s * scale),
                "best_seed": fmt(best_seed_val * scale),
                "worst_seed": fmt(worst_seed_val * scale),
            }
            rows.append(row)

            if m > method_best_val:
                method_best_val = m
                method_best_row = row

        if method_best_row is not None:
            best_by_method[method] = method_best_row

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "config", "metric", "num_seeds", "mean", "std", "best_seed", "worst_seed"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    best_csv = args.best_csv or args.output_csv.with_name(args.output_csv.stem + "_best.csv")
    with best_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "config", "metric", "num_seeds", "mean", "std", "best_seed", "worst_seed"],
        )
        writer.writeheader()
        for method in sorted(best_by_method):
            writer.writerow(best_by_method[method])

    print(f"Saved detailed summary: {args.output_csv}")
    print(f"Saved best-per-method summary: {best_csv}")


if __name__ == "__main__":
    main()