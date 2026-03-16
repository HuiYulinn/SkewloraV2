from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EpochRecord:
    epoch: float
    train_loss: Optional[float] = None
    eval_metric_name: Optional[str] = None
    eval_metric_value: Optional[float] = None


class EpochHistoryLogger:
    """Collects epoch-level metrics and writes them to CSV/JSON."""

    def __init__(self) -> None:
        self.records: List[EpochRecord] = []

    def add(self, epoch: float, train_loss: Optional[float], eval_metric_name: Optional[str], eval_metric_value: Optional[float]) -> None:
        self.records.append(EpochRecord(epoch, train_loss, eval_metric_name, eval_metric_value))

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "epoch_history.csv")
        json_path = os.path.join(output_dir, "epoch_history.json")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "eval_metric_name", "eval_metric_value"],
            )
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.records], f, ensure_ascii=False, indent=2)


def save_final_metrics(metrics: Dict[str, Any], output_dir: str, filename: str = "final_metrics.json") -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
