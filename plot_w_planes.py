#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover
    norm = None


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


def layer_stats(t: torch.Tensor) -> Dict[str, float]:
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


def qq_xy(x: np.ndarray, num_points: int = 4000):
    x = x.reshape(-1)
    if x.size > num_points:
        idx = np.random.RandomState(0).choice(x.size, size=num_points, replace=False)
        x = x[idx]
    x = np.sort(x)
    n = x.size
    p = (np.arange(1, n + 1) - 0.5) / n
    if norm is not None:
        q = norm.ppf(p)
    else:
        # Fallback approximation if scipy is unavailable.
        q = np.sqrt(2.0) * torch.erfinv(torch.tensor(2 * p - 1)).numpy()
    return q, x


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot W plane/shape diagnostics for LoRA target layers.")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/w_plots"))
    parser.add_argument("--max_layers", type=int, default=999)
    parser.add_argument("--dpi", type=int, default=180)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    plotted = 0

    for module_name, module in model.named_modules():
        if plotted >= args.max_layers:
            break
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

        w = base.detach().float().cpu().numpy()
        stats = layer_stats(base)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        im = axes[0].imshow(w, aspect="auto", cmap="coolwarm")
        axes[0].set_title("W Heatmap")
        axes[0].set_xlabel("in_features")
        axes[0].set_ylabel("out_features")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        flat = w.reshape(-1)
        axes[1].hist(flat, bins=120, density=True, alpha=0.85, color="#2a9d8f")
        axes[1].set_title("W Histogram")
        axes[1].set_xlabel("weight value")
        axes[1].set_ylabel("density")

        qx, qy = qq_xy(flat)
        axes[2].scatter(qx, qy, s=6, alpha=0.35, color="#264653")
        lo = min(float(qx.min()), float(qy.min()))
        hi = max(float(qx.max()), float(qy.max()))
        axes[2].plot([lo, hi], [lo, hi], "r--", linewidth=1)
        axes[2].set_title("Q-Q vs Normal")
        axes[2].set_xlabel("normal quantile")
        axes[2].set_ylabel("W quantile")

        fig.suptitle(
            f"{module_name}\nmean={stats['mean']:.4e}, std={stats['std']:.4e}, skew={stats['skew']:.4f}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.0, 1, 0.92])

        safe_name = module_name.replace(".", "_").replace("/", "_")
        out_png = args.output_dir / f"{plotted:02d}_{safe_name}.png"
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)

        rows.append(
            {
                "module_name": module_name,
                "target_modules": ",".join(target_modules),
                "plot_path": str(out_png),
                **stats,
            }
        )
        plotted += 1

    index_json = args.output_dir / "w_plot_index.json"
    with index_json.open("w", encoding="utf-8") as f:
        json.dump({
            "model_name_or_path": args.model_name_or_path,
            "target_modules": target_modules,
            "num_plots": plotted,
            "plots": rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved {plotted} layer plots to {args.output_dir}")
    print(f"Saved plot index: {index_json}")


if __name__ == "__main__":
    main()