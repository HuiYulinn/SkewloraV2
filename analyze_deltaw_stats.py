#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForSequenceClassification

from skewlora.target_modules import infer_target_modules


def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    x = t.detach().float().reshape(-1)
    mu = float(x.mean().item())
    sigma = float(x.std(unbiased=False).item())
    z = (x - mu) / max(sigma, 1e-8)
    skew = float((z.pow(3).mean()).item())
    return {"mean": mu, "std": sigma, "skew": skew}


def resolve_adapter_dir(run_dir: Path) -> Path:
    if (run_dir / "adapter_model.safetensors").exists() or (run_dir / "adapter_model.bin").exists():
        return run_dir
    cands = []
    for d in run_dir.glob("checkpoint-*"):
        if not d.is_dir():
            continue
        if (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists():
            m = re.search(r"checkpoint-(\d+)$", d.name)
            step = int(m.group(1)) if m else -1
            cands.append((step, d))
    if cands:
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands[0][1]
    raise FileNotFoundError(f"No adapter weights found in {run_dir}")


def _load_adapter_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    safe_path = adapter_dir / "adapter_model.safetensors"
    if safe_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safe_path), device="cpu")

    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.exists():
        obj = torch.load(str(bin_path), map_location="cpu")
        if isinstance(obj, dict):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return obj["state_dict"]
            return obj
    return {}


def infer_num_labels_from_adapter(adapter_dir: Path) -> int | None:
    state = _load_adapter_state_dict(adapter_dir)
    if not state:
        return None
    for k, v in state.items():
        if (k.endswith("classifier.modules_to_save.default.out_proj.weight") or k.endswith("classifier.out_proj.weight")) and hasattr(v, "shape"):
            if len(v.shape) >= 1:
                return int(v.shape[0])
    return None


def run_one(
    model_name_or_path: str,
    adapter_dir: Path,
    target_modules: List[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> List[Dict[str, object]]:
    inferred_num_labels = infer_num_labels_from_adapter(adapter_dir)
    kwargs = {}
    if inferred_num_labels is not None:
        kwargs["num_labels"] = inferred_num_labels

    base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
    _ = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)

    rows: List[Dict[str, object]] = []
    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        if not any(module_name.endswith(tok) for tok in target_modules):
            continue

        if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            w = module.base_layer.weight.detach().float()
        elif hasattr(module, "get_base_layer"):
            b = module.get_base_layer()
            if not hasattr(b, "weight"):
                continue
            w = b.weight.detach().float()
        else:
            continue

        for adapter_name, lora_A in module.lora_A.items():
            lora_B = module.lora_B[adapter_name]
            a = lora_A.weight.detach().float()
            b = lora_B.weight.detach().float()
            delta = b @ a
            scaling = 1.0
            if hasattr(module, "scaling") and adapter_name in module.scaling:
                scaling = float(module.scaling[adapter_name])
            delta = delta * scaling

            ds = tensor_stats(delta)
            es = tensor_stats(w + delta)
            rows.append(
                {
                    "module_name": module_name,
                    "module_class": module.__class__.__name__,
                    "adapter_name": adapter_name,
                    "delta_mean": ds["mean"],
                    "delta_std": ds["std"],
                    "delta_skew": ds["skew"],
                    "eff_mean": es["mean"],
                    "eff_std": es["std"],
                    "eff_skew": es["skew"],
                }
            )
    return rows


def aggregate(all_rows: List[List[Dict[str, object]]]) -> List[Dict[str, object]]:
    bucket: Dict[str, List[Dict[str, object]]] = {}
    for rows in all_rows:
        for r in rows:
            bucket.setdefault(str(r["module_name"]), []).append(r)

    out: List[Dict[str, object]] = []
    for module_name in sorted(bucket):
        vals = bucket[module_name]
        out.append(
            {
                "module_name": module_name,
                "module_class": vals[0]["module_class"],
                "adapter_name": vals[0]["adapter_name"],
                "num_runs": len(vals),
                "delta_mean": mean(float(v["delta_mean"]) for v in vals),
                "delta_std": mean(float(v["delta_std"]) for v in vals),
                "delta_skew": mean(float(v["delta_skew"]) for v in vals),
                "eff_mean": mean(float(v["eff_mean"]) for v in vals),
                "eff_std": mean(float(v["eff_std"]) for v in vals),
                "eff_skew": mean(float(v["eff_skew"]) for v in vals),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze DeltaW=BA stats from LoRA runs.")
    p.add_argument("--model_name_or_path", type=str, default="roberta-base")
    p.add_argument("--target_profile", type=str, default="qkvo", choices=["qv", "qkv", "qkvo"])
    p.add_argument("--run_dirs", nargs="+", required=True, help="One or more completed LoRA run dirs")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--init_skew_source", type=str, default="eff_skew", choices=["eff_skew", "delta_skew"])
    p.add_argument("--init_std_source", type=str, default="eff_std", choices=["eff_std", "delta_std"])
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    args = p.parse_args()

    target_modules = infer_target_modules(args.model_name_or_path, args.target_profile)

    per_run_rows: List[List[Dict[str, object]]] = []
    used_run_dirs: List[str] = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        adapter_dir = resolve_adapter_dir(run_dir)
        rows = run_one(
            model_name_or_path=args.model_name_or_path,
            adapter_dir=adapter_dir,
            target_modules=target_modules,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        per_run_rows.append(rows)
        used_run_dirs.append(str(adapter_dir))

    merged = aggregate(per_run_rows)
    for row in merged:
        row["init_skew_source"] = float(row[args.init_skew_source])
        row["init_std_source"] = float(row[args.init_std_source])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "target_profile": args.target_profile,
        "target_modules": target_modules,
        "run_dirs": used_run_dirs,
        "num_layers": len(merged),
        "source_mode": "effective_weight" if args.init_skew_source == "eff_skew" and args.init_std_source == "eff_std" else "custom",
        "init_skew_source": args.init_skew_source,
        "init_std_source": args.init_std_source,
        "layers": merged,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "module_name",
                "module_class",
                "adapter_name",
                "num_runs",
                "delta_mean",
                "delta_std",
                "delta_skew",
                "eff_mean",
                "eff_std",
                "eff_skew",
                "init_skew_source",
                "init_std_source",
            ],
        )
        writer.writeheader()
        for row in merged:
            writer.writerow(row)

    print(f"Saved DeltaW stats JSON: {args.output_json}")
    print(f"Saved DeltaW stats CSV: {args.output_csv}")


if __name__ == "__main__":
    main()