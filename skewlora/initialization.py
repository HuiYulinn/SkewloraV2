"""Utilities for SkewLoRA initialization.

This module keeps the standard LoRA cold-start property by initializing
LoRA A with a skew-normal distribution and LoRA B to zeros, so that the
initial low-rank update is still zero.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def skew_normal_(tensor: torch.Tensor, alpha: float = 3.0, std: float = 0.02) -> torch.Tensor:
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")

    with torch.no_grad():
        u0 = torch.randn_like(tensor)
        v = torch.randn_like(tensor)
        delta = alpha / math.sqrt(1.0 + alpha * alpha)
        z = delta * torch.abs(u0) + math.sqrt(max(1e-12, 1.0 - delta * delta)) * v
        tensor.copy_(std * z)
    return tensor


def _iter_named_lora_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


def _target_match(module_name: str, target_modules: Optional[Sequence[str]]) -> bool:
    if not target_modules:
        return True
    return any(module_name.endswith(token) for token in target_modules)


def _get_base_weight(module: nn.Module) -> Optional[torch.Tensor]:
    if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
        return module.base_layer.weight
    if hasattr(module, "get_base_layer"):
        try:
            base = module.get_base_layer()
            if hasattr(base, "weight"):
                return base.weight
        except Exception:
            pass
    if hasattr(module, "weight"):
        return module.weight
    return None


def _weight_stats(weight: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    w = weight.detach().float().reshape(-1)
    mu = float(w.mean().item())
    sigma = float(w.std(unbiased=False).item())
    sigma_safe = max(sigma, eps)
    z = (w - mu) / sigma_safe
    skew = float((z.pow(3).mean()).item())
    return {"mu": mu, "sigma": sigma, "skew": skew}


def apply_skew_lora_initialization(
    model: nn.Module,
    alpha: float = 3.0,
    std: float = 0.02,
    verbose: bool = True,
) -> int:
    touched = 0
    for _, module in _iter_named_lora_modules(model):
        for adapter_name, lora_A in module.lora_A.items():
            lora_B = module.lora_B[adapter_name]
            skew_normal_(lora_A.weight, alpha=alpha, std=std)
            nn.init.zeros_(lora_B.weight)
            touched += 1
            if verbose:
                print(
                    f"[SkewLoRA] initialized adapter='{adapter_name}' "
                    f"for module={module.__class__.__name__} with alpha={alpha}, std={std}"
                )
    if touched == 0:
        raise RuntimeError("No LoRA adapters were found. Make sure get_peft_model(...) was called first.")
    return touched


def apply_waware_skew_lora_initialization(
    model: nn.Module,
    target_modules: Optional[Sequence[str]] = None,
    base_alpha: float = 3.0,
    base_std: float = 0.02,
    stats_mode: str = "mean_var_skew",
    c1: float = 1.0,
    c2: float = 0.5,
    alpha_max: float = 8.0,
    std_min: float = 0.005,
    std_max: float = 0.05,
    eps: float = 1e-8,
    verbose: bool = True,
) -> Dict[str, object]:
    if stats_mode != "mean_var_skew":
        raise ValueError(f"Unsupported stats_mode={stats_mode}, expected 'mean_var_skew'.")

    layer_entries: List[Dict[str, object]] = []
    sigma_values: List[float] = []

    for module_name, module in _iter_named_lora_modules(model):
        if not _target_match(module_name, target_modules):
            continue
        base_weight = _get_base_weight(module)
        if base_weight is None:
            continue

        ws = _weight_stats(base_weight, eps=eps)
        for adapter_name, lora_A in module.lora_A.items():
            layer_entries.append(
                {
                    "module_name": module_name,
                    "module_class": module.__class__.__name__,
                    "adapter_name": adapter_name,
                    "stats": ws,
                    "lora_A": lora_A,
                    "lora_B": module.lora_B[adapter_name],
                }
            )
            sigma_values.append(float(ws["sigma"]))

    if not layer_entries:
        raise RuntimeError("No matching LoRA adapters were found for W-aware init.")

    median_sigma = float(torch.tensor(sigma_values, dtype=torch.float32).median().item())
    median_sigma = max(median_sigma, eps)

    touched = 0
    layer_stats: List[Dict[str, object]] = []
    for entry in layer_entries:
        ws = entry["stats"]
        skew = float(ws["skew"])
        sigma = float(ws["sigma"])

        alpha_l = base_alpha * math.tanh(c1 * skew)
        alpha_l = max(-alpha_max, min(alpha_max, alpha_l))

        ratio = max(sigma, eps) / median_sigma
        std_l = base_std * (ratio ** c2)
        std_l = max(std_min, min(std_max, std_l))

        lora_A = entry["lora_A"]
        lora_B = entry["lora_B"]
        skew_normal_(lora_A.weight, alpha=alpha_l, std=std_l)
        nn.init.zeros_(lora_B.weight)
        touched += 1

        one = {
            "module_name": entry["module_name"],
            "module_class": entry["module_class"],
            "adapter_name": entry["adapter_name"],
            "mu": float(ws["mu"]),
            "sigma": sigma,
            "skew": skew,
            "alpha_l": float(alpha_l),
            "std_l": float(std_l),
        }
        layer_stats.append(one)

        if verbose:
            print(
                "[SkewLoRA-WAware] "
                f"module={one['module_name']} adapter={one['adapter_name']} "
                f"mu={one['mu']:.5f} sigma={one['sigma']:.5f} skew={one['skew']:.5f} "
                f"alpha_l={one['alpha_l']:.5f} std_l={one['std_l']:.5f}"
            )

    return {
        "touched": touched,
        "stats_mode": stats_mode,
        "base_alpha": base_alpha,
        "base_std": base_std,
        "c1": c1,
        "c2": c2,
        "alpha_max": alpha_max,
        "std_min": std_min,
        "std_max": std_max,
        "median_sigma": median_sigma,
        "target_modules": list(target_modules) if target_modules is not None else None,
        "layers": layer_stats,
    }


def _direction_scale(skew_direction: str, damped_ratio: float) -> float:
    if skew_direction == "same":
        return 1.0
    if skew_direction == "opposite":
        return -1.0
    if skew_direction == "damped":
        return float(damped_ratio)
    raise ValueError(f"Unsupported skew_direction={skew_direction}")


def apply_deltaw_stats_initialization(
    model: nn.Module,
    deltaw_stats: Dict[str, object],
    target_modules: Optional[Sequence[str]] = None,
    base_alpha: float = 3.0,
    base_std: float = 0.02,
    c1: float = 1.0,
    c2: float = 0.5,
    alpha_max: float = 8.0,
    std_min: float = 0.005,
    std_max: float = 0.05,
    skew_direction: str = "same",
    damped_ratio: float = 0.5,
    skew_source: str = "eff_skew",
    std_source: str = "eff_std",
    eps: float = 1e-8,
    verbose: bool = True,
) -> Dict[str, object]:
    if skew_source not in {"eff_skew", "delta_skew"}:
        raise ValueError("skew_source must be one of {'eff_skew', 'delta_skew'}")
    if std_source not in {"eff_std", "delta_std"}:
        raise ValueError("std_source must be one of {'eff_std', 'delta_std'}")

    layers = deltaw_stats.get("layers", [])
    if not isinstance(layers, list) or not layers:
        raise ValueError("deltaw_stats must contain a non-empty 'layers' list.")

    direction_scale = _direction_scale(skew_direction, damped_ratio)

    lookup: Dict[str, Dict[str, object]] = {}
    for row in layers:
        name = row.get("module_name")
        if isinstance(name, str):
            lookup[name] = row

    matched_entries: List[Dict[str, object]] = []
    std_values: List[float] = []

    for module_name, module in _iter_named_lora_modules(model):
        if not _target_match(module_name, target_modules):
            continue
        src = lookup.get(module_name)
        if src is None:
            continue
        init_std = float(src.get(std_source, 0.0))
        init_skew = float(src.get(skew_source, 0.0))

        for adapter_name, lora_A in module.lora_A.items():
            matched_entries.append(
                {
                    "module_name": module_name,
                    "module_class": module.__class__.__name__,
                    "adapter_name": adapter_name,
                    "init_std": init_std,
                    "init_skew": init_skew,
                    "lora_A": lora_A,
                    "lora_B": module.lora_B[adapter_name],
                }
            )
            std_values.append(max(init_std, eps))

    if not matched_entries:
        raise RuntimeError("No overlapping LoRA modules found between model and deltaw_stats.")

    median_std = float(torch.tensor(std_values, dtype=torch.float32).median().item())
    median_std = max(median_std, eps)

    touched = 0
    layer_stats: List[Dict[str, object]] = []
    for entry in matched_entries:
        init_std = float(entry["init_std"])
        init_skew = float(entry["init_skew"])

        alpha_l = base_alpha * math.tanh(c1 * direction_scale * init_skew)
        alpha_l = max(-alpha_max, min(alpha_max, alpha_l))

        ratio = max(init_std, eps) / median_std
        std_l = base_std * (ratio ** c2)
        std_l = max(std_min, min(std_max, std_l))

        lora_A = entry["lora_A"]
        lora_B = entry["lora_B"]
        skew_normal_(lora_A.weight, alpha=alpha_l, std=std_l)
        nn.init.zeros_(lora_B.weight)
        touched += 1

        one = {
            "module_name": entry["module_name"],
            "module_class": entry["module_class"],
            "adapter_name": entry["adapter_name"],
            "init_skew": init_skew,
            "init_std": init_std,
            "alpha_l": float(alpha_l),
            "std_l": float(std_l),
        }
        layer_stats.append(one)

        if verbose:
            print(
                "[SkewLoRA-DeltaW] "
                f"module={one['module_name']} adapter={one['adapter_name']} "
                f"init_std={one['init_std']:.5f} init_skew={one['init_skew']:.5f} "
                f"alpha_l={one['alpha_l']:.5f} std_l={one['std_l']:.5f}"
            )

    return {
        "touched": touched,
        "base_alpha": base_alpha,
        "base_std": base_std,
        "c1": c1,
        "c2": c2,
        "alpha_max": alpha_max,
        "std_min": std_min,
        "std_max": std_max,
        "skew_direction": skew_direction,
        "damped_ratio": damped_ratio,
        "skew_source": skew_source,
        "std_source": std_source,
        "direction_scale": direction_scale,
        "median_init_std": median_std,
        "target_modules": list(target_modules) if target_modules is not None else None,
        "layers": layer_stats,
    }