from __future__ import annotations

from typing import Dict, List


def infer_arch_family(model_name_or_path: str) -> str:
    lower = model_name_or_path.lower()
    if "deberta-v2" in lower:
        return "deberta-v2"
    if "deberta" in lower:
        return "deberta"
    if "roberta" in lower:
        return "roberta"
    if "bert" in lower:
        return "bert"
    return "bert"


# Use full attention-path suffixes to avoid unintended matching of non-attention dense layers.
ARCH_PROFILE_TARGETS: Dict[str, Dict[str, List[str]]] = {
    "roberta": {
        "qv": ["attention.self.query", "attention.self.value"],
        "qkv": ["attention.self.query", "attention.self.key", "attention.self.value"],
        "qkvo": [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ],
    },
    "bert": {
        "qv": ["attention.self.query", "attention.self.value"],
        "qkv": ["attention.self.query", "attention.self.key", "attention.self.value"],
        "qkvo": [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ],
    },
    "deberta": {
        "qv": ["attention.self.query_proj", "attention.self.value_proj"],
        "qkv": [
            "attention.self.query_proj",
            "attention.self.key_proj",
            "attention.self.value_proj",
        ],
        "qkvo": [
            "attention.self.query_proj",
            "attention.self.key_proj",
            "attention.self.value_proj",
            "attention.output.dense",
        ],
    },
    "deberta-v2": {
        "qv": ["attention.self.query_proj", "attention.self.value_proj"],
        "qkv": [
            "attention.self.query_proj",
            "attention.self.key_proj",
            "attention.self.value_proj",
        ],
        "qkvo": [
            "attention.self.query_proj",
            "attention.self.key_proj",
            "attention.self.value_proj",
            "attention.output.dense",
        ],
    },
}


def infer_target_modules(model_name_or_path: str, target_profile: str = "qkvo") -> List[str]:
    if target_profile not in {"qv", "qkv", "qkvo"}:
        raise ValueError(f"Unsupported target_profile={target_profile}")
    arch = infer_arch_family(model_name_or_path)
    return ARCH_PROFILE_TARGETS.get(arch, ARCH_PROFILE_TARGETS["bert"])[target_profile]
