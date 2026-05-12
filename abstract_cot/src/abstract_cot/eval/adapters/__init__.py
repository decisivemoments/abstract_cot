from __future__ import annotations

from abstract_cot.eval.adapters.aime import AimeAdapter
from abstract_cot.eval.adapters.gsm8k import Gsm8kAdapter
from abstract_cot.eval.adapters.math500 import Math500Adapter

ADAPTER_REGISTRY = {
    "gsm8k": Gsm8kAdapter(),
    "math-500": Math500Adapter(),
    "aime": AimeAdapter(),
}


def resolve_adapter(dataset_name: str):
    key = dataset_name.strip().lower()
    if key not in ADAPTER_REGISTRY:
        raise KeyError(f"unsupported eval dataset: {dataset_name}")
    return ADAPTER_REGISTRY[key]

