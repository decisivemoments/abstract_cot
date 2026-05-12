from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    dataset_name: str
    prompt: str
    target_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalPrediction:
    sample_id: str
    dataset_name: str
    model_name: str
    mode: str
    prompt_text: str
    generated_trace_text: str | None
    generated_answer_text: str
    extracted_answer: str | None
    normalized_target_answer: str
    normalized_extracted_answer: str | None
    is_correct: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMetrics:
    dataset_name: str
    model_name: str
    mode: str
    num_samples: int
    num_correct: int
    num_valid_predictions: int
    accuracy: float
    extraction_rate: float
    avg_generated_answer_chars: float
    avg_generated_trace_chars: float


@dataclass(frozen=True)
class ModelEvalSpec:
    run_name: str
    model_path: str
    tokenizer_path: str | None = None
    tokenizer_artifacts_path: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str | None = None
    attn_implementation: str = "flash_attention_2"
    modes: tuple[str, ...] = ("direct-answer",)
    device: str | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_name: str
    dataset_path: str
    split: str = "test"
    subset: str | None = None
    max_samples: int | None = None
    batch_size: int = 8
    max_answer_new_tokens: int = 256
    max_trace_length: int = 128

