from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from abstract_cot.modeling.attention_mask import (
    SEGMENT_ABSTRACT,
    SEGMENT_ANSWER,
    SEGMENT_COT,
    SEGMENT_PROMPT,
)

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None


NEG_INF = -1e9


@dataclass(frozen=True)
class PreparedBatch:
    model_inputs: dict[str, Any]
    metadata: dict[str, Any]


def _to_torch(value: Any, device: str | None = None):
    if torch is None:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for tensor conversion")
    tensor = torch.tensor(value)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _build_additive_bottleneck_mask(
    segment_ids,
    attention_mask,
):
    if torch is None:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to build bottleneck attention masks")

    if not torch.is_tensor(segment_ids):
        segment_ids = torch.tensor(segment_ids)
    if not torch.is_tensor(attention_mask):
        attention_mask = torch.tensor(attention_mask)

    valid_tokens = attention_mask.bool()
    target_segments = segment_ids.unsqueeze(2)
    source_segments = segment_ids.unsqueeze(1)

    prompt_target = target_segments == SEGMENT_PROMPT
    cot_target = target_segments == SEGMENT_COT
    abstract_target = target_segments == SEGMENT_ABSTRACT
    answer_target = target_segments == SEGMENT_ANSWER

    source_prompt = source_segments == SEGMENT_PROMPT
    source_cot = source_segments == SEGMENT_COT
    source_abstract = source_segments == SEGMENT_ABSTRACT
    source_answer = source_segments == SEGMENT_ANSWER

    allowed = (
        (prompt_target & source_prompt)
        | (cot_target & (source_prompt | source_cot))
        | (abstract_target & (source_prompt | source_cot | source_abstract))
        | (answer_target & (source_prompt | source_abstract | source_answer))
    )

    sequence_length = segment_ids.size(1)
    causal = torch.tril(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=segment_ids.device)
    ).unsqueeze(0)
    valid_pairs = valid_tokens.unsqueeze(1) & valid_tokens.unsqueeze(2)
    final_mask = allowed & causal & valid_pairs

    additive = torch.zeros(
        (segment_ids.size(0), 1, sequence_length, sequence_length),
        dtype=torch.float32,
        device=segment_ids.device,
    )
    additive.masked_fill_(~final_mask.unsqueeze(1), NEG_INF)
    return additive


def prepare_distillation_batch(
    batch: dict[str, Any],
    *,
    as_tensors: bool = False,
    device: str | None = None,
) -> PreparedBatch:
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
        "position_ids": batch["position_ids"],
    }
    if as_tensors:
        model_inputs = {key: _to_torch(value, device=device) for key, value in model_inputs.items()}
    metadata = {
        "sample_ids": batch["sample_ids"],
        "round_idxs": batch["round_idxs"],
        "segment_ids": batch["segment_ids"],
    }
    return PreparedBatch(model_inputs=model_inputs, metadata=metadata)


def prepare_bottleneck_batch(
    batch: dict[str, Any],
    *,
    as_tensors: bool = False,
    device: str | None = None,
) -> PreparedBatch:
    additive_mask = _build_additive_bottleneck_mask(
        batch["segment_ids"],
        batch["attention_mask"],
    )
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": additive_mask,
        "labels": batch["labels"],
        "position_ids": batch["position_ids"],
    }
    if as_tensors:
        model_inputs = {
            "input_ids": _to_torch(model_inputs["input_ids"], device=device),
            "attention_mask": model_inputs["attention_mask"].to(device) if device is not None else model_inputs["attention_mask"],
            "labels": _to_torch(model_inputs["labels"], device=device),
            "position_ids": _to_torch(model_inputs["position_ids"], device=device),
        }
    metadata = {
        "sample_ids": batch["sample_ids"],
        "round_idxs": batch["round_idxs"],
        "segment_ids": batch["segment_ids"],
        "token_attention_mask": batch["attention_mask"],
    }
    return PreparedBatch(model_inputs=model_inputs, metadata=metadata)


def compute_causal_lm_loss_from_logits(logits, labels, ignore_index: int = -100):
    if torch is None or F is None:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to compute loss from logits")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
