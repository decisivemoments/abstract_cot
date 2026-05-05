from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    bottleneck_attention_mask: list[list[list[bool]]],
    attention_mask: list[list[int]],
) -> list[list[list[list[float]]]]:
    batch_mask: list[list[list[list[float]]]] = []
    for sample_mask, sample_attention in zip(bottleneck_attention_mask, attention_mask, strict=True):
        length = len(sample_attention)
        rows: list[list[float]] = []
        for row_idx in range(length):
            row: list[float] = []
            target_is_valid = bool(sample_attention[row_idx])
            for col_idx in range(length):
                source_is_valid = bool(sample_attention[col_idx])
                allowed = (
                    target_is_valid
                    and source_is_valid
                    and sample_mask[row_idx][col_idx]
                )
                row.append(0.0 if allowed else NEG_INF)
            rows.append(row)
        batch_mask.append([rows])
    return batch_mask


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
        batch["bottleneck_attention_mask"],
        batch["attention_mask"],
    )
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": additive_mask,
        "labels": batch["labels"],
        "position_ids": batch["position_ids"],
    }
    if as_tensors:
        model_inputs = {key: _to_torch(value, device=device) for key, value in model_inputs.items()}
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
