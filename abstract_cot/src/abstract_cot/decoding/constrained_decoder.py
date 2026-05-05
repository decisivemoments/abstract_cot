from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


@dataclass(frozen=True)
class AbstractDecodingConfig:
    abstract_token_ids: list[int]
    end_token_id: int
    max_trace_length: int


def allowed_abstract_token_ids(config: AbstractDecodingConfig) -> list[int]:
    return [*config.abstract_token_ids, config.end_token_id]


def validate_trace_token_ids(trace_token_ids: Iterable[int], config: AbstractDecodingConfig) -> None:
    allowed = set(allowed_abstract_token_ids(config))
    tokens = list(trace_token_ids)
    if len(tokens) > config.max_trace_length:
        raise ValueError("abstract trace exceeds max_trace_length")
    invalid = [token_id for token_id in tokens if token_id not in allowed]
    if invalid:
        raise ValueError(f"invalid abstract token ids: {invalid}")


def mask_abstract_logits(logits, config: AbstractDecodingConfig):
    if torch is None:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for logits masking")

    allowed = torch.tensor(allowed_abstract_token_ids(config), device=logits.device)
    mask = torch.full_like(logits, float("-inf"))
    mask.index_fill_(dim=-1, index=allowed, value=0.0)
    return logits + mask


def force_end_logits(logits, end_token_id: int):
    if torch is None:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for logits masking")

    forced = torch.full_like(logits, float("-inf"))
    forced[..., end_token_id] = 0.0
    return forced


def next_abstract_logits(logits, generated_length: int, config: AbstractDecodingConfig):
    if generated_length >= config.max_trace_length:
        return force_end_logits(logits, config.end_token_id)
    return mask_abstract_logits(logits, config)
