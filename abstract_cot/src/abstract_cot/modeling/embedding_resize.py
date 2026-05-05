from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingResizeResult:
    original_vocab_size: int
    resized_vocab_size: int
    added_token_count: int


def resize_model_embeddings(model, new_vocab_size: int) -> EmbeddingResizeResult:
    input_embeddings = model.get_input_embeddings()
    original_vocab_size = int(input_embeddings.num_embeddings)
    model.resize_token_embeddings(new_vocab_size)
    return EmbeddingResizeResult(
        original_vocab_size=original_vocab_size,
        resized_vocab_size=new_vocab_size,
        added_token_count=new_vocab_size - original_vocab_size,
    )
