from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None
    nn = object


class SparseLogitsCausalLMWrapper(nn.Module):
    def __init__(self, model) -> None:
        if torch is None or F is None:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is required to construct SparseLogitsCausalLMWrapper")
        super().__init__()
        self.inner_model = model

    @property
    def config(self):
        return self.inner_model.config

    @property
    def model(self):
        return self.inner_model.model

    @property
    def lm_head(self):
        return self.inner_model.lm_head

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.inner_model.gradient_checkpointing_enable(*args, **kwargs)

    def get_input_embeddings(self):
        return self.inner_model.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.inner_model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.inner_model.save_pretrained(*args, **kwargs)

    def forward(self, labels=None, **kwargs):
        if labels is None:
            return self.inner_model(labels=labels, **kwargs)

        base_outputs = self.inner_model.model(**kwargs)
        hidden_states = base_outputs.last_hidden_state
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(-100)

        if valid_mask.any():
            selected_hidden_states = shift_hidden_states[valid_mask]
            selected_labels = shift_labels[valid_mask]
            selected_logits = self.inner_model.lm_head(selected_hidden_states)
            loss = F.cross_entropy(
                selected_logits.float(),
                selected_labels,
                ignore_index=-100,
            )
            logits = selected_logits
        else:
            loss = shift_hidden_states.sum() * 0.0
            logits = shift_hidden_states.new_zeros((0, self.inner_model.config.vocab_size))

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": getattr(base_outputs, "past_key_values", None),
            "hidden_states": getattr(base_outputs, "hidden_states", None),
            "attentions": getattr(base_outputs, "attentions", None),
        }
