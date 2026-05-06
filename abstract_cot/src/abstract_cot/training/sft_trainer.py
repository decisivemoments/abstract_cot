from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .forward_batch import (
    PreparedBatch,
    compute_causal_lm_loss_from_logits,
    prepare_bottleneck_batch,
    prepare_distillation_batch,
)


@dataclass(frozen=True)
class TrainingStepResult:
    loss: Any
    metadata: dict[str, Any]
    model_outputs: Any


def _extract_loss(model_outputs, model_inputs):
    if hasattr(model_outputs, "loss") and model_outputs.loss is not None:
        return model_outputs.loss
    if isinstance(model_outputs, dict) and "loss" in model_outputs and model_outputs["loss"] is not None:
        return model_outputs["loss"]

    logits = getattr(model_outputs, "logits", None)
    if logits is None and isinstance(model_outputs, dict):
        logits = model_outputs.get("logits")
    if logits is None:
        raise ValueError("model output must provide either loss or logits")
    return compute_causal_lm_loss_from_logits(logits, model_inputs["labels"])


class BaseSFTTrainer:
    def __init__(
        self,
        model,
        *,
        prepare_batch: Callable[..., PreparedBatch],
        as_tensors: bool = False,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.prepare_batch = prepare_batch
        self.as_tensors = as_tensors
        self.device = device

    def training_step(self, batch: dict[str, Any]) -> TrainingStepResult:
        prepared = self.prepare_batch(batch, as_tensors=self.as_tensors, device=self.device)
        attention_mask = prepared.model_inputs.get("attention_mask")
        if hasattr(attention_mask, "dtype") and hasattr(attention_mask, "is_floating_point") and attention_mask.is_floating_point():
            first_param = next(self.model.parameters(), None)
            if first_param is not None:
                prepared.model_inputs["attention_mask"] = attention_mask.to(dtype=first_param.dtype)
        model_outputs = self.model(**prepared.model_inputs)
        loss = _extract_loss(model_outputs, prepared.model_inputs)
        return TrainingStepResult(
            loss=loss,
            metadata=prepared.metadata,
            model_outputs=model_outputs,
        )


class BottleneckSFTTrainer(BaseSFTTrainer):
    def __init__(self, model, *, as_tensors: bool = False, device: str | None = None) -> None:
        super().__init__(
            model,
            prepare_batch=prepare_bottleneck_batch,
            as_tensors=as_tensors,
            device=device,
        )


class DistillationSFTTrainer(BaseSFTTrainer):
    def __init__(self, model, *, as_tensors: bool = False, device: str | None = None) -> None:
        super().__init__(
            model,
            prepare_batch=prepare_distillation_batch,
            as_tensors=as_tensors,
            device=device,
        )
