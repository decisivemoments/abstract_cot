from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from abstract_cot.continuous.recurrent_latent import (
    ContinuousLatentConfig,
    compose_continuous_inputs_embeds,
    generate_answer_from_continuous_latent,
    generate_continuous_latent_trace,
)


@dataclass(frozen=True)
class ContinuousSFTResult:
    latent_trace: Any
    model_outputs: Any
    loss: Any


@dataclass(frozen=True)
class ContinuousRLTrajectory:
    latent_trace: Any
    answer_token_ids: Any
    reward: float | None = None


def run_continuous_sft_forward(
    model,
    *,
    prompt_input_ids,
    answer_input_ids,
    config: ContinuousLatentConfig,
    attention_mask=None,
):
    latent_trace = generate_continuous_latent_trace(
        model,
        input_ids=prompt_input_ids,
        attention_mask=attention_mask,
        config=config,
    )
    model_inputs = compose_continuous_inputs_embeds(
        model,
        prompt_input_ids=prompt_input_ids,
        latent_trace=latent_trace,
        answer_input_ids=answer_input_ids,
    )
    model_outputs = model(**model_inputs)
    loss = getattr(model_outputs, "loss", None)
    if loss is None and isinstance(model_outputs, dict):
        loss = model_outputs.get("loss")
    if loss is None:
        raise ValueError("continuous SFT forward expects model to return loss when labels are provided")
    return ContinuousSFTResult(
        latent_trace=latent_trace,
        model_outputs=model_outputs,
        loss=loss,
    )


def collect_continuous_rl_trajectory(
    model,
    *,
    prompt_input_ids,
    config: ContinuousLatentConfig,
    attention_mask=None,
    generation_kwargs: dict[str, Any] | None = None,
):
    latent_trace = generate_continuous_latent_trace(
        model,
        input_ids=prompt_input_ids,
        attention_mask=attention_mask,
        config=config,
    )
    answer_token_ids = generate_answer_from_continuous_latent(
        model,
        prompt_input_ids=prompt_input_ids,
        latent_trace=latent_trace,
        generation_kwargs=generation_kwargs,
    )
    return ContinuousRLTrajectory(
        latent_trace=latent_trace,
        answer_token_ids=answer_token_ids,
    )
