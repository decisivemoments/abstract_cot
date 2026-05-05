from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


@dataclass(frozen=True)
class ContinuousLatentConfig:
    end_token_id: int
    max_reasoning_steps: int
    temperature: float = 1.0
    do_sample: bool = False
    detach_between_steps: bool = False


@dataclass(frozen=True)
class ContinuousLatentTrace:
    latent_states: Any
    stop_logits: Any
    predicted_token_ids: Any
    stopped: Any
    num_steps: int


def _require_torch():
    if torch is None:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for continuous latent reasoning")


def _sample_token(logits, *, do_sample: bool, temperature: float):
    _require_torch()
    if do_sample:
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.argmax(logits, dim=-1)


def _last_hidden_state(model_outputs):
    hidden_states = getattr(model_outputs, "hidden_states", None)
    if hidden_states is None and isinstance(model_outputs, dict):
        hidden_states = model_outputs.get("hidden_states")
    if not hidden_states:
        raise ValueError("model outputs must include hidden_states for continuous latent reasoning")
    return hidden_states[-1][:, -1, :]


def _last_logits(model_outputs):
    logits = getattr(model_outputs, "logits", None)
    if logits is None and isinstance(model_outputs, dict):
        logits = model_outputs.get("logits")
    if logits is None:
        raise ValueError("model outputs must include logits for continuous latent reasoning")
    return logits[:, -1, :]


def generate_continuous_latent_trace(
    model,
    *,
    input_ids,
    attention_mask=None,
    config: ContinuousLatentConfig,
):
    _require_torch()
    device = input_ids.device

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    latent_states = []
    stop_logits = []
    predicted_token_ids = []
    stopped = torch.zeros(input_ids.size(0), dtype=torch.bool, device=device)

    for _ in range(config.max_reasoning_steps):
        current_hidden = _last_hidden_state(outputs)
        current_logits = _last_logits(outputs)
        next_token_ids = _sample_token(
            current_logits,
            do_sample=config.do_sample,
            temperature=config.temperature,
        )

        latent_states.append(current_hidden)
        stop_logits.append(current_logits)
        predicted_token_ids.append(next_token_ids)

        stopped = stopped | (next_token_ids == config.end_token_id)
        if bool(torch.all(stopped).item()):
            break

        next_inputs_embeds = current_hidden.unsqueeze(1)
        if config.detach_between_steps:
            next_inputs_embeds = next_inputs_embeds.detach()
        outputs = model(
            inputs_embeds=next_inputs_embeds,
            attention_mask=torch.ones(
                (input_ids.size(0), 1),
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=device,
            ),
            output_hidden_states=True,
            use_cache=False,
        )

    return ContinuousLatentTrace(
        latent_states=torch.stack(latent_states, dim=1),
        stop_logits=torch.stack(stop_logits, dim=1),
        predicted_token_ids=torch.stack(predicted_token_ids, dim=1),
        stopped=stopped,
        num_steps=len(latent_states),
    )


def compose_continuous_inputs_embeds(
    model,
    *,
    prompt_input_ids,
    latent_trace: ContinuousLatentTrace,
    answer_input_ids=None,
):
    _require_torch()
    embed_layer = model.get_input_embeddings()
    prompt_embeds = embed_layer(prompt_input_ids)
    components = [prompt_embeds, latent_trace.latent_states]
    answer_length = 0
    if answer_input_ids is not None:
        answer_embeds = embed_layer(answer_input_ids)
        components.append(answer_embeds)
        answer_length = int(answer_input_ids.size(1))

    inputs_embeds = torch.cat(components, dim=1)
    attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device)
    labels = None
    if answer_input_ids is not None:
        ignore_prefix = prompt_embeds.size(1) + latent_trace.latent_states.size(1)
        labels = torch.full(
            (inputs_embeds.size(0), inputs_embeds.size(1)),
            -100,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        labels[:, ignore_prefix : ignore_prefix + answer_length] = answer_input_ids
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def generate_answer_from_continuous_latent(
    model,
    *,
    prompt_input_ids,
    latent_trace: ContinuousLatentTrace,
    generation_kwargs: dict[str, Any] | None = None,
):
    _require_torch()
    generation_kwargs = generation_kwargs or {}
    composed = compose_continuous_inputs_embeds(
        model,
        prompt_input_ids=prompt_input_ids,
        latent_trace=latent_trace,
        answer_input_ids=None,
    )
    return model.generate(
        inputs_embeds=composed["inputs_embeds"],
        attention_mask=composed["attention_mask"],
        **generation_kwargs,
    )
