from __future__ import annotations

from dataclasses import dataclass

from abstract_cot.data.prompt_formatter import render_abstract_trace
from abstract_cot.decoding.constrained_decoder import AbstractDecodingConfig, next_abstract_logits
from abstract_cot.tokenization.tokenizer_extension import TokenizerArtifacts


@dataclass(frozen=True)
class GenerationConfig:
    batch_size: int = 8
    max_answer_new_tokens: int = 256
    max_trace_length: int = 128
    do_sample: bool = False
    temperature: float = 1.0


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def generate_text_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> list[str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for text generation") from exc

    if not prompts:
        return []

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_padding_side

    input_ids = encoded["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.long)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = input_ids.size(1)
    return [
        tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
        for row in generated
    ]


def _constrain_next_tokens(logits, generated_lengths: list[int], finished: list[bool], decoding_config: AbstractDecodingConfig):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for abstract trace generation") from exc

    next_token_ids: list[int] = []
    for row_idx, generated_length in enumerate(generated_lengths):
        if finished[row_idx]:
            next_token_ids.append(decoding_config.end_token_id)
            continue
        constrained_logits = next_abstract_logits(logits[row_idx : row_idx + 1], generated_length, decoding_config)
        next_token_ids.append(int(torch.argmax(constrained_logits, dim=-1).item()))
    return next_token_ids


def generate_abstract_traces(
    model,
    tokenizer,
    prompts: list[str],
    *,
    tokenizer_artifacts: TokenizerArtifacts,
    device: str,
    max_trace_length: int,
) -> list[str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for abstract trace generation") from exc

    if not prompts:
        return []

    generated_ids: list[list[int]] = [[] for _ in prompts]
    generated_lengths = [0 for _ in prompts]
    finished = [False for _ in prompts]
    decoding_config = AbstractDecodingConfig(
        abstract_token_ids=tokenizer_artifacts.abstract_token_ids,
        end_token_id=tokenizer_artifacts.end_token_id,
        max_trace_length=max_trace_length,
    )

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_padding_side

    input_tensor = encoded["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.long)
    begin_token_column = torch.full(
        (len(prompts), 1),
        tokenizer_artifacts.begin_token_id,
        device=device,
        dtype=torch.long,
    )
    input_tensor = torch.cat([input_tensor, begin_token_column], dim=1)
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((len(prompts), 1), device=device, dtype=attention_mask.dtype),
        ],
        dim=1,
    )
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=True)
        batch_indices = torch.arange(len(prompts), device=device)
        last_positions = torch.tensor([length - 1 for length in prompt_lengths], device=device, dtype=torch.long)
        for _ in range(max_trace_length + 1):
            if outputs.logits.size(1) == 1:
                logits = outputs.logits[:, -1, :]
            else:
                logits = outputs.logits[batch_indices, last_positions, :]
            next_token_ids = _constrain_next_tokens(logits, generated_lengths, finished, decoding_config)
            next_tensor = torch.tensor(next_token_ids, device=device, dtype=torch.long).unsqueeze(1)

            all_finished = True
            for row_idx, next_token_id in enumerate(next_token_ids):
                if finished[row_idx]:
                    continue
                if next_token_id == tokenizer_artifacts.end_token_id:
                    finished[row_idx] = True
                    continue
                generated_ids[row_idx].append(next_token_id)
                generated_lengths[row_idx] += 1
                all_finished = False
            if all(finished) or all_finished:
                break
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((len(prompts), 1), device=device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            outputs = model(
                input_ids=next_tensor,
                attention_mask=attention_mask,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
            prompt_lengths = [length + 1 for length in prompt_lengths]
    if was_training:
        model.train()

    return [
        render_abstract_trace(
            tokenizer.convert_ids_to_tokens(token_ids),
            begin_token=tokenizer_artifacts.begin_token,
            end_token=tokenizer_artifacts.end_token,
        )
        for token_ids in generated_ids
    ]


def generate_texts_in_batches(
    model,
    tokenizer,
    prompts: list[str],
    *,
    config: GenerationConfig,
    device: str,
    progress_desc: str | None = None,
) -> list[str]:
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        tqdm = None

    outputs: list[str] = []
    batches = _batched(prompts, config.batch_size)
    iterable = batches
    if tqdm is not None:
        iterable = tqdm(batches, desc=progress_desc or "answer-generation", leave=True)
    for prompt_batch in iterable:
        outputs.extend(
            generate_text_batch(
                model,
                tokenizer,
                prompt_batch,
                device=device,
                max_new_tokens=config.max_answer_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
            )
        )
    return outputs


def generate_traces_in_batches(
    model,
    tokenizer,
    prompts: list[str],
    *,
    tokenizer_artifacts: TokenizerArtifacts,
    config: GenerationConfig,
    device: str,
    progress_desc: str | None = None,
) -> list[str]:
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        tqdm = None

    outputs: list[str] = []
    batches = _batched(prompts, config.batch_size)
    iterable = batches
    if tqdm is not None:
        iterable = tqdm(batches, desc=progress_desc or "trace-generation", leave=True)
    for prompt_batch in iterable:
        outputs.extend(
            generate_abstract_traces(
                model,
                tokenizer,
                prompt_batch,
                tokenizer_artifacts=tokenizer_artifacts,
                device=device,
                max_trace_length=config.max_trace_length,
            )
        )
    return outputs
