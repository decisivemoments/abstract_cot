from __future__ import annotations

import random
import re
from dataclasses import dataclass

from abstract_cot.data.collator import collate_bottleneck_features, collate_distillation_features
from abstract_cot.data.distill_dataset import build_distillation_example
from abstract_cot.data.schema import SupervisedSample
from abstract_cot.data.tokenized_features import (
    build_bottleneck_feature,
    build_distillation_feature,
    resolve_pad_token_id,
)
from abstract_cot.data.warmup_dataset import build_bottleneck_sft_example, initialize_random_trace
from abstract_cot.training.sft_trainer import BottleneckSFTTrainer, DistillationSFTTrainer
from abstract_cot.utils.io import write_json

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)


@dataclass(frozen=True)
class WarmupRuntimeConfig:
    dataset_path: str
    output_dir: str
    batch_size: int
    max_samples: int
    learning_rate: float
    epochs: int
    seed: int


def _parse_messages_row(row: dict, idx: int) -> SupervisedSample | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    user_messages = [message for message in messages if message.get("role") == "user"]
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not user_messages or not assistant_messages:
        return None

    prompt = str(user_messages[-1].get("content", "")).strip()
    assistant_content = str(assistant_messages[-1].get("content", "")).strip()
    if not prompt or not assistant_content:
        return None

    match = _THINK_RE.match(assistant_content)
    if match:
        cot = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        cot = ""
        answer = assistant_content

    if not answer:
        return None

    return SupervisedSample(
        sample_id=str(row.get("id", f"sample-{idx}")),
        prompt=prompt,
        cot=cot,
        answer=answer,
        task_type=str(row.get("dataset_source", "generic")),
        meta={"source_row": idx},
    )


def load_supervised_samples(dataset_path: str, max_samples: int) -> list[SupervisedSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required to load Hugging Face datasets") from exc

    dataset = load_dataset("parquet", data_dir=dataset_path, split="train")
    samples: list[SupervisedSample] = []
    for idx, row in enumerate(dataset):
        if idx >= max_samples:
            break
        sample = _parse_messages_row(row, idx)
        if sample is not None:
            samples.append(sample)
            continue

        sample_id = str(row.get("id", f"sample-{idx}"))
        prompt = row.get("prompt") or row.get("x") or ""
        cot = row.get("cot") or row.get("c") or row.get("reasoning") or ""
        answer = row.get("answer") or row.get("y") or row.get("response") or ""
        if not prompt or not answer:
            continue
        samples.append(
            SupervisedSample(
                sample_id=sample_id,
                prompt=prompt,
                cot=cot,
                answer=answer,
                task_type=str(row.get("task_type", "generic")),
                meta={"source_row": idx},
            )
        )
    if not samples:
        raise ValueError(f"no valid supervised samples found in {dataset_path}; dataset columns: {dataset.column_names}")
    return samples


def build_bottleneck_batches(samples, tokenizer, abstract_tokens, batch_size: int, seed: int):
    rng = random.Random(seed)
    features = []
    for sample in samples:
        if not sample.cot:
            continue
        trace = initialize_random_trace(sample, abstract_tokens, rng, round_idx=1)
        example = build_bottleneck_sft_example(sample, trace)
        features.append(build_bottleneck_feature(example, tokenizer))
    pad_token_id = resolve_pad_token_id(tokenizer)
    if not features:
        raise ValueError("no bottleneck features were built; dataset may not contain CoT fields")
    return [
        collate_bottleneck_features(features[i : i + batch_size], pad_token_id)
        for i in range(0, len(features), batch_size)
    ]


def build_distillation_batches(samples, tokenizer, abstract_tokens, batch_size: int, seed: int):
    rng = random.Random(seed)
    features = []
    for sample in samples:
        trace = initialize_random_trace(sample, abstract_tokens, rng, round_idx=1)
        example = build_distillation_example(sample, trace)
        features.append(build_distillation_feature(example, tokenizer))
    pad_token_id = resolve_pad_token_id(tokenizer)
    if not features:
        raise ValueError("no distillation features were built")
    return [
        collate_distillation_features(features[i : i + batch_size], pad_token_id)
        for i in range(0, len(features), batch_size)
    ]


def _loss_scalar(loss) -> float:
    return float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)


def run_minimal_sft_epoch(trainer, optimizer, batches):
    losses: list[float] = []
    for batch in batches:
        optimizer.zero_grad()
        result = trainer.training_step(batch)
        loss = result.loss
        if hasattr(loss, "backward"):
            loss.backward()
            optimizer.step()
        losses.append(_loss_scalar(loss))
    return {
        "num_steps": len(losses),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "last_loss": losses[-1] if losses else None,
    }


def run_minimal_warmup(
    *,
    model,
    tokenizer,
    abstract_tokens: list[str],
    runtime: WarmupRuntimeConfig,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for training") from exc

    samples = load_supervised_samples(runtime.dataset_path, runtime.max_samples)
    bottleneck_batches = build_bottleneck_batches(
        samples, tokenizer, abstract_tokens, runtime.batch_size, runtime.seed
    )
    distill_batches = build_distillation_batches(
        samples, tokenizer, abstract_tokens, runtime.batch_size, runtime.seed + 1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=runtime.learning_rate)
    bottleneck_trainer = BottleneckSFTTrainer(model, as_tensors=True)
    distill_trainer = DistillationSFTTrainer(model, as_tensors=True)

    summary = {
        "num_samples": len(samples),
        "bottleneck_batches": len(bottleneck_batches),
        "distill_batches": len(distill_batches),
        "epochs": [],
    }
    for epoch in range(1, runtime.epochs + 1):
        model.train()
        bottleneck_metrics = run_minimal_sft_epoch(bottleneck_trainer, optimizer, bottleneck_batches)
        distill_metrics = run_minimal_sft_epoch(distill_trainer, optimizer, distill_batches)
        summary["epochs"].append(
            {
                "epoch": epoch,
                "bottleneck": bottleneck_metrics,
                "distill": distill_metrics,
            }
        )

    write_json(f"{runtime.output_dir}/warmup_summary.json", summary)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(runtime.output_dir)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(runtime.output_dir)
    return summary
