from __future__ import annotations

import os
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
from abstract_cot.training.distributed import DistributedContext, reduce_scalar_sum
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
    device: str = "cpu"
    use_fsdp: bool = False


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


def _target_samples_per_rank(max_samples: int, distributed: DistributedContext) -> int:
    if distributed.world_size <= 1:
        return max_samples
    if max_samples < distributed.world_size:
        raise ValueError(
            f"max_samples={max_samples} is smaller than world_size={distributed.world_size}; "
            f"increase max_samples or reduce GPU count"
        )
    if max_samples % distributed.world_size != 0:
        raise ValueError(
            f"max_samples={max_samples} must be divisible by world_size={distributed.world_size} "
            f"for distributed warm-up with equal per-rank work"
        )
    return max_samples // distributed.world_size


def load_supervised_samples(
    dataset_path: str,
    max_samples: int,
    distributed: DistributedContext,
) -> list[SupervisedSample]:
    try:
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required to load Hugging Face datasets") from exc

    target_samples = _target_samples_per_rank(max_samples, distributed)
    if distributed.is_main_process:
        print(
            {
                "dataset_path": dataset_path,
                "dataset_loading": "huggingface-streaming",
                "world_size": distributed.world_size,
                "target_samples_per_rank": target_samples,
            }
        )
    dataset = load_dataset(
        "parquet",
        data_dir=dataset_path,
        split="train",
        streaming=True,
    )
    if distributed.world_size > 1:
        dataset = split_dataset_by_node(dataset, rank=distributed.rank, world_size=distributed.world_size)

    samples: list[SupervisedSample] = []
    for idx, row in enumerate(dataset):
        sample = _parse_messages_row(row, idx)
        if sample is not None and sample.cot:
            samples.append(sample)
            if len(samples) >= target_samples:
                break
            continue

        sample_id = str(row.get("id", f"sample-{idx}"))
        prompt = row.get("prompt") or row.get("x") or ""
        cot = row.get("cot") or row.get("c") or row.get("reasoning") or ""
        answer = row.get("answer") or row.get("y") or row.get("response") or ""
        if not prompt or not answer or not cot:
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
        if len(samples) >= target_samples:
            break
    if len(samples) < target_samples:
        raise ValueError(
            f"only collected {len(samples)} valid samples on rank {distributed.rank}, "
            f"but need {target_samples}. Increase max_samples or reduce GPU count."
        )
    if distributed.is_main_process:
        print({"parsed_samples_per_rank": len(samples)})
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


def run_minimal_sft_epoch(
    trainer,
    optimizer,
    batches,
    *,
    distributed: DistributedContext,
    phase_name: str,
):
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        tqdm = None

    losses: list[float] = []
    iterable = batches
    if tqdm is not None and distributed.is_main_process:
        iterable = tqdm(batches, desc=phase_name, leave=False)
    for batch in iterable:
        optimizer.zero_grad()
        result = trainer.training_step(batch)
        loss = result.loss
        if hasattr(loss, "backward"):
            loss.backward()
            optimizer.step()
        losses.append(_loss_scalar(loss))
        if tqdm is not None and distributed.is_main_process and hasattr(iterable, "set_postfix"):
            iterable.set_postfix(loss=f"{losses[-1]:.4f}")

    local_steps = len(losses)
    local_loss_sum = sum(losses)
    total_steps = int(reduce_scalar_sum(float(local_steps), distributed.device))
    total_loss_sum = reduce_scalar_sum(local_loss_sum, distributed.device)
    return {
        "num_steps": total_steps,
        "mean_loss": total_loss_sum / max(total_steps, 1),
        "last_loss": losses[-1] if losses else None,
    }


def maybe_wrap_fsdp(model, *, distributed: DistributedContext, use_fsdp: bool):
    if not use_fsdp:
        return model
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FSDP is not available in this torch build") from exc
    if not distributed.enabled:
        raise RuntimeError("FSDP requested but distributed process group is not initialized")
    return FSDP(model, device_id=distributed.local_rank if distributed.device.startswith("cuda") else None)


def run_minimal_warmup(
    *,
    model,
    tokenizer,
    abstract_tokens: list[str],
    runtime: WarmupRuntimeConfig,
    distributed: DistributedContext,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for training") from exc

    samples = load_supervised_samples(runtime.dataset_path, runtime.max_samples, distributed)
    bottleneck_batches = build_bottleneck_batches(
        samples, tokenizer, abstract_tokens, runtime.batch_size, runtime.seed
    )
    distill_batches = build_distillation_batches(
        samples, tokenizer, abstract_tokens, runtime.batch_size, runtime.seed + 1
    )

    if distributed.is_main_process:
        print(
            {
                "per_rank_bottleneck_batches": len(bottleneck_batches),
                "per_rank_distill_batches": len(distill_batches),
            }
        )

    model = maybe_wrap_fsdp(model, distributed=distributed, use_fsdp=runtime.use_fsdp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=runtime.learning_rate)
    bottleneck_trainer = BottleneckSFTTrainer(model, as_tensors=True, device=runtime.device)
    distill_trainer = DistillationSFTTrainer(model, as_tensors=True, device=runtime.device)

    summary = {
        "num_samples": len(samples),
        "bottleneck_batches": len(bottleneck_batches),
        "distill_batches": len(distill_batches),
        "device": runtime.device,
        "distributed": {
            "enabled": distributed.enabled,
            "rank": distributed.rank,
            "world_size": distributed.world_size,
            "use_fsdp": runtime.use_fsdp,
        },
        "epochs": [],
    }
    for epoch in range(1, runtime.epochs + 1):
        model.train()
        bottleneck_metrics = run_minimal_sft_epoch(
            bottleneck_trainer,
            optimizer,
            bottleneck_batches,
            distributed=distributed,
            phase_name=f"epoch{epoch}-bottleneck",
        )
        distill_metrics = run_minimal_sft_epoch(
            distill_trainer,
            optimizer,
            distill_batches,
            distributed=distributed,
            phase_name=f"epoch{epoch}-distill",
        )
        summary["epochs"].append(
            {
                "epoch": epoch,
                "bottleneck": bottleneck_metrics,
                "distill": distill_metrics,
            }
        )

    if distributed.is_main_process:
        write_json(f"{runtime.output_dir}/warmup_summary.json", summary)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(runtime.output_dir)
        stateful_model = getattr(model, "module", model)
        if hasattr(stateful_model, "save_pretrained"):
            stateful_model.save_pretrained(runtime.output_dir)
    return summary
