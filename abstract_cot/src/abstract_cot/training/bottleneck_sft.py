from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

from abstract_cot.data.collator import collate_bottleneck_features, collate_distillation_features
from abstract_cot.data.prompt_formatter import render_abstract_trace
from abstract_cot.data.distill_dataset import build_distillation_example
from abstract_cot.data.schema import SupervisedSample, TraceSample
from abstract_cot.data.tokenized_features import (
    build_bottleneck_feature,
    build_distillation_feature,
    resolve_pad_token_id,
)
from abstract_cot.data.warmup_dataset import build_bottleneck_sft_example, initialize_random_trace
from abstract_cot.decoding.constrained_decoder import AbstractDecodingConfig, next_abstract_logits
from abstract_cot.training.distributed import DistributedContext, reduce_scalar_sum
from abstract_cot.training.sft_trainer import BottleneckSFTTrainer, DistillationSFTTrainer
from abstract_cot.tokenization.tokenizer_extension import TokenizerArtifacts
from abstract_cot.utils.io import write_json


@dataclass(frozen=True)
class WarmupRuntimeConfig:
    dataset_path: str
    output_dir: str
    batch_size: int
    max_samples: int
    max_trace_length: int
    learning_rate: float
    rounds: int
    seed: int
    device: str = "cpu"
    use_fsdp: bool = False


def _rows_to_supervised_samples(rows: list[dict[str, Any]]) -> list[SupervisedSample]:
    return [
        SupervisedSample(
            sample_id=str(row["sample_id"]).strip(),
            prompt=str(row["prompt"]).strip(),
            cot=str(row["cot"]).strip(),
            answer=str(row["answer"]).strip(),
            task_type=str(row.get("task_type", "generic")).strip() or "generic",
            meta={},
        )
        for row in rows
    ]


def _target_total_samples(
    max_samples: int,
    *,
    rounds: int,
    batch_size: int,
    distributed: DistributedContext,
) -> int:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    granularity = 2 * rounds * batch_size * distributed.world_size
    if max_samples < granularity:
        raise ValueError(
            f"max_samples={max_samples} is too small for rounds={rounds}, batch_size={batch_size}, "
            f"world_size={distributed.world_size}; need at least {granularity}"
        )
    usable = max_samples - (max_samples % granularity)
    if usable <= 0:
        raise ValueError("no usable samples remain after enforcing equal round and batch partitioning")
    return usable


def load_warmup_round_datasets(
    dataset_path: str,
    *,
    max_samples: int,
    rounds: int,
    batch_size: int,
    distributed: DistributedContext,
):
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required to load Hugging Face datasets") from exc

    total_target = _target_total_samples(
        max_samples,
        rounds=rounds,
        batch_size=batch_size,
        distributed=distributed,
    )

    if distributed.is_main_process:
        print(
            {
                "dataset_path": dataset_path,
                "dataset_loading": "huggingface-load_from_disk",
                "world_size": distributed.world_size,
                "target_total_samples": total_target,
            }
        )

    dataset = load_from_disk(dataset_path)
    required_columns = {"sample_id", "prompt", "cot", "answer"}
    missing = sorted(required_columns.difference(dataset.column_names))
    if missing:
        raise ValueError(
            f"preprocessed warm-up dataset is missing required columns: {missing}; "
            f"expected a -cot dataset prepared by scripts/preprocess_dolci_think_sft.py"
        )
    if len(dataset) < total_target:
        raise ValueError(
            f"preprocessed dataset only has {len(dataset)} rows, but warm-up needs {total_target}"
        )

    dataset = dataset.select(range(total_target))

    if distributed.world_size > 1:
        dataset = dataset.shard(
            num_shards=distributed.world_size,
            index=distributed.rank,
            contiguous=True,
        )

    local_count = len(dataset)
    if local_count % (2 * rounds * batch_size) != 0:
        raise ValueError(
            f"local dataset size {local_count} on rank {distributed.rank} must be divisible by "
            f"2 * rounds * batch_size = {2 * rounds * batch_size}"
        )

    if distributed.is_main_process:
        print({"parsed_samples_per_rank": local_count})

    per_subset_size = local_count // (2 * rounds)
    round_datasets = []
    for round_idx in range(rounds):
        first_start = round_idx * 2 * per_subset_size
        second_start = first_start + per_subset_size
        d_t1 = dataset.select(range(first_start, first_start + per_subset_size))
        d_t2 = dataset.select(range(second_start, second_start + per_subset_size))
        round_datasets.append((d_t1, d_t2))
    return round_datasets


def _trace_text_from_ids(tokenizer, token_ids: list[int], artifacts: TokenizerArtifacts) -> str:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return render_abstract_trace(tokens, begin_token=artifacts.begin_token, end_token=artifacts.end_token)


def generate_on_policy_trace(
    model,
    tokenizer,
    sample: SupervisedSample,
    *,
    artifacts: TokenizerArtifacts,
    max_trace_length: int,
    device: str,
    include_cot: bool,
    round_idx: int,
) -> TraceSample:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for on-policy abstract trace generation") from exc

    prompt_text = sample.prompt
    if include_cot and sample.cot:
        prompt_text = f"{sample.prompt}\n{sample.cot}"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = prompt_ids + [artifacts.begin_token_id]
    generated_ids: list[int] = []
    decoding_config = AbstractDecodingConfig(
        abstract_token_ids=artifacts.abstract_token_ids,
        end_token_id=artifacts.end_token_id,
        max_trace_length=max_trace_length,
    )

    was_training = model.training
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        outputs = model(input_ids=tensor, use_cache=True)
        for _ in range(max_trace_length + 1):
            logits = outputs.logits[:, -1, :]
            constrained_logits = next_abstract_logits(logits, len(generated_ids), decoding_config)
            next_token_id = int(torch.argmax(constrained_logits, dim=-1).item())
            if next_token_id == artifacts.end_token_id:
                break
            generated_ids.append(next_token_id)
            next_tensor = torch.tensor([[next_token_id]], device=device, dtype=torch.long)
            outputs = model(
                input_ids=next_tensor,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
    if was_training:
        model.train()

    return TraceSample(
        sample_id=sample.sample_id,
        prompt=sample.prompt,
        answer=sample.answer,
        abstract_trace_text=_trace_text_from_ids(tokenizer, generated_ids, artifacts),
        abstract_trace_ids=generated_ids,
        stage="on_policy",
        round_idx=round_idx,
        cot=sample.cot,
    )


def _loss_scalar(loss) -> float:
    return float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)


def _supervised_collate(rows: list[dict[str, Any]]) -> list[SupervisedSample]:
    return _rows_to_supervised_samples(rows)


def _build_phase_sample_loader(dataset, batch_size: int):
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to construct dataloaders") from exc

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        collate_fn=_supervised_collate,
    )


class _BottleneckFeatureDataset:
    def __init__(self, samples: list[SupervisedSample], traces: list[TraceSample], tokenizer) -> None:
        self.features = [
            build_bottleneck_feature(build_bottleneck_sft_example(sample, trace), tokenizer)
            for sample, trace in zip(samples, traces, strict=True)
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx]


class _DistillationFeatureDataset:
    def __init__(self, samples: list[SupervisedSample], traces: list[TraceSample], tokenizer) -> None:
        self.features = [
            build_distillation_feature(build_distillation_example(sample, trace), tokenizer)
            for sample, trace in zip(samples, traces, strict=True)
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx]


def _build_bottleneck_train_loader(
    samples: list[SupervisedSample],
    traces: list[TraceSample],
    tokenizer,
    batch_size: int,
):
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to construct dataloaders") from exc

    dataset = _BottleneckFeatureDataset(samples, traces, tokenizer)
    pad_token_id = resolve_pad_token_id(tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda features: collate_bottleneck_features(features, pad_token_id),
    )


def _build_distillation_train_loader(
    samples: list[SupervisedSample],
    traces: list[TraceSample],
    tokenizer,
    batch_size: int,
):
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to construct dataloaders") from exc

    dataset = _DistillationFeatureDataset(samples, traces, tokenizer)
    pad_token_id = resolve_pad_token_id(tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda features: collate_distillation_features(features, pad_token_id),
    )


def _collect_trace_inputs(sample_loader) -> list[SupervisedSample]:
    samples: list[SupervisedSample] = []
    for batch in sample_loader:
        samples.extend(batch)
    return samples


def _build_bottleneck_traces(
    samples: list[SupervisedSample],
    *,
    model,
    tokenizer,
    tokenizer_artifacts: TokenizerArtifacts,
    runtime: WarmupRuntimeConfig,
    round_idx: int,
) -> list[TraceSample]:
    rng = random.Random(runtime.seed + round_idx)
    if round_idx == 1:
        return [
            initialize_random_trace(sample, tokenizer_artifacts.abstract_tokens, rng, round_idx=round_idx)
            for sample in samples
        ]
    return [
        generate_on_policy_trace(
            model,
            tokenizer,
            sample,
            artifacts=tokenizer_artifacts,
            max_trace_length=runtime.max_trace_length,
            device=runtime.device,
            include_cot=True,
            round_idx=round_idx,
        )
        for sample in samples
    ]


def _build_distillation_traces(
    samples: list[SupervisedSample],
    *,
    model,
    tokenizer,
    tokenizer_artifacts: TokenizerArtifacts,
    runtime: WarmupRuntimeConfig,
    round_idx: int,
) -> list[TraceSample]:
    return [
        generate_on_policy_trace(
            model,
            tokenizer,
            sample,
            artifacts=tokenizer_artifacts,
            max_trace_length=runtime.max_trace_length,
            device=runtime.device,
            include_cot=False,
            round_idx=round_idx,
        )
        for sample in samples
    ]


def run_sft_phase(
    trainer,
    optimizer,
    dataloader,
    *,
    distributed: DistributedContext,
    phase_name: str,
):
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        tqdm = None

    losses: list[float] = []
    iterable = dataloader
    if tqdm is not None and distributed.is_main_process:
        iterable = tqdm(dataloader, desc=phase_name, leave=False)
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
        "per_rank_num_steps": local_steps,
        "global_num_steps": total_steps,
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
    tokenizer_artifacts: TokenizerArtifacts,
    runtime: WarmupRuntimeConfig,
    distributed: DistributedContext,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for training") from exc

    round_datasets = load_warmup_round_datasets(
        runtime.dataset_path,
        max_samples=runtime.max_samples,
        rounds=runtime.rounds,
        batch_size=runtime.batch_size,
        distributed=distributed,
    )

    model = maybe_wrap_fsdp(model, distributed=distributed, use_fsdp=runtime.use_fsdp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=runtime.learning_rate)
    bottleneck_trainer = BottleneckSFTTrainer(model, as_tensors=True, device=runtime.device)
    distill_trainer = DistillationSFTTrainer(model, as_tensors=True, device=runtime.device)

    local_total_samples = sum(len(d_t1) + len(d_t2) for d_t1, d_t2 in round_datasets)
    summary = {
        "num_samples_per_rank": local_total_samples,
        "rounds": runtime.rounds,
        "batch_size_per_rank": runtime.batch_size,
        "global_effective_batch_size": runtime.batch_size * distributed.world_size,
        "device": runtime.device,
        "distributed": {
            "enabled": distributed.enabled,
            "rank": distributed.rank,
            "world_size": distributed.world_size,
            "use_fsdp": runtime.use_fsdp,
        },
        "round_summaries": [],
    }

    for round_idx, (d_t1, d_t2) in enumerate(round_datasets, start=1):
        bottleneck_source_loader = _build_phase_sample_loader(d_t1, runtime.batch_size)
        bottleneck_samples = _collect_trace_inputs(bottleneck_source_loader)
        bottleneck_traces = _build_bottleneck_traces(
            bottleneck_samples,
            model=model,
            tokenizer=tokenizer,
            tokenizer_artifacts=tokenizer_artifacts,
            runtime=runtime,
            round_idx=round_idx,
        )
        bottleneck_train_loader = _build_bottleneck_train_loader(
            bottleneck_samples,
            bottleneck_traces,
            tokenizer,
            runtime.batch_size,
        )

        if distributed.is_main_process:
            print(
                {
                    "round": round_idx,
                    "D_t1_size": len(d_t1),
                    "D_t2_size": len(d_t2),
                    "per_rank_bottleneck_batches": len(bottleneck_train_loader),
                }
            )

        model.train()
        bottleneck_metrics = run_sft_phase(
            bottleneck_trainer,
            optimizer,
            bottleneck_train_loader,
            distributed=distributed,
            phase_name=f"round{round_idx}-bottleneck",
        )

        distill_source_loader = _build_phase_sample_loader(d_t2, runtime.batch_size)
        distill_samples = _collect_trace_inputs(distill_source_loader)
        distill_traces = _build_distillation_traces(
            distill_samples,
            model=model,
            tokenizer=tokenizer,
            tokenizer_artifacts=tokenizer_artifacts,
            runtime=runtime,
            round_idx=round_idx,
        )
        distill_train_loader = _build_distillation_train_loader(
            distill_samples,
            distill_traces,
            tokenizer,
            runtime.batch_size,
        )

        if distributed.is_main_process:
            print(
                {
                    "round": round_idx,
                    "per_rank_distill_batches": len(distill_train_loader),
                }
            )

        model.train()
        distill_metrics = run_sft_phase(
            distill_trainer,
            optimizer,
            distill_train_loader,
            distributed=distributed,
            phase_name=f"round{round_idx}-distill",
        )

        summary["round_summaries"].append(
            {
                "round": round_idx,
                "D_t1_size": len(d_t1),
                "D_t2_size": len(d_t2),
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
