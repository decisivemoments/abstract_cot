from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

from abstract_cot.data.collator import collate_distillation_features
from abstract_cot.data.prompt_formatter import render_abstract_trace
from abstract_cot.data.distill_dataset import build_distillation_example
from abstract_cot.data.schema import SupervisedSample, TraceSample
from abstract_cot.data.tokenized_features import (
    build_bottleneck_y_feature,
    build_bottleneck_z_feature,
    build_distillation_feature,
    resolve_pad_token_id,
)
from abstract_cot.data.warmup_dataset import build_bottleneck_sft_example, initialize_random_trace
from abstract_cot.decoding.constrained_decoder import AbstractDecodingConfig, next_abstract_logits
from abstract_cot.training.distributed import DistributedContext, reduce_scalar_sum, scatter_object
from abstract_cot.training.sft_trainer import DistillationSFTTrainer
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
    max_steps_per_phase: int | None = None
    memory_snapshot_enabled: bool = False
    memory_snapshot_max_entries: int = 100000


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


def _materialize_rows(dataset) -> list[dict[str, Any]]:
    columns = dataset[:]
    row_count = len(next(iter(columns.values()))) if columns else 0
    return [
        {key: values[row_idx] for key, values in columns.items()}
        for row_idx in range(row_count)
    ]


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

    local_rows: list[dict[str, Any]]
    if distributed.enabled:
        per_rank_rows = None
        if distributed.is_main_process:
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
            rows = _materialize_rows(dataset)
            per_rank_size = total_target // distributed.world_size
            per_rank_rows = [
                rows[rank_idx * per_rank_size : (rank_idx + 1) * per_rank_size]
                for rank_idx in range(distributed.world_size)
            ]
        local_rows = scatter_object(per_rank_rows, distributed=distributed, src=0)
    else:
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
        local_rows = _materialize_rows(dataset)

    local_count = len(local_rows)
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
        d_t1 = local_rows[first_start : first_start + per_subset_size]
        d_t2 = local_rows[second_start : second_start + per_subset_size]
        round_datasets.append((d_t1, d_t2))
    return round_datasets


def _trace_text_from_ids(tokenizer, token_ids: list[int], artifacts: TokenizerArtifacts) -> str:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return render_abstract_trace(tokens, begin_token=artifacts.begin_token, end_token=artifacts.end_token)


def _constrain_next_tokens(logits, generated_lengths: list[int], finished: list[bool], decoding_config: AbstractDecodingConfig):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for batched abstract trace generation") from exc

    next_token_ids: list[int] = []
    for row_idx, generated_length in enumerate(generated_lengths):
        if finished[row_idx]:
            next_token_ids.append(decoding_config.end_token_id)
            continue
        constrained_logits = next_abstract_logits(logits[row_idx : row_idx + 1], generated_length, decoding_config)
        next_token_ids.append(int(torch.argmax(constrained_logits, dim=-1).item()))
    return next_token_ids


def generate_on_policy_traces(
    model,
    tokenizer,
    samples: list[SupervisedSample],
    *,
    artifacts: TokenizerArtifacts,
    max_trace_length: int,
    device: str,
    include_cot: bool,
    round_idx: int,
) -> list[TraceSample]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for on-policy abstract trace generation") from exc

    if not samples:
        return []

    prompt_texts: list[str] = []
    for sample in samples:
        prompt_text = sample.prompt
        if include_cot and sample.cot:
            prompt_text = f"{sample.prompt}\n{sample.cot}"
        prompt_texts.append(prompt_text)

    generated_ids: list[list[int]] = [[] for _ in samples]
    generated_lengths = [0 for _ in samples]
    finished = [False for _ in samples]
    decoding_config = AbstractDecodingConfig(
        abstract_token_ids=artifacts.abstract_token_ids,
        end_token_id=artifacts.end_token_id,
        max_trace_length=max_trace_length,
    )

    was_training = model.training
    model.eval()
    with torch.no_grad():
        original_padding_side = getattr(tokenizer, "padding_side", "right")
        try:
            tokenizer.padding_side = "left"
            encoded = tokenizer(
                prompt_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = original_padding_side

        input_tensor = encoded["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.long)
        begin_token_column = torch.full(
            (len(samples), 1),
            artifacts.begin_token_id,
            device=device,
            dtype=torch.long,
        )
        input_tensor = torch.cat([input_tensor, begin_token_column], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((len(samples), 1), device=device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        outputs = model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=True)
        batch_indices = torch.arange(len(samples), device=device)
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
                if next_token_id == artifacts.end_token_id:
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
                    torch.ones((len(samples), 1), device=device, dtype=attention_mask.dtype),
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
        TraceSample(
            sample_id=sample.sample_id,
            prompt=sample.prompt,
            answer=sample.answer,
            abstract_trace_text=_trace_text_from_ids(tokenizer, sample_generated_ids, artifacts),
            abstract_trace_ids=sample_generated_ids,
            stage="on_policy",
            round_idx=round_idx,
            cot=sample.cot,
        )
        for sample, sample_generated_ids in zip(samples, generated_ids, strict=True)
    ]


def _loss_scalar(loss) -> float:
    return float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)


def _debug_cuda_enabled() -> bool:
    return os.environ.get("ABSTRACT_COT_DEBUG_CUDA_MEM", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _cuda_memory_snapshot(device: str) -> dict[str, float | str]:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        return {}
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize(device)
    return {
        "device": device,
        "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2),
    }


def _log_cuda_mem(stage: str, *, device: str, phase_name: str, **extra) -> None:
    if not _debug_cuda_enabled():
        return
    payload = {
        "stage": stage,
        "phase": phase_name,
        "cuda_mem": _cuda_memory_snapshot(device),
    }
    payload.update(extra)
    print(payload)


class MemorySnapshotRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str,
        device: str,
        rank: int,
        max_entries: int,
    ) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.device = device
        self.rank = rank
        self.max_entries = max_entries

    def start(self, phase_name: str) -> None:
        if not self.enabled:
            return
        try:
            import torch
        except ImportError:  # pragma: no cover - optional dependency
            return
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return
        memory = getattr(torch.cuda, "memory", None)
        if memory is None or not hasattr(memory, "_record_memory_history"):
            print({"memory_snapshot": "unsupported", "phase": phase_name, "rank": self.rank})
            return
        torch.cuda.synchronize(self.device)
        memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=self.max_entries,
        )
        print({"memory_snapshot": "recording_started", "phase": phase_name, "rank": self.rank})

    def dump(self, phase_name: str, *, tag: str) -> str | None:
        if not self.enabled:
            return None
        try:
            import torch
        except ImportError:  # pragma: no cover - optional dependency
            return None
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return None
        memory = getattr(torch.cuda, "memory", None)
        if memory is None or not hasattr(memory, "_dump_snapshot"):
            return None
        snapshot_dir = os.path.join(self.output_dir, "memory_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        safe_phase_name = phase_name.replace("/", "_")
        filename = f"rank{self.rank}-{safe_phase_name}-{tag}.pickle"
        path = os.path.join(snapshot_dir, filename)
        torch.cuda.synchronize(self.device)
        memory._dump_snapshot(path)
        print({"memory_snapshot": "dumped", "phase": phase_name, "rank": self.rank, "path": path})
        return path

    def stop(self, phase_name: str) -> None:
        if not self.enabled:
            return
        try:
            import torch
        except ImportError:  # pragma: no cover - optional dependency
            return
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return
        memory = getattr(torch.cuda, "memory", None)
        if memory is None or not hasattr(memory, "_record_memory_history"):
            return
        torch.cuda.synchronize(self.device)
        memory._record_memory_history(enabled=None)
        print({"memory_snapshot": "recording_stopped", "phase": phase_name, "rank": self.rank})


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


def _build_bottleneck_batch_traces(
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
    return generate_on_policy_traces(
        model,
        tokenizer,
        samples,
        artifacts=tokenizer_artifacts,
        max_trace_length=runtime.max_trace_length,
        device=runtime.device,
        include_cot=True,
        round_idx=round_idx,
    )


def _build_distillation_batch_traces(
    samples: list[SupervisedSample],
    *,
    model,
    tokenizer,
    tokenizer_artifacts: TokenizerArtifacts,
    runtime: WarmupRuntimeConfig,
    round_idx: int,
) -> list[TraceSample]:
    return generate_on_policy_traces(
        model,
        tokenizer,
        samples,
        artifacts=tokenizer_artifacts,
        max_trace_length=runtime.max_trace_length,
        device=runtime.device,
        include_cot=False,
        round_idx=round_idx,
    )


def _collate_phase_features(features, pad_token_id: int) -> dict[str, Any]:
    return collate_distillation_features(features, pad_token_id)


def _count_supervised_tokens(batch: dict[str, Any]) -> int:
    return sum(
        1
        for labels in batch["labels"]
        for label in labels[1:]
        if label != -100
    )


def _build_bottleneck_step_batches(
    samples: list[SupervisedSample],
    traces: list[TraceSample],
    tokenizer,
    pad_token_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    z_features = []
    y_features = []
    for sample, trace in zip(samples, traces, strict=True):
        example = build_bottleneck_sft_example(sample, trace)
        z_features.append(build_bottleneck_z_feature(example, tokenizer))
        y_features.append(build_bottleneck_y_feature(example, tokenizer))
    return (
        _collate_phase_features(z_features, pad_token_id),
        _collate_phase_features(y_features, pad_token_id),
    )


def _build_distillation_step_batch(
    samples: list[SupervisedSample],
    traces: list[TraceSample],
    tokenizer,
    pad_token_id: int,
) -> dict[str, Any]:
    features = [
        build_distillation_feature(build_distillation_example(sample, trace), tokenizer)
        for sample, trace in zip(samples, traces, strict=True)
    ]
    return _collate_phase_features(features, pad_token_id)


def run_online_bottleneck_phase(
    trainer,
    optimizer,
    sample_loader,
    *,
    model,
    tokenizer,
    tokenizer_artifacts: TokenizerArtifacts,
    runtime: WarmupRuntimeConfig,
    distributed: DistributedContext,
    round_idx: int,
    phase_name: str,
    writer=None,
    global_step: int = 0,
    max_steps: int | None = None,
    memory_snapshot: MemorySnapshotRecorder | None = None,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for bottleneck training") from exc
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        tqdm = None

    pad_token_id = resolve_pad_token_id(tokenizer)
    losses: list[float] = []
    iterable = sample_loader
    if tqdm is not None and distributed.is_main_process:
        iterable = tqdm(sample_loader, desc=phase_name, leave=True)
    step = 0
    phase_completed = False
    if memory_snapshot is not None:
        memory_snapshot.start(phase_name)
    _log_cuda_mem("phase_start", device=distributed.device, phase_name=phase_name, step=step)
    try:
        for sample_batch in iterable:
            optimizer.zero_grad()
            try:
                _log_cuda_mem(
                    "before_trace_generation",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    batch_size=len(sample_batch),
                    prompt_chars=[len(sample.prompt) for sample in sample_batch],
                    cot_chars=[len(sample.cot or "") for sample in sample_batch],
                    answer_chars=[len(sample.answer) for sample in sample_batch],
                )
                traces = _build_bottleneck_batch_traces(
                    sample_batch,
                    model=model,
                    tokenizer=tokenizer,
                    tokenizer_artifacts=tokenizer_artifacts,
                    runtime=runtime,
                    round_idx=round_idx,
                )
                _log_cuda_mem(
                    "after_trace_generation",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    trace_token_lengths=[len(trace.abstract_trace_ids) for trace in traces],
                    trace_text_chars=[len(trace.abstract_trace_text) for trace in traces],
                )
                z_batch, y_batch = _build_bottleneck_step_batches(sample_batch, traces, tokenizer, pad_token_id)
                _log_cuda_mem(
                    "after_build_step_batches",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    z_seq_lens=[sum(row) for row in z_batch["attention_mask"]],
                    y_seq_lens=[sum(row) for row in y_batch["attention_mask"]],
                )
                z_tokens = _count_supervised_tokens(z_batch)
                y_tokens = _count_supervised_tokens(y_batch)
                total_tokens = z_tokens + y_tokens
                if total_tokens <= 0:
                    raise ValueError("bottleneck phase produced no supervised tokens")

                z_result = trainer.training_step(z_batch)
                _log_cuda_mem(
                    "after_z_forward_and_loss",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    z_supervised_tokens=z_tokens,
                )
                z_weight = z_tokens / total_tokens
                z_loss = z_result.loss * z_weight
                if hasattr(z_loss, "backward"):
                    _log_cuda_mem(
                        "before_z_backward",
                        device=distributed.device,
                        phase_name=phase_name,
                        step=step,
                        z_weight=z_weight,
                    )
                    z_loss.backward()
                    del z_result
                    _log_cuda_mem("after_z_backward", device=distributed.device, phase_name=phase_name, step=step)
                y_result = trainer.training_step(y_batch)
                _log_cuda_mem(
                    "after_y_forward_and_loss",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    y_supervised_tokens=y_tokens,
                )
                y_weight = y_tokens / total_tokens
                y_loss = y_result.loss * y_weight
                combined_loss = z_loss + y_loss
                _log_cuda_mem(
                    "after_combined_loss",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    total_supervised_tokens=total_tokens,
                )
                if hasattr(y_loss, "backward"):
                    _log_cuda_mem(
                        "before_y_backward",
                        device=distributed.device,
                        phase_name=phase_name,
                        step=step,
                        y_weight=y_weight,
                    )
                    y_loss.backward()
                    del y_result
                    _log_cuda_mem("after_y_backward", device=distributed.device, phase_name=phase_name, step=step)
                    optimizer.step()
                    _log_cuda_mem("after_optimizer_step", device=distributed.device, phase_name=phase_name, step=step)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(
                        {
                            "stage": "oom",
                            "phase": phase_name,
                            "device": distributed.device,
                            "batch_size": len(sample_batch),
                            "sample_prompt_chars": [len(sample.prompt) for sample in sample_batch],
                            "sample_cot_chars": [len(sample.cot or "") for sample in sample_batch],
                            "sample_answer_chars": [len(sample.answer) for sample in sample_batch],
                            "cuda_mem": _cuda_memory_snapshot(distributed.device),
                        }
                    )
                    if memory_snapshot is not None:
                        memory_snapshot.dump(phase_name, tag=f"oom-step{step}")
                raise

            loss_val = _loss_scalar(combined_loss)
            losses.append(loss_val)
            if writer is not None and distributed.is_main_process:
                writer.add_scalar(f"loss/{phase_name}", loss_val, global_step + step)
            step += 1
            _log_cuda_mem(
                "end_of_step",
                device=distributed.device,
                phase_name=phase_name,
                step=step,
                loss=loss_val,
            )
            if tqdm is not None and distributed.is_main_process and hasattr(iterable, "set_postfix"):
                iterable.set_postfix(loss=f"{losses[-1]:.4f}")
            if max_steps is not None and step >= max_steps:
                break
        phase_completed = True
    finally:
        if memory_snapshot is not None and not phase_completed:
            memory_snapshot.stop(phase_name)

    local_steps = len(losses)
    local_loss_sum = sum(losses)
    total_steps = int(reduce_scalar_sum(float(local_steps), distributed.device))
    total_loss_sum = reduce_scalar_sum(local_loss_sum, distributed.device)
    if writer is not None and distributed.is_main_process and losses:
        writer.add_scalar(f"loss/{phase_name}_mean", local_loss_sum / max(local_steps, 1), global_step)
        writer.flush()
    if memory_snapshot is not None:
        memory_snapshot.dump(phase_name, tag=f"steps{step}")
        memory_snapshot.stop(phase_name)
    _log_cuda_mem("phase_end", device=distributed.device, phase_name=phase_name, step=step)
    return {
        "per_rank_num_steps": local_steps,
        "global_num_steps": total_steps,
        "mean_loss": total_loss_sum / max(total_steps, 1),
        "last_loss": losses[-1] if losses else None,
    }, step


def run_online_distillation_phase(
    trainer,
    optimizer,
    sample_loader,
    *,
    model,
    tokenizer,
    tokenizer_artifacts: TokenizerArtifacts,
    runtime: WarmupRuntimeConfig,
    distributed: DistributedContext,
    round_idx: int,
    phase_name: str,
    writer=None,
    global_step: int = 0,
    max_steps: int | None = None,
    memory_snapshot: MemorySnapshotRecorder | None = None,
):
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        tqdm = None

    pad_token_id = resolve_pad_token_id(tokenizer)
    losses: list[float] = []
    iterable = sample_loader
    if tqdm is not None and distributed.is_main_process:
        iterable = tqdm(sample_loader, desc=phase_name, leave=True)
    step = 0
    phase_completed = False
    if memory_snapshot is not None:
        memory_snapshot.start(phase_name)
    _log_cuda_mem("phase_start", device=distributed.device, phase_name=phase_name, step=step)
    try:
        for sample_batch in iterable:
            optimizer.zero_grad()
            try:
                _log_cuda_mem(
                    "before_trace_generation",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    batch_size=len(sample_batch),
                    prompt_chars=[len(sample.prompt) for sample in sample_batch],
                    answer_chars=[len(sample.answer) for sample in sample_batch],
                )
                traces = _build_distillation_batch_traces(
                    sample_batch,
                    model=model,
                    tokenizer=tokenizer,
                    tokenizer_artifacts=tokenizer_artifacts,
                    runtime=runtime,
                    round_idx=round_idx,
                )
                _log_cuda_mem(
                    "after_trace_generation",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    trace_token_lengths=[len(trace.abstract_trace_ids) for trace in traces],
                    trace_text_chars=[len(trace.abstract_trace_text) for trace in traces],
                )
                train_batch = _build_distillation_step_batch(sample_batch, traces, tokenizer, pad_token_id)
                _log_cuda_mem(
                    "after_build_step_batch",
                    device=distributed.device,
                    phase_name=phase_name,
                    step=step,
                    seq_lens=[sum(row) for row in train_batch["attention_mask"]],
                    supervised_tokens=_count_supervised_tokens(train_batch),
                )
                result = trainer.training_step(train_batch)
                loss = result.loss
                _log_cuda_mem("after_forward_and_loss", device=distributed.device, phase_name=phase_name, step=step)
                if hasattr(loss, "backward"):
                    _log_cuda_mem("before_backward", device=distributed.device, phase_name=phase_name, step=step)
                    loss.backward()
                    del result
                    _log_cuda_mem("after_backward", device=distributed.device, phase_name=phase_name, step=step)
                    optimizer.step()
                    _log_cuda_mem("after_optimizer_step", device=distributed.device, phase_name=phase_name, step=step)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(
                        {
                            "stage": "oom",
                            "phase": phase_name,
                            "device": distributed.device,
                            "batch_size": len(sample_batch),
                            "sample_prompt_chars": [len(sample.prompt) for sample in sample_batch],
                            "sample_answer_chars": [len(sample.answer) for sample in sample_batch],
                            "cuda_mem": _cuda_memory_snapshot(distributed.device),
                        }
                    )
                    if memory_snapshot is not None:
                        memory_snapshot.dump(phase_name, tag=f"oom-step{step}")
                raise

            loss_val = _loss_scalar(loss)
            losses.append(loss_val)
            if writer is not None and distributed.is_main_process:
                writer.add_scalar(f"loss/{phase_name}", loss_val, global_step + step)
            step += 1
            _log_cuda_mem(
                "end_of_step",
                device=distributed.device,
                phase_name=phase_name,
                step=step,
                loss=loss_val,
            )
            if tqdm is not None and distributed.is_main_process and hasattr(iterable, "set_postfix"):
                iterable.set_postfix(loss=f"{losses[-1]:.4f}")
            if max_steps is not None and step >= max_steps:
                break
        phase_completed = True
    finally:
        if memory_snapshot is not None and not phase_completed:
            memory_snapshot.stop(phase_name)

    local_steps = len(losses)
    local_loss_sum = sum(losses)
    total_steps = int(reduce_scalar_sum(float(local_steps), distributed.device))
    total_loss_sum = reduce_scalar_sum(local_loss_sum, distributed.device)
    if writer is not None and distributed.is_main_process and losses:
        writer.add_scalar(f"loss/{phase_name}_mean", local_loss_sum / max(local_steps, 1), global_step)
        writer.flush()
    if memory_snapshot is not None:
        memory_snapshot.dump(phase_name, tag=f"steps{step}")
        memory_snapshot.stop(phase_name)
    _log_cuda_mem("phase_end", device=distributed.device, phase_name=phase_name, step=step)
    return {
        "per_rank_num_steps": local_steps,
        "global_num_steps": total_steps,
        "mean_loss": total_loss_sum / max(total_steps, 1),
        "last_loss": losses[-1] if losses else None,
    }, step


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


def _save_checkpoint(
    model,
    optimizer,
    *,
    output_dir: str,
    round_idx: int,
    phase_name: str,
    global_step: int,
    distributed: DistributedContext,
):
    if not distributed.is_main_process:
        return
    import os
    try:
        import torch
    except ImportError:
        return

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f"round{round_idx}-{phase_name}-step{global_step}.pt"

    stateful_model = getattr(model, "module", model)
    state_dict = stateful_model.state_dict() if hasattr(stateful_model, "state_dict") else None
    optim_state = optimizer.state_dict() if hasattr(optimizer, "state_dict") else None

    checkpoint = {
        "round_idx": round_idx,
        "phase_name": phase_name,
        "global_step": global_step,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optim_state,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print({"checkpoint_saved": filename, "round": round_idx, "phase": phase_name, "global_step": global_step})


def _unwrap_stateful_model(model):
    stateful_model = getattr(model, "module", model)
    return getattr(stateful_model, "inner_model", stateful_model)


def _normalize_export_state_dict_keys(state_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    if state_dict is None:
        return None
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("inner_model."):
            normalized_key = normalized_key[len("inner_model.") :]
        normalized[normalized_key] = value
    return normalized


def _gather_full_model_state_dict(model, *, distributed: DistributedContext):
    if distributed.enabled:
        try:
            from torch.distributed.fsdp import (
                FullStateDictConfig,
                FullyShardedDataParallel as FSDP,
                StateDictType,
            )
        except ImportError:  # pragma: no cover - optional dependency
            FSDP = None

        if FSDP is not None and isinstance(model, FSDP):
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            print(
                {
                    "save_stage": "before_fsdp_full_state_dict_gather",
                    "rank": distributed.rank,
                    "world_size": distributed.world_size,
                }
            )
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = model.state_dict()
            print(
                {
                    "save_stage": "after_fsdp_full_state_dict_gather",
                    "rank": distributed.rank,
                    "has_state_dict": bool(state_dict),
                }
            )
            return _normalize_export_state_dict_keys(state_dict) if distributed.is_main_process else None

    stateful_model = _unwrap_stateful_model(model)
    if not hasattr(stateful_model, "state_dict"):
        return None
    print({"save_stage": "before_state_dict", "rank": distributed.rank})
    state_dict = stateful_model.state_dict()
    print({"save_stage": "after_state_dict", "rank": distributed.rank, "has_state_dict": bool(state_dict)})
    return _normalize_export_state_dict_keys(state_dict)


def _save_pretrained_model(model, output_dir: str, *, distributed: DistributedContext) -> None:
    base_model = _unwrap_stateful_model(model)
    if not hasattr(base_model, "config"):
        return
    state_dict = _gather_full_model_state_dict(model, distributed=distributed)
    if not distributed.is_main_process:
        return
    print({"save_stage": "before_save_pretrained", "rank": distributed.rank, "output_dir": output_dir})
    if state_dict is None:
        if hasattr(base_model, "save_pretrained"):
            base_model.save_pretrained(output_dir)
        print({"save_stage": "after_save_pretrained", "rank": distributed.rank, "output_dir": output_dir})
        return

    from safetensors.torch import save_file
    from transformers.utils import SAFE_WEIGHTS_NAME

    os.makedirs(output_dir, exist_ok=True)
    tensor_state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if hasattr(value, "detach")
    }
    weights_path = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
    save_file(tensor_state_dict, weights_path, metadata={"format": "pt"})
    base_model.config.save_pretrained(output_dir)
    generation_config = getattr(base_model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "save_pretrained"):
        generation_config.save_pretrained(output_dir)
    print({"save_stage": "after_save_pretrained", "rank": distributed.rank, "output_dir": output_dir})


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
    causal_trainer = DistillationSFTTrainer(model, as_tensors=True, device=runtime.device)
    memory_snapshot = MemorySnapshotRecorder(
        enabled=runtime.memory_snapshot_enabled,
        output_dir=runtime.output_dir,
        device=runtime.device,
        rank=distributed.rank,
        max_entries=runtime.memory_snapshot_max_entries,
    )

    writer = None
    tensorboard_dir = os.path.join(runtime.output_dir, "tensorboard")
    if distributed.is_main_process:
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(tensorboard_dir)
        except ImportError:
            pass

    global_step = 0

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
        if distributed.is_main_process:
            print(
                {
                    "round": round_idx,
                    "D_t1_size": len(d_t1),
                    "D_t2_size": len(d_t2),
                    "per_rank_bottleneck_batches": len(bottleneck_source_loader),
                }
            )

        if runtime.device.startswith("cuda"):
            torch.cuda.empty_cache()
        model.train()
        bottleneck_metrics, bottleneck_steps = run_online_bottleneck_phase(
            causal_trainer,
            optimizer,
            bottleneck_source_loader,
            model=model,
            tokenizer=tokenizer,
            tokenizer_artifacts=tokenizer_artifacts,
            runtime=runtime,
            distributed=distributed,
            round_idx=round_idx,
            phase_name=f"round{round_idx}-bottleneck",
            writer=writer,
            global_step=global_step,
            max_steps=runtime.max_steps_per_phase,
            memory_snapshot=memory_snapshot,
        )
        global_step += bottleneck_steps
        _save_checkpoint(
            model,
            optimizer,
            output_dir=runtime.output_dir,
            round_idx=round_idx,
            phase_name="bottleneck",
            global_step=global_step,
            distributed=distributed,
        )

        if runtime.device.startswith("cuda"):
            torch.cuda.empty_cache()
        distill_source_loader = _build_phase_sample_loader(d_t2, runtime.batch_size)

        if distributed.is_main_process:
            print(
                {
                    "round": round_idx,
                    "per_rank_distill_batches": len(distill_source_loader),
                }
            )

        if runtime.device.startswith("cuda"):
            torch.cuda.empty_cache()
        model.train()
        distill_metrics, distill_steps = run_online_distillation_phase(
            causal_trainer,
            optimizer,
            distill_source_loader,
            model=model,
            tokenizer=tokenizer,
            tokenizer_artifacts=tokenizer_artifacts,
            runtime=runtime,
            distributed=distributed,
            round_idx=round_idx,
            phase_name=f"round{round_idx}-distill",
            writer=writer,
            global_step=global_step,
            max_steps=runtime.max_steps_per_phase,
            memory_snapshot=memory_snapshot,
        )
        global_step += distill_steps
        _save_checkpoint(
            model,
            optimizer,
            output_dir=runtime.output_dir,
            round_idx=round_idx,
            phase_name="distill",
            global_step=global_step,
            distributed=distributed,
        )

        if runtime.device.startswith("cuda"):
            torch.cuda.empty_cache()
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
            print({"save_stage": "before_tokenizer_save", "rank": distributed.rank, "output_dir": runtime.output_dir})
            tokenizer.save_pretrained(runtime.output_dir)
            print({"save_stage": "after_tokenizer_save", "rank": distributed.rank, "output_dir": runtime.output_dir})
    _save_pretrained_model(model, runtime.output_dir, distributed=distributed)
    if distributed.is_main_process and writer is not None:
        writer.close()
    return summary
