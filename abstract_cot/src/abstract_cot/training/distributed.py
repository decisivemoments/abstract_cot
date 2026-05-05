from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    try:
        import torch
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - optional dependency
        return DistributedContext(enabled=False)

    if not dist.is_available() or not dist.is_initialized():
        if torch.cuda.is_available():
            return DistributedContext(enabled=False, device="cuda:0")
        return DistributedContext(enabled=False, device="cpu")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def init_process_group_if_needed() -> DistributedContext:
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - optional dependency
        return setup_distributed()

    import os

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        backend = "nccl" if __import__("torch").cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return setup_distributed()


def cleanup_process_group() -> None:
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - optional dependency
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def shard_sequence_for_rank(sequence: list, rank: int, world_size: int) -> list:
    if world_size <= 1:
        return sequence
    return sequence[rank::world_size]


def reduce_scalar_sum(value: float, device: str) -> float:
    try:
        import torch
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - optional dependency
        return value
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())
