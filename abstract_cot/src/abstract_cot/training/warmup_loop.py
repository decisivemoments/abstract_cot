from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupPlan:
    rounds: int
    bottleneck_epochs: int
    distill_epochs: int


def build_warmup_schedule(plan: WarmupPlan) -> list[dict[str, int]]:
    if plan.rounds <= 0:
        raise ValueError("rounds must be positive")
    if plan.bottleneck_epochs <= 0 or plan.distill_epochs <= 0:
        raise ValueError("epoch counts must be positive")

    schedule: list[dict[str, int]] = []
    for round_idx in range(1, plan.rounds + 1):
        schedule.append(
            {
                "round": round_idx,
                "bottleneck_epochs": plan.bottleneck_epochs,
                "distill_epochs": plan.distill_epochs,
            }
        )
    return schedule
