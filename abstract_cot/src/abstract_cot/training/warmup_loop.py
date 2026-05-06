from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupPlan:
    rounds: int


def build_warmup_schedule(plan: WarmupPlan) -> list[dict[str, int]]:
    if plan.rounds <= 0:
        raise ValueError("rounds must be positive")

    schedule: list[dict[str, str | int]] = []
    for round_idx in range(1, plan.rounds + 1):
        schedule.append(
            {
                "round": round_idx,
                "stage_1": "bottleneck_sft_on_D_t1",
                "stage_2": "self_distillation_on_D_t2",
            }
        )
    return schedule
