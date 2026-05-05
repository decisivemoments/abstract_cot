from __future__ import annotations

import argparse
import json

from abstract_cot.training.warmup_loop import WarmupPlan, build_warmup_schedule
from abstract_cot.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the warm-up schedule for an experiment config.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    plan = WarmupPlan(
        rounds=int(config["warmup"]["rounds"]),
        bottleneck_epochs=int(config["warmup"]["bottleneck_epochs"]),
        distill_epochs=int(config["warmup"]["distill_epochs"]),
    )
    print(json.dumps(build_warmup_schedule(plan), indent=2))


if __name__ == "__main__":
    main()
