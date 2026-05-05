from __future__ import annotations

import argparse
import json
import random

from abstract_cot.data.schema import SupervisedSample
from abstract_cot.data.warmup_dataset import build_bottleneck_sft_example, initialize_random_trace
from abstract_cot.tokenization.abstract_vocab import build_abstract_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an example random warm-up sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=8)
    args = parser.parse_args()

    sample = SupervisedSample(
        sample_id="demo-1",
        prompt="If Alice has 3 apples and buys 2 more, how many apples does she have?",
        cot="1. Start with 3 apples.\n2. Add 2 apples.\n3. The total is 5.",
        answer="Alice has 5 apples.",
    )
    abstract_tokens = build_abstract_vocabulary(args.vocab_size)
    trace = initialize_random_trace(
        sample=sample,
        abstract_tokens=abstract_tokens,
        rng=random.Random(args.seed),
        round_idx=1,
    )
    example = build_bottleneck_sft_example(sample, trace)
    print(
        json.dumps(
            {
                "sample_id": example.sample_id,
                "round_idx": example.round_idx,
                "segments": {
                    "prompt": example.segments.prompt,
                    "cot": example.segments.cot,
                    "abstract_trace": example.segments.abstract_trace,
                    "answer": example.segments.answer,
                },
                "cot_steps": example.cot_steps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
