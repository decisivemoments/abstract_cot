from __future__ import annotations

import argparse
import json
import random

from abstract_cot.data.collator import collate_bottleneck_features
from abstract_cot.data.schema import SupervisedSample
from abstract_cot.data.tokenized_features import build_bottleneck_feature
from abstract_cot.data.warmup_dataset import build_bottleneck_sft_example, initialize_random_trace
from abstract_cot.tokenization.abstract_vocab import build_abstract_vocabulary


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        if not text:
            return []
        ids: list[int] = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[token])
        return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tokenized bottleneck warm-up samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=8)
    args = parser.parse_args()

    sample = SupervisedSample(
        sample_id="demo-1",
        prompt="If Alice has 3 apples and buys 2 more, how many apples does she have?",
        cot="1. Start with 3 apples.\n2. Add 2 apples.\n3. The total is 5.",
        answer="Alice has 5 apples.",
    )
    tokenizer = WhitespaceTokenizer()
    trace = initialize_random_trace(
        sample=sample,
        abstract_tokens=build_abstract_vocabulary(args.vocab_size),
        rng=random.Random(args.seed),
        round_idx=1,
    )
    feature = build_bottleneck_feature(build_bottleneck_sft_example(sample, trace), tokenizer)
    batch = collate_bottleneck_features([feature], pad_token_id=tokenizer.pad_token_id)
    print(
        json.dumps(
            {
                "input_ids": batch["input_ids"][0],
                "labels": batch["labels"][0],
                "segment_ids": batch["segment_ids"][0],
                "attention_mask": batch["attention_mask"][0],
                "bottleneck_attention_mask_shape": [
                    len(batch["bottleneck_attention_mask"][0]),
                    len(batch["bottleneck_attention_mask"][0][0]),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
