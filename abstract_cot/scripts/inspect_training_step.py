from __future__ import annotations

import json
import random

from abstract_cot.data.collator import collate_distillation_features
from abstract_cot.data.distill_dataset import build_distillation_example
from abstract_cot.data.schema import SupervisedSample
from abstract_cot.data.tokenized_features import (
    build_bottleneck_y_feature,
    build_bottleneck_z_feature,
    build_distillation_feature,
)
from abstract_cot.data.warmup_dataset import build_bottleneck_sft_example, initialize_random_trace
from abstract_cot.tokenization.abstract_vocab import build_abstract_vocabulary
from abstract_cot.training.sft_trainer import DistillationSFTTrainer


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


class FakeModel:
    def __call__(self, **kwargs):
        labels = kwargs["labels"]
        loss = sum(1 for row in labels for token in row if token != -100)
        return {"loss": float(loss), "received_keys": sorted(kwargs.keys())}


def main() -> None:
    sample = SupervisedSample(
        sample_id="demo-1",
        prompt="If Alice has 3 apples and buys 2 more, how many apples does she have?",
        cot="1. Start with 3 apples.\n2. Add 2 apples.\n3. The total is 5.",
        answer="Alice has 5 apples.",
    )
    tokenizer = WhitespaceTokenizer()
    trace = initialize_random_trace(
        sample=sample,
        abstract_tokens=build_abstract_vocabulary(4),
        rng=random.Random(7),
        round_idx=1,
    )

    bottleneck_example = build_bottleneck_sft_example(sample, trace)
    bottleneck_z_batch = collate_distillation_features(
        [build_bottleneck_z_feature(bottleneck_example, tokenizer)],
        pad_token_id=tokenizer.pad_token_id,
    )
    bottleneck_y_batch = collate_distillation_features(
        [build_bottleneck_y_feature(bottleneck_example, tokenizer)],
        pad_token_id=tokenizer.pad_token_id,
    )
    bottleneck_z_result = DistillationSFTTrainer(FakeModel()).training_step(bottleneck_z_batch)
    bottleneck_y_result = DistillationSFTTrainer(FakeModel()).training_step(bottleneck_y_batch)

    distill_feature = build_distillation_feature(build_distillation_example(sample, trace), tokenizer)
    distill_batch = collate_distillation_features([distill_feature], pad_token_id=tokenizer.pad_token_id)
    distill_result = DistillationSFTTrainer(FakeModel()).training_step(distill_batch)

    print(
        json.dumps(
            {
                "bottleneck_z_loss": bottleneck_z_result.loss,
                "bottleneck_y_loss": bottleneck_y_result.loss,
                "distill_loss": distill_result.loss,
                "bottleneck_z_received_keys": bottleneck_z_result.model_outputs["received_keys"],
                "bottleneck_y_received_keys": bottleneck_y_result.model_outputs["received_keys"],
                "distill_received_keys": distill_result.model_outputs["received_keys"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
