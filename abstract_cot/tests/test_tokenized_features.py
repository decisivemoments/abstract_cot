from abstract_cot.data.collator import collate_bottleneck_features, collate_distillation_features
from abstract_cot.data.schema import BottleneckSFTExample, DistillationExample, SegmentText
from abstract_cot.data.tokenized_features import (
    IGNORE_INDEX,
    build_bottleneck_feature,
    build_distillation_feature,
)
from abstract_cot.modeling.attention_mask import SEGMENT_ABSTRACT, SEGMENT_ANSWER, SEGMENT_COT, SEGMENT_PROMPT


class FakeTokenizer:
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


def test_bottleneck_feature_masks_prompt_and_cot_labels():
    tokenizer = FakeTokenizer()
    example = BottleneckSFTExample(
        sample_id="s1",
        round_idx=1,
        segments=SegmentText(
            prompt="prompt text",
            cot="cot text",
            abstract_trace="<beginabstract> <TOKEN_A> <endabstract>",
            answer="final answer",
        ),
        abstract_tokens=["<beginabstract>", "<TOKEN_A>", "<endabstract>"],
        cot_steps=["cot text"],
    )

    feature = build_bottleneck_feature(example, tokenizer)
    assert len(feature.input_ids) == len(feature.labels) == len(feature.segment_ids)
    assert feature.segment_ids[:2] == [SEGMENT_PROMPT, SEGMENT_PROMPT]
    assert feature.segment_ids[2:4] == [SEGMENT_COT, SEGMENT_COT]
    assert all(label == IGNORE_INDEX for label in feature.labels[:4])
    assert feature.segment_ids[4:7] == [SEGMENT_ABSTRACT, SEGMENT_ABSTRACT, SEGMENT_ABSTRACT]
    assert feature.segment_ids[7:] == [SEGMENT_ANSWER, SEGMENT_ANSWER]
    assert feature.labels[4:] == feature.input_ids[4:]
    assert feature.bottleneck_attention_mask[-1][2] is False
    assert feature.bottleneck_attention_mask[-1][4] is True


def test_distillation_feature_only_masks_prompt_labels():
    tokenizer = FakeTokenizer()
    example = DistillationExample(
        sample_id="s2",
        round_idx=1,
        prompt="prompt text",
        abstract_trace="<beginabstract> <TOKEN_B> <endabstract>",
        abstract_tokens=["<beginabstract>", "<TOKEN_B>", "<endabstract>"],
        answer="final answer",
    )
    feature = build_distillation_feature(example, tokenizer)
    assert feature.segment_ids[:2] == [SEGMENT_PROMPT, SEGMENT_PROMPT]
    assert all(label == IGNORE_INDEX for label in feature.labels[:2])
    assert feature.segment_ids[2:5] == [SEGMENT_ABSTRACT, SEGMENT_ABSTRACT, SEGMENT_ABSTRACT]
    assert feature.segment_ids[5:] == [SEGMENT_ANSWER, SEGMENT_ANSWER]
    assert feature.labels[2:] == feature.input_ids[2:]


def test_collator_pads_features_and_masks():
    tokenizer = FakeTokenizer()
    feature_a = build_distillation_feature(
        DistillationExample(
            sample_id="a",
            round_idx=1,
            prompt="p",
            abstract_trace="<beginabstract> <TOKEN_A> <endabstract>",
            abstract_tokens=["<beginabstract>", "<TOKEN_A>", "<endabstract>"],
            answer="ans",
        ),
        tokenizer,
    )
    feature_b = build_distillation_feature(
        DistillationExample(
            sample_id="b",
            round_idx=2,
            prompt="longer prompt",
            abstract_trace="<beginabstract> <TOKEN_A> <TOKEN_B> <endabstract>",
            abstract_tokens=["<beginabstract>", "<TOKEN_A>", "<TOKEN_B>", "<endabstract>"],
            answer="final answer",
        ),
        tokenizer,
    )
    batch = collate_distillation_features([feature_a, feature_b], pad_token_id=tokenizer.pad_token_id)
    assert batch["sample_ids"] == ["a", "b"]
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert batch["attention_mask"][0][-1] == 0
    assert batch["labels"][0][-1] == IGNORE_INDEX


def test_bottleneck_collator_pads_square_attention_masks():
    tokenizer = FakeTokenizer()
    feature = build_bottleneck_feature(
        BottleneckSFTExample(
            sample_id="c",
            round_idx=1,
            segments=SegmentText(
                prompt="p",
                cot="c",
                abstract_trace="<beginabstract> <TOKEN_A> <endabstract>",
                answer="a",
            ),
            abstract_tokens=["<beginabstract>", "<TOKEN_A>", "<endabstract>"],
            cot_steps=["c"],
        ),
        tokenizer,
    )
    batch = collate_bottleneck_features([feature], pad_token_id=tokenizer.pad_token_id)
    mask = batch["bottleneck_attention_mask"][0]
    assert len(mask) == len(batch["input_ids"][0])
    assert len(mask[0]) == len(batch["input_ids"][0])
