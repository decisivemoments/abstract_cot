from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from abstract_cot.data.prompt_formatter import normalize_text_block
from abstract_cot.data.schema import (
    BottleneckSFTExample,
    BottleneckTokenizedFeature,
    DistillationExample,
    DistillationTokenizedFeature,
)
from abstract_cot.modeling.attention_mask import (
    SEGMENT_ABSTRACT,
    SEGMENT_ANSWER,
    SEGMENT_COT,
    SEGMENT_PROMPT,
)

IGNORE_INDEX = -100


class SupportsEncode(Protocol):
    pad_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class SegmentEncoding:
    input_ids: list[int]
    segment_id: int
    predict_labels: bool


def _encode_segment(
    tokenizer: SupportsEncode,
    text: str,
    segment_id: int,
    predict_labels: bool,
) -> SegmentEncoding:
    return SegmentEncoding(
        input_ids=tokenizer.encode(normalize_text_block(text), add_special_tokens=False),
        segment_id=segment_id,
        predict_labels=predict_labels,
    )


def _merge_segment_encodings(segments: list[SegmentEncoding]) -> tuple[list[int], list[int], list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    position_ids: list[int] = []
    segment_ids: list[int] = []

    for segment in segments:
        for token_id in segment.input_ids:
            input_ids.append(token_id)
            labels.append(token_id if segment.predict_labels else IGNORE_INDEX)
            position_ids.append(len(position_ids))
            segment_ids.append(segment.segment_id)
    attention_mask = [1] * len(input_ids)
    return input_ids, labels, position_ids, segment_ids, attention_mask


def build_bottleneck_feature(
    example: BottleneckSFTExample,
    tokenizer: SupportsEncode,
) -> BottleneckTokenizedFeature:
    prompt_ids = tokenizer.encode(example.segments.prompt, add_special_tokens=False)
    cot_ids = tokenizer.encode(example.segments.cot, add_special_tokens=False)
    abstract_ids = tokenizer.encode(example.segments.abstract_trace, add_special_tokens=False)
    answer_ids = tokenizer.encode(example.segments.answer, add_special_tokens=False)

    segments = [
        SegmentEncoding(prompt_ids, SEGMENT_PROMPT, False),
        SegmentEncoding(cot_ids, SEGMENT_COT, False),
        SegmentEncoding(abstract_ids, SEGMENT_ABSTRACT, True),
        SegmentEncoding(answer_ids, SEGMENT_ANSWER, True),
    ]
    input_ids, labels, position_ids, segment_ids, attention_mask = _merge_segment_encodings(segments)
    return BottleneckTokenizedFeature(
        sample_id=example.sample_id,
        round_idx=example.round_idx,
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        segment_ids=segment_ids,
        attention_mask=attention_mask,
    )


def build_distillation_feature(
    example: DistillationExample,
    tokenizer: SupportsEncode,
) -> DistillationTokenizedFeature:
    segments = [
        _encode_segment(tokenizer, example.prompt, SEGMENT_PROMPT, False),
        _encode_segment(tokenizer, example.abstract_trace, SEGMENT_ABSTRACT, True),
        _encode_segment(tokenizer, example.answer, SEGMENT_ANSWER, True),
    ]
    input_ids, labels, position_ids, segment_ids, attention_mask = _merge_segment_encodings(segments)
    return DistillationTokenizedFeature(
        sample_id=example.sample_id,
        round_idx=example.round_idx,
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        segment_ids=segment_ids,
        attention_mask=attention_mask,
    )


def resolve_pad_token_id(tokenizer: SupportsEncode) -> int:
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set for collation")
    return int(tokenizer.pad_token_id)
