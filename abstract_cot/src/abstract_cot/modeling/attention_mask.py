from __future__ import annotations

from dataclasses import dataclass

SEGMENT_PROMPT = 0
SEGMENT_COT = 1
SEGMENT_ABSTRACT = 2
SEGMENT_ANSWER = 3


@dataclass(frozen=True)
class SegmentBoundaries:
    prompt_length: int
    cot_length: int
    abstract_length: int
    answer_length: int

    @property
    def total_length(self) -> int:
        return self.prompt_length + self.cot_length + self.abstract_length + self.answer_length


def build_segment_ids(boundaries: SegmentBoundaries) -> list[int]:
    return (
        [SEGMENT_PROMPT] * boundaries.prompt_length
        + [SEGMENT_COT] * boundaries.cot_length
        + [SEGMENT_ABSTRACT] * boundaries.abstract_length
        + [SEGMENT_ANSWER] * boundaries.answer_length
    )


def build_bottleneck_attention_mask(boundaries: SegmentBoundaries) -> list[list[bool]]:
    segment_ids = build_segment_ids(boundaries)
    total_length = boundaries.total_length
    mask: list[list[bool]] = [[False for _ in range(total_length)] for _ in range(total_length)]

    for target in range(total_length):
        target_segment = segment_ids[target]
        for source in range(target + 1):
            source_segment = segment_ids[source]
            allowed = False

            if target_segment == SEGMENT_PROMPT:
                allowed = source_segment == SEGMENT_PROMPT
            elif target_segment == SEGMENT_COT:
                allowed = source_segment in {SEGMENT_PROMPT, SEGMENT_COT}
            elif target_segment == SEGMENT_ABSTRACT:
                allowed = source_segment in {SEGMENT_PROMPT, SEGMENT_COT, SEGMENT_ABSTRACT}
            elif target_segment == SEGMENT_ANSWER:
                allowed = source_segment in {SEGMENT_PROMPT, SEGMENT_ABSTRACT, SEGMENT_ANSWER}

            mask[target][source] = allowed
    return mask
