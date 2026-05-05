from abstract_cot.modeling.attention_mask import (
    SEGMENT_ABSTRACT,
    SEGMENT_ANSWER,
    SEGMENT_COT,
    SEGMENT_PROMPT,
    SegmentBoundaries,
    build_bottleneck_attention_mask,
    build_segment_ids,
)


def test_segment_ids_follow_expected_order():
    boundaries = SegmentBoundaries(prompt_length=2, cot_length=1, abstract_length=2, answer_length=1)
    assert build_segment_ids(boundaries) == [
        SEGMENT_PROMPT,
        SEGMENT_PROMPT,
        SEGMENT_COT,
        SEGMENT_ABSTRACT,
        SEGMENT_ABSTRACT,
        SEGMENT_ANSWER,
    ]


def test_answer_cannot_attend_to_cot_tokens():
    boundaries = SegmentBoundaries(prompt_length=1, cot_length=2, abstract_length=1, answer_length=1)
    mask = build_bottleneck_attention_mask(boundaries)
    answer_row = mask[-1]
    assert answer_row[0] is True
    assert answer_row[1] is False
    assert answer_row[2] is False
    assert answer_row[3] is True
    assert answer_row[4] is True


def test_abstract_tokens_can_attend_to_prompt_and_cot():
    boundaries = SegmentBoundaries(prompt_length=1, cot_length=1, abstract_length=2, answer_length=0)
    mask = build_bottleneck_attention_mask(boundaries)
    abstract_row = mask[2]
    assert abstract_row[0] is True
    assert abstract_row[1] is True
    assert abstract_row[2] is True
    assert abstract_row[3] is False
