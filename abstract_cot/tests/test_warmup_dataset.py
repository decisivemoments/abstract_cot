import random

from abstract_cot.data.distill_dataset import build_distillation_example
from abstract_cot.data.schema import SupervisedSample, TraceSample
from abstract_cot.data.warmup_dataset import (
    build_bottleneck_sft_example,
    initialize_random_trace,
    split_cot_into_steps,
)


def test_split_cot_into_steps_strips_common_prefixes():
    cot = """
    1. Compute the sum.
    2. Verify the result.
    Step 3: Return the answer.
    """
    assert split_cot_into_steps(cot) == [
        "Compute the sum.",
        "Verify the result.",
        "Return the answer.",
    ]


def test_initialize_random_trace_is_seedable():
    sample = SupervisedSample(
        sample_id="s1",
        prompt="What is 2 + 2?",
        cot="1. Add the numbers.\n2. Return four.",
        answer="4",
    )
    trace = initialize_random_trace(
        sample=sample,
        abstract_tokens=["<TOKEN_A>", "<TOKEN_B>", "<TOKEN_C>"],
        rng=random.Random(7),
        round_idx=1,
    )
    assert trace.stage == "warmup_init"
    assert trace.round_idx == 1
    assert trace.abstract_trace_text.startswith("<beginabstract>")
    assert trace.abstract_trace_text.endswith("<endabstract>")


def test_build_bottleneck_sft_example_keeps_all_segments():
    sample = SupervisedSample(
        sample_id="s2",
        prompt="Solve it",
        cot="1. reason\n2. finish",
        answer="done",
    )
    trace_sample = TraceSample(
        sample_id="s2",
        prompt="Solve it",
        answer="done",
        abstract_trace_text="<beginabstract> <TOKEN_A> <TOKEN_B> <endabstract>",
        abstract_trace_ids=[],
        stage="warmup_init",
        round_idx=2,
        cot=sample.cot,
    )
    example = build_bottleneck_sft_example(sample, trace_sample)
    assert example.segments.prompt == "Solve it"
    assert example.segments.cot == "1. reason\n2. finish"
    assert example.segments.abstract_trace == "<beginabstract> <TOKEN_A> <TOKEN_B> <endabstract>"
    assert example.segments.answer == "done"
    assert example.cot_steps == ["reason", "finish"]


def test_build_distillation_example_omits_cot_and_keeps_trace():
    sample = SupervisedSample(
        sample_id="s3",
        prompt="Prompt",
        cot="unused",
        answer="Answer",
    )
    trace_sample = TraceSample(
        sample_id="s3",
        prompt="Prompt",
        answer="Answer",
        abstract_trace_text="<beginabstract> <TOKEN_A> <endabstract>",
        abstract_trace_ids=[],
        stage="distill",
        round_idx=1,
    )
    example = build_distillation_example(sample, trace_sample)
    assert example.prompt == "Prompt"
    assert example.abstract_trace == "<beginabstract> <TOKEN_A> <endabstract>"
    assert example.abstract_tokens == ["<beginabstract>", "<TOKEN_A>", "<endabstract>"]
    assert example.answer == "Answer"
