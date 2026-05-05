from __future__ import annotations

import random
import re
from dataclasses import dataclass

from abstract_cot.data.prompt_formatter import normalize_text_block, render_abstract_trace
from abstract_cot.data.schema import BottleneckSFTExample, SegmentText, SupervisedSample, TraceSample

_STEP_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)]|step\s+\d+[:.)]?)\s*", re.IGNORECASE)


def split_cot_into_steps(cot: str | None) -> list[str]:
    text = normalize_text_block(cot)
    if not text:
        return []

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(raw_lines) <= 1:
        return [text]

    steps: list[str] = []
    current: list[str] = []
    for line in raw_lines:
        starts_new_step = bool(_STEP_PREFIX_RE.match(line))
        if starts_new_step and current:
            steps.append(" ".join(current).strip())
            current = []
        current.append(_STEP_PREFIX_RE.sub("", line).strip())
    if current:
        steps.append(" ".join(current).strip())
    return [step for step in steps if step]


@dataclass(frozen=True)
class RandomTraceConfig:
    max_tokens_per_step_divisor: int = 2
    min_tokens_per_step: int = 1


def sample_random_abstract_tokens_for_steps(
    cot_steps: list[str],
    abstract_tokens: list[str],
    rng: random.Random,
    config: RandomTraceConfig | None = None,
) -> list[str]:
    if not abstract_tokens:
        raise ValueError("abstract_tokens must not be empty")

    trace_config = config or RandomTraceConfig()
    sampled_tokens: list[str] = []
    for step in cot_steps:
        step_token_estimate = max(len(step.split()), trace_config.min_tokens_per_step)
        upper = max(trace_config.min_tokens_per_step, step_token_estimate // trace_config.max_tokens_per_step_divisor)
        sample_count = rng.randint(trace_config.min_tokens_per_step, upper)
        sampled_tokens.extend(rng.choice(abstract_tokens) for _ in range(sample_count))
    return sampled_tokens


def initialize_random_trace(
    sample: SupervisedSample,
    abstract_tokens: list[str],
    rng: random.Random,
    round_idx: int,
    config: RandomTraceConfig | None = None,
) -> TraceSample:
    cot_steps = split_cot_into_steps(sample.cot)
    sampled_tokens = sample_random_abstract_tokens_for_steps(cot_steps, abstract_tokens, rng, config)
    abstract_trace_text = render_abstract_trace(sampled_tokens)
    return TraceSample(
        sample_id=sample.sample_id,
        prompt=sample.prompt,
        answer=sample.answer,
        abstract_trace_text=abstract_trace_text,
        abstract_trace_ids=[],
        stage="warmup_init",
        round_idx=round_idx,
        cot=sample.cot,
    )


def build_bottleneck_sft_example(sample: SupervisedSample, trace_sample: TraceSample) -> BottleneckSFTExample:
    if not sample.cot:
        raise ValueError("bottleneck SFT requires verbal CoT")
    if sample.sample_id != trace_sample.sample_id:
        raise ValueError("sample and trace_sample must share the same sample_id")

    cot_text = normalize_text_block(sample.cot)
    cot_steps = split_cot_into_steps(cot_text)
    return BottleneckSFTExample(
        sample_id=sample.sample_id,
        round_idx=trace_sample.round_idx,
        segments=SegmentText(
            prompt=normalize_text_block(sample.prompt),
            cot=cot_text,
            abstract_trace=normalize_text_block(trace_sample.abstract_trace_text),
            answer=normalize_text_block(sample.answer),
        ),
        abstract_tokens=normalize_text_block(trace_sample.abstract_trace_text).split(),
        cot_steps=cot_steps,
    )
