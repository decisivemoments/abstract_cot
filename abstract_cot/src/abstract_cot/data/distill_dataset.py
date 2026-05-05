from __future__ import annotations

from abstract_cot.data.prompt_formatter import normalize_text_block
from abstract_cot.data.schema import DistillationExample, SupervisedSample, TraceSample


def build_distillation_example(sample: SupervisedSample, trace_sample: TraceSample) -> DistillationExample:
    if sample.sample_id != trace_sample.sample_id:
        raise ValueError("sample and trace_sample must share the same sample_id")

    abstract_trace = normalize_text_block(trace_sample.abstract_trace_text)
    return DistillationExample(
        sample_id=sample.sample_id,
        round_idx=trace_sample.round_idx,
        prompt=normalize_text_block(sample.prompt),
        abstract_trace=abstract_trace,
        abstract_tokens=abstract_trace.split(),
        answer=normalize_text_block(sample.answer),
    )
