from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SupervisedSample:
    sample_id: str
    prompt: str
    answer: str
    cot: str | None = None
    task_type: str = "generic"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceSample:
    sample_id: str
    prompt: str
    answer: str
    abstract_trace_text: str
    abstract_trace_ids: list[int]
    stage: str
    round_idx: int
    cot: str | None = None
    reward: float | None = None


@dataclass(frozen=True)
class SegmentText:
    prompt: str
    cot: str
    abstract_trace: str
    answer: str


@dataclass(frozen=True)
class BottleneckSFTExample:
    sample_id: str
    round_idx: int
    segments: SegmentText
    abstract_tokens: list[str]
    cot_steps: list[str]


@dataclass(frozen=True)
class DistillationExample:
    sample_id: str
    round_idx: int
    prompt: str
    abstract_trace: str
    abstract_tokens: list[str]
    answer: str


@dataclass(frozen=True)
class TokenizedFeature:
    sample_id: str
    round_idx: int
    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    segment_ids: list[int]
    attention_mask: list[int]


@dataclass(frozen=True)
class BottleneckTokenizedFeature(TokenizedFeature):
    pass


@dataclass(frozen=True)
class DistillationTokenizedFeature(TokenizedFeature):
    pass
