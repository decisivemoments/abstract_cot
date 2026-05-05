from __future__ import annotations

from typing import Any

from abstract_cot.data.schema import BottleneckTokenizedFeature, DistillationTokenizedFeature
from abstract_cot.data.tokenized_features import IGNORE_INDEX


def _pad_1d(values: list[int], target_length: int, pad_value: int) -> list[int]:
    return values + [pad_value] * (target_length - len(values))


def _pad_2d_bool(mask: list[list[bool]], target_length: int) -> list[list[bool]]:
    padded_rows: list[list[bool]] = []
    for row in mask:
        padded_rows.append(row + [False] * (target_length - len(row)))
    for _ in range(target_length - len(mask)):
        padded_rows.append([False] * target_length)
    return padded_rows


def collate_distillation_features(
    features: list[DistillationTokenizedFeature],
    pad_token_id: int,
) -> dict[str, Any]:
    max_length = max(len(feature.input_ids) for feature in features)
    return {
        "sample_ids": [feature.sample_id for feature in features],
        "round_idxs": [feature.round_idx for feature in features],
        "input_ids": [_pad_1d(feature.input_ids, max_length, pad_token_id) for feature in features],
        "labels": [_pad_1d(feature.labels, max_length, IGNORE_INDEX) for feature in features],
        "position_ids": [_pad_1d(feature.position_ids, max_length, 0) for feature in features],
        "segment_ids": [_pad_1d(feature.segment_ids, max_length, -1) for feature in features],
        "attention_mask": [_pad_1d(feature.attention_mask, max_length, 0) for feature in features],
    }


def collate_bottleneck_features(
    features: list[BottleneckTokenizedFeature],
    pad_token_id: int,
) -> dict[str, Any]:
    batch = collate_distillation_features(features, pad_token_id)
    max_length = max(len(feature.input_ids) for feature in features)
    batch["bottleneck_attention_mask"] = [
        _pad_2d_bool(feature.bottleneck_attention_mask, max_length) for feature in features
    ]
    return batch
