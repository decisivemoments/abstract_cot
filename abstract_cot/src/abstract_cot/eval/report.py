from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from abstract_cot.eval.schema import DatasetMetrics, EvalPrediction
from abstract_cot.utils.io import write_json


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_predictions(path: str | Path, predictions: list[EvalPrediction]) -> None:
    write_jsonl(path, [asdict(prediction) for prediction in predictions])


def write_metrics(path: str | Path, metrics: list[DatasetMetrics]) -> None:
    write_json(path, [asdict(metric) for metric in metrics])


def write_markdown_summary(path: str | Path, metrics: list[DatasetMetrics]) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        "| Model | Mode | Dataset | Accuracy | Valid Pred Rate | Samples |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric.model_name} | {metric.mode} | {metric.dataset_name} | "
            f"{metric.accuracy:.4f} | {metric.extraction_rate:.4f} | {metric.num_samples} |"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

