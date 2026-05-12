from __future__ import annotations

from abstract_cot.eval.answer_extraction import normalize_answer_text
from abstract_cot.eval.schema import DatasetMetrics, EvalPrediction


def normalized_exact_match(predicted: str | None, target: str | None) -> bool:
    normalized_predicted = normalize_answer_text(predicted)
    normalized_target = normalize_answer_text(target)
    if normalized_predicted is None or normalized_target is None:
        return False
    return normalized_predicted == normalized_target


def build_dataset_metrics(
    dataset_name: str,
    model_name: str,
    mode: str,
    predictions: list[EvalPrediction],
) -> DatasetMetrics:
    num_samples = len(predictions)
    num_correct = sum(1 for prediction in predictions if prediction.is_correct)
    num_valid_predictions = sum(1 for prediction in predictions if prediction.extracted_answer is not None)
    avg_generated_answer_chars = (
        sum(len(prediction.generated_answer_text) for prediction in predictions) / num_samples
        if num_samples
        else 0.0
    )
    avg_generated_trace_chars = (
        sum(len(prediction.generated_trace_text or "") for prediction in predictions) / num_samples
        if num_samples
        else 0.0
    )
    accuracy = num_correct / num_samples if num_samples else 0.0
    extraction_rate = num_valid_predictions / num_samples if num_samples else 0.0
    return DatasetMetrics(
        dataset_name=dataset_name,
        model_name=model_name,
        mode=mode,
        num_samples=num_samples,
        num_correct=num_correct,
        num_valid_predictions=num_valid_predictions,
        accuracy=accuracy,
        extraction_rate=extraction_rate,
        avg_generated_answer_chars=avg_generated_answer_chars,
        avg_generated_trace_chars=avg_generated_trace_chars,
    )

