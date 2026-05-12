from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from abstract_cot.eval.adapters import resolve_adapter
from abstract_cot.eval.answer_extraction import normalize_answer_text
from abstract_cot.eval.generation import GenerationConfig, generate_texts_in_batches, generate_traces_in_batches
from abstract_cot.eval.metrics import build_dataset_metrics, normalized_exact_match
from abstract_cot.eval.prompts import (
    render_abstract_answer_prompt,
    render_direct_answer_prompt,
    render_natural_cot_prompt,
)
from abstract_cot.eval.report import write_markdown_summary, write_metrics, write_predictions
from abstract_cot.eval.schema import BenchmarkSpec, DatasetMetrics, EvalPrediction, ModelEvalSpec
from abstract_cot.tokenization.tokenizer_extension import load_tokenizer_artifacts
from abstract_cot.utils.io import write_json


def _resolve_eval_device(model_spec: ModelEvalSpec) -> str:
    if model_spec.device:
        return model_spec.device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_tokenizer_artifacts_path(model_spec: ModelEvalSpec) -> str | None:
    search_roots = []
    tokenizer_path = model_spec.tokenizer_path or model_spec.model_path
    search_roots.append(Path(tokenizer_path))
    search_roots.append(Path(model_spec.model_path) / "tokenizer")
    for root in search_roots:
        candidate = root / "abstract_tokenizer_artifacts.json"
        if candidate.exists():
            return str(candidate)
    return None


def _build_prompts(mode: str, questions: list[str], traces: list[str] | None = None) -> list[str]:
    if mode == "direct-answer":
        return [render_direct_answer_prompt(question) for question in questions]
    if mode == "natural-cot":
        return [render_natural_cot_prompt(question) for question in questions]
    if mode == "abstract-cot":
        if traces is None:
            raise ValueError("abstract-cot mode requires traces")
        return [
            render_abstract_answer_prompt(question, trace)
            for question, trace in zip(questions, traces, strict=True)
        ]
    raise ValueError(f"unsupported eval mode: {mode}")


def evaluate_mode_on_benchmark(
    model,
    tokenizer,
    model_spec: ModelEvalSpec,
    benchmark_spec: BenchmarkSpec,
    *,
    mode: str,
) -> tuple[list[EvalPrediction], DatasetMetrics]:
    adapter = resolve_adapter(benchmark_spec.dataset_name)
    samples = adapter.load_samples(benchmark_spec)
    questions = [sample.prompt for sample in samples]
    generation_config = GenerationConfig(
        batch_size=benchmark_spec.batch_size,
        max_answer_new_tokens=benchmark_spec.max_answer_new_tokens,
        max_trace_length=benchmark_spec.max_trace_length,
        do_sample=False,
        temperature=1.0,
    )
    device = _resolve_eval_device(model_spec)
    traces: list[str] | None = None
    if mode == "abstract-cot":
        tokenizer_artifacts_path = model_spec.tokenizer_artifacts_path or _default_tokenizer_artifacts_path(model_spec)
        if tokenizer_artifacts_path is None:
            raise ValueError(f"abstract-cot mode requires tokenizer_artifacts_path for model run {model_spec.run_name}")
        tokenizer_artifacts = load_tokenizer_artifacts(tokenizer_artifacts_path)
        traces = generate_traces_in_batches(
            model,
            tokenizer,
            questions,
            tokenizer_artifacts=tokenizer_artifacts,
            config=generation_config,
            device=device,
            progress_desc=f"{model_spec.run_name}:{benchmark_spec.dataset_name}:{mode}:trace",
        )
    prompts = _build_prompts(mode, questions, traces)
    generated_answers = generate_texts_in_batches(
        model,
        tokenizer,
        prompts,
        config=generation_config,
        device=device,
        progress_desc=f"{model_spec.run_name}:{benchmark_spec.dataset_name}:{mode}:answer",
    )
    predictions: list[EvalPrediction] = []
    for idx, sample in enumerate(samples):
        generated_answer = generated_answers[idx]
        extracted_answer = adapter.extract_prediction_answer(generated_answer)
        normalized_target = normalize_answer_text(sample.target_answer) or sample.target_answer
        normalized_extracted = normalize_answer_text(extracted_answer)
        predictions.append(
            EvalPrediction(
                sample_id=sample.sample_id,
                dataset_name=sample.dataset_name,
                model_name=model_spec.run_name,
                mode=mode,
                prompt_text=prompts[idx],
                generated_trace_text=None if traces is None else traces[idx],
                generated_answer_text=generated_answer,
                extracted_answer=extracted_answer,
                normalized_target_answer=normalized_target,
                normalized_extracted_answer=normalized_extracted,
                is_correct=normalized_exact_match(extracted_answer, sample.target_answer),
                metadata=sample.metadata,
            )
        )
    metrics = build_dataset_metrics(adapter.dataset_name, model_spec.run_name, mode, predictions)
    return predictions, metrics


def run_eval_suite(
    model,
    tokenizer,
    model_spec: ModelEvalSpec,
    benchmark_specs: list[BenchmarkSpec],
    *,
    output_dir: str,
) -> dict[str, object]:
    all_predictions: list[EvalPrediction] = []
    all_metrics: list[DatasetMetrics] = []
    per_run_dir = Path(output_dir) / model_spec.run_name
    per_run_dir.mkdir(parents=True, exist_ok=True)

    for mode in model_spec.modes:
        for benchmark_spec in benchmark_specs:
            print(
                {
                    "eval_stage": "start",
                    "model_run": model_spec.run_name,
                    "mode": mode,
                    "dataset": benchmark_spec.dataset_name,
                },
                flush=True,
            )
            predictions, metrics = evaluate_mode_on_benchmark(
                model,
                tokenizer,
                model_spec,
                benchmark_spec,
                mode=mode,
            )
            all_predictions.extend(predictions)
            all_metrics.append(metrics)
            print(
                {
                    "eval_stage": "done",
                    "model_run": model_spec.run_name,
                    "mode": mode,
                    "dataset": benchmark_spec.dataset_name,
                    "accuracy": metrics.accuracy,
                    "num_samples": metrics.num_samples,
                },
                flush=True,
            )

    write_predictions(per_run_dir / "predictions.jsonl", all_predictions)
    write_metrics(per_run_dir / "metrics.json", all_metrics)
    write_markdown_summary(per_run_dir / "summary.md", all_metrics)
    summary = {
        "model_run": asdict(model_spec),
        "benchmarks": [asdict(spec) for spec in benchmark_specs],
        "metrics": [asdict(metric) for metric in all_metrics],
    }
    write_json(per_run_dir / "eval_config.json", summary)
    return summary
