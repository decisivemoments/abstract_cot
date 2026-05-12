from __future__ import annotations

import argparse
from pathlib import Path

from abstract_cot.eval.runner import run_eval_suite
from abstract_cot.eval.schema import BenchmarkSpec, ModelEvalSpec
from abstract_cot.modeling.model_loader import load_causal_lm, load_tokenizer, resolve_torch_dtype
from abstract_cot.utils.config import load_config
from abstract_cot.utils.io import write_json
from abstract_cot.utils.seed import set_seed


def _parse_model_specs(config: dict) -> list[ModelEvalSpec]:
    runs = config["eval"]["model_runs"]
    specs: list[ModelEvalSpec] = []
    for row in runs:
        specs.append(
            ModelEvalSpec(
                run_name=str(row["run_name"]),
                model_path=str(row["model_path"]),
                tokenizer_path=str(row.get("tokenizer_path") or row["model_path"]),
                tokenizer_artifacts_path=(
                    str(row["tokenizer_artifacts_path"])
                    if row.get("tokenizer_artifacts_path") is not None
                    else None
                ),
                trust_remote_code=bool(row.get("trust_remote_code", False)),
                torch_dtype=str(row["torch_dtype"]) if row.get("torch_dtype") is not None else None,
                attn_implementation=str(row.get("attn_implementation", "flash_attention_2")),
                modes=tuple(str(mode) for mode in row.get("modes", ["direct-answer"])),
                device=str(row["device"]) if row.get("device") is not None else None,
            )
        )
    return specs


def _parse_benchmark_specs(config: dict) -> list[BenchmarkSpec]:
    rows = config["eval"]["benchmarks"]
    specs: list[BenchmarkSpec] = []
    for row in rows:
        specs.append(
            BenchmarkSpec(
                dataset_name=str(row["dataset_name"]),
                dataset_path=str(row["dataset_path"]),
                split=str(row.get("split", "test")),
                subset=str(row["subset"]) if row.get("subset") is not None else None,
                max_samples=int(row["max_samples"]) if row.get("max_samples") is not None else None,
                batch_size=int(row.get("batch_size", 8)),
                max_answer_new_tokens=int(row.get("max_answer_new_tokens", 256)),
                max_trace_length=int(row.get("max_trace_length", config.get("abstract", {}).get("max_trace_length", 128))),
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and warmup-only evaluation.")
    parser.add_argument("--config", required=True, help="Eval YAML config")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to outputs/evals/<experiment_id>",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_id = str(config["experiment_id"])
    output_root = Path(args.output_dir or Path(config["output"]["root_dir"]) / experiment_id)
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.get("seed", 42)))

    model_specs = _parse_model_specs(config)
    benchmark_specs = _parse_benchmark_specs(config)
    run_summaries: list[dict[str, object]] = []

    for model_spec in model_specs:
        tokenizer = load_tokenizer(
            model_spec.tokenizer_path or model_spec.model_path,
            trust_remote_code=model_spec.trust_remote_code,
        )
        model = load_causal_lm(
            model_spec.model_path,
            trust_remote_code=model_spec.trust_remote_code,
            torch_dtype=resolve_torch_dtype(model_spec.torch_dtype),
            attn_implementation=model_spec.attn_implementation,
        )
        device = model_spec.device
        if device is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("torch is required for evaluation") from exc
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            model = model.to(device)
        model.eval()
        summary = run_eval_suite(
            model,
            tokenizer,
            model_spec,
            benchmark_specs,
            output_dir=str(output_root),
        )
        run_summaries.append(summary)
        del model
        del tokenizer
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(output_root / "all_runs_summary.json", run_summaries)
    print({"eval_output_dir": str(output_root), "num_model_runs": len(run_summaries)})


if __name__ == "__main__":
    main()
