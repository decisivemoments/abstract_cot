from __future__ import annotations

import argparse
from pathlib import Path

from abstract_cot.modeling.embedding_resize import resize_model_embeddings
from abstract_cot.modeling.model_loader import load_causal_lm, load_tokenizer, resolve_torch_dtype
from abstract_cot.tokenization.abstract_vocab import AbstractTokenSpec, build_abstract_vocabulary
from abstract_cot.tokenization.tokenizer_extension import extend_tokenizer
from abstract_cot.training.bottleneck_sft import WarmupRuntimeConfig, run_minimal_warmup
from abstract_cot.training.distributed import cleanup_process_group, init_process_group_if_needed
from abstract_cot.utils.config import load_config
from abstract_cot.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Abstract-CoT warm-up SFT smoke run.")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to outputs/experiments/<experiment_id>",
    )
    args = parser.parse_args()

    distributed = init_process_group_if_needed()
    try:
        config = load_config(args.config)
        experiment_id = config["experiment_id"]
        output_dir = Path(args.output_dir or Path(config["output"]["root_dir"]) / experiment_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        set_seed(int(config["seed"]))

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is required to run warm-up") from exc

        if distributed.is_main_process:
            print(
                {
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                    "device": distributed.device,
                    "distributed_enabled": distributed.enabled,
                    "world_size": distributed.world_size,
                }
            )

        model_cfg = config["model"]
        tokenizer = load_tokenizer(
            model_cfg["tokenizer_path"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        model = load_causal_lm(
            model_cfg["base_model"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            dtype=resolve_torch_dtype(model_cfg.get("torch_dtype")),
        )
        if bool(model_cfg.get("gradient_checkpointing", True)):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False

        abstract_cfg = config["abstract"]
        token_spec = AbstractTokenSpec(
            abstract_tokens=build_abstract_vocabulary(
                int(abstract_cfg["vocab_size"]),
                str(abstract_cfg.get("naming_scheme", "excel")),
            )
        )
        tokenizer_dir = output_dir / "tokenizer"
        artifacts = extend_tokenizer(tokenizer, token_spec, tokenizer_dir)
        resize_model_embeddings(model, len(tokenizer))

        if distributed.device.startswith("cuda"):
            model = model.to(distributed.device)

        warmup_cfg = config["warmup"]
        use_fsdp = bool(config.get("distributed", {}).get("use_fsdp", False) or distributed.enabled)
        summary = run_minimal_warmup(
            model=model,
            tokenizer=tokenizer,
            tokenizer_artifacts=artifacts,
            runtime=WarmupRuntimeConfig(
                dataset_path=str(warmup_cfg["dataset_path"]),
                output_dir=str(output_dir),
                batch_size=int(warmup_cfg["batch_size"]),
                max_samples=int(warmup_cfg["max_samples"]),
                max_trace_length=int(config["abstract"]["max_trace_length"]),
                learning_rate=float(warmup_cfg["learning_rate"]),
                rounds=int(warmup_cfg["rounds"]),
                seed=int(config["seed"]),
                device=distributed.device,
                use_fsdp=use_fsdp,
            ),
            distributed=distributed,
        )
        if distributed.is_main_process:
            print(summary)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
