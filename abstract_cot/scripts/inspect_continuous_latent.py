from __future__ import annotations

import argparse

from abstract_cot.continuous.recurrent_latent import ContinuousLatentConfig, generate_continuous_latent_trace
from abstract_cot.modeling.model_loader import load_causal_lm, load_tokenizer, resolve_torch_dtype
from abstract_cot.tokenization.abstract_vocab import END_ABSTRACT
from abstract_cot.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect experimental continuous latent reasoning.")
    parser.add_argument("--config", required=True, help="Path to continuous experiment config")
    parser.add_argument("--prompt", required=True, help="Prompt used for the smoke trace")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for continuous latent inspection") from exc

    config = load_config(args.config)
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

    if tokenizer.convert_tokens_to_ids(END_ABSTRACT) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [END_ABSTRACT]})
        model.resize_token_embeddings(len(tokenizer))
    end_token_id = int(tokenizer.convert_tokens_to_ids(END_ABSTRACT))

    encoded = tokenizer(args.prompt, return_tensors="pt")
    trace = generate_continuous_latent_trace(
        model,
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        config=ContinuousLatentConfig(
            end_token_id=end_token_id,
            max_reasoning_steps=int(config["continuous"]["max_reasoning_steps"]),
            temperature=float(config["continuous"].get("temperature", 1.0)),
            do_sample=bool(config["continuous"].get("do_sample", False)),
            detach_between_steps=bool(config["continuous"].get("detach_between_steps", False)),
        ),
    )
    print(
        {
            "num_steps": trace.num_steps,
            "latent_shape": list(trace.latent_states.shape),
            "predicted_token_ids": trace.predicted_token_ids.tolist(),
            "stopped": trace.stopped.tolist(),
        }
    )


if __name__ == "__main__":
    main()
