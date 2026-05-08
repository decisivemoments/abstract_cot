from __future__ import annotations


def resolve_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required to resolve torch_dtype") from exc
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_causal_lm(model_name_or_path: str, attn_implementation: str = "sdpa", **kwargs):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required to load the model") from exc
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_implementation,
        **kwargs,
    )


def load_tokenizer(tokenizer_name_or_path: str, **kwargs):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required to load the tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer
