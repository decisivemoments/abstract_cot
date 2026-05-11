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


def load_causal_lm(model_name_or_path: str, attn_implementation: str = "flash_attention_2", **kwargs):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required to load the model") from exc
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_implementation,
        **kwargs,
    )


def inspect_attention_backend(model, *, requested_attn_implementation: str) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "requested_attn_implementation": requested_attn_implementation,
    }

    try:
        from transformers.utils import is_flash_attn_2_available
        diagnostics["transformers_flash_attn_2_available"] = bool(is_flash_attn_2_available())
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        diagnostics["transformers_flash_attn_2_available"] = f"error: {exc}"

    try:
        import flash_attn
        diagnostics["flash_attn_importable"] = True
        diagnostics["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        diagnostics["flash_attn_importable"] = False
        diagnostics["flash_attn_import_error"] = str(exc)

    config = getattr(model, "config", None)
    diagnostics["model_config_attn_implementation"] = getattr(config, "_attn_implementation", None)
    diagnostics["model_config_attn_implementation_internal"] = getattr(config, "_attn_implementation_internal", None)

    model_core = getattr(model, "model", model)
    diagnostics["model_gradient_checkpointing"] = bool(getattr(model_core, "gradient_checkpointing", False))

    attention_modules: dict[str, str] = {}
    for module_name, module in model.named_modules():
        if module_name.endswith("self_attn"):
            attention_modules[module_name] = module.__class__.__name__
            if len(attention_modules) >= 3:
                break
    diagnostics["sample_attention_module_classes"] = attention_modules

    return diagnostics


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
