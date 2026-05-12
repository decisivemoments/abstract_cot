from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)

# 和现有粗筛脚本保持一致，先用字符数做保守预估
_CHARS_PER_TOKEN_LOWER_BOUND = 2.5


def _project_messages_batch(
    batch: dict,
    indices: list[int],
    max_chars: int,
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {
        "sample_id": [],
        "prompt": [],
        "cot": [],
        "answer": [],
        "task_type": [],
    }

    n = len(indices)
    for pos in range(n):
        idx = indices[pos]
        row = {key: values[pos] for key, values in batch.items()}

        messages = row.get("messages")
        if not isinstance(messages, list):
            continue

        user_msgs = [m for m in messages if m.get("role") == "user"]
        asst_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not user_msgs or not asst_msgs:
            continue

        prompt = str(user_msgs[-1].get("content", "")).strip()
        asst_content = str(asst_msgs[-1].get("content", "")).strip()
        if not prompt or not asst_content:
            continue

        match = _THINK_RE.match(asst_content)
        if match is None:
            continue

        cot = match.group(1).strip()
        answer = match.group(2).strip()
        if not cot or not answer:
            continue

        total_chars = len(prompt) + len(cot) + len(answer)
        if total_chars > max_chars:
            continue

        output["sample_id"].append(str(row.get("id", f"sample-{idx}")))
        output["prompt"].append(prompt)
        output["cot"].append(cot)
        output["answer"].append(answer)
        output["task_type"].append(str(row.get("dataset_source", "generic")))

    return output


def _build_precise_length_batch_fn(tokenizer, abstract_budget_tokens: int):
    def _compute_precise_lengths(batch: dict) -> dict[str, list[int]]:
        prompt_token_lengths = [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in batch["prompt"]
        ]
        cot_token_lengths = [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in batch["cot"]
        ]
        answer_token_lengths = [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in batch["answer"]
        ]

        total_token_lengths = [
            prompt_len + cot_len + answer_len + abstract_budget_tokens
            for prompt_len, cot_len, answer_len in zip(
                prompt_token_lengths,
                cot_token_lengths,
                answer_token_lengths,
                strict=True,
            )
        ]
        return {
            "prompt_token_length": prompt_token_lengths,
            "cot_token_length": cot_token_lengths,
            "answer_token_length": answer_token_lengths,
            "total_projected_token_length": total_token_lengths,
        }

    return _compute_precise_lengths


def _write_dataset_disk(dataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Dolci-Think-SFT-7B with char prefilter + precise tokenizer length filtering."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1000,
        help="Final projected token budget after adding abstract token reservation.",
    )
    parser.add_argument(
        "--max-trace-length",
        type=int,
        default=128,
        help="Reserved abstract trace token count; final budget adds 2 + max_trace_length.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for preprocessing") from exc

    from abstract_cot.modeling.model_loader import load_tokenizer

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    num_proc = max(1, int(args.num_proc))
    max_seq_length = int(args.max_seq_length)
    max_trace_length = int(args.max_trace_length)

    if max_seq_length <= 0:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
    if max_trace_length < 0:
        raise ValueError(f"max_trace_length must be non-negative, got {max_trace_length}")
    if not input_dir.exists():
        raise FileNotFoundError(f"input dataset directory does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")

    abstract_budget_tokens = 2 + max_trace_length
    if abstract_budget_tokens >= max_seq_length:
        raise ValueError(
            f"abstract budget {abstract_budget_tokens} must be smaller than max_seq_length {max_seq_length}"
        )

    max_chars = int(math.floor((max_seq_length - abstract_budget_tokens) * _CHARS_PER_TOKEN_LOWER_BOUND))

    tokenizer = load_tokenizer(args.tokenizer_path, trust_remote_code=True)

    dataset = load_dataset("parquet", data_dir=str(input_dir), split="train", num_proc=num_proc)
    original_num_rows = len(dataset)

    extracted_dataset = dataset.map(
        lambda batch, indices: _project_messages_batch(batch, indices, max_chars=max_chars),
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="extract_and_char_prefilter",
    )

    precise_length_fn = _build_precise_length_batch_fn(tokenizer, abstract_budget_tokens)
    extracted_dataset = extracted_dataset.map(
        precise_length_fn,
        batched=True,
        num_proc=num_proc,
        desc="compute_precise_token_lengths",
    )

    filtered_dataset = extracted_dataset.filter(
        lambda row: int(row["total_projected_token_length"]) < max_seq_length,
        num_proc=num_proc,
        desc="precise_token_filter",
    )

    _write_dataset_disk(filtered_dataset, output_dir)

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tokenizer_path": args.tokenizer_path,
        "num_proc": num_proc,
        "max_seq_length": max_seq_length,
        "max_trace_length": max_trace_length,
        "abstract_budget_tokens": abstract_budget_tokens,
        "max_chars_threshold": max_chars,
        "chars_per_token_lower_bound": _CHARS_PER_TOKEN_LOWER_BOUND,
        "num_rows_raw": original_num_rows,
        "num_rows_after_char_prefilter": len(extracted_dataset),
        "num_rows_filtered": len(filtered_dataset),
        "columns": list(filtered_dataset.column_names),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(metadata)


if __name__ == "__main__":
    main()
