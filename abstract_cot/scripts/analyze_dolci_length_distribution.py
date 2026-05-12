from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import math
import os

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)
_CHARS_PER_TOKEN_LOWER_BOUND = 2.5


def _extract_prompt_cot_answer(row: dict, row_idx: int) -> tuple[str, str, str] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None

    user_msgs = [message for message in messages if message.get("role") == "user"]
    asst_msgs = [message for message in messages if message.get("role") == "assistant"]
    if not user_msgs or not asst_msgs:
        return None

    prompt = str(user_msgs[-1].get("content", "")).strip()
    asst_content = str(asst_msgs[-1].get("content", "")).strip()
    if not prompt or not asst_content:
        return None

    match = _THINK_RE.match(asst_content)
    if match is None:
        return None

    cot = match.group(1).strip()
    answer = match.group(2).strip()
    if not cot or not answer:
        return None
    return prompt, cot, answer


def _estimate_lengths_batch(batch: dict, indices: list[int]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {
        "estimated_total_tokens": [],
        "prompt_chars": [],
        "cot_chars": [],
        "answer_chars": [],
        "total_chars": [],
    }

    n = len(indices)
    for pos in range(n):
        row = {key: values[pos] for key, values in batch.items()}
        extracted = _extract_prompt_cot_answer(row, indices[pos])
        if extracted is None:
            continue
        prompt, cot, answer = extracted
        prompt_chars = len(prompt)
        cot_chars = len(cot)
        answer_chars = len(answer)
        total_chars = prompt_chars + cot_chars + answer_chars
        estimated_tokens = int(math.ceil(total_chars / _CHARS_PER_TOKEN_LOWER_BOUND))

        output["estimated_total_tokens"].append(estimated_tokens)
        output["prompt_chars"].append(prompt_chars)
        output["cot_chars"].append(cot_chars)
        output["answer_chars"].append(answer_chars)
        output["total_chars"].append(total_chars)
    return output


def _percentile(sorted_values: list[int], ratio: float) -> int:
    if not sorted_values:
        return 0
    if ratio <= 0:
        return sorted_values[0]
    if ratio >= 1:
        return sorted_values[-1]
    idx = int((len(sorted_values) - 1) * ratio)
    return sorted_values[idx]


def _bucket_label(start: int, width: int) -> str:
    start_k = start / 1000
    end_k = (start + width) / 1000
    if width % 1000 == 0:
        return f"[{int(start_k)}k, {int(end_k)}k)"
    return f"[{start_k:.1f}k, {end_k:.1f}k)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Dolci-Think-SFT-7B prompt/cot/answer length distribution using char/token ratio estimate."
    )
    parser.add_argument("--input-dir", required=True, help="Raw Dolci-Think-SFT-7B dataset directory")
    parser.add_argument("--split", default="train")
    parser.add_argument("--bin-size", type=int, default=1000, help="Histogram bin size in estimated tokens")
    parser.add_argument("--output-json", default=None, help="Optional path to write summary JSON")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required for length analysis") from exc

    num_proc = max(1, os.cpu_count() or 1)

    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input dataset directory does not exist: {input_dir}")
    dataset = load_dataset("parquet", data_dir=str(input_dir), split=args.split)
    original_num_rows = len(dataset)
    dataset = dataset.map(
        _estimate_lengths_batch,
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="estimate_lengths",
    )

    columns = dataset[:]
    estimated_token_lengths = list(columns.get("estimated_total_tokens", []))
    prompt_char_lengths = list(columns.get("prompt_chars", []))
    cot_char_lengths = list(columns.get("cot_chars", []))
    answer_char_lengths = list(columns.get("answer_chars", []))
    total_char_lengths = list(columns.get("total_chars", []))
    invalid_rows = original_num_rows - len(estimated_token_lengths)

    sorted_total_lengths = sorted(estimated_token_lengths)
    histogram = Counter((length // args.bin_size) * args.bin_size for length in estimated_token_lengths)
    histogram_rows = [
        {
            "bucket_start": bucket_start,
            "bucket_end": bucket_start + args.bin_size,
            "bucket_label": _bucket_label(bucket_start, args.bin_size),
            "count": histogram[bucket_start],
            "ratio": (histogram[bucket_start] / len(estimated_token_lengths)) if estimated_token_lengths else 0.0,
        }
        for bucket_start in sorted(histogram)
    ]

    summary = {
        "input_dir": str(input_dir),
        "split": args.split,
        "bin_size": args.bin_size,
        "num_proc": num_proc,
        "chars_per_token_lower_bound": _CHARS_PER_TOKEN_LOWER_BOUND,
        "num_rows_raw": original_num_rows,
        "num_rows_valid": len(estimated_token_lengths),
        "num_rows_invalid": invalid_rows,
        "estimated_total_length_stats": {
            "min": min(estimated_token_lengths) if estimated_token_lengths else 0,
            "p50": _percentile(sorted_total_lengths, 0.50),
            "p90": _percentile(sorted_total_lengths, 0.90),
            "p95": _percentile(sorted_total_lengths, 0.95),
            "p99": _percentile(sorted_total_lengths, 0.99),
            "max": max(estimated_token_lengths) if estimated_token_lengths else 0,
            "mean": (sum(estimated_token_lengths) / len(estimated_token_lengths)) if estimated_token_lengths else 0.0,
        },
        "char_length_stats": {
            "prompt_mean": (sum(prompt_char_lengths) / len(prompt_char_lengths)) if prompt_char_lengths else 0.0,
            "cot_mean": (sum(cot_char_lengths) / len(cot_char_lengths)) if cot_char_lengths else 0.0,
            "answer_mean": (sum(answer_char_lengths) / len(answer_char_lengths)) if answer_char_lengths else 0.0,
            "total_mean": (sum(total_char_lengths) / len(total_char_lengths)) if total_char_lengths else 0.0,
        },
        "threshold_counts": {
            "le_4k": sum(1 for length in estimated_token_lengths if length <= 4096),
            "le_8k": sum(1 for length in estimated_token_lengths if length <= 8192),
            "gt_8k": sum(1 for length in estimated_token_lengths if length > 8192),
            "gt_12k": sum(1 for length in estimated_token_lengths if length > 12288),
            "gt_16k": sum(1 for length in estimated_token_lengths if length > 16384),
        },
        "histogram": histogram_rows,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
