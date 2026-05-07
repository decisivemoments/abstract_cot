from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)

# 保守估计：每个 token 对应的最少字符数
# 英文 ~4, 中文 ~1.5，取 2.5 作为下界，宁可多留不误删
_CHARS_PER_TOKEN_LOWER_BOUND = 2.5


def _project_messages_batch(
    batch: dict,
    indices: list[int],
    max_chars: int,
) -> dict[str, list[str]]:
    """
    单次迭代完成：
      1. messages -> prompt / cot / answer 解析
      2. 字符数预过滤（替代 tokenize 过滤）
    """
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

        # ── 1. 解析 messages ──────────────────────────────────────────
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

        # ── 2. 字符数预过滤（O(1) 相对 tokenize）────────────────────
        total_chars = len(prompt) + len(cot) + len(answer)
        if total_chars > max_chars:
            continue

        output["sample_id"].append(str(row.get("id", f"sample-{idx}")))
        output["prompt"].append(prompt)
        output["cot"].append(cot)
        output["answer"].append(answer)
        output["task_type"].append(str(row.get("dataset_source", "generic")))

    return output


def _write_dataset_disk(dataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Dolci-Think-SFT-7B into a flat -cot Hugging Face dataset."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Token 数上限；内部转换为字符数上限做预过滤",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for preprocessing") from exc

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    num_proc = max(1, int(args.num_proc))
    max_seq_length = int(args.max_seq_length)

    if max_seq_length <= 0:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
    if not input_dir.exists():
        raise FileNotFoundError(f"input dataset directory does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")

    # token 数上限 → 字符数上限（保守下界，宁多留不误删）
    max_chars = int(max_seq_length * _CHARS_PER_TOKEN_LOWER_BOUND)

    dataset = load_dataset("parquet", data_dir=str(input_dir), split="train", num_proc=num_proc)
    original_num_rows = len(dataset)

    dataset = dataset.map(
        lambda batch, indices: _project_messages_batch(batch, indices, max_chars=max_chars),
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="extract_and_filter",
    )

    _write_dataset_disk(dataset, output_dir)

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_proc": num_proc,
        "max_seq_length": max_seq_length,
        "max_chars_threshold": max_chars,
        "chars_per_token_lower_bound": _CHARS_PER_TOKEN_LOWER_BOUND,
        "num_rows_raw": original_num_rows,
        "num_rows_filtered": len(dataset),
        "columns": list(dataset.column_names),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(metadata)


if __name__ == "__main__":
    main()
