from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)


def _extract_messages_sample(row: dict, idx: int) -> dict[str, str] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None

    user_messages = [message for message in messages if message.get("role") == "user"]
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not user_messages or not assistant_messages:
        return None

    prompt = str(user_messages[-1].get("content", "")).strip()
    assistant_content = str(assistant_messages[-1].get("content", "")).strip()
    if not prompt or not assistant_content:
        return None

    match = _THINK_RE.match(assistant_content)
    if match is None:
        return None

    cot = match.group(1).strip()
    answer = match.group(2).strip()
    if not cot or not answer:
        return None

    return {
        "sample_id": str(row.get("id", f"sample-{idx}")),
        "prompt": prompt,
        "cot": cot,
        "answer": answer,
        "task_type": str(row.get("dataset_source", "generic")),
    }


def _project_messages_batch(batch: dict, indices: list[int]) -> dict[str, list[str]]:
    output = {
        "sample_id": [],
        "prompt": [],
        "cot": [],
        "answer": [],
        "task_type": [],
    }
    rows = [
        {key: values[position] for key, values in batch.items()}
        for position in range(len(indices))
    ]
    for idx, row in zip(indices, rows, strict=True):
        parsed = _extract_messages_sample(row, idx)
        if parsed is None:
            continue
        output["sample_id"].append(parsed["sample_id"])
        output["prompt"].append(parsed["prompt"])
        output["cot"].append(parsed["cot"])
        output["answer"].append(parsed["answer"])
        output["task_type"].append(parsed["task_type"])
    return output


def _write_dataset_disk(dataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Dolci-Think-SFT-7B into a flat -cot Hugging Face dataset.")
    parser.add_argument("--input-dir", required=True, help="Directory containing the original parquet dataset")
    parser.add_argument("--output-dir", required=True, help="Directory to write the flattened -cot Hugging Face dataset")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="CPU worker count for datasets filter/map",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required for preprocessing") from exc

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    num_proc = max(1, int(args.num_proc))
    if not input_dir.exists():
        raise FileNotFoundError(f"input dataset directory does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")

    dataset = load_dataset("parquet", data_dir=str(input_dir), split="train", num_proc=num_proc)
    dataset = dataset.map(
        _project_messages_batch,
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="extract_cot_examples",
    )
    _write_dataset_disk(dataset, output_dir)

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_proc": num_proc,
        "num_rows": len(dataset),
        "columns": list(dataset.column_names),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()
