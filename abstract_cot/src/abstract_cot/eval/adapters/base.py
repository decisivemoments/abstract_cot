from __future__ import annotations

from pathlib import Path
from typing import Any

from abstract_cot.eval.schema import BenchmarkSpec, EvalSample


def load_dataset_rows(spec: BenchmarkSpec):
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required for evaluation") from exc

    source_path = Path(spec.dataset_path)
    if source_path.exists():
        if source_path.is_dir():
            try:
                dataset = load_from_disk(str(source_path))
                if isinstance(dataset, DatasetDict):
                    return dataset[spec.split]
                return dataset
            except Exception:
                pass

            jsonl_candidate = source_path / f"{spec.split}.jsonl"
            if jsonl_candidate.exists():
                return load_dataset("json", data_files=str(jsonl_candidate), split="train")

            json_candidate = source_path / f"{spec.split}.json"
            if json_candidate.exists():
                return load_dataset("json", data_files=str(json_candidate), split="train")

            parquet_candidate = source_path / f"{spec.split}.parquet"
            if parquet_candidate.exists():
                return load_dataset("parquet", data_files=str(parquet_candidate), split="train")

            return load_dataset(str(source_path), spec.subset, split=spec.split)
        if source_path.suffix in {".jsonl", ".json"}:
            dataset = load_dataset("json", data_files=str(source_path), split="train")
            return dataset
        if source_path.suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(source_path), split="train")
            return dataset

    if spec.subset:
        return load_dataset(spec.dataset_path, spec.subset, split=spec.split)
    return load_dataset(spec.dataset_path, split=spec.split)


def limit_samples(samples: list[EvalSample], max_samples: int | None) -> list[EvalSample]:
    if max_samples is None:
        return samples
    return samples[: max(0, max_samples)]


class BaseEvalAdapter:
    dataset_name: str

    def load_samples(self, spec: BenchmarkSpec) -> list[EvalSample]:
        raise NotImplementedError

    def extract_target_answer(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_prediction_answer(self, text: str) -> str | None:
        raise NotImplementedError
