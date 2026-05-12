from __future__ import annotations

from typing import Any

from abstract_cot.eval.adapters.base import BaseEvalAdapter, limit_samples, load_dataset_rows
from abstract_cot.eval.answer_extraction import extract_generic_answer, normalize_answer_text
from abstract_cot.eval.schema import BenchmarkSpec, EvalSample


class AimeAdapter(BaseEvalAdapter):
    dataset_name = "aime"

    def _extract_prompt(self, row: dict[str, Any]) -> str:
        for key in ("question", "problem", "prompt"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        raise KeyError(f"AIME row is missing a prompt-like field; available keys: {sorted(row.keys())}")

    def load_samples(self, spec: BenchmarkSpec) -> list[EvalSample]:
        dataset = load_dataset_rows(spec)
        samples = [
            EvalSample(
                sample_id=str(
                    row.get(
                        "unique_id",
                        row.get("id", row.get("metadata", {}).get("problem_idx", row_idx)),
                    )
                ),
                dataset_name=self.dataset_name,
                prompt=self._extract_prompt(row),
                target_answer=self.extract_target_answer(row),
                metadata={
                    "raw_metadata": row.get("metadata", {}),
                    "year": row.get("year", row.get("metadata", {}).get("year")),
                    "problem_idx": row.get("metadata", {}).get("problem_idx"),
                },
            )
            for row_idx, row in enumerate(dataset)
        ]
        return limit_samples(samples, spec.max_samples)

    def extract_target_answer(self, row: dict[str, Any]) -> str:
        answer = normalize_answer_text(str(row["answer"]))
        if answer is None:
            raise ValueError("failed to normalize AIME gold answer")
        return answer

    def extract_prediction_answer(self, text: str) -> str | None:
        return extract_generic_answer(text)
