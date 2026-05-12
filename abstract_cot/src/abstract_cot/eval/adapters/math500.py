from __future__ import annotations

from typing import Any

from abstract_cot.eval.adapters.base import BaseEvalAdapter, limit_samples, load_dataset_rows
from abstract_cot.eval.answer_extraction import extract_boxed_answer, extract_generic_answer, normalize_answer_text
from abstract_cot.eval.schema import BenchmarkSpec, EvalSample


class Math500Adapter(BaseEvalAdapter):
    dataset_name = "math-500"

    def load_samples(self, spec: BenchmarkSpec) -> list[EvalSample]:
        dataset = load_dataset_rows(spec)
        samples = [
            EvalSample(
                sample_id=str(row.get("unique_id", row_idx)),
                dataset_name=self.dataset_name,
                prompt=str(row["problem"]).strip(),
                target_answer=self.extract_target_answer(row),
                metadata={
                    "subject": row.get("subject"),
                    "level": row.get("level"),
                },
            )
            for row_idx, row in enumerate(dataset)
        ]
        return limit_samples(samples, spec.max_samples)

    def extract_target_answer(self, row: dict[str, Any]) -> str:
        answer = normalize_answer_text(str(row["answer"]))
        if answer is None:
            raise ValueError("failed to normalize MATH-500 gold answer")
        return answer

    def extract_prediction_answer(self, text: str) -> str | None:
        return extract_boxed_answer(text) or extract_generic_answer(text)

