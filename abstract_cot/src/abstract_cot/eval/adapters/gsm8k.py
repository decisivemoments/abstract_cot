from __future__ import annotations

from typing import Any

from abstract_cot.eval.adapters.base import BaseEvalAdapter, limit_samples, load_dataset_rows
from abstract_cot.eval.answer_extraction import extract_generic_answer, extract_hash_answer
from abstract_cot.eval.schema import BenchmarkSpec, EvalSample


class Gsm8kAdapter(BaseEvalAdapter):
    dataset_name = "gsm8k"

    def load_samples(self, spec: BenchmarkSpec) -> list[EvalSample]:
        dataset = load_dataset_rows(spec)
        samples = [
            EvalSample(
                sample_id=str(row.get("id", row_idx)),
                dataset_name=self.dataset_name,
                prompt=str(row["question"]).strip(),
                target_answer=self.extract_target_answer(row),
                metadata={},
            )
            for row_idx, row in enumerate(dataset)
        ]
        return limit_samples(samples, spec.max_samples)

    def extract_target_answer(self, row: dict[str, Any]) -> str:
        answer = extract_hash_answer(str(row["answer"]))
        if answer is None:
            raise ValueError("failed to extract GSM8K target answer from gold solution")
        return answer

    def extract_prediction_answer(self, text: str) -> str | None:
        return extract_generic_answer(text)

