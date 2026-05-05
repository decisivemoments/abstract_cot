from abstract_cot.data.collator import collate_bottleneck_features, collate_distillation_features
from abstract_cot.data.schema import BottleneckSFTExample, DistillationExample, SegmentText
from abstract_cot.data.tokenized_features import build_bottleneck_feature, build_distillation_feature
from abstract_cot.training.forward_batch import prepare_bottleneck_batch, prepare_distillation_batch
from abstract_cot.training.sft_trainer import BottleneckSFTTrainer, DistillationSFTTrainer


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        if not text:
            return []
        ids: list[int] = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[token])
        return ids


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        labels = kwargs["labels"]
        active = sum(1 for row in labels for token in row if token != -100)
        return {"loss": float(active)}


def test_prepare_bottleneck_batch_builds_4d_additive_mask():
    tokenizer = FakeTokenizer()
    feature = build_bottleneck_feature(
        BottleneckSFTExample(
            sample_id="s1",
            round_idx=1,
            segments=SegmentText(
                prompt="p",
                cot="c",
                abstract_trace="<beginabstract> <TOKEN_A> <endabstract>",
                answer="a",
            ),
            abstract_tokens=[],
            cot_steps=[],
        ),
        tokenizer,
    )
    batch = collate_bottleneck_features([feature], pad_token_id=tokenizer.pad_token_id)
    prepared = prepare_bottleneck_batch(batch)
    attention_mask = prepared.model_inputs["attention_mask"]
    assert len(attention_mask) == 1
    assert len(attention_mask[0]) == 1
    assert len(attention_mask[0][0]) == len(batch["input_ids"][0])
    assert len(attention_mask[0][0][0]) == len(batch["input_ids"][0])


def test_bottleneck_trainer_passes_prepared_batch_to_model():
    tokenizer = FakeTokenizer()
    feature = build_bottleneck_feature(
        BottleneckSFTExample(
            sample_id="s2",
            round_idx=1,
            segments=SegmentText(
                prompt="p1 p2",
                cot="c1",
                abstract_trace="<beginabstract> <TOKEN_A> <endabstract>",
                answer="a1",
            ),
            abstract_tokens=[],
            cot_steps=[],
        ),
        tokenizer,
    )
    batch = collate_bottleneck_features([feature], pad_token_id=tokenizer.pad_token_id)
    model = RecordingModel()
    result = BottleneckSFTTrainer(model).training_step(batch)
    assert result.loss > 0
    assert "attention_mask" in model.calls[0]
    assert "labels" in model.calls[0]


def test_distillation_trainer_uses_standard_attention_mask():
    tokenizer = FakeTokenizer()
    feature = build_distillation_feature(
        DistillationExample(
            sample_id="s3",
            round_idx=1,
            prompt="p",
            abstract_trace="<beginabstract> <TOKEN_B> <endabstract>",
            abstract_tokens=[],
            answer="a",
        ),
        tokenizer,
    )
    batch = collate_distillation_features([feature], pad_token_id=tokenizer.pad_token_id)
    prepared = prepare_distillation_batch(batch)
    assert prepared.model_inputs["attention_mask"] == batch["attention_mask"]
    model = RecordingModel()
    result = DistillationSFTTrainer(model).training_step(batch)
    assert result.loss > 0
