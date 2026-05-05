from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .abstract_vocab import AbstractTokenSpec


class SupportsTokenizer(Protocol):
    def add_special_tokens(self, special_tokens_dict: dict[str, list[str] | str]) -> int: ...
    def add_tokens(self, new_tokens: list[str]) -> int: ...
    def convert_tokens_to_ids(self, tokens: list[str] | str) -> list[int] | int: ...
    def save_pretrained(self, save_directory: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class TokenizerArtifacts:
    tokenizer_dir: str
    begin_token: str
    end_token: str
    begin_token_id: int
    end_token_id: int
    abstract_tokens: list[str]
    abstract_token_ids: list[int]


def extend_tokenizer(
    tokenizer: SupportsTokenizer,
    token_spec: AbstractTokenSpec,
    output_dir: str | Path,
) -> TokenizerArtifacts:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [token_spec.begin_token, token_spec.end_token]}
    )
    tokenizer.add_tokens(token_spec.abstract_tokens)

    begin_token_id = int(tokenizer.convert_tokens_to_ids(token_spec.begin_token))
    end_token_id = int(tokenizer.convert_tokens_to_ids(token_spec.end_token))
    abstract_token_ids = [int(token_id) for token_id in tokenizer.convert_tokens_to_ids(token_spec.abstract_tokens)]

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(save_dir))

    artifacts = TokenizerArtifacts(
        tokenizer_dir=str(save_dir),
        begin_token=token_spec.begin_token,
        end_token=token_spec.end_token,
        begin_token_id=begin_token_id,
        end_token_id=end_token_id,
        abstract_tokens=token_spec.abstract_tokens,
        abstract_token_ids=abstract_token_ids,
    )
    (save_dir / "abstract_tokenizer_artifacts.json").write_text(
        json.dumps(asdict(artifacts), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return artifacts


def load_tokenizer_artifacts(path: str | Path) -> TokenizerArtifacts:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return TokenizerArtifacts(**data)
