from __future__ import annotations

from abstract_cot.tokenization.abstract_vocab import BEGIN_ABSTRACT, END_ABSTRACT


def normalize_text_block(text: str | None) -> str:
    if not text:
        return ""
    return text.strip()


def render_abstract_trace(
    abstract_tokens: list[str],
    begin_token: str = BEGIN_ABSTRACT,
    end_token: str = END_ABSTRACT,
) -> str:
    body = " ".join(token.strip() for token in abstract_tokens if token.strip())
    if body:
        return f"{begin_token} {body} {end_token}"
    return f"{begin_token} {end_token}"
