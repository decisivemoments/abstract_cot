from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_RE = re.compile(r"####\s*(.+)")
_NUMBERISH_RE = re.compile(
    r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:,\d{3})*(?:\.\d+)?)?(?:\s*,\s*-?\d+(?:,\d{3})*(?:\.\d+)?)*"
)


def strip_boxed(text: str) -> str:
    value = text.strip()
    while True:
        match = _BOXED_RE.search(value)
        if match is None:
            return value
        value = match.group(1).strip()


def normalize_answer_text(text: str | None) -> str | None:
    if text is None:
        return None
    value = strip_boxed(text.strip())
    value = value.replace("$", "")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .,\n\t")
    if not value:
        return None
    return value


def extract_hash_answer(text: str) -> str | None:
    matches = _HASH_RE.findall(text)
    if not matches:
        return None
    return normalize_answer_text(matches[-1])


def extract_boxed_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return normalize_answer_text(matches[-1])


def extract_last_numberish_answer(text: str) -> str | None:
    matches = _NUMBERISH_RE.findall(text)
    if not matches:
        return None
    return normalize_answer_text(matches[-1])


def extract_generic_answer(text: str) -> str | None:
    for extractor in (extract_hash_answer, extract_boxed_answer, extract_last_numberish_answer):
        answer = extractor(text)
        if answer:
            return answer
    normalized = normalize_answer_text(text)
    return normalized

