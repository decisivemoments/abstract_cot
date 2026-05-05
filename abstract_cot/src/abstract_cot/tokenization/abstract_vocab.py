from __future__ import annotations

from dataclasses import dataclass

BEGIN_ABSTRACT = "<beginabstract>"
END_ABSTRACT = "<endabstract>"


def _excel_style_label(index: int) -> str:
    if index < 0:
        raise ValueError("index must be non-negative")

    chars: list[str] = []
    value = index
    while True:
        value, rem = divmod(value, 26)
        chars.append(chr(ord("A") + rem))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(chars))


def build_abstract_vocabulary(size: int, scheme: str = "excel") -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if scheme != "excel":
        raise ValueError(f"unsupported scheme: {scheme}")
    return [f"<TOKEN_{_excel_style_label(i)}>" for i in range(size)]


@dataclass(frozen=True)
class AbstractTokenSpec:
    abstract_tokens: list[str]
    begin_token: str = BEGIN_ABSTRACT
    end_token: str = END_ABSTRACT

    @property
    def special_tokens(self) -> list[str]:
        return [self.begin_token, self.end_token]

    @property
    def all_added_tokens(self) -> list[str]:
        return self.special_tokens + self.abstract_tokens
