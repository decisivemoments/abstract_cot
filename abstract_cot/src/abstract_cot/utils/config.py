from __future__ import annotations

from pathlib import Path


def _parse_scalar(raw: str):
    value = raw.strip()
    if value == "":
        return {}
    if value.isdigit():
        return int(value)
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    return value


def _load_simple_yaml(text: str):
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]

    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, sep, raw_value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"invalid config line: {line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        value = _parse_scalar(raw_value)
        current[key] = value

        if value == {}:
            child: dict[str, object] = {}
            current[key] = child
            stack.append((indent, child))

    return root


def load_config(path: str | Path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        _ = exc
        return _load_simple_yaml(text)
    return yaml.safe_load(text)
