from __future__ import annotations

import argparse

from abstract_cot.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a dotted key from a runtime YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    value = load_config(args.config)
    for part in args.key.split("."):
        value = value[part]
    print(value)


if __name__ == "__main__":
    main()
