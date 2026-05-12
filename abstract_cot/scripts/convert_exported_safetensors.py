from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def _load_safetensors(path: Path) -> tuple[dict[str, object], dict[str, str]]:
    tensors: dict[str, object] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


def _normalize_keys(state_dict: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("inner_model."):
            normalized_key = normalized_key[len("inner_model.") :]
        normalized[normalized_key] = value
    return normalized


def _copy_sidecar_files(input_dir: Path, output_dir: Path) -> None:
    for child in input_dir.iterdir():
        if child.name == "model.safetensors":
            continue
        target = output_dir / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert exported model.safetensors keys to bare HF names.")
    parser.add_argument("--input-dir", required=True, help="Directory containing model.safetensors")
    parser.add_argument("--output-dir", required=True, help="Directory to write converted export")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_weights = input_dir / "model.safetensors"
    if not input_weights.exists():
        raise FileNotFoundError(f"missing {input_weights}")

    tensors, metadata = _load_safetensors(input_weights)
    normalized = _normalize_keys(tensors)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_weights = output_dir / "model.safetensors"
    save_file(normalized, str(output_weights), metadata=metadata or {"format": "pt"})
    _copy_sidecar_files(input_dir, output_dir)
    print(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "num_tensors": len(normalized),
            "weights_path": str(output_weights),
        }
    )


if __name__ == "__main__":
    main()
