from __future__ import annotations

import argparse
from pathlib import Path

from abstract_cot.utils.config import load_config


def _snapshot_download(*, repo_id: str, repo_type: str, local_dir: str, endpoint: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("huggingface_hub is required for mirrored downloads") from exc

    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        endpoint=endpoint,
        resume_download=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models and datasets on the server only.")
    parser.add_argument("--assets-config", required=True, help="Path to asset config YAML")
    parser.add_argument("--asset-group", required=True, choices=["models", "datasets"])
    parser.add_argument("--asset-name", required=True, help="Name under the selected asset group")
    parser.add_argument("--project-root", default=".", help="Project root for local asset directories")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    config = load_config(args.assets_config)
    group = config[args.asset_group]
    asset = group[args.asset_name]
    target_dir = Path(args.project_root) / "server_assets" / asset["local_subdir"]
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    download_path = _snapshot_download(
        repo_id=asset["repo_id"],
        repo_type=asset["repo_type"],
        local_dir=str(target_dir),
        endpoint=args.hf_endpoint,
    )
    print(download_path)


if __name__ == "__main__":
    main()
