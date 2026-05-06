#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ASSET_CONFIG="${ASSET_CONFIG:-${PROJECT_ROOT}/configs/data/reproduction_assets.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${PROJECT_ROOT}/configs/runtime/server.yaml}"

read_runtime_value() {
  PYTHONPATH="${PROJECT_ROOT}/src" python "${PROJECT_ROOT}/scripts/read_runtime_config.py" \
    --config "${RUNTIME_CONFIG}" \
    --key "$1"
}

CONDA_ENV_NAME="${CONDA_ENV_NAME:-$(read_runtime_value server.conda_env_name)}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONPATH="${PROJECT_ROOT}/src"

export HF_ENDPOINT
export HF_HOME="${HF_HOME:-$HOME/workspace/.cache/abstract_cot/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/workspace/.cache/abstract_cot/datasets}"

python "${PROJECT_ROOT}/scripts/download_assets.py" \
  --assets-config "${ASSET_CONFIG}" \
  --asset-group models \
  --asset-name qwen3_0p6b \
  --project-root "${PROJECT_ROOT}" \
  --hf-endpoint "${HF_ENDPOINT}"

python "${PROJECT_ROOT}/scripts/download_assets.py" \
  --assets-config "${ASSET_CONFIG}" \
  --asset-group datasets \
  --asset-name warmup_train \
  --project-root "${PROJECT_ROOT}" \
  --hf-endpoint "${HF_ENDPOINT}"

python "${PROJECT_ROOT}/scripts/download_assets.py" \
  --assets-config "${ASSET_CONFIG}" \
  --asset-group datasets \
  --asset-name rl_train \
  --project-root "${PROJECT_ROOT}" \
  --hf-endpoint "${HF_ENDPOINT}"

cat <<EOF
Downloaded baseline assets:
  model: qwen3_0p6b
  datasets: warmup_train, rl_train
EOF
