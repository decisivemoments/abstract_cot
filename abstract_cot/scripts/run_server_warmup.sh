#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/experiment/mvp_warmup.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/experiments/mvp_warmup}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${PROJECT_ROOT}/configs/runtime/cisl113.yaml}"
USE_FSDP="${USE_FSDP:-false}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

read_runtime_value() {
  PYTHONPATH="${PROJECT_ROOT}/src" python "${PROJECT_ROOT}/scripts/read_runtime_config.py" \
    --config "${RUNTIME_CONFIG}" \
    --key "$1"
}

CONDA_ENV_NAME="${CONDA_ENV_NAME:-$(read_runtime_value server.conda_env_name)}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONPATH="${PROJECT_ROOT}/src"

if [[ "${USE_FSDP}" == "true" ]]; then
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${PROJECT_ROOT}/scripts/run_warmup_mvp.py" \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}"
else
  python "${PROJECT_ROOT}/scripts/run_warmup_mvp.py" \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}"
fi
