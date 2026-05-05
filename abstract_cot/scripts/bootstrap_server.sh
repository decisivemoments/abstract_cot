#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/workspace/.cache/abstract_cot}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${PROJECT_ROOT}/configs/runtime/cisl113.yaml}"

read_runtime_value() {
  local key="$1"
  PYTHONPATH="${PROJECT_ROOT}/src" python "${PROJECT_ROOT}/scripts/read_runtime_config.py" \
    --config "${RUNTIME_CONFIG}" \
    --key "${key}"
}

CONDA_ENV_NAME="${CONDA_ENV_NAME:-$(read_runtime_value server.conda_env_name)}"
PYTHON_VERSION="${PYTHON_VERSION:-$(read_runtime_value server.python_version)}"

mkdir -p "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/datasets" "${CACHE_ROOT}/modelscope"
mkdir -p "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/server_assets/models" "${PROJECT_ROOT}/server_assets/datasets"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${CONDA_ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install --upgrade uv
uv pip install -r "${PROJECT_ROOT}/requirements-dev.txt"

cat <<EOF
Server bootstrap complete.
Project root: ${PROJECT_ROOT}
Conda env: ${CONDA_ENV_NAME}
Cache root: ${CACHE_ROOT}
EOF
