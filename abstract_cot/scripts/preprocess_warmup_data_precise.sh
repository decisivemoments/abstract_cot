#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${PROJECT_ROOT}/configs/runtime/server.yaml}"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/server_assets/datasets/Dolci-Think-SFT-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/server_assets/datasets/Dolci-Think-SFT-7B-cot-precise}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${PROJECT_ROOT}/server_assets/models/gpt2-xl}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1000}"
MAX_TRACE_LENGTH="${MAX_TRACE_LENGTH:-128}"
NUM_PROC="${NUM_PROC:-}"

read_runtime_value() {
  PYTHONPATH="${PROJECT_ROOT}/src" python "${PROJECT_ROOT}/scripts/read_runtime_config.py" \
    --config "${RUNTIME_CONFIG}" \
    --key "$1"
}

CONDA_ENV_NAME="${CONDA_ENV_NAME:-$(read_runtime_value server.conda_env_name)}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONPATH="${PROJECT_ROOT}/src"

if [[ -z "${NUM_PROC}" ]]; then
  NUM_PROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
fi
if [[ -z "${NUM_PROC}" ]]; then
  NUM_PROC="1"
fi

python "${PROJECT_ROOT}/scripts/preprocess_dolci_think_sft_precise.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --max-trace-length "${MAX_TRACE_LENGTH}" \
  --num-proc "${NUM_PROC}"
