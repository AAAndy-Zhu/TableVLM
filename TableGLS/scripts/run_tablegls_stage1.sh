#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

MODEL_TYPE="${MODEL_TYPE:-qwen3vl}"
MODEL_PATH="${MODEL_PATH:-}"
TEMPERATURE="${TEMPERATURE:-0}"

EVAL_FILE_PATH="${EVAL_FILE_PATH:-${ROOT_DIR}/test_data/MMTab-eval_test_data_22K.json}"
IMG_PATH="${IMG_PATH:-${ROOT_DIR}/data/images}"
ANSWERS_FILE="${ANSWERS_FILE:-${ROOT_DIR}/outputs/${MODEL_TYPE}/tablegls_stage1.jsonl}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Please set MODEL_PATH, for example:"
  echo "MODEL_PATH=/path/to/model MODEL_TYPE=qwen3vl bash $0"
  exit 1
fi

python "${SRC_DIR}/eval_mmtab_vllm_tablegls_stage1.py" \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --temperature "${TEMPERATURE}" \
  --eval_file_path "${EVAL_FILE_PATH}" \
  --img_path "${IMG_PATH}" \
  --answers_file "${ANSWERS_FILE}"
