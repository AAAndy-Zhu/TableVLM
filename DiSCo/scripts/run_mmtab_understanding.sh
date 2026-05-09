#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

MODEL_TYPE="${MODEL_TYPE:-qwen3vl}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

EVAL_FILE_PATH="${EVAL_FILE_PATH:-${ROOT_DIR}/test_data/MMTab-eval_understanding_test_data_6K.json}"
IMG_PATH="${IMG_PATH:-${ROOT_DIR}/data/images}"
ANSWERS_FILE="${ANSWERS_FILE:-${ROOT_DIR}/outputs/${MODEL_TYPE}/mmtab_understanding.jsonl}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Please set MODEL_PATH, for example:"
  echo "MODEL_PATH=/path/to/model MODEL_TYPE=qwen3vl bash $0"
  exit 1
fi

python "${SRC_DIR}/eval_mmtab_understanding.py" \
  --model_path "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --eval_file_path "${EVAL_FILE_PATH}" \
  --img_path "${IMG_PATH}" \
  --answers_file "${ANSWERS_FILE}"
