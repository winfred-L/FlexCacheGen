#!/usr/bin/env bash
set -e

# load .env from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${SCRIPT_DIR}/../.env"
set +a

# ========== config ==========

TASK="8_sub_scene"    # 8_sub_scene / 9_summary
METRICS="all"     # gpt / bert / rouge / all / "bert rouge"

PRED_DIR="${OUTPUT_ROOT}/pred"
PRED_FILE_NAMES=(
  "qwen3vl-8b--MLVU_8_sub_scene--2026-05-05_14-22-58"
)
OUTPUT_DIR="${OUTPUT_ROOT}/acc"

# ========== run ==========

mkdir -p "${OUTPUT_DIR}"

python3 "./test/MLVU/evaluation.py" \
  --task "${TASK}" \
  --metrics ${METRICS} \
  --pred-dir "${PRED_DIR}" \
  --pred-file-names "${PRED_FILE_NAMES[@]}" \
  --output-dir "${OUTPUT_DIR}"