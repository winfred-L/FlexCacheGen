#!/usr/bin/env bash
set -e

export API_KEY="sk-JZTZcaSrvIvSQUpdnVCUyVTOzCLBSSVyxF0wbkUfDzFaKVTz"

# ========== config ==========

TASK="8_sub_scene"    # 8_sub_scene / 9_summary
METRICS="all"     # gpt / bert / rouge / all / "gpt bert rouge"

PRED_DIR="/data1/lyc/flexcachegen_outputs/pred"
PRED_FILE_NAMES=(
  "qwen3vl-8b--MLVU_8_sub_scene--2026-05-05_14-22-58"
)
OUTPUT_DIR="/data1/lyc/flexcachegen_outputs/acc"

# ========== run ==========

mkdir -p "${OUTPUT_DIR}"

python "./test/MLVU/evaluation.py" \
  --task "${TASK}" \
  --metrics ${METRICS} \
  --pred-dir "${PRED_DIR}" \
  --pred-file-names "${PRED_FILE_NAMES[@]}" \
  --output-dir "${OUTPUT_DIR}"