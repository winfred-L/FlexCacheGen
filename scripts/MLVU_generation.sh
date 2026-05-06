#!/usr/bin/env bash
set -e

# ========== config ==========

TASK="8_sub_scene"    # 8_sub_scene / 9_summary
PRED_DIR="/data1/lyc/flexcachegen_outputs/pred"
LIMIT=3  # 0 for no limit

# ========== run ==========

mkdir -p "${PRED_DIR}"

python "./test/MLVU/generation.py" \
  --task "${TASK}" \
  --pred-dir "${PRED_DIR}" \
  --limit "${LIMIT}"