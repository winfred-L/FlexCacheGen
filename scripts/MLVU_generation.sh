#!/usr/bin/env bash
set -e

# ========== config ==========

TASK="8_sub_scene"    # 8_sub_scene / 9_summary
PRED_DIR="/data1/lyc/flexcachegen_outputs/pred"
LIMIT=3  # 0 for no limit

# sparse threshold options (empty = no sparsity)
STATIC_SPARSE_THRESHOLD=""   # e.g. "0.3"
DYNAMIC_SPARSE_THRESHOLD=""  # e.g. 0.3

# ========== run ==========

mkdir -p "${PRED_DIR}"

python3 "./test/MLVU/generation.py" \
  --task "${TASK}" \
  --pred-dir "${PRED_DIR}" \
  --limit "${LIMIT}" \
  ${STATIC_SPARSE_THRESHOLD:+--static-sparse-threshold "${STATIC_SPARSE_THRESHOLD}"} \
  ${DYNAMIC_SPARSE_THRESHOLD:+--dynamic-sparse-threshold "${DYNAMIC_SPARSE_THRESHOLD}"}