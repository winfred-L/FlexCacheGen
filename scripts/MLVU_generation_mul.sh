#!/usr/bin/env bash
set -e

# load .env from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${SCRIPT_DIR}/../.env"
set +a

# ========== config ==========

TASK="8_sub_scene"    # 8_sub_scene / 9_summary
PRED_DIR="${OUTPUT_ROOT}/pred"
LIMIT=0  # 0 for no limit

DYNAMIC_SPARSE_THRESHOLD=""  # e.g. 0.3

# iterate over multiple static sparse thresholds
for STATIC_SPARSE_THRESHOLD in 0.1 0.2 0.3; do

  mkdir -p "${PRED_DIR}"

  python3 "./test/MLVU/generation.py" \
    --task "${TASK}" \
    --pred-dir "${PRED_DIR}" \
    --limit "${LIMIT}" \
    ${STATIC_SPARSE_THRESHOLD:+--static-sparse-threshold "${STATIC_SPARSE_THRESHOLD}"} \
    ${DYNAMIC_SPARSE_THRESHOLD:+--dynamic-sparse-threshold "${DYNAMIC_SPARSE_THRESHOLD}"}

done