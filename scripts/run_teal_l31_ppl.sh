#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
TEAL_PATH="${TEAL_PATH:-/mnt/data2/lbc/TEAL/runs/llama31_teal}"
SPARSITY="${SPARSITY:-0.5}"
DTYPE="${DTYPE:-bfloat16}"
ARC_SAVED_DIR="${ARC_SAVED_DIR:-}"
ARC_DATASET="${ARC_DATASET:-wikitext2}"
ARC_METRIC="${ARC_METRIC:-max}"
ARC_QUANT_TYPE="${ARC_QUANT_TYPE:-NVFP4}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DISABLE_PREFILL_PROTECTION="${DISABLE_PREFILL_PROTECTION:-0}"

ARGS=(
  --model_name "${MODEL_NAME}"
  --teal_path "${TEAL_PATH}"
  --sparsity "${SPARSITY}"
  --dtype "${DTYPE}"
  --attn_implementation "${ATTN_IMPL}"
  --dataset_name "wikitext"
  --subset "wikitext-2-raw-v1"
  --split "test"
  --size 0
)

if [[ -n "${ARC_SAVED_DIR}" ]]; then
  ARGS+=(
    --arc_saved_dir "${ARC_SAVED_DIR}"
    --arc_dataset "${ARC_DATASET}"
    --arc_metric "${ARC_METRIC}"
    --arc_quant_type "${ARC_QUANT_TYPE}"
  )
fi

if [[ "${DISABLE_PREFILL_PROTECTION}" == "1" ]]; then
  ARGS+=(--disable_prefill_protection)
fi

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 python -u /mnt/data2/lbc/TEAL/teal/ppl_test.py "${ARGS[@]}"
