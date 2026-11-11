#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/code/ecommerce-retrieval/data/train/all_data_package}"
SPLIT_DIR="${2:-artifacts/train_v0/split_quarantine_duplicates}"
ITEM_INDEX="${3:-artifacts/train_v0/train_item_index}"
DUPLICATE_GROUPS="${4:-artifacts/train_v0/proposed_item_duplicate_groups.jsonl}"
OUTPUT_DIR="${5:-artifacts/train_v0/hard_negative_candidates}"
LABELS="${ECOM_CLEAN_LABELS:-artifacts/train_v0/clean_labels.jsonl}"
GPU_COUNT="${GPU_COUNT:-8}"

mkdir -p "${OUTPUT_DIR}/logs"

pids=()
for ((worker = 0; worker < GPU_COUNT; worker++)); do
  CUDA_VISIBLE_DEVICES="${worker}" python -u scripts/mine_hard_negatives.py \
    --data-root "${DATA_ROOT}" \
    --labels "${LABELS}" \
    --photo-ids "${SPLIT_DIR}/train_photo_ids.txt" \
    --item-index "${ITEM_INDEX}" \
    --duplicate-groups "${DUPLICATE_GROUPS}" \
    --output-dir "${OUTPUT_DIR}" \
    --partitions "${GPU_COUNT}" \
    --partition-index "${worker}" \
    > "${OUTPUT_DIR}/logs/worker_${worker}.log" 2>&1 &
  worker_pid="$!"
  pids+=("${worker_pid}")
  echo "worker=${worker} pid=${worker_pid} log=${OUTPUT_DIR}/logs/worker_${worker}.log"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if ((failed)); then
  echo "one or more hard-negative workers failed; inspect ${OUTPUT_DIR}/logs" >&2
  exit 1
fi

python scripts/mine_hard_negatives.py \
  --data-root "${DATA_ROOT}" \
  --labels "${LABELS}" \
  --photo-ids "${SPLIT_DIR}/train_photo_ids.txt" \
  --item-index "${ITEM_INDEX}" \
  --duplicate-groups "${DUPLICATE_GROUPS}" \
  --output-dir "${OUTPUT_DIR}" \
  --finalize-only

echo "hard-negative mining complete: ${OUTPUT_DIR}"
