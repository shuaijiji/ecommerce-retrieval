#!/usr/bin/env bash
set -euo pipefail

# Build the frozen-retriever item index over training items only.  Shards are
# disjoint across GPUs, atomically published, and safe to resume.
DATA_ROOT="${1:-/code/ecommerce-retrieval/data/train/all_data_package}"
ITEM_IDS="${2:-artifacts/train_v0/split_quarantine_duplicates/train_item_ids.txt}"
OUTPUT_DIR="${3:-artifacts/train_v0/train_item_index}"
GPU_COUNT="${GPU_COUNT:-8}"

mkdir -p "${OUTPUT_DIR}/logs"

pids=()
for ((worker = 0; worker < GPU_COUNT; worker++)); do
  CUDA_VISIBLE_DEVICES="${worker}" python scripts/build_item_index.py \
    --data-root "${DATA_ROOT}" \
    --item-ids "${ITEM_IDS}" \
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
  echo "one or more item-index workers failed; inspect ${OUTPUT_DIR}/logs" >&2
  exit 1
fi

# With every shard present, this validates the whole index and publishes the
# manifest without loading the model again.
CUDA_VISIBLE_DEVICES=0 python scripts/build_item_index.py \
  --data-root "${DATA_ROOT}" \
  --item-ids "${ITEM_IDS}" \
  --output-dir "${OUTPUT_DIR}"

echo "train item index complete: ${OUTPUT_DIR}"
