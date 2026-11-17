#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/code/ecommerce-retrieval}
OUTPUT_ROOT=${2:-$ROOT/artifacts/train_v0/item_disjoint_eval_v1}
DATA_ROOT=${ECOM_TRAIN_DATA_ROOT:-$ROOT/data/train/all_data_package}
SPLIT_DIR=${ECOM_SPLIT_DIR:-$ROOT/artifacts/train_v0/split_quarantine_duplicates}
TRAIN_INDEX=${ECOM_TRAIN_INDEX:-$ROOT/artifacts/train_v0/train_item_index}
DEV_INDEX=${ECOM_DEV_INDEX:-$ROOT/artifacts/train_v0/dev_item_index}
LABELS=${ECOM_CLEAN_LABELS:-$ROOT/artifacts/train_v0/clean_labels.jsonl}
EMBED_MODEL=${ECOM_EMBED_MODEL:-$ROOT/models/Qwen3-VL-Embedding-2B}
EVAL_GPU=${EVAL_GPU:-1}
NUM_QUERIES=${NUM_QUERIES:-2000}
SEED=${SEED:-20261117}

DEV_PHOTO_IDS=$SPLIT_DIR/dev_photo_ids.txt
DEV_ITEM_IDS=$SPLIT_DIR/dev_item_ids.txt
RETRIEVAL_DIR=$OUTPUT_ROOT/retrieval
CANDIDATES=$OUTPUT_ROOT/candidates_top10.jsonl

test -d "$DATA_ROOT" || { echo "missing training data root: $DATA_ROOT" >&2; exit 2; }
test -f "$LABELS" || { echo "missing clean labels: $LABELS" >&2; exit 2; }
test -f "$DEV_PHOTO_IDS" || { echo "missing dev photo IDs: $DEV_PHOTO_IDS" >&2; exit 2; }
test -f "$DEV_ITEM_IDS" || { echo "missing dev item IDs: $DEV_ITEM_IDS" >&2; exit 2; }
test -f "$TRAIN_INDEX/manifest.json" || { echo "missing train item index manifest: $TRAIN_INDEX" >&2; exit 2; }
test -d "$EMBED_MODEL" || { echo "missing embedding model: $EMBED_MODEL" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not available" >&2; exit 2; }

GPU_UUID=$(nvidia-smi --id="$EVAL_GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
test -n "$GPU_UUID" || { echo "failed to resolve GPU $EVAL_GPU to UUID" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs"

echo "building or validating the dev item index"
CUDA_VISIBLE_DEVICES=$GPU_UUID python "$ROOT/scripts/build_item_index.py" \
  --data-root "$DATA_ROOT" \
  --item-ids "$DEV_ITEM_IDS" \
  --output-dir "$DEV_INDEX" \
  > "$OUTPUT_ROOT/logs/dev_item_index.log" 2>&1

echo "running frozen retrieval for $NUM_QUERIES item-disjoint queries"
CUDA_VISIBLE_DEVICES=$GPU_UUID python "$ROOT/scripts/evaluate_retrieval.py" \
  --data-root "$DATA_ROOT" \
  --model "$EMBED_MODEL" \
  --item-index "$TRAIN_INDEX" \
  --item-index "$DEV_INDEX" \
  --labels "$LABELS" \
  --photo-ids "$DEV_PHOTO_IDS" \
  --selection-strategy uniform \
  --num-queries "$NUM_QUERIES" \
  --seed "$SEED" \
  --top-k 100 \
  --modes vision multimodal \
  --output-dir "$RETRIEVAL_DIR" \
  > "$OUTPUT_ROOT/logs/retrieval.log" 2>&1

echo "freezing the union of vision/multimodal Top-10 candidates"
python "$ROOT/scripts/rerank_pipeline.py" candidates \
  --retrieval-dir "$RETRIEVAL_DIR" \
  --top-k 10 \
  --expected-count "$NUM_QUERIES" \
  --output "$CANDIDATES"

python "$ROOT/scripts/rerank_pipeline.py" rrf \
  --candidates "$CANDIDATES" \
  --output "$OUTPUT_ROOT/rrf.jsonl"

python "$ROOT/scripts/rerank_pipeline.py" evaluate \
  --candidates "$CANDIDATES" \
  --predictions "$OUTPUT_ROOT/rrf.jsonl" \
  --labels "$LABELS" \
  --expected-count "$NUM_QUERIES" \
  --output "$OUTPUT_ROOT/rrf_metrics.json"

echo "item-disjoint retrieval and candidate generation complete: $CANDIDATES"
