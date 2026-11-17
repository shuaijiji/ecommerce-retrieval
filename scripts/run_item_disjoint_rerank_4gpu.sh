#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/code/ecommerce-retrieval}
CANDIDATES=${2:-$ROOT/artifacts/train_v0/item_disjoint_eval_v1/candidates_top10.jsonl}
LABELS=${3:-$ROOT/artifacts/train_v0/clean_labels.jsonl}
MERGED_ROOT=${4:-$ROOT/artifacts/train_v0/reranker_listwise_train_v1_merged}
OUTPUT_ROOT=${5:-$ROOT/artifacts/train_v0/item_disjoint_eval_v1/rerank}
DATA_ROOT=${ECOM_TRAIN_DATA_ROOT:-$ROOT/data/train/all_data_package}
BASE_MODEL=${ECOM_RERANK_MODEL:-$ROOT/models/Qwen3-VL-Reranker-2B}
RERANK_GPU_LIST=${RERANK_GPU_LIST:-1,2,3,4}
BATCH_SIZE=${BATCH_SIZE:-4}

test -f "$CANDIDATES" || { echo "missing candidates: $CANDIDATES" >&2; exit 2; }
test -f "$LABELS" || { echo "missing labels: $LABELS" >&2; exit 2; }
test -d "$DATA_ROOT" || { echo "missing data root: $DATA_ROOT" >&2; exit 2; }
test -d "$BASE_MODEL" || { echo "missing base model: $BASE_MODEL" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not available" >&2; exit 2; }

IFS=',' read -r -a physical_gpus <<< "$RERANK_GPU_LIST"
test "${#physical_gpus[@]}" -eq 4 || { echo "RERANK_GPU_LIST must contain exactly four physical indexes" >&2; exit 2; }
models=(
  "$BASE_MODEL"
  "$MERGED_ROOT/checkpoint-300-merged"
  "$MERGED_ROOT/checkpoint-400-merged"
  "$MERGED_ROOT/checkpoint-500-merged"
)
names=(base checkpoint_300 checkpoint_400 checkpoint_500)
for model in "${models[@]}"; do
  test -d "$model" || { echo "missing reranker model: $model" >&2; exit 2; }
done

query_count=$(wc -l < "$CANDIDATES" | tr -d ' ')
test "$query_count" -ge 100 || { echo "candidate set must contain at least 100 queries" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"

pids=()
for index in 0 1 2 3; do
  gpu=${physical_gpus[$index]}
  uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
  test -n "$uuid" || { echo "failed to resolve GPU $gpu to UUID" >&2; exit 2; }
  name=${names[$index]}
  model=${models[$index]}
  output=$OUTPUT_ROOT/$name
  (
    CUDA_VISIBLE_DEVICES=$uuid python "$ROOT/scripts/rerank_multimodal.py" \
      --candidates "$CANDIDATES" \
      --output-dir "$output" \
      --data-root "$DATA_ROOT" \
      --model "$model" \
      --num-queries 100 \
      --batch-size "$BATCH_SIZE" \
      > "$output.pilot.log" 2>&1
    CUDA_VISIBLE_DEVICES=$uuid python "$ROOT/scripts/rerank_multimodal.py" \
      --candidates "$CANDIDATES" \
      --output-dir "$output" \
      --data-root "$DATA_ROOT" \
      --model "$model" \
      --num-queries "$query_count" \
      --batch-size "$BATCH_SIZE" \
      > "$output.full.log" 2>&1
    python "$ROOT/scripts/rerank_pipeline.py" evaluate \
      --candidates "$CANDIDATES" \
      --predictions "$output/predictions_${query_count}.jsonl" \
      --labels "$LABELS" \
      --expected-count "$query_count" \
      --output "$output.metrics.json"
  ) &
  worker_pid=$!
  pids+=("$worker_pid")
  echo "started $name on physical GPU $gpu pid=$worker_pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
test "$failed" -eq 0 || { echo "one or more reranker evaluations failed; inspect $OUTPUT_ROOT/*.log" >&2; exit 1; }

python "$ROOT/scripts/compare_reranker_predictions.py" \
  --candidates "$CANDIDATES" \
  --labels "$LABELS" \
  --prediction "base=$OUTPUT_ROOT/base/predictions_${query_count}.jsonl" \
  --prediction "checkpoint_300=$OUTPUT_ROOT/checkpoint_300/predictions_${query_count}.jsonl" \
  --prediction "checkpoint_400=$OUTPUT_ROOT/checkpoint_400/predictions_${query_count}.jsonl" \
  --prediction "checkpoint_500=$OUTPUT_ROOT/checkpoint_500/predictions_${query_count}.jsonl" \
  --output "$OUTPUT_ROOT/comparison.json"

echo "item-disjoint reranker comparison complete: $OUTPUT_ROOT/comparison.json"
