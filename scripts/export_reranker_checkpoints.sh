#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/code/ecommerce-retrieval}
TRAIN_RUN=${2:-$ROOT/artifacts/train_v0/reranker_listwise_train_v1}
OUTPUT_ROOT=${3:-$ROOT/artifacts/train_v0/reranker_listwise_train_v1_merged}
EXPORT_GPU=${EXPORT_GPU:-1}

command -v swift >/dev/null || { echo "swift is not installed in the active environment" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not available" >&2; exit 2; }
test -d "$TRAIN_RUN/model" || { echo "missing training model directory: $TRAIN_RUN/model" >&2; exit 2; }

GPU_UUID=$(nvidia-smi --id="$EXPORT_GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
test -n "$GPU_UUID" || { echo "failed to resolve GPU $EXPORT_GPU to UUID" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs"

for step in 300 400 500; do
  mapfile -t matches < <(find "$TRAIN_RUN/model" -type d -name "checkpoint-$step" -print)
  test "${#matches[@]}" -eq 1 || {
    echo "expected exactly one checkpoint-$step under $TRAIN_RUN/model, found ${#matches[@]}" >&2
    exit 2
  }
  checkpoint=${matches[0]}
  destination=$OUTPUT_ROOT/checkpoint-$step-merged
  log=$OUTPUT_ROOT/logs/checkpoint-$step.log
  if test -e "$destination"; then
    test -f "$destination/config.json" || { echo "incomplete existing export: $destination" >&2; exit 2; }
    test -n "$(find "$destination" -maxdepth 1 -name '*.safetensors' -print -quit)" || {
      echo "existing export has no safetensors: $destination" >&2
      exit 2
    }
    echo "validated existing merged checkpoint: $destination"
    continue
  fi
  echo "exporting $checkpoint -> $destination"
  CUDA_VISIBLE_DEVICES=$GPU_UUID swift export \
    --adapters "$checkpoint" \
    --merge_lora true \
    --output_dir "$destination" \
    > "$log" 2>&1
  test -f "$destination/config.json" || { echo "export missing config.json: $destination" >&2; exit 3; }
  test -n "$(find "$destination" -maxdepth 1 -name '*.safetensors' -print -quit)" || {
    echo "export missing safetensors: $destination" >&2
    exit 3
  }
  echo "export complete: $destination"
done

echo "all requested checkpoints exported: $OUTPUT_ROOT"
