#!/usr/bin/env bash
# Rebuild BOTH item pools with the trained embedding and reuse the exact 2,000 queries.
set -euo pipefail
ROOT=${1:-/code/ecommerce-retrieval}
PILOT=${2:-$ROOT/artifacts/train_v0/embedding_pilot_v1}
OUT=${3:-$ROOT/artifacts/train_v0/embedding_pilot_eval_v1}
BASE=$ROOT/artifacts/train_v0/item_disjoint_eval_v1/retrieval
SPLIT=$ROOT/artifacts/train_v0/split_quarantine_duplicates
DATA=$ROOT/data/train/all_data_package
GPU=${EVAL_GPU:-1}
python - "$PILOT" "$BASE" <<'PY'
import json, sys
from pathlib import Path
pilot, baseline = map(Path, sys.argv[1:])
assert json.loads((pilot/'summary.json').read_text())['functional_gate'], 'Pilot gate failed'
ids=json.loads((baseline/'selected_photo_ids.json').read_text())
assert len(ids)==len(set(ids))==2000, 'Wrong baseline selection'
assert (pilot/'merged/config.json').is_file(), 'Missing merged model'
PY
GPU_UUID=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if ((USED > 1024)); then echo "GPU $GPU is occupied" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES=$GPU_UUID
mkdir -p "$OUT/logs"
for side in train dev; do
  python -u "$ROOT/scripts/build_item_index.py" \
    --model "$PILOT/merged" --record-weight-hashes \
    --data-root "$DATA" --item-ids "$SPLIT/${side}_item_ids.txt" \
    --output-dir "$OUT/${side}_item_index" \
    > "$OUT/logs/${side}_index.log" 2>&1
done
python -u "$ROOT/scripts/evaluate_retrieval.py" \
  --model "$PILOT/merged" --require-weight-hashes \
  --data-root "$DATA" --labels "$ROOT/artifacts/train_v0/clean_labels.jsonl" \
  --photo-ids "$SPLIT/dev_photo_ids.txt" \
  --fixed-selection "$BASE/selected_photo_ids.json" \
  --item-index "$OUT/train_item_index" --item-index "$OUT/dev_item_index" \
  --num-queries 2000 --seed 20261117 --selection-strategy uniform \
  --top-k 100 --modes vision multimodal --output-dir "$OUT/retrieval" \
  > "$OUT/logs/retrieval.log" 2>&1
python "$ROOT/scripts/compare_embedding_recall.py" \
  --baseline "$BASE" --trained "$OUT/retrieval" --output "$OUT/comparison.json"
