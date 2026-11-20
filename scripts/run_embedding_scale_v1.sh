#!/usr/bin/env bash
# Explicit stages: data -> perf -> train. No development-set-driven parameter search.
set -euo pipefail
STAGE=${1:?usage: bash scripts/run_embedding_scale_v1.sh data|perf|train [ROOT]}
ROOT=${2:-/code/ecommerce-retrieval}
cd "$ROOT"
DATASET=$ROOT/artifacts/train_v0/embedding_data_1k_v1
OUTPUT=$ROOT/artifacts/train_v0/embedding_train_1k_v1
PERF=$ROOT/artifacts/train_v0/embedding_perf_1k_v1
LOGS=$ROOT/artifacts/train_v0/embedding_scale_logs
mkdir -p "$LOGS"
if [[ "$STAGE" == data ]]; then
  python -u scripts/prepare_embedding_pilot.py \
    --data-root data/train/all_data_package \
    --split-dir artifacts/train_v0/split_quarantine_duplicates \
    --labels artifacts/train_v0/clean_labels.jsonl \
    --candidates artifacts/train_v0/hard_negative_candidates \
    --duplicate-groups artifacts/train_v0/proposed_item_duplicate_groups.jsonl \
    --scale-up --num-queries 1000 --negatives 4 --seed 20261119 \
    --output-dir "$DATASET" > "$LOGS/data.log" 2>&1
  echo "dataset audit and construction complete: $DATASET/manifest.json"
  exit 0
fi
case "$STAGE" in perf|train) ;; *) echo 'stage must be data, perf, or train' >&2; exit 2;; esac
GPU=${TRAIN_GPU:-1}
GPU_UUID=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if ((USED > 1024)); then echo "GPU $GPU occupied ($USED MiB)" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES=$GPU_UUID
if [[ "$STAGE" == perf ]]; then
  TARGET=$PERF
  STEPS=100
  EXTRA=(--performance-only)
else
  python - "$PERF" "$DATASET" "$ROOT/models/Qwen3-VL-Embedding-2B" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from retrieval_common import sha256_file, model_weight_hashes
perf, dataset, model = map(Path, sys.argv[1:])
r=json.loads((perf/'performance.json').read_text())
c=json.loads((perf/'config.json').read_text())
assert r['performance_only'] and r['completed_steps']==100 and r['steady_queries_per_second']>0
assert r['peak_reserved_gib'] < 70, 'insufficient memory headroom for this single-GPU protocol'
assert c['dataset_manifest_sha256']==sha256_file(dataset/'manifest.json'), 'dataset changed since performance pilot'
assert c['base_model_weight_hashes']==model_weight_hashes(model), 'model changed since performance pilot'
assert c['upstream_source_sha256']==sha256_file(Path('Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py'))
assert c['training_script_sha256']==sha256_file(Path('scripts/train_embedding_pilot.py'))
assert c['retrieval_common_sha256']==sha256_file(Path('scripts/retrieval_common.py'))
for key, expected in [('lr',1e-5),('temperature',0.07),('seed',20261119),('scale_up',True)]:
    assert c['args'][key]==expected, f'performance config differs: {key}'
PY
  TARGET=$OUTPUT
  STEPS=1000
  EXTRA=()
fi
# Each stage keeps a separate resource trace. Rows include initialization/monitoring;
# correlate with timestamped steps.jsonl for the pure training window.
TRACE=$LOGS/${STAGE}_gpu_$(date +%Y%m%d-%H%M%S).csv
nvidia-smi --id="$GPU_UUID" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw \
  --format=csv --loop=2 > "$TRACE" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
python -u scripts/train_embedding_pilot.py \
  --repo Qwen3-VL-Embedding --model models/Qwen3-VL-Embedding-2B \
  --dataset "$DATASET" --output-dir "$TARGET" --scale-up --monitor-queries 100 \
  --steps "$STEPS" --lr 1e-5 --temperature 0.07 --seed 20261119 \
  "${EXTRA[@]}" > "$LOGS/${STAGE}.log" 2>&1
echo "$STAGE complete: $TARGET; GPU resource trace: $TRACE"
