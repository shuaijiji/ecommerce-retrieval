#!/usr/bin/env bash
set -euo pipefail
# Run from /code/ecommerce-retrieval. First argument is existing full retrieval output.
RETRIEVAL_DIR=${1:?Usage: bash scripts/run_rerank.sh FULL_RETRIEVAL_OUTPUT_DIR}
OUT=${2:-artifacts/rerank_v0}
DATA=${ECOM_DATA_ROOT:-/code/ecommerce-retrieval/data/eval/test_small_package}
MODEL=${ECOM_RERANK_MODEL:-/code/ecommerce-retrieval/models/Qwen3-VL-Reranker-2B}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT"
python scripts/rerank_pipeline.py candidates --retrieval-dir "$RETRIEVAL_DIR" --output "$OUT/candidates.jsonl"
python scripts/rerank_pipeline.py rrf --candidates "$OUT/candidates.jsonl" --output "$OUT/rrf.jsonl"
python scripts/rerank_pipeline.py evaluate --candidates "$OUT/candidates.jsonl" --predictions "$OUT/rrf.jsonl" --labels "$DATA/label.json" --expected-count 10000 --expect-recall 0.9039 --output "$OUT/rrf_metrics.json"
python scripts/rerank_multimodal.py --candidates "$OUT/candidates.jsonl" --output-dir "$OUT/vlm" --data-root "$DATA" --model "$MODEL" --num-queries 100 2>&1 | tee "$OUT/pilot.log"
python scripts/rerank_pipeline.py evaluate --candidates "$OUT/candidates.jsonl" --predictions "$OUT/vlm/predictions_100.jsonl" --labels "$DATA/label.json" --expected-count 100 --output "$OUT/vlm_pilot_metrics.json"
python scripts/rerank_multimodal.py --candidates "$OUT/candidates.jsonl" --output-dir "$OUT/vlm" --data-root "$DATA" --model "$MODEL" --num-queries 10000 2>&1 | tee "$OUT/full.log"
python scripts/rerank_pipeline.py evaluate --candidates "$OUT/candidates.jsonl" --predictions "$OUT/vlm/predictions_10000.jsonl" --labels "$DATA/label.json" --expected-count 10000 --expect-recall 0.9039 --output "$OUT/vlm_full_metrics.json"
