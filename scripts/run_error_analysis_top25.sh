#!/usr/bin/env bash
set -euo pipefail
# Run after the completed Top-10 experiment. Uses a fresh output directory for Top-25.
RETRIEVAL_DIR=${1:?Usage: bash scripts/run_error_analysis_top25.sh FULL_RETRIEVAL_OUTPUT_DIR [TOP10_OUT] [TOP25_OUT]}
BASE=${2:-artifacts/rerank_v0}
OUT=${3:-artifacts/rerank_top25_v0}
DATA=${ECOM_DATA_ROOT:-/code/ecommerce-retrieval/data/eval/test_small_package}
MODEL=${ECOM_RERANK_MODEL:-/code/ecommerce-retrieval/models/Qwen3-VL-Reranker-2B}
BATCH_SIZE=${ECOM_RERANK_BATCH_SIZE:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$BASE/analysis" "$OUT/analysis"

# First decompose the already-completed Top-10 run. This is label-only post-evaluation.
python scripts/analyze_rerank_errors.py analyze \
  --candidates "$BASE/candidates.jsonl" \
  --predictions "$BASE/vlm/predictions_10000.jsonl" \
  --labels "$DATA/label.json" \
  --retrieval-dir "$RETRIEVAL_DIR" \
  --photo-info "$DATA/photo_info.jsonl" \
  --output-summary "$BASE/analysis/error_summary.json" \
  --output-cases "$BASE/analysis/error_cases.jsonl"

# Build the label-free Top-25 union and run the same frozen zero-shot reranker.
python scripts/rerank_pipeline.py candidates \
  --retrieval-dir "$RETRIEVAL_DIR" \
  --top-k 25 \
  --output "$OUT/candidates.jsonl"
python scripts/rerank_pipeline.py rrf \
  --candidates "$OUT/candidates.jsonl" \
  --output "$OUT/rrf.jsonl"
python scripts/rerank_pipeline.py evaluate \
  --candidates "$OUT/candidates.jsonl" \
  --predictions "$OUT/rrf.jsonl" \
  --labels "$DATA/label.json" \
  --expected-count 10000 \
  --expect-recall 0.9362 \
  --output "$OUT/rrf_metrics.json"

python scripts/rerank_multimodal.py \
  --candidates "$OUT/candidates.jsonl" \
  --output-dir "$OUT/vlm" \
  --data-root "$DATA" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --num-queries 100 2>&1 | tee "$OUT/pilot.log"
python scripts/rerank_pipeline.py evaluate \
  --candidates "$OUT/candidates.jsonl" \
  --predictions "$OUT/vlm/predictions_100.jsonl" \
  --labels "$DATA/label.json" \
  --expected-count 100 \
  --output "$OUT/vlm_pilot_metrics.json"

python scripts/rerank_multimodal.py \
  --candidates "$OUT/candidates.jsonl" \
  --output-dir "$OUT/vlm" \
  --data-root "$DATA" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --num-queries 10000 2>&1 | tee "$OUT/full.log"
python scripts/rerank_pipeline.py evaluate \
  --candidates "$OUT/candidates.jsonl" \
  --predictions "$OUT/vlm/predictions_10000.jsonl" \
  --labels "$DATA/label.json" \
  --expected-count 10000 \
  --expect-recall 0.9362 \
  --output "$OUT/vlm_full_metrics.json"

python scripts/analyze_rerank_errors.py analyze \
  --candidates "$OUT/candidates.jsonl" \
  --predictions "$OUT/vlm/predictions_10000.jsonl" \
  --labels "$DATA/label.json" \
  --retrieval-dir "$RETRIEVAL_DIR" \
  --photo-info "$DATA/photo_info.jsonl" \
  --output-summary "$OUT/analysis/error_summary.json" \
  --output-cases "$OUT/analysis/error_cases.jsonl"
python scripts/analyze_rerank_errors.py compare \
  --base-candidates "$BASE/candidates.jsonl" \
  --base-predictions "$BASE/vlm/predictions_10000.jsonl" \
  --new-candidates "$OUT/candidates.jsonl" \
  --new-predictions "$OUT/vlm/predictions_10000.jsonl" \
  --labels "$DATA/label.json" \
  --output-summary "$OUT/analysis/top10_vs_top25_summary.json" \
  --output-cases "$OUT/analysis/top10_vs_top25_cases.jsonl"
