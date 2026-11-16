#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/code/ecommerce-retrieval}
DATA_DIR=${2:-$ROOT/artifacts/train_v0/reranker_sft_smoke_40k}
OUTPUT_DIR=${3:-$ROOT/artifacts/train_v0/reranker_listwise_perf_1gpu}
NUM_GPUS=${4:-1}
MODEL=${ECOM_RERANK_MODEL:-$ROOT/models/Qwen3-VL-Reranker-2B}
MAX_STEPS=${MAX_STEPS:-200}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-1}
MASTER_PORT=${MASTER_PORT:-29500}

TRAIN_DATA=$DATA_DIR/swift_train.jsonl
MANIFEST=$DATA_DIR/manifest.json
SWIFT_OUTPUT=$OUTPUT_DIR/model
TRAIN_LOG=$OUTPUT_DIR/train.log
GPU_LOG=$OUTPUT_DIR/gpu_metrics.csv
CPU_LOG=$OUTPUT_DIR/cpu_metrics.csv
RUN_INFO=$OUTPUT_DIR/run_info.txt

case "$NUM_GPUS" in
  1|2|4|8) ;;
  *) echo "NUM_GPUS must be one of 1, 2, 4, 8: $NUM_GPUS" >&2; exit 2 ;;
esac

test "$MAX_STEPS" -ge 100 || { echo "MAX_STEPS must be at least 100" >&2; exit 2; }
test -d "$MODEL" || { echo "missing model directory: $MODEL" >&2; exit 2; }
test -f "$TRAIN_DATA" || { echo "missing training data: $TRAIN_DATA" >&2; exit 2; }
test -f "$MANIFEST" || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
command -v swift >/dev/null || { echo "swift is not installed in the active environment" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not available" >&2; exit 2; }

python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m.get("complete") is True; assert m.get("effective_pointwise_pairs")==40000; assert m.get("rows",{}).get("swift_train")==4500; assert m.get("rows",{}).get("swift_smoke_eval")==500' "$MANIFEST"

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
test "$AVAILABLE_GPUS" -ge "$NUM_GPUS" || {
  echo "requested $NUM_GPUS GPUs, but nvidia-smi reports $AVAILABLE_GPUS" >&2
  exit 2
}

if test -e "$OUTPUT_DIR"; then
  echo "output already exists; use a new directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

GPU_LIST=$(python -c 'import sys; print(",".join(map(str, range(int(sys.argv[1])))))' "$NUM_GPUS")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_POSITIVE_SAMPLES=1
export MAX_NEGATIVE_SAMPLES=7
export LISTWISE_RERANKER_TEMPERATURE=1.0
export LISTWISE_RERANKER_MIN_GROUP_SIZE=2
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

{
  echo "objective=listwise_reranker"
  echo "root=$ROOT"
  echo "model=$MODEL"
  echo "train_data=$TRAIN_DATA"
  echo "output_dir=$OUTPUT_DIR"
  echo "num_gpus=$NUM_GPUS"
  echo "gpu_list=$GPU_LIST"
  echo "max_steps=$MAX_STEPS"
  echo "per_device_train_batch_size=1"
  echo "gradient_accumulation_steps=1"
  echo "effective_pairs_per_query=8"
  echo "learning_rate=$LEARNING_RATE"
  echo "master_port=$MASTER_PORT"
  echo "monitor_interval=$MONITOR_INTERVAL"
  echo "start_time=$(date --iso-8601=seconds)"
  python -V
  python -c 'import torch,transformers,swift,flash_attn; print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda)); print("transformers="+transformers.__version__); print("swift="+swift.__version__); print("flash_attn="+flash_attn.__version__)'
} > "$RUN_INFO"

echo "timestamp,index,utilization_gpu_percent,memory_used_mib,power_draw_w" > "$GPU_LOG"
echo "timestamp,cpu_percent_sum,rss_mib" > "$CPU_LOG"

monitor_gpu() {
  nvidia-smi \
    --id="$GPU_LIST" \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits \
    --loop="$MONITOR_INTERVAL" >> "$GPU_LOG"
}

monitor_cpu() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    timestamp=$(date --iso-8601=seconds)
    ps -eo pcpu=,rss=,args= | awk -v ts="$timestamp" '
      /swift|torchrun|torch\.distributed/ && !/awk/ {cpu += $1; rss += $2}
      END {printf "%s,%.2f,%.2f\n", ts, cpu + 0, rss / 1024}
    '
    sleep "$MONITOR_INTERVAL"
  done
}

cleanup() {
  test -n "${GPU_MONITOR_PID:-}" && kill "$GPU_MONITOR_PID" 2>/dev/null || true
  test -n "${CPU_MONITOR_PID:-}" && kill "$CPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "starting ${NUM_GPUS}-GPU listwise performance pilot; log: $TRAIN_LOG"
CUDA_VISIBLE_DEVICES=$GPU_LIST \
NPROC_PER_NODE=$NUM_GPUS \
MASTER_PORT=$MASTER_PORT \
swift sft \
  --model "$MODEL" \
  --task_type generative_reranker \
  --loss_type listwise_reranker \
  --label_names labels \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --learning_rate "$LEARNING_RATE" \
  --target_modules all-linear \
  --dataset "$TRAIN_DATA" \
  --split_dataset_ratio 0 \
  --attn_impl flash_attn \
  --padding_free true \
  --torch_dtype bfloat16 \
  --load_from_cache_file false \
  --eval_strategy no \
  --save_strategy no \
  --output_dir "$SWIFT_OUTPUT" \
  --logging_steps 5 \
  --max_steps "$MAX_STEPS" \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --warmup_ratio 0.05 \
  --report_to none \
  > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

monitor_gpu &
GPU_MONITOR_PID=$!
monitor_cpu &
CPU_MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
cleanup
trap - EXIT INT TERM

echo "end_time=$(date --iso-8601=seconds)" >> "$RUN_INFO"
echo "exit_status=$TRAIN_STATUS" >> "$RUN_INFO"

if test "$TRAIN_STATUS" -ne 0; then
  echo "performance pilot failed with status $TRAIN_STATUS; inspect $TRAIN_LOG" >&2
  exit "$TRAIN_STATUS"
fi

grep -q "'train_runtime'" "$TRAIN_LOG" || {
  echo "training exited successfully but train_runtime is missing: $TRAIN_LOG" >&2
  exit 3
}

echo "performance pilot complete: $OUTPUT_DIR"
grep "'train_runtime'" "$TRAIN_LOG" | tail -n 1
