#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/code/ecommerce-retrieval}
DATA_DIR=${2:-$ROOT/artifacts/train_v0/reranker_sft_smoke_40k}
OUTPUT_DIR=${3:-$ROOT/artifacts/train_v0/reranker_listwise_train_v1}
MODEL=${ECOM_RERANK_MODEL:-$ROOT/models/Qwen3-VL-Reranker-2B}
TRAIN_GPU_LIST=${TRAIN_GPU_LIST:-0,1,2,3,4,5,6,7}
MAX_STEPS=${MAX_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MASTER_PORT=${MASTER_PORT:-29500}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
MAX_PREEXISTING_MEMORY_MIB=${MAX_PREEXISTING_MEMORY_MIB:-1024}
ALLOW_BUSY_GPUS=${ALLOW_BUSY_GPUS:-0}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

NUM_GPUS=8
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
TRAIN_DATA=$DATA_DIR/swift_train.jsonl
VAL_DATA=$DATA_DIR/swift_smoke_eval.jsonl
MANIFEST=$DATA_DIR/manifest.json
SWIFT_OUTPUT=$OUTPUT_DIR/model
TRAIN_LOG=$OUTPUT_DIR/train.log
GPU_LOG=$OUTPUT_DIR/gpu_metrics.csv
CPU_LOG=$OUTPUT_DIR/cpu_metrics.csv
RUN_INFO=$OUTPUT_DIR/run_info.txt

test "$MAX_STEPS" -gt 0 || { echo "MAX_STEPS must be positive" >&2; exit 2; }
test "$EVAL_STEPS" -gt 0 || { echo "EVAL_STEPS must be positive" >&2; exit 2; }
test "$SAVE_STEPS" -gt 0 || { echo "SAVE_STEPS must be positive" >&2; exit 2; }
test "$SAVE_TOTAL_LIMIT" -ge 5 || { echo "SAVE_TOTAL_LIMIT must be at least 5 to retain steps 100-500" >&2; exit 2; }
test "$MAX_STEPS" -eq 500 || { echo "formal v1 protocol requires MAX_STEPS=500" >&2; exit 2; }
test "$EVAL_STEPS" -eq 100 || { echo "formal v1 protocol requires EVAL_STEPS=100" >&2; exit 2; }
test "$SAVE_STEPS" -eq 100 || { echo "formal v1 protocol requires SAVE_STEPS=100" >&2; exit 2; }
test "$LEARNING_RATE" = "1e-5" || { echo "formal v1 protocol requires LEARNING_RATE=1e-5" >&2; exit 2; }
test "$DATALOADER_NUM_WORKERS" -ge 0 || { echo "DATALOADER_NUM_WORKERS must be non-negative" >&2; exit 2; }
test "$DATASET_NUM_PROC" -ge 1 || { echo "DATASET_NUM_PROC must be positive" >&2; exit 2; }
test "$MAX_PREEXISTING_MEMORY_MIB" -ge 0 || { echo "MAX_PREEXISTING_MEMORY_MIB must be non-negative" >&2; exit 2; }
case "$ALLOW_BUSY_GPUS" in
  0|1) ;;
  *) echo "ALLOW_BUSY_GPUS must be 0 or 1" >&2; exit 2 ;;
esac

test -d "$MODEL" || { echo "missing model directory: $MODEL" >&2; exit 2; }
test -f "$TRAIN_DATA" || { echo "missing training data: $TRAIN_DATA" >&2; exit 2; }
test -f "$VAL_DATA" || { echo "missing validation data: $VAL_DATA" >&2; exit 2; }
test -f "$MANIFEST" || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
command -v swift >/dev/null || { echo "swift is not installed in the active environment" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not available" >&2; exit 2; }

python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m.get("complete") is True; assert m.get("effective_pointwise_pairs")==40000; assert m.get("rows",{}).get("swift_train")==4500; assert m.get("rows",{}).get("swift_smoke_eval")==500' "$MANIFEST"

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
test "$AVAILABLE_GPUS" -ge "$NUM_GPUS" || {
  echo "formal training requires 8 GPUs, but nvidia-smi reports $AVAILABLE_GPUS" >&2
  exit 2
}

GPU_LIST=$(python -c 'import sys; xs=sys.argv[1].split(","); n=int(sys.argv[2]); assert len(xs)==n and len(set(xs))==n and all(x.isdigit() for x in xs); print(",".join(xs))' "$TRAIN_GPU_LIST" "$NUM_GPUS") || {
  echo "TRAIN_GPU_LIST must contain exactly 8 unique numeric physical indexes: $TRAIN_GPU_LIST" >&2
  exit 2
}
python -c 'import sys; xs=map(int,sys.argv[1].split(",")); available=int(sys.argv[2]); assert all(0 <= x < available for x in xs)' "$GPU_LIST" "$AVAILABLE_GPUS" || {
  echo "TRAIN_GPU_LIST contains an unavailable physical GPU index: $GPU_LIST" >&2
  exit 2
}

PREEXISTING_MEMORY=$(nvidia-smi --id="$GPU_LIST" --query-gpu=index,memory.used --format=csv,noheader,nounits)
echo "selected physical GPUs and pre-existing memory (MiB):"
echo "$PREEXISTING_MEMORY"
if test "$ALLOW_BUSY_GPUS" -ne 1; then
  python -c 'import sys
limit=float(sys.argv[1])
busy=[]
for line in sys.stdin:
    index, memory=(part.strip() for part in line.split(",", 1))
    if float(memory) > limit:
        busy.append((index, float(memory)))
if busy:
    raise SystemExit("selected GPUs exceed pre-existing memory limit: " + repr(busy))
' "$MAX_PREEXISTING_MEMORY_MIB" <<< "$PREEXISTING_MEMORY" || {
    echo "confirm the owners of existing GPU contexts; rerun with ALLOW_BUSY_GPUS=1 only if sharing is intentional" >&2
    exit 2
  }
fi

CUDA_VISIBLE_LIST=""
for gpu_index in ${GPU_LIST//,/ }; do
  gpu_uuid=$(nvidia-smi --id="$gpu_index" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
  test -n "$gpu_uuid" || { echo "failed to resolve UUID for GPU index $gpu_index" >&2; exit 2; }
  if test -n "$CUDA_VISIBLE_LIST"; then
    CUDA_VISIBLE_LIST="$CUDA_VISIBLE_LIST,$gpu_uuid"
  else
    CUDA_VISIBLE_LIST=$gpu_uuid
  fi
done

if test -n "$RESUME_FROM_CHECKPOINT"; then
  test -d "$OUTPUT_DIR" || { echo "resume output directory does not exist: $OUTPUT_DIR" >&2; exit 2; }
  test -d "$RESUME_FROM_CHECKPOINT" || { echo "resume checkpoint does not exist: $RESUME_FROM_CHECKPOINT" >&2; exit 2; }
  RESUME_TAG=$(basename "$RESUME_FROM_CHECKPOINT")
  TRAIN_LOG=$OUTPUT_DIR/train_resume_$RESUME_TAG.log
  GPU_LOG=$OUTPUT_DIR/gpu_metrics_resume_$RESUME_TAG.csv
  CPU_LOG=$OUTPUT_DIR/cpu_metrics_resume_$RESUME_TAG.csv
  RUN_INFO=$OUTPUT_DIR/run_info_resume_$RESUME_TAG.txt
  test ! -e "$TRAIN_LOG" || { echo "resume log already exists; preserve it and choose the correct checkpoint: $TRAIN_LOG" >&2; exit 2; }
else
  if test -e "$OUTPUT_DIR"; then
    echo "output already exists; use a new directory or set RESUME_FROM_CHECKPOINT: $OUTPUT_DIR" >&2
    exit 2
  fi
  mkdir -p "$OUTPUT_DIR"
fi

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
  echo "objective=listwise_reranker_formal_v1"
  echo "root=$ROOT"
  echo "model=$MODEL"
  echo "train_data=$TRAIN_DATA"
  echo "val_data=$VAL_DATA"
  echo "output_dir=$OUTPUT_DIR"
  echo "num_gpus=$NUM_GPUS"
  echo "gpu_list=$GPU_LIST"
  echo "cuda_visible_devices=$CUDA_VISIBLE_LIST"
  echo "preexisting_memory_mib=$(echo "$PREEXISTING_MEMORY" | tr '\n' ';')"
  echo "allow_busy_gpus=$ALLOW_BUSY_GPUS"
  echo "max_steps=$MAX_STEPS"
  echo "eval_steps=$EVAL_STEPS"
  echo "save_steps=$SAVE_STEPS"
  echo "save_total_limit=$SAVE_TOTAL_LIMIT"
  echo "per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
  echo "global_batch_size=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
  echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  echo "dataloader_num_workers=$DATALOADER_NUM_WORKERS"
  echo "dataset_num_proc=$DATASET_NUM_PROC"
  echo "effective_pairs_per_query=8"
  echo "learning_rate=$LEARNING_RATE"
  echo "master_port=$MASTER_PORT"
  echo "monitor_interval=$MONITOR_INTERVAL"
  echo "resume_from_checkpoint=$RESUME_FROM_CHECKPOINT"
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
  while true; do
    timestamp=$(date --iso-8601=seconds)
    if ! ps -eo pcpu=,rss=,args= 2>/dev/null | awk -v ts="$timestamp" '
        /swift|torchrun|torch\.distributed/ && !/awk/ {cpu += $1; rss += $2}
        END {printf "%s,%.2f,%.2f\n", ts, cpu + 0, rss / 1024}
      '
    then
      echo "$timestamp,0.00,0.00"
    fi
    sleep "$MONITOR_INTERVAL"
  done
}

cleanup() {
  test -n "${GPU_MONITOR_PID:-}" && kill "$GPU_MONITOR_PID" 2>/dev/null || true
  test -n "${CPU_MONITOR_PID:-}" && kill "$CPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

SWIFT_ARGS=(
  --model "$MODEL"
  --task_type generative_reranker
  --loss_type listwise_reranker
  --label_names labels
  --tuner_type lora
  --lora_rank 8
  --lora_alpha 32
  --learning_rate "$LEARNING_RATE"
  --target_modules all-linear
  --dataset "$TRAIN_DATA"
  --val_dataset "$VAL_DATA"
  --split_dataset_ratio 0
  --attn_impl flash_attn
  --padding_free true
  --torch_dtype bfloat16
  --load_from_cache_file false
  --eval_strategy steps
  --eval_on_start true
  --eval_steps "$EVAL_STEPS"
  --save_strategy steps
  --save_steps "$SAVE_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --output_dir "$SWIFT_OUTPUT"
  --logging_steps 5
  --max_steps "$MAX_STEPS"
  --max_length 4096
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --gradient_checkpointing true
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --dataset_num_proc "$DATASET_NUM_PROC"
  --warmup_ratio 0.05
  --report_to none
)
if test -n "$RESUME_FROM_CHECKPOINT"; then
  SWIFT_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "starting formal 8-GPU listwise training; log: $TRAIN_LOG"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_LIST \
NPROC_PER_NODE=$NUM_GPUS \
MASTER_PORT=$MASTER_PORT \
swift sft "${SWIFT_ARGS[@]}" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

monitor_gpu &
GPU_MONITOR_PID=$!
monitor_cpu >> "$CPU_LOG" &
CPU_MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
cleanup
trap - EXIT INT TERM

{
  echo "end_time=$(date --iso-8601=seconds)"
  echo "exit_status=$TRAIN_STATUS"
} >> "$RUN_INFO"

if test "$TRAIN_STATUS" -ne 0; then
  echo "formal training failed with status $TRAIN_STATUS; inspect $TRAIN_LOG" >&2
  exit "$TRAIN_STATUS"
fi

grep -q "'train_runtime'" "$TRAIN_LOG" || {
  echo "training exited successfully but train_runtime is missing: $TRAIN_LOG" >&2
  exit 3
}
FINAL_CHECKPOINT=$(find "$SWIFT_OUTPUT" -type d -name "checkpoint-$MAX_STEPS" -print -quit)
test -n "$FINAL_CHECKPOINT" || {
  echo "training exited successfully but checkpoint-$MAX_STEPS is missing under $SWIFT_OUTPUT" >&2
  exit 3
}
test "$(wc -l < "$GPU_LOG")" -gt 1 || { echo "GPU monitoring produced no samples: $GPU_LOG" >&2; exit 3; }
test "$(wc -l < "$CPU_LOG")" -gt 1 || { echo "CPU monitoring produced no samples: $CPU_LOG" >&2; exit 3; }

python -c '
import csv, sys
expected=set(sys.argv[2].split(","))
stats={index: {"memory": 0.0, "util": 0.0} for index in expected}
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        index=row["index"].strip()
        if index in stats:
            stats[index]["memory"]=max(stats[index]["memory"], float(row["memory_used_mib"]))
            stats[index]["util"]=max(stats[index]["util"], float(row["utilization_gpu_percent"]))
inactive={index: values for index, values in stats.items() if values["memory"] < 1024 or values["util"] <= 0}
assert not inactive, f"selected GPUs showed no training activity: {inactive}"
' "$GPU_LOG" "$GPU_LIST" || {
  echo "GPU activity validation failed; inspect UUID mapping and $GPU_LOG" >&2
  exit 3
}

echo "formal listwise training complete"
echo "final checkpoint: $FINAL_CHECKPOINT"
grep "'train_runtime'" "$TRAIN_LOG" | tail -n 1
