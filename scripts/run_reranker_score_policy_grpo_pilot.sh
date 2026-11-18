#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/code/ecommerce-retrieval}
DATA_DIR=${2:-$ROOT/artifacts/train_v0/reranker_sft_smoke_40k}
OUTPUT_DIR=${3:-$ROOT/artifacts/train_v0/reranker_score_policy_grpo_pilot_v1}
MODEL=${ECOM_RERANK_MODEL:-$ROOT/artifacts/train_v0/reranker_listwise_train_v1_merged/checkpoint-500-merged}
POLICY_GPU=${POLICY_GPU:-1}
CONTROL_GPU=${CONTROL_GPU:-2}
MAX_STEPS=${MAX_STEPS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
SEED=${SEED:-20261118}

SOURCE_DATA=$DATA_DIR/swift_train.jsonl
PILOT_DATA=$OUTPUT_DIR/data/train_100.jsonl
PILOT_MANIFEST=$OUTPUT_DIR/data/manifest.json
PLUGIN=$ROOT/scripts/reranker_score_policy_grpo_plugin.py
POLICY_OUTPUT=$OUTPUT_DIR/score_policy
CONTROL_OUTPUT=$OUTPUT_DIR/listwise_control

test -d "$MODEL" || { echo "missing merged checkpoint-500: $MODEL" >&2; exit 2; }
test -f "$SOURCE_DATA" || { echo "missing training groups: $SOURCE_DATA" >&2; exit 2; }
test -f "$PLUGIN" || { echo "missing score-policy plugin: $PLUGIN" >&2; exit 2; }
command -v swift >/dev/null || { echo "swift is not installed in the active environment" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not available" >&2; exit 2; }
if test "$POLICY_GPU" = "$CONTROL_GPU"; then
  echo "POLICY_GPU and CONTROL_GPU must be different" >&2
  exit 2
fi
if test -e "$OUTPUT_DIR"; then
  echo "output already exists; use a new directory: $OUTPUT_DIR" >&2
  exit 2
fi

for gpu in "$POLICY_GPU" "$CONTROL_GPU"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if test -z "$used" || awk -v value="$used" 'BEGIN {exit !(value > 1024)}'; then
    echo "GPU $gpu has ${used:-unknown} MiB pre-existing memory; choose an idle GPU" >&2
    exit 2
  fi
done
POLICY_UUID=$(nvidia-smi --id="$POLICY_GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
CONTROL_UUID=$(nvidia-smi --id="$CONTROL_GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')

mkdir -p "$OUTPUT_DIR/data" "$POLICY_OUTPUT" "$CONTROL_OUTPUT"
PYTHONPATH=$ROOT/scripts python "$ROOT/scripts/prepare_score_policy_grpo_pilot.py" \
  --source "$SOURCE_DATA" \
  --output "$PILOT_DATA" \
  --manifest "$PILOT_MANIFEST" \
  --count 100 \
  --seed "$SEED"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_POSITIVE_SAMPLES=1
export MAX_NEGATIVE_SAMPLES=7
export LISTWISE_RERANKER_TEMPERATURE=1.0
export LISTWISE_RERANKER_MIN_GROUP_SIZE=2
export SCORE_POLICY_GRPO_GROUP_SIZE=8
export SCORE_POLICY_GRPO_ROLLOUT_SAMPLES=8
export SCORE_POLICY_GRPO_TEMPERATURE=1.0
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

COMMON_ARGS=(
  --model "$MODEL"
  --task_type generative_reranker
  --label_names labels
  --tuner_type lora
  --lora_rank 8
  --lora_alpha 32
  --learning_rate "$LEARNING_RATE"
  --target_modules all-linear
  --dataset "$PILOT_DATA"
  --split_dataset_ratio 0
  --attn_impl flash_attn
  --padding_free true
  --torch_dtype bfloat16
  --load_from_cache_file false
  --eval_strategy no
  --save_strategy steps
  --save_steps "$MAX_STEPS"
  --save_total_limit 1
  --logging_steps 1
  --max_steps "$MAX_STEPS"
  --max_length 4096
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --gradient_checkpointing true
  --dataloader_num_workers 4
  --dataset_num_proc 4
  --warmup_ratio 0.05
  --seed "$SEED"
  --data_seed "$SEED"
  --report_to none
)

echo "starting paired 100-query Pilot from $MODEL"
echo "score-policy GPU=$POLICY_GPU listwise-control GPU=$CONTROL_GPU steps=$MAX_STEPS lr=$LEARNING_RATE"

CUDA_VISIBLE_DEVICES=$POLICY_UUID \
SCORE_POLICY_GRPO_STATS_PATH=$POLICY_OUTPUT/rollout_stats \
swift sft \
  --external_plugins "$PLUGIN" \
  --loss_type score_policy_grpo \
  --output_dir "$POLICY_OUTPUT/model" \
  "${COMMON_ARGS[@]}" \
  > "$POLICY_OUTPUT/train.log" 2>&1 &
POLICY_PID=$!

CUDA_VISIBLE_DEVICES=$CONTROL_UUID \
swift sft \
  --loss_type listwise_reranker \
  --output_dir "$CONTROL_OUTPUT/model" \
  "${COMMON_ARGS[@]}" \
  > "$CONTROL_OUTPUT/train.log" 2>&1 &
CONTROL_PID=$!

set +e
wait "$POLICY_PID"
POLICY_STATUS=$?
wait "$CONTROL_PID"
CONTROL_STATUS=$?
set -e
if test "$POLICY_STATUS" -ne 0 || test "$CONTROL_STATUS" -ne 0; then
  echo "Pilot failed: score-policy status=$POLICY_STATUS control status=$CONTROL_STATUS" >&2
  echo "inspect $POLICY_OUTPUT/train.log and $CONTROL_OUTPUT/train.log" >&2
  exit 3
fi

mapfile -t STATS_FILES < <(find "$POLICY_OUTPUT" -maxdepth 1 -name 'rollout_stats.rank*.jsonl' -print | sort)
test "${#STATS_FILES[@]}" -gt 0 || { echo "missing rollout statistics" >&2; exit 3; }
PYTHONPATH=$ROOT/scripts python "$ROOT/scripts/summarize_score_policy_grpo_stats.py" \
  "${STATS_FILES[@]}" \
  --output "$POLICY_OUTPUT/rollout_summary.json"

python - "$POLICY_OUTPUT/rollout_summary.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["all_losses_finite"], result
assert result["passes_training_signal_gate"], result
PY

echo "paired score-policy/listwise Pilot complete: $OUTPUT_DIR"
echo "next: merge both adapters and compare only on the fixed item-disjoint development set"

