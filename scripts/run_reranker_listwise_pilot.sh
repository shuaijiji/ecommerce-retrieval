#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/code/ecommerce-retrieval}
DATA_DIR=${2:-$ROOT/artifacts/train_v0/reranker_sft_smoke_40k}
OUTPUT_DIR=${3:-$ROOT/artifacts/train_v0/reranker_listwise_pilot_v1}
MODEL=${ECOM_RERANK_MODEL:-$ROOT/models/Qwen3-VL-Reranker-2B}
PILOT_GPU=${PILOT_GPU:-0}
MAX_STEPS=${MAX_STEPS:-50}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

TRAIN_DATA=$DATA_DIR/swift_train.jsonl
VAL_DATA=$DATA_DIR/swift_smoke_eval.jsonl
MANIFEST=$DATA_DIR/manifest.json

test -d "$MODEL" || { echo "missing model directory: $MODEL" >&2; exit 2; }
test -f "$TRAIN_DATA" || { echo "missing training data: $TRAIN_DATA" >&2; exit 2; }
test -f "$VAL_DATA" || { echo "missing validation data: $VAL_DATA" >&2; exit 2; }
test -f "$MANIFEST" || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
command -v swift >/dev/null || { echo "swift is not installed in the active environment" >&2; exit 2; }

python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m.get("complete") is True; assert m.get("effective_pointwise_pairs")==40000; assert m.get("rows",{}).get("swift_train")==4500; assert m.get("rows",{}).get("swift_smoke_eval")==500' "$MANIFEST"

if test -e "$OUTPUT_DIR"; then
  echo "output already exists; use a new directory: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

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

echo "objective=listwise_reranker"
echo "model=$MODEL"
echo "train_data=$TRAIN_DATA"
echo "val_data=$VAL_DATA"
echo "output_dir=$OUTPUT_DIR"
echo "gpu=$PILOT_GPU max_steps=$MAX_STEPS learning_rate=$LEARNING_RATE"
python -V
python -c 'import torch,transformers,swift,flash_attn; print("torch",torch.__version__,"cuda",torch.version.cuda); print("transformers",transformers.__version__); print("swift",swift.__version__); print("flash_attn",flash_attn.__version__); print("cuda_available",torch.cuda.is_available())'

CUDA_VISIBLE_DEVICES=$PILOT_GPU swift sft \
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
  --val_dataset "$VAL_DATA" \
  --split_dataset_ratio 0 \
  --attn_impl flash_attn \
  --padding_free true \
  --torch_dtype bfloat16 \
  --load_from_cache_file false \
  --eval_strategy steps \
  --eval_on_start true \
  --output_dir "$OUTPUT_DIR" \
  --save_steps "$MAX_STEPS" \
  --eval_steps 25 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --max_steps "$MAX_STEPS" \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --warmup_ratio 0.05 \
  --report_to none

echo "reranker listwise single-GPU pilot complete: $OUTPUT_DIR"
