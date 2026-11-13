#!/usr/bin/env bash
set -euo pipefail

ROOT=\${1:-/code/ecommerce-retrieval}
DATA_DIR=\${2:-$ROOT/artifacts/train_v0/reranker_sft_smoke_pilot}
OUTPUT_DIR=\${3:-$ROOT/artifacts/train_v0/reranker_sft_lora_pilot}
MODEL=\${ECOM_RERANK_MODEL:-$ROOT/models/Qwen3-VL-Reranker-2B}
PILOT_GPU=\${PILOT_GPU:-0}

TRAIN_DATA=$DATA_DIR/swift_train.jsonl
VAL_DATA=$DATA_DIR/swift_smoke_eval.jsonl

test -d "$MODEL"
test -f "$TRAIN_DATA"
test -f "$VAL_DATA"
test -f "$DATA_DIR/manifest.json"
command -v swift >/dev/null

python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m.get("complete") is True; assert m.get("effective_pointwise_pairs")==800' "$DATA_DIR/manifest.json"

if test -e "$OUTPUT_DIR"; then
  echo "output already exists; use a new directory: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_POSITIVE_SAMPLES=1
export MAX_NEGATIVE_SAMPLES=7
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=\${MKL_NUM_THREADS:-8}

echo "model=$MODEL"
echo "train_data=$TRAIN_DATA"
echo "val_data=$VAL_DATA"
echo "output_dir=$OUTPUT_DIR"
echo "gpu=$PILOT_GPU"
python -V
python -c 'import torch, transformers, swift; print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__); print("swift", swift.__version__); print("cuda_available", torch.cuda.is_available())'

CUDA_VISIBLE_DEVICES=$PILOT_GPU swift sft \
  --model "$MODEL" \
  --task_type generative_reranker \
  --loss_type pointwise_reranker \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --learning_rate 5e-5 \
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
  --save_steps 10 \
  --eval_steps 5 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --max_steps 10 \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --warmup_ratio 0 \
  --report_to none

echo "reranker SFT single-GPU pilot complete: $OUTPUT_DIR"
