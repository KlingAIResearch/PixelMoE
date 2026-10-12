#!/usr/bin/env bash
set -e

source "$1"
CHECKPOINT=$2
OUTPUT_DIR=$3
shift 3

torchrun --nproc_per_node="${GPUS:-8}" --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" main_pixelmoe.py \
  "${MODEL_ARGS[@]}" \
  --use_second_ema --use_flash_attn \
  --hierarchical_logit_mean 0.0 --hierarchical_logit_std 1.0 \
  --hierarchical_stage_logit_means 0.6 0.6 --hierarchical_stage_logit_stds 0.8 0.8 \
  --num_sampling_steps 25 25 --evaluate_gen --gen_bsz 640 \
  --data_path "${IMAGENET_PATH:-./data/imagenet}" \
  --output_dir "${OUTPUT_DIR}" --resume_ckpt "${CHECKPOINT}" \
  "${SAMPLE_ARGS[@]}" "$@"
