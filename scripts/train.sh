#!/usr/bin/env bash
set -e

source "$1"
shift
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${CONFIG_NAME}}

if [[ "${STANDALONE:-0}" == "1" ]]; then
  TORCHRUN_ARGS=(--standalone --nnodes=1 --nproc_per_node="${GPUS:-8}")
else
  TORCHRUN_ARGS=(--nproc_per_node="${GPUS:-8}" --nnodes="${NNODES:-1}"
    --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}"
    --master_port="${MASTER_PORT:-29500}")
fi

torchrun "${TORCHRUN_ARGS[@]}" main_pixelmoe.py \
  "${MODEL_ARGS[@]}" \
  --epochs 321 --min_lr 2e-5 --lr_schedule cosine_tail --lr_decay_epochs 40 \
  --max_grad_norm 0.5 --warmup_max_grad_norm 5.0 --grad_clip_start_epoch 1 \
  --skip_grad_norm_threshold 0.5 --skip_grad_norm_start_epoch 10 \
  --max_skipped_grad_steps_per_epoch 10 \
  --use_second_ema --sigma_min 0.05 --use_flash_attn \
  --hierarchical_stage_logit_means 0.6 0.6 --hierarchical_stage_logit_stds 0.8 0.8 \
  --hierarchical_min_stage_sigma 0.04 --num_workers 16 \
  --data_path "${IMAGENET_PATH}" \
  --output_dir "${OUTPUT_DIR}" --resume "${OUTPUT_DIR}" --save_last_freq 1 \
  "${TRAIN_ARGS[@]}" "$@"
