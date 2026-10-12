CONFIG_NAME=pixelmoe_h_e32a4_512

MODEL_ARGS=(
  --model PixelMoE-H-E32A4/16 --img_size 512
  --noise_scale 2.0
)

TRAIN_ARGS=(
  --proj_dropout 0.2 --flux_attn_proj_dropout
  --batch_size 32
  --random_crop_scales 1.10 --data_noise_std 0.005 --regularization_start_epoch 200
)

SAMPLE_ARGS=(
  --sigma_min 0.04
  --cfg 3.7 --interval_max 0.8
  --eval_use_second_ema
)
