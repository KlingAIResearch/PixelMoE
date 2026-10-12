CONFIG_NAME=pixelmoe_b_e32a4_512

MODEL_ARGS=(
  --model PixelMoE-B-E32A4/16 --img_size 512
  --noise_scale 2.0
)

TRAIN_ARGS=(
  --batch_size 128
)

SAMPLE_ARGS=(
  --sigma_min 0.03
  --cfg 3.7 --interval_max 0.8
  --eval_use_second_ema
)
