CONFIG_NAME=pixelmoe_b_e32a4_256

MODEL_ARGS=(
  --model PixelMoE-B-E32A4/8 --img_size 256
  --noise_scale 1.0
)

TRAIN_ARGS=(
  --batch_size 128
)

SAMPLE_ARGS=(
  --sigma_min 0.03
  --cfg 3.6 --interval_max 0.78
)
