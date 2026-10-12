CONFIG_NAME=pixelmoe_l_e32a4_256

MODEL_ARGS=(
  --model PixelMoE-L-E32A4/8 --img_size 256
  --noise_scale 1.0
)

TRAIN_ARGS=(
  --batch_size 64
)

SAMPLE_ARGS=(
  --sigma_min 0.03
  --cfg 3.2 --interval_max 0.8
)
