CONFIG_NAME=pixelmoe_l_e32a4_512

MODEL_ARGS=(
  --model PixelMoE-L-E32A4/16 --img_size 512
  --noise_scale 2.0
)

TRAIN_ARGS=(
  --batch_size 64
)

SAMPLE_ARGS=(
  --sigma_min 0.03
  --cfg 3.8 --interval_max 0.8
)
