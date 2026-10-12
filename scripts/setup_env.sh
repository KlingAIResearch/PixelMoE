#!/usr/bin/env bash
set -e

conda env create -f environment.yaml
MAX_JOBS=4 conda run -n pixelmoe \
  python -m pip install flash-attn==2.8.3 --no-build-isolation --no-deps
