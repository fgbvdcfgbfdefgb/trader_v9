#!/usr/bin/env bash
# Offline training entrypoint (Snowflake container / any machine with this repo).
# The repo already contains the dataset - only pip packages are fetched, then
# training runs with zero network access.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
# CPU-only torch (much smaller download) - comment out if you want the CUDA build:
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt

# Hardware is auto-detected (multi-GPU / single GPU / CPU) and models are sized
# accordingly. Override with e.g. --device cuda --hidden 512.
python3 train.py --epochs "${EPOCHS:-2000}" --device "${DEVICE:-auto}" --out "${OUT:-runs}"
