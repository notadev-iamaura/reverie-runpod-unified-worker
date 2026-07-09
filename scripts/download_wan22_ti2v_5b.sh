#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${WAN22_MODEL_DIR:-/runpod-volume/models/Wan2.2-TI2V-5B}"

mkdir -p "$(dirname "$MODEL_DIR")"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$MODEL_DIR"

echo "Wan2.2 TI2V-5B model ready at $MODEL_DIR"
