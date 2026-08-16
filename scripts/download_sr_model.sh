#!/usr/bin/env bash
# Fetch the Real-ESRGAN compact ONNX model used for auto-Full-HD upscaling.
# Without it the bot still works and falls back to Lanczos resizing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${SR_MODEL_PATH:-$REPO_ROOT/models/realesr-general-x4v3.onnx}"
MODEL_URL="https://huggingface.co/tamnvcc/Real-ESRGAN-General-x4v3_float/resolve/main/onnx/model.onnx"

if [ -s "$MODEL_PATH" ]; then
    echo "SR model already present: $MODEL_PATH"
    exit 0
fi

mkdir -p "$(dirname "$MODEL_PATH")"
echo "Downloading SR model to $MODEL_PATH ..."
curl -fsSL -o "$MODEL_PATH" "$MODEL_URL"
echo "Done."
