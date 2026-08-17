#!/usr/bin/env bash
# Fetch the MI-GAN ONNX inpainting model used for local, zero-spend overlay removal.
# Without it the bot still works but overlay removal fails after detection
# (see src/local_inpaint.py) since Gemini's image-generation models no longer
# have a free tier.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MIGAN_MODEL_PATH:-$REPO_ROOT/models/migan_pipeline_v2.onnx}"
MODEL_URL="https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx"

if [ -s "$MODEL_PATH" ]; then
    echo "MI-GAN model already present: $MODEL_PATH"
    exit 0
fi

mkdir -p "$(dirname "$MODEL_PATH")"
echo "Downloading MI-GAN model to $MODEL_PATH ..."
curl -fsSL -o "$MODEL_PATH" "$MODEL_URL"
echo "Done."
