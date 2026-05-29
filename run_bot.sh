#!/usr/bin/env bash
#
# Simple runner script
# Usage:
#   ./run_bot.sh                    # legacy Telegram bot (OpenRouter)
#   ./run_bot.sh --gemini <images>  # Gemini 2.5 free tool (recommended)
#

set -euo pipefail

# Go to script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists (recommended)
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating venv..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found at .venv or venv"
fi

# Load environment variables from .env if present
if [ -f ".env" ]; then
    echo "Loading .env..."
    set -a
    source .env
    set +a
fi

echo "Starting Mr Prompter Bot..."

if [[ "${1:-}" == "--gemini" ]]; then
    shift
    echo "Running Gemini 2.5 free tool..."
    exec python scripts/gemini_25_free_watermark_remover.py "$@"
else
    exec python -m src.main
fi
