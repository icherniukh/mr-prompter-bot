#!/usr/bin/env bash
#
# Simple runner script for Mr Prompter Bot
# Usage: ./run_bot.sh
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
exec python -m src.main
