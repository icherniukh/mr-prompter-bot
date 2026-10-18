#!/usr/bin/env bash
#
# Runner script for Mr Prompter Bot
#

set -euo pipefail

cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating venv..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found at .venv or venv"
fi

if [ -f ".env" ]; then
    echo "Loading .env..."
    set -a
    source .env
    set +a
fi

echo "Starting Mr Prompter Bot..."
exec python -m src.main
