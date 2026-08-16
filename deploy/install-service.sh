#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/../scripts/download_sr_model.sh"

sudo cp "$SCRIPT_DIR/mr-prompter-bot.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mr-prompter-bot
sudo systemctl start mr-prompter-bot
sudo systemctl status mr-prompter-bot --no-pager
