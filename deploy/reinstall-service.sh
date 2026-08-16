#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

sudo cp "$SCRIPT_DIR/mr-prompter-bot.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart mr-prompter-bot
sudo systemctl status mr-prompter-bot --no-pager
