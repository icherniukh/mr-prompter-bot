#!/usr/bin/env bash
set -euo pipefail
sudo systemctl restart mr-prompter-bot
sudo systemctl status mr-prompter-bot --no-pager
