#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run as a normal user with sudo privileges, not root."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip mongodb

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Bootstrap complete. Copy .env.example to .env and add API keys."
