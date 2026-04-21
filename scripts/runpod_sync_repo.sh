#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <user@host> [remote_dir]" >&2
  exit 1
fi

HOST="$1"
REMOTE_DIR="${2:-/workspace/parameter-golf}"
ROOT="/Users/sanilbaweja/Projects/parameter-golf"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude 'logs/' \
  --exclude 'data/datasets/' \
  "$ROOT/" "$HOST:$REMOTE_DIR/"

echo "synced repo to $HOST:$REMOTE_DIR"
