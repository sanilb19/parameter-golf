#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <user@host> [remote_dir]" >&2
  exit 1
fi

HOST="$1"
REMOTE_DIR="${2:-/workspace/parameter-golf}"
ROOT="/Users/sanilbaweja/Projects/parameter-golf"

mkdir -p "$ROOT/logs"
rsync -az "$HOST:$REMOTE_DIR/logs/runpod_first_h100/" "$ROOT/logs/runpod_first_h100/"

echo "fetched $HOST:$REMOTE_DIR/logs/runpod_first_h100 -> $ROOT/logs/runpod_first_h100"
