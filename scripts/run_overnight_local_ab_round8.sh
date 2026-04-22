#!/bin/zsh
set -euo pipefail

cd /Users/sanilbaweja/Projects/parameter-golf
PYTHON_BIN="/Users/sanilbaweja/Projects/parameter-golf/.venv/bin/python"
LOG_DIR="/Users/sanilbaweja/Projects/parameter-golf/logs/overnight_local_ab_round8"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing venv python: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

"$PYTHON_BIN" /Users/sanilbaweja/Projects/parameter-golf/overnight_local_ab_round8.py >> "$LOG_DIR/launcher.log" 2>&1
