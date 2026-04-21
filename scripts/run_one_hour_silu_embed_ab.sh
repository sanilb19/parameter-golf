#!/bin/zsh
set -euo pipefail

cd /Users/sanilbaweja/Projects/parameter-golf
PYTHON_BIN="/Users/sanilbaweja/Projects/parameter-golf/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing venv python: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" /Users/sanilbaweja/Projects/parameter-golf/one_hour_silu_embed_ab.py
