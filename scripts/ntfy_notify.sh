#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <title> <message>" >&2
  exit 1
fi

TOPIC="${NTFY_TOPIC:-sb-parameter-golf-19}"
TITLE="$1"
MESSAGE="$2"

curl -fsS \
  -H "Title: ${TITLE}" \
  -H "Tags: warning" \
  -d "${MESSAGE}" \
  "https://ntfy.sh/${TOPIC}"
