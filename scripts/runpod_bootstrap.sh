#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/parameter-golf}"
DATA_VARIANT="${DATA_VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-}"
PERSIST_MOUNT="${PERSIST_MOUNT:-/workspace/persist}"
PERSIST_CACHE_ROOT="${PERSIST_CACHE_ROOT:-$PERSIST_MOUNT/parameter-golf-cache}"
USE_PERSIST_CACHE="${USE_PERSIST_CACHE:-auto}"

cd "$ROOT"

echo "[bootstrap] root=$ROOT"
python3 -V
python3 - <<'PY'
import torch
print(f"[bootstrap] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[bootstrap] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[bootstrap] nvidia-smi not available; continuing data setup only"
fi

use_persist=0
if [[ "$USE_PERSIST_CACHE" == "1" || "$USE_PERSIST_CACHE" == "true" ]]; then
  use_persist=1
elif [[ "$USE_PERSIST_CACHE" == "auto" && -d "$PERSIST_MOUNT" && -w "$PERSIST_MOUNT" ]]; then
  use_persist=1
fi

if [[ "$use_persist" == "1" ]]; then
  echo "[bootstrap] using persistent cache at $PERSIST_CACHE_ROOT"
  mkdir -p "$PERSIST_CACHE_ROOT/data/datasets" "$PERSIST_CACHE_ROOT/data/tokenizers" "$PERSIST_CACHE_ROOT/hf-cache"
  export HF_HOME="$PERSIST_CACHE_ROOT/hf-cache"

  mkdir -p "$ROOT/data"
  if [[ -d "$ROOT/data/datasets" && ! -L "$ROOT/data/datasets" && ! -e "$PERSIST_CACHE_ROOT/data/datasets/.seeded_from_repo" ]]; then
    rsync -a "$ROOT/data/datasets/" "$PERSIST_CACHE_ROOT/data/datasets/" 2>/dev/null || true
    touch "$PERSIST_CACHE_ROOT/data/datasets/.seeded_from_repo"
  fi
  if [[ -d "$ROOT/data/tokenizers" && ! -L "$ROOT/data/tokenizers" && ! -e "$PERSIST_CACHE_ROOT/data/tokenizers/.seeded_from_repo" ]]; then
    rsync -a "$ROOT/data/tokenizers/" "$PERSIST_CACHE_ROOT/data/tokenizers/" 2>/dev/null || true
    touch "$PERSIST_CACHE_ROOT/data/tokenizers/.seeded_from_repo"
  fi
  rm -rf "$ROOT/data/datasets" "$ROOT/data/tokenizers"
  ln -s "$PERSIST_CACHE_ROOT/data/datasets" "$ROOT/data/datasets"
  ln -s "$PERSIST_CACHE_ROOT/data/tokenizers" "$ROOT/data/tokenizers"
else
  echo "[bootstrap] persistent cache disabled or unavailable"
fi

if [[ -n "$TRAIN_SHARDS" ]]; then
  python3 data/cached_challenge_fineweb.py --variant "$DATA_VARIANT" --train-shards "$TRAIN_SHARDS"
else
  python3 data/cached_challenge_fineweb.py --variant "$DATA_VARIANT"
fi

if [[ "$use_persist" == "1" ]]; then
  find "$PERSIST_CACHE_ROOT/data" -maxdepth 3 -type f | wc -l | awk '{print "[bootstrap] persistent data files:" $1}'
fi
echo "[bootstrap] data ready"
