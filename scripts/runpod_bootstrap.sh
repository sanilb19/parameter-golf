#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/parameter-golf}"
DATA_VARIANT="${DATA_VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-}"

cd "$ROOT"

echo "[bootstrap] root=$ROOT"
python3 -V
python3 - <<'PY'
import torch
print(f"[bootstrap] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[bootstrap] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
PY
nvidia-smi

if [[ -n "$TRAIN_SHARDS" ]]; then
  python3 data/cached_challenge_fineweb.py --variant "$DATA_VARIANT" --train-shards "$TRAIN_SHARDS"
else
  python3 data/cached_challenge_fineweb.py --variant "$DATA_VARIANT"
fi

echo "[bootstrap] data ready"
