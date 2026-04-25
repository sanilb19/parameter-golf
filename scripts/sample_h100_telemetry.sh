#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <run_id> <out_dir> [interval_seconds]" >&2
  exit 1
fi

RUN_ID="$1"
OUT_DIR="$2"
INTERVAL="${3:-10}"

mkdir -p "$OUT_DIR"

GPU_CSV="$OUT_DIR/${RUN_ID}.nvidia_smi.csv"
PROC_LOG="$OUT_DIR/${RUN_ID}.proc_snapshots.log"
META_JSON="$OUT_DIR/${RUN_ID}.telemetry_meta.json"

python3 - <<'PY' "$META_JSON" "$RUN_ID" "$INTERVAL"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
run_id = sys.argv[2]
interval = int(sys.argv[3])
path.write_text(
    json.dumps(
        {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "sample_interval_seconds": interval,
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "timestamp_utc,index,name,utilization_gpu_pct,utilization_memory_pct,memory_used_mb,memory_total_mb,power_w,temperature_c" > "$GPU_CSV"

while true; do
  TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  nvidia-smi \
    --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits \
    | sed "s/^/${TS},/" >> "$GPU_CSV"

  {
    echo "[$TS] top"
    COLUMNS=200 top -b -n 1 | sed -n '1,25p'
    echo
    echo "[$TS] ps"
    ps -eo pid,ppid,%cpu,%mem,state,comm,args --sort=-%cpu | sed -n '1,20p'
    echo
  } >> "$PROC_LOG"

  sleep "$INTERVAL"
done
