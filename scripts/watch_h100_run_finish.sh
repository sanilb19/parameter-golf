#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sanilbaweja/Projects/parameter-golf"
STATE_PATH="$ROOT/.research/runpod_h100_campaign_state.json"
NTFY_SCRIPT="$ROOT/scripts/ntfy_notify.sh"
POLL_SCRIPT="$ROOT/scripts/poll_h100_run_status.sh"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <run_id> [poll_seconds]" >&2
  exit 1
fi

RUN_ID="$1"
POLL_SECONDS="${2:-30}"
WATCH_DIR="$ROOT/logs/h100_campaign_20260421"
mkdir -p "$WATCH_DIR"
WATCH_LOG="$WATCH_DIR/watch_${RUN_ID}.log"
PID_FILE="$WATCH_DIR/watch_${RUN_ID}.pid"

echo $$ > "$PID_FILE"
echo \"[$(date -u +%FT%TZ)] watch_start run_id=$RUN_ID poll_seconds=$POLL_SECONDS\" >> "$WATCH_LOG"

cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT

while true; do
  STATUS_JSON="$("$POLL_SCRIPT" "$RUN_ID")"
  printf '[%s] poll %s\n' "$(date -u +%FT%TZ)" "$STATUS_JSON" >> "$WATCH_LOG"

  LATEST_RUN_ID="$(python3 - <<'PY' "$STATUS_JSON"
import json, sys
obj = json.loads(sys.argv[1])
record = obj.get("latest_record") or {}
print(record.get("run_id", ""))
PY
)"

  ACTIVE_COUNT="$(python3 - <<'PY' "$STATUS_JSON"
import json, sys
obj = json.loads(sys.argv[1])
print(len(obj.get("active_train_processes") or []))
PY
)"

  if [[ "$LATEST_RUN_ID" == "$RUN_ID" && "$ACTIVE_COUNT" == "0" ]]; then
    SUMMARY="$(python3 - <<'PY' "$STATUS_JSON"
import json, sys
obj = json.loads(sys.argv[1])
record = obj.get("latest_record") or {}
parts = [
    f"run_id={record.get('run_id','')}",
    f"status={record.get('status','')}",
    f"val_bpb={record.get('val_bpb','')}",
    f"val_loss={record.get('val_loss','')}",
    f"step_avg_ms={record.get('step_avg_ms','')}",
]
print(" ".join(parts))
PY
)"
    "$NTFY_SCRIPT" "H100 run finished" "$SUMMARY"
    echo "[$(date -u +%FT%TZ)] watch_done $SUMMARY" >> "$WATCH_LOG"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
