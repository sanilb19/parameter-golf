#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <pod_id> <delay_minutes> [reason]" >&2
  exit 1
fi

POD_ID="$1"
DELAY_MINUTES="$2"
REASON="${3:-batch_failsafe}"

ROOT="/Users/sanilbaweja/Projects/parameter-golf"
LOG_DIR="$ROOT/logs/h100_campaign_20260421"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_DIR/stop_${POD_ID}_${STAMP}.log"
PID_PATH="$LOG_DIR/stop_${POD_ID}_${STAMP}.pid"

SECONDS_TOTAL=$(( DELAY_MINUTES * 60 ))

nohup bash -lc "
  echo \$\$ > '$PID_PATH'
  echo '[\$(date -u +%FT%TZ)] scheduled_stop_start pod_id=$POD_ID delay_minutes=$DELAY_MINUTES reason=$REASON' >> '$LOG_PATH'
  sleep $SECONDS_TOTAL
  echo '[\$(date -u +%FT%TZ)] scheduled_stop_fire pod_id=$POD_ID reason=$REASON' >> '$LOG_PATH'
  runpodctl pod stop '$POD_ID' >> '$LOG_PATH' 2>&1
" >/dev/null 2>&1 &

echo "scheduled pod stop for $POD_ID in $DELAY_MINUTES minutes"
echo "pid file: $PID_PATH"
echo "log file: $LOG_PATH"
