#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sanilbaweja/Projects/parameter-golf"
NTFY_SCRIPT="$ROOT/scripts/ntfy_notify.sh"
KEY="/Users/sanilbaweja/.runpod/ssh/RunPod-Key-Go"

if [[ $# -lt 4 || $# -gt 8 ]]; then
  echo "usage: $0 <pod_id> <ip> <port> <remote_pid_path> [remote_fetch_dir] [local_fetch_dir] [label] [poll_seconds]" >&2
  exit 1
fi

POD_ID="$1"
IP="$2"
PORT="$3"
REMOTE_PID_PATH="$4"
REMOTE_FETCH_DIR="${5:-}"
LOCAL_FETCH_DIR="${6:-}"
LABEL="${7:-remote_run}"
POLL_SECONDS="${8:-15}"

while true; do
  if ! ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    root@"$IP" "bash -lc 'test -f \"$REMOTE_PID_PATH\" && kill -0 \$(cat \"$REMOTE_PID_PATH\")'" >/dev/null 2>&1; then
    FETCHED="no"
    if [[ -n "$REMOTE_FETCH_DIR" && -n "$LOCAL_FETCH_DIR" ]]; then
      mkdir -p "$LOCAL_FETCH_DIR"
      rsync -az --no-owner --no-group -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
        "root@$IP:$REMOTE_FETCH_DIR/" "$LOCAL_FETCH_DIR/" >/dev/null
      FETCHED="yes"
    fi
    runpodctl pod stop "$POD_ID" >/dev/null
    "$NTFY_SCRIPT" "H100 pod auto-stopped" "label=$LABEL pod_id=$POD_ID stop_reason=remote_run_complete fetched=$FETCHED"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
