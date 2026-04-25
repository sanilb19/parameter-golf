#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sanilbaweja/Projects/parameter-golf"
KEY="/Users/sanilbaweja/.runpod/ssh/RunPod-Key-Go"
NTFY_SCRIPT="$ROOT/scripts/ntfy_notify.sh"
RUNPODCTL="${RUNPODCTL:-/opt/homebrew/bin/runpodctl}"

if [[ $# -lt 11 ]]; then
  echo "usage: $0 <pod_id> <ip> <port> <remote_campaign_dir> <local_fetch_dir> <batch_name> <run_id> <category> <description> <parent_recipe> <decision> [ENV=VALUE ...]" >&2
  exit 1
fi

POD_ID="$1"
IP="$2"
PORT="$3"
REMOTE_CAMPAIGN_DIR="$4"
LOCAL_FETCH_DIR="$5"
BATCH_NAME="$6"
RUN_ID="$7"
CATEGORY="$8"
DESCRIPTION="$9"
PARENT_RECIPE="${10}"
DECISION="${11}"
shift 11

mkdir -p "$LOCAL_FETCH_DIR"

RUN_STATUS=255
FETCHED=no
REMOTE_ARG_FILE="/tmp/${RUN_ID}.args.sh"
LOCAL_ARG_FILE="$(mktemp "${TMPDIR:-/tmp}/runpod-h100-args.XXXXXX")"

cleanup() {
  local stop_status=0
  rsync -az --no-owner --no-group \
    -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "root@$IP:$REMOTE_CAMPAIGN_DIR/" "$LOCAL_FETCH_DIR/" >/dev/null 2>&1 && FETCHED=yes || FETCHED=no
  "$RUNPODCTL" pod stop "$POD_ID" >/dev/null 2>&1 || stop_status=$?
  "$NTFY_SCRIPT" "H100 run finished and pod stopped" "run_id=$RUN_ID pod_id=$POD_ID exit_status=$RUN_STATUS fetched=$FETCHED stop_status=$stop_status" || true
  rm -f "$LOCAL_ARG_FILE"
}
trap cleanup EXIT

{
  printf 'REMOTE_CAMPAIGN_DIR=%q\n' "$REMOTE_CAMPAIGN_DIR"
  printf 'BATCH_NAME=%q\n' "$BATCH_NAME"
  printf 'RUN_ID=%q\n' "$RUN_ID"
  printf 'CATEGORY=%q\n' "$CATEGORY"
  printf 'DESCRIPTION=%q\n' "$DESCRIPTION"
  printf 'PARENT_RECIPE=%q\n' "$PARENT_RECIPE"
  printf 'DECISION=%q\n' "$DECISION"
  printf 'ENV_ARGS=('
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf ')\n'
} >"$LOCAL_ARG_FILE"

scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$LOCAL_ARG_FILE" "root@$IP:$REMOTE_ARG_FILE" >/dev/null

set +e
ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$IP" \
  "bash -s" -- "$REMOTE_ARG_FILE" <<'EOS'
set -euo pipefail
REMOTE_ARG_FILE="$1"
source "$REMOTE_ARG_FILE"

cd /workspace/parameter-golf
chmod +x scripts/*.sh scripts/*.py
rm -rf "$REMOTE_CAMPAIGN_DIR"
mkdir -p "$REMOTE_CAMPAIGN_DIR"
CAMPAIGN_DIR="$REMOTE_CAMPAIGN_DIR" BATCH_NAME="$BATCH_NAME" bash scripts/run_h100_campaign_case.sh \
  "$RUN_ID" "$CATEGORY" "$DESCRIPTION" "$PARENT_RECIPE" "$DECISION" "${ENV_ARGS[@]}"
EOS
RUN_STATUS=$?
set -e

exit "$RUN_STATUS"
