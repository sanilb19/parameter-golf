#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sanilbaweja/Projects/parameter-golf"
STATE_PATH="$ROOT/.research/runpod_h100_campaign_state.json"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <run_id> [log_relpath]" >&2
  exit 1
fi

RUN_ID="$1"
LOG_RELPATH="${2:-logs/h100_campaign_20260421/${RUN_ID}.log}"

STATE_RAW="$(
  python3 - <<'PY' "$STATE_PATH"
import json, sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text())
pod = state["runpod"]["active_pod"]
print(pod["ip"])
print(pod["ssh_port"])
PY
)"

IP="$(printf '%s\n' "$STATE_RAW" | sed -n '1p')"
PORT="$(printf '%s\n' "$STATE_RAW" | sed -n '2p')"
KEY="/Users/sanilbaweja/.runpod/ssh/RunPod-Key-Go"

ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no root@"$IP" \
  "RUN_ID='$RUN_ID' LOG_RELPATH='$LOG_RELPATH' python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path

run_id = os.environ['RUN_ID']
log_path = Path('/workspace/parameter-golf') / os.environ['LOG_RELPATH']
jsonl_path = Path('/workspace/parameter-golf/logs/h100_campaign_20260421/experiments_master.jsonl')

ps = subprocess.run(
    \"ps -ef | grep -E 'train_gpt.py|torchrun' | grep -v grep\",
    shell=True,
    capture_output=True,
    text=True,
)
process_lines = [line for line in ps.stdout.splitlines() if line.strip()]

last_record = None
if jsonl_path.exists():
    for line in jsonl_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('run_id') == run_id:
            last_record = row

tail_lines = []
if log_path.exists():
    tail_lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-8:]

out = {
    'run_id': run_id,
    'remote_log_path': str(log_path),
    'log_exists': log_path.exists(),
    'active_train_processes': process_lines,
    'latest_record': last_record,
    'tail': tail_lines,
}
print(json.dumps(out, indent=2))
PY"
