#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <run_id> <category> <description> <parent_recipe> <decision> [ENV=VALUE ...]" >&2
  exit 1
fi

ROOT="${ROOT:-/workspace/parameter-golf}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-$ROOT/logs/h100_campaign_20260421}"
DATA_PATH="${DATA_PATH:-$ROOT/data/datasets/fineweb10B_sp1024/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/data/tokenizers/fineweb_1024_bpe.model}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PRIORITY="${PRIORITY:-1}"
STATUS_DEFAULT="${STATUS_DEFAULT:-ok}"
BATCH_NAME="${BATCH_NAME:-phase_0_noise_estimation}"
TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-10}"

RUN_ID="$1"
CATEGORY="$2"
DESCRIPTION="$3"
PARENT_RECIPE="$4"
DECISION="$5"
shift 5

mkdir -p "$CAMPAIGN_DIR"
LOG_PATH="$CAMPAIGN_DIR/${RUN_ID}.log"
JSONL_PATH="$CAMPAIGN_DIR/experiments_master.jsonl"
GPU_TELEMETRY_CSV="$CAMPAIGN_DIR/${RUN_ID}.nvidia_smi.csv"
PROC_SNAPSHOT_LOG="$CAMPAIGN_DIR/${RUN_ID}.proc_snapshots.log"
TELEMETRY_META_JSON="$CAMPAIGN_DIR/${RUN_ID}.telemetry_meta.json"

cd "$ROOT"

STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

TELEMETRY_PID=""
cleanup() {
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$ROOT/scripts/sample_h100_telemetry.sh" "$RUN_ID" "$CAMPAIGN_DIR" "$TELEMETRY_INTERVAL_SECONDS" &
TELEMETRY_PID=$!

set +e
env \
  RUN_ID="$RUN_ID" \
  DATA_PATH="$DATA_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  VOCAB_SIZE=1024 \
  TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
  VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  "$@" \
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py >"$LOG_PATH" 2>&1
EXIT_CODE=$?
set -e

FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

python3 - "$ROOT" "$LOG_PATH" "$JSONL_PATH" "$RUN_ID" "$CATEGORY" "$DESCRIPTION" "$PARENT_RECIPE" "$DECISION" "$PRIORITY" "$STATUS_DEFAULT" "$BATCH_NAME" "$STARTED_AT" "$FINISHED_AT" "$EXIT_CODE" "$GPU_TELEMETRY_CSV" "$PROC_SNAPSHOT_LOG" "$TELEMETRY_META_JSON" "$NPROC_PER_NODE" "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
log_path = Path(sys.argv[2])
jsonl_path = Path(sys.argv[3])
run_id = sys.argv[4]
category = sys.argv[5]
description = sys.argv[6]
parent_recipe = sys.argv[7]
decision = sys.argv[8]
priority = int(sys.argv[9])
status_default = sys.argv[10]
batch_name = sys.argv[11]
started_at = sys.argv[12]
finished_at = sys.argv[13]
exit_code = int(sys.argv[14])
gpu_telemetry_csv = sys.argv[15]
proc_snapshot_log = sys.argv[16]
telemetry_meta_json = sys.argv[17]
nproc_per_node = int(sys.argv[18])
env_pairs = sys.argv[19:]

sys.path.insert(0, str((root / "scripts").resolve()))
from parse_train_log import parse_train_log

parsed = parse_train_log(log_path)
env_overrides = {}
for pair in env_pairs:
    if "=" in pair:
        key, value = pair.split("=", 1)
        env_overrides[key] = value

status = parsed.get("status") or status_default
if exit_code != 0:
    status = "crash"

telemetry_summary = {}
gpu_rows = []
by_gpu_index = {}
gpu_csv_path = Path(gpu_telemetry_csv)
if gpu_csv_path.exists():
    lines = gpu_csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            continue
        try:
            row = {
                "index": parts[1],
                "utilization_gpu_pct": float(parts[3]),
                "utilization_memory_pct": float(parts[4]),
                "memory_used_mb": float(parts[5]),
                "memory_total_mb": float(parts[6]),
                "power_w": float(parts[7]),
                "temperature_c": float(parts[8]),
            }
            gpu_rows.append(row)
            by_gpu_index.setdefault(parts[1], []).append(row)
        except ValueError:
            continue

if gpu_rows:
    vram_used_pcts = [
        (r["memory_used_mb"] / r["memory_total_mb"]) * 100.0
        for r in gpu_rows
        if r["memory_total_mb"] > 0
    ]
    telemetry_summary = {
        "gpu_util_mean_pct": round(sum(r["utilization_gpu_pct"] for r in gpu_rows) / len(gpu_rows), 2),
        "gpu_util_max_pct": round(max(r["utilization_gpu_pct"] for r in gpu_rows), 2),
        "memory_controller_util_mean_pct": round(
            sum(r["utilization_memory_pct"] for r in gpu_rows) / len(gpu_rows), 2
        ),
        "memory_controller_util_max_pct": round(max(r["utilization_memory_pct"] for r in gpu_rows), 2),
        "gpu_mem_used_max_mb": round(max(r["memory_used_mb"] for r in gpu_rows), 2),
        "gpu_mem_total_mb": round(max(r["memory_total_mb"] for r in gpu_rows), 2),
        "power_mean_w": round(sum(r["power_w"] for r in gpu_rows) / len(gpu_rows), 2),
        "power_max_w": round(max(r["power_w"] for r in gpu_rows), 2),
        "temperature_max_c": round(max(r["temperature_c"] for r in gpu_rows), 2),
        "telemetry_samples": len(gpu_rows),
    }
    if vram_used_pcts:
        telemetry_summary["vram_used_mean_pct"] = round(sum(vram_used_pcts) / len(vram_used_pcts), 2)
        telemetry_summary["vram_used_max_pct"] = round(max(vram_used_pcts), 2)
    if by_gpu_index:
        per_gpu_util_means = [
            sum(item["utilization_gpu_pct"] for item in items) / len(items)
            for items in by_gpu_index.values()
        ]
        per_gpu_vram_means = [
            sum((item["memory_used_mb"] / item["memory_total_mb"]) * 100.0 for item in items if item["memory_total_mb"] > 0)
            / len([item for item in items if item["memory_total_mb"] > 0])
            for items in by_gpu_index.values()
            if any(item["memory_total_mb"] > 0 for item in items)
        ]
        telemetry_summary["gpu_count"] = len(by_gpu_index)
        telemetry_summary["gpu_util_device_mean_min_pct"] = round(min(per_gpu_util_means), 2)
        telemetry_summary["gpu_util_device_mean_max_pct"] = round(max(per_gpu_util_means), 2)
        if per_gpu_vram_means:
            telemetry_summary["vram_used_device_mean_min_pct"] = round(min(per_gpu_vram_means), 2)
            telemetry_summary["vram_used_device_mean_max_pct"] = round(max(per_gpu_vram_means), 2)

existing_rows = []
if jsonl_path.exists():
    existing_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

world_size = parsed.get("world_size") or 1
train_batch_tokens = parsed.get("train_batch_tokens")
step_avg_ms = parsed.get("step_avg_ms")
tok_s = None
tok_s_per_gpu = None
if train_batch_tokens is not None and step_avg_ms not in (None, 0):
    tok_s = round((float(train_batch_tokens) * 1000.0) / float(step_avg_ms), 2)
    tok_s_per_gpu = round(tok_s / max(int(world_size), 1), 2)

baseline_row = next((row for row in existing_rows if row.get("parent_recipe") == "baseline_anchor"), None)
parent_row = next((row for row in reversed(existing_rows) if row.get("run_id") == parent_recipe), None)

delta_vs_baseline = None
if baseline_row and baseline_row.get("val_bpb") is not None and parsed.get("val_bpb") is not None:
    delta_vs_baseline = round(float(parsed["val_bpb"]) - float(baseline_row["val_bpb"]), 8)

delta_vs_parent = None
if parent_row and parent_row.get("val_bpb") is not None and parsed.get("val_bpb") is not None:
    delta_vs_parent = round(float(parsed["val_bpb"]) - float(parent_row["val_bpb"]), 8)

record = {
    "run_id": run_id,
    "batch": batch_name,
    "category": category,
    "priority": priority,
    "description": description,
    "status": status,
    "decision": decision,
    "parent_recipe": parent_recipe,
    "seed": int(env_overrides.get("SEED", "1337")),
    "started_at": started_at,
    "finished_at": finished_at,
    "duration_seconds": round((parsed.get("train_time_ms") or 0) / 1000.0, 3) if parsed.get("train_time_ms") is not None else None,
    "command": " ".join([
        *(f"{k}={v}" for k, v in env_overrides.items()),
        f"torchrun --standalone --nproc_per_node={nproc_per_node} train_gpt.py",
    ]).strip(),
    "env_overrides": env_overrides,
    "path_log": str(log_path),
    "exit_code": exit_code,
    "iterations": parsed.get("iterations"),
    "step": parsed.get("step"),
    "step_avg_ms": parsed.get("step_avg_ms"),
    "train_time_ms": parsed.get("train_time_ms"),
    "tok_s": tok_s,
    "tok_s_per_gpu": tok_s_per_gpu,
    "val_loss": parsed.get("val_loss"),
    "val_bpb": parsed.get("val_bpb"),
    "model_params": parsed.get("model_params"),
    "world_size": parsed.get("world_size"),
    "grad_accum_steps": parsed.get("grad_accum_steps"),
    "train_batch_tokens": parsed.get("train_batch_tokens"),
    "train_seq_len": parsed.get("train_seq_len"),
    "train_microbatch_tokens_per_gpu": parsed.get("train_microbatch_tokens_per_gpu"),
    "warmup_steps": parsed.get("warmup_steps"),
    "max_wallclock_seconds": parsed.get("max_wallclock_seconds"),
    "peak_memory_allocated_mib": parsed.get("peak_memory_allocated_mib"),
    "peak_memory_reserved_mib": parsed.get("peak_memory_reserved_mib"),
    "final_eval_time_ms": parsed.get("final_eval_time_ms"),
    "bytes_model_int8_zlib": parsed.get("bytes_model_int8_zlib"),
    "bytes_total_int8_zlib": parsed.get("bytes_total_int8_zlib"),
    "delta_vs_baseline": delta_vs_baseline,
    "delta_vs_parent": delta_vs_parent,
    "notes": "",
    "telemetry": {
        "gpu_csv_path": gpu_telemetry_csv,
        "proc_snapshot_log": proc_snapshot_log,
        "meta_json_path": telemetry_meta_json,
        **telemetry_summary,
    },
}

with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")

print(json.dumps(record, indent=2, sort_keys=True))
PY

python3 "$ROOT/scripts/build_h100_campaign_dashboard.py"

exit "$EXIT_CODE"
