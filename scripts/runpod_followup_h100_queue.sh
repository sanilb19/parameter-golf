#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/parameter-golf}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/runpod_followup_h100}"
DATA_PATH="${DATA_PATH:-$ROOT/data/datasets/fineweb10B_sp1024/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/data/tokenizers/fineweb_1024_bpe.model}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"

MODE="${1:-}"
BEST_RUN_ID="${2:-}"

if [[ -z "$MODE" ]]; then
  echo "usage: $0 <rerun_best_seed2024|dim384_vs_mlp1> [best_run_id]" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_JSONL="$LOG_DIR/summary.jsonl"
CONSOLE_LOG="$LOG_DIR/queue.log"

cd "$ROOT"

if [[ ! -f "$SUMMARY_TSV" ]]; then
  printf "run_id\tval_bpb\tval_loss\tbytes_total_int8_zlib\tbytes_model_int8_zlib\ttrain_time_ms\tstep\tstep_avg_ms\tstatus\tdescription\n" > "$SUMMARY_TSV"
fi
: > "$CONSOLE_LOG"

run_case() {
  local run_id="$1"
  local description="$2"
  shift 2
  local log_path="$LOG_DIR/$run_id.log"

  echo "[queue] start run_id=$run_id desc=$description" | tee -a "$CONSOLE_LOG"
  set +e
  env \
    RUN_ID="$run_id" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
    VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
    MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
    "$@" \
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py > "$log_path" 2>&1
  local exit_code=$?
  set -e

  python3 - "$log_path" "$run_id" "$description" "$exit_code" "$SUMMARY_TSV" "$SUMMARY_JSONL" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from parse_train_log import parse_train_log

log_path = Path(sys.argv[1])
run_id = sys.argv[2]
description = sys.argv[3]
exit_code = int(sys.argv[4])
summary_tsv = Path(sys.argv[5])
summary_jsonl = Path(sys.argv[6])

parsed = parse_train_log(log_path)
parsed["run_id"] = run_id
parsed["description"] = description
parsed["exit_code"] = exit_code
if exit_code != 0 and parsed["status"] == "ok":
    parsed["status"] = f"exit_{exit_code}"

with summary_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(parsed, sort_keys=True) + "\n")

with summary_tsv.open("a", encoding="utf-8") as f:
    f.write(
        "\t".join(
            [
                run_id,
                str(parsed.get("val_bpb", "")),
                str(parsed.get("val_loss", "")),
                str(parsed.get("bytes_total_int8_zlib", "")),
                str(parsed.get("bytes_model_int8_zlib", "")),
                str(parsed.get("train_time_ms", "")),
                str(parsed.get("step", "")),
                str(parsed.get("step_avg_ms", "")),
                str(parsed.get("status", "")),
                description,
            ]
        )
        + "\n"
    )

print(json.dumps(parsed, indent=2, sort_keys=True))
PY
  echo "[queue] done run_id=$run_id exit_code=$exit_code" | tee -a "$CONSOLE_LOG"
}

case "$MODE" in
  rerun_best_seed2024)
    if [[ -z "$BEST_RUN_ID" ]]; then
      echo "best_run_id required for rerun_best_seed2024" >&2
      exit 1
    fi
    case "$BEST_RUN_ID" in
      h100_baseline_sp1024)
        run_case "h100_baseline_sp1024_seed2024" "Baseline rerun with seed 2024" SEED=2024
        ;;
      h100_silu_sp1024)
        run_case "h100_silu_sp1024_seed2024" "MLP_ACT=silu rerun with seed 2024" SEED=2024 MLP_ACT=silu
        ;;
      h100_silu_embed004_sp1024)
        run_case "h100_silu_embed004_sp1024_seed2024" "MLP_ACT=silu + TIED_EMBED_LR=0.04 rerun with seed 2024" SEED=2024 MLP_ACT=silu TIED_EMBED_LR=0.04
        ;;
      h100_silu_embed004_dim384_sp1024)
        run_case "h100_silu_embed004_dim384_sp1024_seed2024" "MLP_ACT=silu + TIED_EMBED_LR=0.04 + MODEL_DIM=384 rerun with seed 2024" SEED=2024 MLP_ACT=silu TIED_EMBED_LR=0.04 MODEL_DIM=384 NUM_HEADS=8 NUM_KV_HEADS=4
        ;;
      *)
        echo "unknown best_run_id: $BEST_RUN_ID" >&2
        exit 1
        ;;
    esac
    ;;
  dim384_vs_mlp1)
    run_case \
      "h100_silu_embed004_dim384_sp1024" \
      "Compact architecture candidate: silu + tied_embed_lr=0.04 + dim384" \
      SEED=1337 \
      MLP_ACT=silu \
      TIED_EMBED_LR=0.04 \
      MODEL_DIM=384 \
      NUM_HEADS=8 \
      NUM_KV_HEADS=4

    run_case \
      "h100_silu_embed004_dim384_mlp1_sp1024" \
      "Compact follow-up: silu + tied_embed_lr=0.04 + dim384 + mlp_mult=1" \
      SEED=1337 \
      MLP_ACT=silu \
      TIED_EMBED_LR=0.04 \
      MODEL_DIM=384 \
      NUM_HEADS=8 \
      NUM_KV_HEADS=4 \
      MLP_MULT=1
    ;;
  *)
    echo "unsupported mode: $MODE" >&2
    exit 1
    ;;
esac

echo "[queue] complete summary=$SUMMARY_TSV"
