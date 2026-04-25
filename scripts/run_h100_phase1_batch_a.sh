#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/parameter-golf}"
cd "$ROOT"

run_one() {
  local run_id="$1"
  local category="$2"
  local description="$3"
  local parent="$4"
  local decision="$5"
  shift 5
  BATCH_NAME=phase_1_knobs_a TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-10}" \
    "$ROOT/scripts/run_h100_campaign_case.sh" \
    "$run_id" "$category" "$description" "$parent" "$decision" "$@"
}

PARENT="h100_baseline_sp1024"

run_one \
  "h100_qk_gain_125" \
  "hparam_optimizer" \
  "QK gain 1.25" \
  "$PARENT" \
  "pending" \
  SEED=1337 \
  QK_GAIN_INIT=1.25

run_one \
  "h100_qk_gain_175" \
  "hparam_optimizer" \
  "QK gain 1.75" \
  "$PARENT" \
  "pending" \
  SEED=1337 \
  QK_GAIN_INIT=1.75

run_one \
  "h100_scalar_lr_003" \
  "hparam_optimizer" \
  "Scalar LR 0.03" \
  "$PARENT" \
  "pending" \
  SEED=1337 \
  SCALAR_LR=0.03

run_one \
  "h100_grad_clip_10" \
  "hparam_optimizer" \
  "Grad clip 1.0" \
  "$PARENT" \
  "pending" \
  SEED=1337 \
  GRAD_CLIP_NORM=1.0
