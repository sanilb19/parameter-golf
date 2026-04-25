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
  BATCH_NAME=phase_1_knobs_b TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-10}" \
    "$ROOT/scripts/run_h100_campaign_case.sh" \
    "$run_id" "$category" "$description" "$parent" "$decision" "$@"
}

PARENT="h100_baseline_sp1024"

run_one \
  "h100_muon_momentum_0975" \
  "hparam_optimizer" \
  "Muon momentum 0.975" \
  "$PARENT" \
  "pending" \
  SEED=1337 \
  MUON_MOMENTUM=0.975

run_one \
  "h100_muon_warmup_1000" \
  "hparam_optimizer" \
  "Muon momentum warmup steps 1000" \
  "$PARENT" \
  "pending" \
  SEED=1337 \
  MUON_MOMENTUM_WARMUP_STEPS=1000

run_one \
  "h100_logit_softcap_20" \
  "hparam_schedule" \
  "Logit softcap 20" \
  "$PARENT" \
  "pending" \
  SEED=1337 \
  LOGIT_SOFTCAP=20

run_one \
  "h100_logit_softcap_40" \
  "hparam_schedule" \
  "Logit softcap 40" \
  "$PARENT" \
  "pending" \
  SEED=1337 \
  LOGIT_SOFTCAP=40
