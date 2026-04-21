#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "logs" / "overnight_local_ab_round6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSONL = OUT_DIR / "ab_results.jsonl"
RESULTS_TSV = OUT_DIR / "ab_results.tsv"
SUMMARY_MD = OUT_DIR / "summary.md"
CONSOLE_LOG = OUT_DIR / "console.txt"

VAL_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)")
ARTIFACT_RE = re.compile(r"serialized_model_int8_zlib:(\d+) bytes")
TRAIN_RE = re.compile(r"step:(\d+)/(\d+) train_loss:([0-9.]+) train_time:([0-9.]+)ms step_avg:([0-9.]+)ms tok_s:([0-9.]+)")


@dataclass(frozen=True)
class Candidate:
    key: str
    description: str
    env: dict[str, str]
    phase: str = "stage1"


BASE_ENV = {
    "ITERATIONS": "5000",
    "MAX_WALLCLOCK_SECONDS": "600",
    "TRAIN_BATCH_TOKENS": "8192",
    "VAL_LOSS_EVERY": "0",
    "VAL_BATCH_SIZE": "8192",
    "PROXY_VAL_BATCHES": "1024",
    "PROXY_SKIP_QUANT_EVAL": "0",
    "SCALAR_LR": "0.02",
    "GRAD_CLIP_NORM": "1.0",
    "LOGIT_SOFTCAP": "20.0",
    "QK_GAIN_INIT": "1.5",
    "ROPE_BASE": "10000.0",
    "MUON_BACKEND_STEPS": "5",
    "MUON_MOMENTUM": "0.95",
    "MUON_MOMENTUM_WARMUP_START": "0.85",
    "MUON_MOMENTUM_WARMUP_STEPS": "500",
    "MATRIX_LR": "0.04",
    "TIED_EMBED_LR": "0.05",
    "WARMUP_STEPS": "20",
    "WARMDOWN_ITERS": "3200",
    "MLP_ACT": "relu2",
    "NUM_KV_HEADS": "4",
    "SEED": "1337",
}

STAGE1_CANDIDATES = [
    Candidate("silu", "MLP_ACT=silu", {"MLP_ACT": "silu"}),
    Candidate("kv2", "NUM_KV_HEADS=2", {"NUM_KV_HEADS": "2"}),
    Candidate("embed_004", "TIED_EMBED_LR=0.04", {"TIED_EMBED_LR": "0.04"}),
    Candidate("embed_0035", "TIED_EMBED_LR=0.035", {"TIED_EMBED_LR": "0.035"}),
    Candidate("silu_kv2", "MLP_ACT=silu + NUM_KV_HEADS=2", {"MLP_ACT": "silu", "NUM_KV_HEADS": "2"}),
    Candidate("silu_embed_004", "MLP_ACT=silu + TIED_EMBED_LR=0.04", {"MLP_ACT": "silu", "TIED_EMBED_LR": "0.04"}),
    Candidate("silu_embed_0035", "MLP_ACT=silu + TIED_EMBED_LR=0.035", {"MLP_ACT": "silu", "TIED_EMBED_LR": "0.035"}),
    Candidate("silu_warmdown_4000", "MLP_ACT=silu + WARMDOWN_ITERS=4000", {"MLP_ACT": "silu", "WARMDOWN_ITERS": "4000"}),
    Candidate("silu_warmdown_2800", "MLP_ACT=silu + WARMDOWN_ITERS=2800", {"MLP_ACT": "silu", "WARMDOWN_ITERS": "2800"}),
    Candidate(
        "silu_kv2_embed_004",
        "MLP_ACT=silu + NUM_KV_HEADS=2 + TIED_EMBED_LR=0.04",
        {"MLP_ACT": "silu", "NUM_KV_HEADS": "2", "TIED_EMBED_LR": "0.04"},
    ),
]

MAX_DURATION_HOURS = 8
COOLDOWN_SECONDS = 45
BASELINE_SPREAD_CONTAMINATED = 0.02
DELTA_IMPROVE = -0.01
DELTA_WORSE = 0.01
NTFY_TOPIC = "sb-parameter-golf-19"
SECOND_SEED = "2024"
MAX_SECOND_SEED_BLOCKS = 2

SESSION_START = datetime.now().astimezone()


def now() -> datetime:
    return datetime.now().astimezone()


def stop_time() -> datetime:
    return SESSION_START + timedelta(hours=MAX_DURATION_HOURS)


def log(msg: str) -> None:
    line = f"[{now().strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}"
    print(line)
    with CONSOLE_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_metrics(path: Path, exit_code: int) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    val_match = VAL_RE.findall(text)
    artifact_match = ARTIFACT_RE.findall(text)
    train_match = TRAIN_RE.findall(text)
    if not val_match or not artifact_match or not train_match:
        if exit_code != 0:
            raise RuntimeError(f"Run crashed before metrics were available: {path}")
        raise RuntimeError(f"Missing metrics in {path}")
    val_loss, val_bpb = val_match[-1]
    _, _, _, train_time_ms, step_avg_ms, tok_s = train_match[-1]
    return {
        "val_loss": float(val_loss),
        "val_bpb": float(val_bpb),
        "artifact_int8_bytes": int(artifact_match[-1]),
        "train_time_ms": float(train_time_ms),
        "step_avg_ms": float(step_avg_ms),
        "tok_s": float(tok_s),
        "log_path": str(path),
    }


def run_one(run_id: str, env_overrides: dict[str, str]) -> dict[str, object]:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(env_overrides)
    env["RUN_ID"] = run_id
    env["OUT_DIR"] = str(OUT_DIR)
    log(f"run_start id={run_id} env={json.dumps(env_overrides, sort_keys=True)}")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "train_gpt_mlx.py"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    wall_s = time.time() - started
    log_path = OUT_DIR / f"{run_id}.txt"
    combined_output = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(combined_output, encoding="utf-8")
    if proc.stdout:
        with CONSOLE_LOG.open("a", encoding="utf-8") as f:
            f.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                f.write("\n")
    if proc.stderr:
        with CONSOLE_LOG.open("a", encoding="utf-8") as f:
            f.write(proc.stderr)
            if not proc.stderr.endswith("\n"):
                f.write("\n")
    metrics = parse_metrics(log_path, proc.returncode)
    metrics["run_id"] = run_id
    metrics["exit_code"] = proc.returncode
    metrics["wall_s"] = round(wall_s, 1)
    metrics["env"] = env_overrides
    log(
        f"run_done id={run_id} exit={proc.returncode} "
        f"val_bpb={metrics['val_bpb']:.8f} artifact={metrics['artifact_int8_bytes']} tok_s={metrics['tok_s']:.0f}"
    )
    return metrics


def classify(mean_base_bpb: float, candidate_bpb: float, baseline_spread: float) -> str:
    delta = candidate_bpb - mean_base_bpb
    if baseline_spread > BASELINE_SPREAD_CONTAMINATED:
        return "contaminated"
    if delta <= DELTA_IMPROVE:
        return "improve"
    if delta >= DELTA_WORSE:
        return "worse"
    return "mixed"


def append_result(row: dict[str, object]) -> None:
    header = [
        "phase",
        "candidate",
        "status",
        "baseline_mean_bpb",
        "candidate_bpb",
        "delta_bpb",
        "baseline_spread",
        "candidate_artifact_int8_bytes",
        "candidate_tok_s",
        "baseline_run_a",
        "candidate_run",
        "baseline_run_b",
    ]
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("\t".join(header) + "\n", encoding="utf-8")
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        f.write("\t".join(str(row[col]) for col in header) + "\n")
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_existing_rows() -> list[dict[str, object]]:
    if not RESULTS_JSONL.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Overnight Local A/B Round 6 Summary",
        "",
        "- Focus: full-night silu-centered queue with second-seed confirmations",
        "- Method: paired baseline -> candidate -> baseline",
        f"- Base env: `{json.dumps(BASE_ENV, sort_keys=True)}`",
        f"- Stop target: `{stop_time().strftime('%Y-%m-%d %H:%M %Z')}`",
        "",
        "| Phase | Candidate | Status | Candidate BPB | Delta vs Baseline Mean | Baseline Spread | Artifact | tok/s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['candidate']} | {row['status']} | {row['candidate_bpb']:.8f} | "
            f"{row['delta_bpb']:+.8f} | {row['baseline_spread']:.8f} | "
            f"{row['candidate_artifact_int8_bytes']} | {row['candidate_tok_s']:.0f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def notify_completion(rows: list[dict[str, object]], status_text: str) -> None:
    if rows:
        ranked = sorted(rows, key=lambda row: (0 if row["status"] == "improve" else 1, float(row["candidate_bpb"])))
        top = ranked[0]
        body = (
            f"{status_text}\n"
            f"round6 rows={len(rows)}\n"
            f"top={top['candidate']}\n"
            f"phase={top['phase']}\n"
            f"status={top['status']}\n"
            f"candidate_bpb={top['candidate_bpb']:.8f}\n"
            f"delta_bpb={top['delta_bpb']:+.8f}\n"
            f"artifact={top['candidate_artifact_int8_bytes']}\n"
            f"summary={SUMMARY_MD}"
        )
    else:
        body = f"{status_text}\nround6 completed with no candidate rows\nsummary={SUMMARY_MD}"
    req = urllib.request.Request(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=body.encode("utf-8"),
        headers={
            "Title": "Parameter Golf Round 6 Complete",
            "Priority": "default",
            "Tags": "robot,abacus",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            log(f"ntfy_sent status={resp.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log(f"ntfy_failed error={exc}")


def should_stop() -> bool:
    return now() >= stop_time()


def run_candidate_block(candidate: Candidate, idx: int, seed_override: str | None = None) -> dict[str, object]:
    extra = dict(candidate.env)
    phase = candidate.phase if seed_override is None else f"{candidate.phase}_seed{seed_override}"
    if seed_override is not None:
        extra["SEED"] = seed_override
    suffix = "" if seed_override is None else f"_seed{seed_override}"
    base_a_id = f"r6_{idx:02d}_{candidate.key}{suffix}_base_a"
    cand_id = f"r6_{idx:02d}_{candidate.key}{suffix}_cand"
    base_b_id = f"r6_{idx:02d}_{candidate.key}{suffix}_base_b"

    base_a = run_one(base_a_id, {"SEED": extra.get("SEED", BASE_ENV["SEED"])})
    time.sleep(COOLDOWN_SECONDS)
    candidate_metrics = run_one(cand_id, extra)
    time.sleep(COOLDOWN_SECONDS)
    base_b = run_one(base_b_id, {"SEED": extra.get("SEED", BASE_ENV["SEED"])})

    baseline_mean = (float(base_a["val_bpb"]) + float(base_b["val_bpb"])) / 2.0
    baseline_spread = abs(float(base_a["val_bpb"]) - float(base_b["val_bpb"]))
    delta = float(candidate_metrics["val_bpb"]) - baseline_mean
    status = classify(baseline_mean, float(candidate_metrics["val_bpb"]), baseline_spread)

    row = {
        "phase": phase,
        "candidate": candidate.description,
        "status": status,
        "baseline_mean_bpb": round(baseline_mean, 8),
        "candidate_bpb": round(float(candidate_metrics["val_bpb"]), 8),
        "delta_bpb": round(delta, 8),
        "baseline_spread": round(baseline_spread, 8),
        "candidate_artifact_int8_bytes": int(candidate_metrics["artifact_int8_bytes"]),
        "candidate_tok_s": round(float(candidate_metrics["tok_s"]), 0),
        "baseline_run_a": base_a_id,
        "candidate_run": cand_id,
        "baseline_run_b": base_b_id,
    }
    return row


def main() -> None:
    log("overnight_local_ab_round6_start")
    rows = load_existing_rows()
    completed = {(str(row["phase"]), str(row["candidate"])) for row in rows}
    if rows:
        write_summary(rows)
        log(f"resume_loaded completed_rows={len(completed)}")

    for idx, candidate in enumerate(STAGE1_CANDIDATES, start=1):
        if (candidate.phase, candidate.description) in completed:
            log(f"skip_completed key={candidate.key} phase={candidate.phase}")
            continue
        if should_stop():
            log("stop_window_reached before next stage1 candidate")
            break
        row = run_candidate_block(candidate, idx)
        rows.append(row)
        append_result(row)
        write_summary(rows)
        log(
            f"candidate_complete key={candidate.key} phase={row['phase']} status={row['status']} "
            f"delta_bpb={row['delta_bpb']:+.8f} baseline_spread={row['baseline_spread']:.8f}"
        )
        time.sleep(COOLDOWN_SECONDS)

    stage1_rows = [row for row in rows if row["phase"] == "stage1" and row["status"] != "contaminated"]
    stage1_ranked = sorted(stage1_rows, key=lambda row: (float(row["delta_bpb"]), float(row["candidate_bpb"])))
    second_seed_candidates: list[Candidate] = []
    for row in stage1_ranked[:MAX_SECOND_SEED_BLOCKS]:
        for candidate in STAGE1_CANDIDATES:
            if candidate.description == row["candidate"]:
                second_seed_candidates.append(Candidate(candidate.key, candidate.description, candidate.env, phase="stage2"))
                break

    for offset, candidate in enumerate(second_seed_candidates, start=1):
        phase_name = f"stage2_seed{SECOND_SEED}"
        if (phase_name, candidate.description) in completed:
            log(f"skip_completed key={candidate.key} phase={phase_name}")
            continue
        if should_stop():
            log("stop_window_reached before next stage2 candidate")
            break
        row = run_candidate_block(candidate, len(STAGE1_CANDIDATES) + offset, seed_override=SECOND_SEED)
        row["phase"] = phase_name
        rows.append(row)
        append_result(row)
        write_summary(rows)
        log(
            f"candidate_complete key={candidate.key} phase={row['phase']} status={row['status']} "
            f"delta_bpb={row['delta_bpb']:+.8f} baseline_spread={row['baseline_spread']:.8f}"
        )
        time.sleep(COOLDOWN_SECONDS)

    log("overnight_local_ab_round6_done")
    write_summary(rows)
    notify_completion(rows, "round6 complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"round6_fatal error={exc}")
        notify_completion(load_existing_rows(), f"round6 failed: {exc}")
        raise
