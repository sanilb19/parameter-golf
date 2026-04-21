#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "logs" / "one_hour_silu_embed_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSONL = OUT_DIR / "results.jsonl"
RESULTS_TSV = OUT_DIR / "results.tsv"
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


BASE_ENV = {
    "ITERATIONS": "5000",
    "MAX_WALLCLOCK_SECONDS": "480",
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
    "MLP_ACT": "silu",
    "NUM_KV_HEADS": "4",
    "SEED": "1337",
}

CANDIDATES = [
    Candidate("embed_004", "MLP_ACT=silu + TIED_EMBED_LR=0.04", {"TIED_EMBED_LR": "0.04"}),
    Candidate("embed_0035", "MLP_ACT=silu + TIED_EMBED_LR=0.035", {"TIED_EMBED_LR": "0.035"}),
]

COOLDOWN_SECONDS = 30
BASELINE_SPREAD_CONTAMINATED = 0.02
DELTA_IMPROVE = -0.01
DELTA_WORSE = 0.01
MAX_DURATION_MINUTES = 60
SESSION_START = datetime.now().astimezone()


def now() -> datetime:
    return datetime.now().astimezone()


def stop_time() -> datetime:
    return SESSION_START + timedelta(minutes=MAX_DURATION_MINUTES)


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


def write_summary(rows: list[dict[str, object]]) -> None:
    lines = [
        "# One Hour silu + Embed LR A/B",
        "",
        "- Focus: on top of `MLP_ACT=silu`, compare `TIED_EMBED_LR=0.04` and `0.035` against the default `0.05`",
        f"- Base env: `{json.dumps(BASE_ENV, sort_keys=True)}`",
        f"- Stop target: `{stop_time().strftime('%Y-%m-%d %H:%M %Z')}`",
        "",
        "| Candidate | Status | Candidate BPB | Delta vs Baseline Mean | Baseline Spread | Artifact | tok/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['status']} | {row['candidate_bpb']:.8f} | "
            f"{row['delta_bpb']:+.8f} | {row['baseline_spread']:.8f} | "
            f"{row['candidate_artifact_int8_bytes']} | {row['candidate_tok_s']:.0f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def should_stop() -> bool:
    return now() >= stop_time()


def main() -> None:
    log("one_hour_silu_embed_ab_start")
    rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(CANDIDATES, start=1):
        if should_stop():
            log("stop_window_reached before next candidate")
            break

        base_a_id = f"h1_{idx:02d}_{candidate.key}_base_a"
        cand_id = f"h1_{idx:02d}_{candidate.key}_cand"
        base_b_id = f"h1_{idx:02d}_{candidate.key}_base_b"

        base_a = run_one(base_a_id, {})
        time.sleep(COOLDOWN_SECONDS)
        candidate_metrics = run_one(cand_id, candidate.env)
        time.sleep(COOLDOWN_SECONDS)
        base_b = run_one(base_b_id, {})

        baseline_mean = (float(base_a["val_bpb"]) + float(base_b["val_bpb"])) / 2.0
        baseline_spread = abs(float(base_a["val_bpb"]) - float(base_b["val_bpb"]))
        delta = float(candidate_metrics["val_bpb"]) - baseline_mean
        status = classify(baseline_mean, float(candidate_metrics["val_bpb"]), baseline_spread)

        row = {
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
        rows.append(row)
        append_result(row)
        write_summary(rows)
        log(
            f"candidate_complete key={candidate.key} status={status} "
            f"delta_bpb={delta:+.8f} baseline_spread={baseline_spread:.8f}"
        )
        time.sleep(COOLDOWN_SECONDS)

    log("one_hour_silu_embed_ab_done")


if __name__ == "__main__":
    main()
