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
OUT_DIR = ROOT / "logs" / "overnight_local_ab_round8"
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
    phase: str


BASE_ENV = {
    "ITERATIONS": "5000",
    "MAX_WALLCLOCK_SECONDS": "360",
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
    "TIED_EMBED_LR": "0.04",
    "WARMUP_STEPS": "20",
    "WARMDOWN_ITERS": "3200",
    "MLP_ACT": "silu",
    "NUM_KV_HEADS": "4",
    "NUM_HEADS": "8",
    "NUM_LAYERS": "9",
    "MODEL_DIM": "384",
    "MLP_MULT": "2",
    "SEED": "1337",
}

STAGE1_CANDIDATES = [
    Candidate("mlp_mult_1", "MODEL_DIM=384 + MLP_MULT=1", {"MLP_MULT": "1"}, "stage1"),
    Candidate("layers_8", "MODEL_DIM=384 + NUM_LAYERS=8", {"NUM_LAYERS": "8"}, "stage1"),
    Candidate("layers_10", "MODEL_DIM=384 + NUM_LAYERS=10", {"NUM_LAYERS": "10"}, "stage1"),
    Candidate("dim_320", "MODEL_DIM=320 NUM_HEADS=8 NUM_KV_HEADS=4", {"MODEL_DIM": "320", "NUM_HEADS": "8", "NUM_KV_HEADS": "4"}, "stage1"),
    Candidate("dim_448", "MODEL_DIM=448 NUM_HEADS=8 NUM_KV_HEADS=4", {"MODEL_DIM": "448", "NUM_HEADS": "8", "NUM_KV_HEADS": "4"}, "stage1"),
    Candidate("mlp1_layers8", "MODEL_DIM=384 + MLP_MULT=1 + NUM_LAYERS=8", {"MLP_MULT": "1", "NUM_LAYERS": "8"}, "stage1"),
    Candidate("mlp1_layers10", "MODEL_DIM=384 + MLP_MULT=1 + NUM_LAYERS=10", {"MLP_MULT": "1", "NUM_LAYERS": "10"}, "stage1"),
    Candidate("dim320_mlp1", "MODEL_DIM=320 + MLP_MULT=1", {"MODEL_DIM": "320", "NUM_HEADS": "8", "NUM_KV_HEADS": "4", "MLP_MULT": "1"}, "stage1"),
    Candidate("dim448_mlp1", "MODEL_DIM=448 + MLP_MULT=1", {"MODEL_DIM": "448", "NUM_HEADS": "8", "NUM_KV_HEADS": "4", "MLP_MULT": "1"}, "stage1"),
    Candidate("embed_0035", "MODEL_DIM=384 + TIED_EMBED_LR=0.035", {"TIED_EMBED_LR": "0.035"}, "stage1"),
    Candidate("embed_0045", "MODEL_DIM=384 + TIED_EMBED_LR=0.045", {"TIED_EMBED_LR": "0.045"}, "stage1"),
    Candidate("warmdown_2800", "MODEL_DIM=384 + WARMDOWN_ITERS=2800", {"WARMDOWN_ITERS": "2800"}, "stage1"),
    Candidate("warmdown_4000", "MODEL_DIM=384 + WARMDOWN_ITERS=4000", {"WARMDOWN_ITERS": "4000"}, "stage1"),
]

MAX_DURATION_HOURS = 9
COOLDOWN_SECONDS = 30
BASELINE_SPREAD_CONTAMINATED = 0.02
DELTA_IMPROVE = -0.01
DELTA_WORSE = 0.01
NTFY_TOPIC = "sb-parameter-golf-19"
SECOND_SEED = "2024"
THIRD_SEED = "7"
TAIL_SEEDS = ("1337", "2024", "7")
MAX_STAGE2_BLOCKS = 4
MAX_STAGE3_BLOCKS = 3

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
        if line:
            rows.append(json.loads(line))
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Overnight Local A/B Round 8 Summary",
        "",
        "- Focus: last high-signal M3 night centered on transferable compact-architecture ablations",
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
            f"round8 rows={len(rows)}\n"
            f"top={top['candidate']}\n"
            f"phase={top['phase']}\n"
            f"status={top['status']}\n"
            f"candidate_bpb={top['candidate_bpb']:.8f}\n"
            f"delta_bpb={top['delta_bpb']:+.8f}\n"
            f"artifact={top['candidate_artifact_int8_bytes']}\n"
            f"summary={SUMMARY_MD}"
        )
    else:
        body = f"{status_text}\nround8 completed with no candidate rows\nsummary={SUMMARY_MD}"
    req = urllib.request.Request(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=body.encode("utf-8"),
        headers={"Title": "Parameter Golf Round 8 Complete", "Priority": "default", "Tags": "robot,abacus"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            log(f"ntfy_sent status={resp.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log(f"ntfy_failed error={exc}")


def should_stop() -> bool:
    return now() >= stop_time()


def run_candidate_block(
    candidate: Candidate,
    idx: int,
    seed_override: str | None = None,
    long_confirm: bool = False,
    phase_override: str | None = None,
) -> dict[str, object]:
    extra = dict(candidate.env)
    phase = phase_override or candidate.phase
    max_wallclock = BASE_ENV["MAX_WALLCLOCK_SECONDS"]
    if seed_override is not None:
        extra["SEED"] = seed_override
        if phase_override is None:
            phase = f"{phase}_seed{seed_override}"
    if long_confirm:
        max_wallclock = "600"
        extra["MAX_WALLCLOCK_SECONDS"] = max_wallclock
        if phase_override is None:
            phase = f"{phase}_10m"
    seed_value = extra.get("SEED", BASE_ENV["SEED"])
    suffix = ""
    if seed_override is not None:
        suffix += f"_seed{seed_override}"
    if long_confirm:
        suffix += "_10m"
    base_a_id = f"r8_{idx:02d}_{candidate.key}{suffix}_base_a"
    cand_id = f"r8_{idx:02d}_{candidate.key}{suffix}_cand"
    base_b_id = f"r8_{idx:02d}_{candidate.key}{suffix}_base_b"

    base_env = {"SEED": seed_value, "MAX_WALLCLOCK_SECONDS": max_wallclock}
    base_a = run_one(base_a_id, base_env)
    time.sleep(COOLDOWN_SECONDS)
    candidate_metrics = run_one(cand_id, extra)
    time.sleep(COOLDOWN_SECONDS)
    base_b = run_one(base_b_id, base_env)

    baseline_mean = (float(base_a["val_bpb"]) + float(base_b["val_bpb"])) / 2.0
    baseline_spread = abs(float(base_a["val_bpb"]) - float(base_b["val_bpb"]))
    delta = float(candidate_metrics["val_bpb"]) - baseline_mean
    status = classify(baseline_mean, float(candidate_metrics["val_bpb"]), baseline_spread)
    return {
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


def stage_rank(rows: list[dict[str, object]], phase_prefix: str) -> list[dict[str, object]]:
    subset = [row for row in rows if str(row["phase"]).startswith(phase_prefix) and row["status"] != "contaminated"]
    return sorted(subset, key=lambda row: (float(row["delta_bpb"]), float(row["candidate_bpb"])))


def candidate_by_description(description: str) -> Candidate:
    for candidate in STAGE1_CANDIDATES:
        if candidate.description == description:
            return candidate
    raise KeyError(description)


def main() -> None:
    log("overnight_local_ab_round8_start")
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
        log(f"candidate_complete key={candidate.key} phase={row['phase']} status={row['status']} delta_bpb={row['delta_bpb']:+.8f}")
        time.sleep(COOLDOWN_SECONDS)

    ranked_stage1 = stage_rank(rows, "stage1")
    stage2_candidates = [candidate_by_description(row["candidate"]) for row in ranked_stage1[:MAX_STAGE2_BLOCKS]]

    for offset, candidate in enumerate(stage2_candidates, start=1):
        phase_name = f"stage2_seed{SECOND_SEED}"
        if (phase_name, candidate.description) in completed:
            log(f"skip_completed key={candidate.key} phase={phase_name}")
            continue
        if should_stop():
            log("stop_window_reached before next stage2 candidate")
            break
        row = run_candidate_block(candidate, len(STAGE1_CANDIDATES) + offset, seed_override=SECOND_SEED, phase_override=phase_name)
        rows.append(row)
        append_result(row)
        write_summary(rows)
        log(f"candidate_complete key={candidate.key} phase={row['phase']} status={row['status']} delta_bpb={row['delta_bpb']:+.8f}")
        time.sleep(COOLDOWN_SECONDS)

    ranked_stage2 = stage_rank(rows, f"stage2_seed{SECOND_SEED}") or ranked_stage1
    stage3_candidates = [candidate_by_description(row["candidate"]) for row in ranked_stage2[:MAX_STAGE3_BLOCKS]]

    for offset, candidate in enumerate(stage3_candidates, start=1):
        phase_name = f"stage3_seed{THIRD_SEED}_10m"
        if (phase_name, candidate.description) in completed:
            log(f"skip_completed key={candidate.key} phase={phase_name}")
            continue
        if should_stop():
            log("stop_window_reached before next stage3 candidate")
            break
        row = run_candidate_block(
            candidate,
            len(STAGE1_CANDIDATES) + MAX_STAGE2_BLOCKS + offset,
            seed_override=THIRD_SEED,
            long_confirm=True,
            phase_override=phase_name,
        )
        rows.append(row)
        append_result(row)
        write_summary(rows)
        log(f"candidate_complete key={candidate.key} phase={row['phase']} status={row['status']} delta_bpb={row['delta_bpb']:+.8f}")
        time.sleep(COOLDOWN_SECONDS)

    tail_rank_source = stage_rank(rows, f"stage3_seed{THIRD_SEED}_10m") or stage_rank(rows, f"stage2_seed{SECOND_SEED}") or ranked_stage1
    tail_candidates = [candidate_by_description(row["candidate"]) for row in tail_rank_source[:2]]
    tail_iter = 0
    while tail_candidates and not should_stop():
        candidate = tail_candidates[tail_iter % len(tail_candidates)]
        seed_value = TAIL_SEEDS[tail_iter % len(TAIL_SEEDS)]
        phase_name = f"tail_seed{seed_value}_10m"
        tail_count_for_phase = sum(1 for row in rows if str(row["phase"]).startswith(phase_name))
        candidate_label = f"{candidate.description} [tail-{tail_count_for_phase + 1}]"
        synthetic = Candidate(candidate.key, candidate_label, candidate.env, "tail")
        row = run_candidate_block(
            synthetic,
            len(STAGE1_CANDIDATES) + MAX_STAGE2_BLOCKS + MAX_STAGE3_BLOCKS + tail_iter + 1,
            seed_override=seed_value,
            long_confirm=True,
            phase_override=phase_name,
        )
        rows.append(row)
        append_result(row)
        write_summary(rows)
        log(f"tail_complete key={candidate.key} phase={row['phase']} status={row['status']} delta_bpb={row['delta_bpb']:+.8f}")
        tail_iter += 1
        time.sleep(COOLDOWN_SECONDS)

    log("overnight_local_ab_round8_done")
    write_summary(rows)
    notify_completion(rows, "round8 complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"round8_fatal error={exc}")
        notify_completion(load_existing_rows(), f"round8 failed: {exc}")
        raise
