#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FINAL_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
FINAL_EVAL_RE = re.compile(
    r"final_int8_zlib_roundtrip val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+) "
    r"eval_time:(?P<eval_time_ms>\d+)ms"
)
TOTAL_BYTES_RE = re.compile(r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes")
MODEL_BYTES_RE = re.compile(r"Serialized model int8\+zlib: (?P<bytes>\d+) bytes")
STOP_RE = re.compile(r"stopping_early: wallclock_cap train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?P<iterations>\d+)")
STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) (?:train_loss|val_loss):.*train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms"
)
PARAMS_RE = re.compile(r"model_params:(?P<model_params>\d+)")
WORLD_RE = re.compile(r"world_size:(?P<world_size>\d+) grad_accum_steps:(?P<grad_accum_steps>\d+)")
TRAIN_CFG_RE = re.compile(
    r"train_batch_tokens:(?P<train_batch_tokens>\d+) train_seq_len:(?P<train_seq_len>\d+) "
    r"iterations:(?P<iterations>\d+) warmup_steps:(?P<warmup_steps>\d+) "
    r"max_wallclock_seconds:(?P<max_wallclock_seconds>[-+0-9.eE]+)"
)
MICROBATCH_RE = re.compile(r"train_microbatch_tokens_per_gpu:(?P<train_microbatch_tokens_per_gpu>\d+)")
SEED_RE = re.compile(r"seed:(?P<seed>\d+)")
PEAK_MEM_RE = re.compile(
    r"peak memory allocated: (?P<peak_memory_allocated_mib>\d+) MiB reserved: (?P<peak_memory_reserved_mib>\d+) MiB"
)


def parse_train_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, object] = {
        "path": str(path),
        "status": "ok",
        "val_loss": None,
        "val_bpb": None,
        "bytes_total_int8_zlib": None,
        "bytes_model_int8_zlib": None,
        "train_time_ms": None,
        "step": None,
        "iterations": None,
        "step_avg_ms": None,
        "model_params": None,
        "world_size": None,
        "grad_accum_steps": None,
        "train_batch_tokens": None,
        "train_seq_len": None,
        "warmup_steps": None,
        "max_wallclock_seconds": None,
        "train_microbatch_tokens_per_gpu": None,
        "seed": None,
        "peak_memory_allocated_mib": None,
        "peak_memory_reserved_mib": None,
        "final_eval_time_ms": None,
    }

    final = None
    for match in FINAL_EXACT_RE.finditer(text):
        final = match
    if final is not None:
        out["val_loss"] = float(final.group("val_loss"))
        out["val_bpb"] = float(final.group("val_bpb"))
    else:
        out["status"] = "missing_final"

    total = None
    for match in TOTAL_BYTES_RE.finditer(text):
        total = match
    if total is not None:
        out["bytes_total_int8_zlib"] = int(total.group("bytes"))

    model = None
    for match in MODEL_BYTES_RE.finditer(text):
        model = match
    if model is not None:
        out["bytes_model_int8_zlib"] = int(model.group("bytes"))

    stop = None
    for match in STOP_RE.finditer(text):
        stop = match
    if stop is not None:
        out["train_time_ms"] = int(stop.group("train_time_ms"))
        out["step"] = int(stop.group("step"))
        out["iterations"] = int(stop.group("iterations"))

    last_step = None
    for match in STEP_RE.finditer(text):
        last_step = match
    if last_step is not None:
        out["step_avg_ms"] = float(last_step.group("step_avg_ms"))
        if out["train_time_ms"] is None:
            out["train_time_ms"] = int(last_step.group("train_time_ms"))
        if out["step"] is None:
            out["step"] = int(last_step.group("step"))
            out["iterations"] = int(last_step.group("iterations"))

    params = None
    for match in PARAMS_RE.finditer(text):
        params = match
    if params is not None:
        out["model_params"] = int(params.group("model_params"))

    world = None
    for match in WORLD_RE.finditer(text):
        world = match
    if world is not None:
        out["world_size"] = int(world.group("world_size"))
        out["grad_accum_steps"] = int(world.group("grad_accum_steps"))

    train_cfg = None
    for match in TRAIN_CFG_RE.finditer(text):
        train_cfg = match
    if train_cfg is not None:
        out["train_batch_tokens"] = int(train_cfg.group("train_batch_tokens"))
        out["train_seq_len"] = int(train_cfg.group("train_seq_len"))
        out["iterations"] = int(train_cfg.group("iterations"))
        out["warmup_steps"] = int(train_cfg.group("warmup_steps"))
        out["max_wallclock_seconds"] = float(train_cfg.group("max_wallclock_seconds"))

    microbatch = None
    for match in MICROBATCH_RE.finditer(text):
        microbatch = match
    if microbatch is not None:
        out["train_microbatch_tokens_per_gpu"] = int(microbatch.group("train_microbatch_tokens_per_gpu"))

    seed = None
    for match in SEED_RE.finditer(text):
        seed = match
    if seed is not None:
        out["seed"] = int(seed.group("seed"))

    peak = None
    for match in PEAK_MEM_RE.finditer(text):
        peak = match
    if peak is not None:
        out["peak_memory_allocated_mib"] = int(peak.group("peak_memory_allocated_mib"))
        out["peak_memory_reserved_mib"] = int(peak.group("peak_memory_reserved_mib"))

    final_eval = None
    for match in FINAL_EVAL_RE.finditer(text):
        final_eval = match
    if final_eval is not None:
        out["final_eval_time_ms"] = int(final_eval.group("eval_time_ms"))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--field", type=str, default=None)
    args = parser.parse_args()

    parsed = parse_train_log(args.log_path)
    if args.field is not None:
        value = parsed.get(args.field)
        if isinstance(value, (dict, list)):
            print(json.dumps(value))
        elif value is None:
            print("")
        else:
            print(value)
        return

    print(json.dumps(parsed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
