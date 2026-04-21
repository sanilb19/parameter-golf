#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FINAL_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
TOTAL_BYTES_RE = re.compile(r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes")
MODEL_BYTES_RE = re.compile(r"Serialized model int8\+zlib: (?P<bytes>\d+) bytes")
STOP_RE = re.compile(r"stopping_early: wallclock_cap train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?P<iterations>\d+)")
STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) (?:train_loss|val_loss):.*train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms"
)
PARAMS_RE = re.compile(r"model_params:(?P<model_params>\d+)")


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
