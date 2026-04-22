#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SIZE_CAP = 16_000_000


def parse_float(value: str) -> float:
    return float(value) if value else float("inf")


def parse_int(value: str) -> int:
    return int(value) if value else 1 << 60


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_tsv", type=Path)
    args = parser.parse_args()

    rows = []
    with args.summary_tsv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["val_bpb_num"] = parse_float(row.get("val_bpb", ""))
            row["bytes_total_num"] = parse_int(row.get("bytes_total_int8_zlib", ""))
            row["clean"] = row.get("status", "") == "ok" and row["bytes_total_num"] <= SIZE_CAP
            rows.append(row)

    if not rows:
        raise SystemExit("No rows found")

    clean = [row for row in rows if row["clean"]]
    ranked = sorted(clean if clean else rows, key=lambda row: (row["val_bpb_num"], row["bytes_total_num"]))
    best = ranked[0]
    log_name = f"{best['run_id']}.log"
    out = {
        "best_run_id": best["run_id"],
        "best_log": log_name,
        "val_bpb": best.get("val_bpb"),
        "val_loss": best.get("val_loss"),
        "bytes_total_int8_zlib": best.get("bytes_total_int8_zlib"),
        "status": best.get("status"),
        "description": best.get("description"),
        "clean_under_cap": best["clean"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
