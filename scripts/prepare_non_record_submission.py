#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from parse_train_log import parse_train_log


README_TEMPLATE = """# Non-Record Submission: {name}

## Summary

{blurb}

## Run

- Track: non-record-16mb
- GPU: {gpu}
- Source log: `{log_name}`
- Source script: `train_gpt.py`

## Metrics

- `val_bpb`: {val_bpb}
- `val_loss`: {val_loss}
- `bytes_total`: {bytes_total}
- `bytes_model_int8_zlib`: {bytes_model}
- `train_time_ms`: {train_time_ms}
- `step`: {step}
- `step_avg_ms`: {step_avg_ms}

## Approach

- Base path: published `train_gpt.py`
- Remote validation target: 1xH100 RunPod screen before a larger 8xH100 push
- Main idea:
  - TODO replace with the exact hypothesis this run was testing
  - TODO summarize the kept local evidence leading to this run

## Notes

- This is intended as a same-day non-record submission scaffold.
- Replace the TODO bullets above before opening the PR.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--github-id", required=True)
    parser.add_argument("--blurb", required=True)
    parser.add_argument("--gpu", default="1xH100")
    parser.add_argument("--script", type=Path, default=Path("train_gpt.py"))
    args = parser.parse_args()

    parsed = parse_train_log(args.log)
    if parsed.get("val_bpb") is None or parsed.get("bytes_total_int8_zlib") is None:
        raise SystemExit(f"Could not parse final int8 metrics from {args.log}")

    target = args.folder
    target.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.script, target / "train_gpt.py")
    shutil.copy2(args.log, target / "train.log")

    readme = README_TEMPLATE.format(
        name=args.name,
        blurb=args.blurb,
        gpu=args.gpu,
        log_name=args.log.name,
        val_bpb=parsed["val_bpb"],
        val_loss=parsed["val_loss"],
        bytes_total=parsed["bytes_total_int8_zlib"],
        bytes_model=parsed["bytes_model_int8_zlib"],
        train_time_ms=parsed["train_time_ms"],
        step=parsed["step"],
        step_avg_ms=parsed["step_avg_ms"],
    )
    (target / "README.md").write_text(readme, encoding="utf-8")

    submission = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.name,
        "blurb": args.blurb,
        "date": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "track": "non-record-16mb",
        "val_loss": parsed["val_loss"],
        "val_bpb": parsed["val_bpb"],
        "step_stop": parsed["step"],
        "wallclock_seconds": round((parsed["train_time_ms"] or 0) / 1000.0, 3),
        "bytes_total": parsed["bytes_total_int8_zlib"],
        "bytes_model_int8_zlib": parsed["bytes_model_int8_zlib"],
        "bytes_code": args.script.stat().st_size,
        "gpu": args.gpu,
    }
    (target / "submission.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")

    print(target)


if __name__ == "__main__":
    main()
