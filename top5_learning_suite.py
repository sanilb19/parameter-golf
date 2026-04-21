from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


MANIFEST_DEFAULT = Path(".research/top5_learning_suite.json")
PREQUANT_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+)")
ROUNDTRIP_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)")
SERIALIZED_RE = re.compile(r"serialized_model_int8_zlib:(\d+) bytes")
TRAIN_STEP_RE = re.compile(r"step:(\d+)/(\d+) train_loss:([0-9.]+) train_time:([0-9.]+)ms step_avg:([0-9.]+)ms tok_s:([0-9.]+)")


def maybe_run_top5_learning_suite(script_path: Path, python_executable: str, run_subprocess) -> bool:
    if os.environ.get("TOP5_SUITE_CHILD") == "1":
        return False
    manifest_path = Path(os.environ.get("TOP5_SUITE_MANIFEST", str(MANIFEST_DEFAULT)))
    if not manifest_path.is_file():
        return False
    run_top5_learning_suite(script_path=script_path, python_executable=python_executable, run_subprocess=run_subprocess, manifest_path=manifest_path)
    return True


def run_top5_learning_suite(script_path: Path, python_executable: str, run_subprocess, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite_name = manifest.get("name", "top5_learning_suite")
    output_dir = Path(manifest.get("output_dir", "logs/top5_learning_suite"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{suite_name}_summary.md"
    results_jsonl = output_dir / f"{suite_name}_results.jsonl"
    results_tsv = output_dir / f"{suite_name}_results.tsv"
    chart_path = output_dir / f"{suite_name}_summary.svg"
    console_log = output_dir / f"{suite_name}_console.txt"

    base_env = {str(k): str(v) for k, v in manifest.get("base_env", {}).items()}
    experiments = manifest["experiments"]
    reuse_existing = bool(manifest.get("reuse_existing_logs", True))
    records: list[dict[str, object]] = []

    with console_log.open("w", encoding="utf-8") as console:
        _log(console, f"suite_start:{suite_name} experiments:{len(experiments)}")
        for idx, experiment in enumerate(experiments, start=1):
            log_path = output_dir / f"{experiment['run_id']}.txt"
            if reuse_existing and log_path.is_file():
                record = build_record_from_log(experiment, base_env, log_path, wall_ms=0.0, exit_code=None, status="cached")
                records.append(record)
                _log(
                    console,
                    f"experiment_cached:{idx}/{len(experiments)} id:{experiment['id']} "
                    f"plot_bpb:{record['plot_val_bpb']:.8f} artifact:{record['artifact_int8_bytes']} status:{record['status']}",
                )
                continue
            env = os.environ.copy()
            env.update(base_env)
            env.update({str(k): str(v) for k, v in experiment.get("env", {}).items()})
            env["TOP5_SUITE_CHILD"] = "1"
            env["RUN_ID"] = experiment["run_id"]
            env["OUT_DIR"] = str(output_dir)
            _log(console, f"experiment_start:{idx}/{len(experiments)} id:{experiment['id']} run_id:{experiment['run_id']}")
            started = time.perf_counter()
            proc = run_subprocess(
                [python_executable, str(script_path)],
                cwd=str(script_path.parent),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            wall_ms = 1000.0 * (time.perf_counter() - started)
            if proc.stdout:
                console.write(proc.stdout)
                if not proc.stdout.endswith("\n"):
                    console.write("\n")
            if proc.stderr:
                console.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    console.write("\n")
            status = "ok" if proc.returncode == 0 else "crash"
            if proc.returncode != 0:
                _log(console, f"experiment_error:{idx}/{len(experiments)} id:{experiment['id']} exit_code:{proc.returncode}")
            record = build_record_from_log(
                experiment,
                base_env,
                log_path,
                wall_ms=wall_ms,
                exit_code=proc.returncode,
                status=status,
            )
            records.append(record)
            _log(
                console,
                f"experiment_done:{idx}/{len(experiments)} id:{experiment['id']} "
                f"plot_bpb:{record['plot_val_bpb']:.8f} "
                f"artifact:{record['artifact_int8_bytes']} tok_s:{record['tok_s']:.0f} status:{record['status']}",
            )

    baseline = records[0]
    for record in records:
        record["delta_prequant_bpb"] = round(float(record["prequant_val_bpb"]) - float(baseline["prequant_val_bpb"]), 8)
        record["delta_postquant_bpb"] = round(float(record["plot_val_bpb"]) - float(baseline["plot_val_bpb"]), 8)
        record["delta_artifact_bytes"] = int(record["artifact_int8_bytes"]) - int(baseline["artifact_int8_bytes"])

    write_results_jsonl(results_jsonl, records)
    write_results_tsv(results_tsv, records)
    write_summary_markdown(summary_path, records, manifest, chart_path)
    write_summary_svg(chart_path, records)

    print(f"top5_learning_suite_summary:{summary_path}")
    print(f"top5_learning_suite_chart:{chart_path}")
    print(f"top5_learning_suite_results:{results_tsv}")


def _log(console, msg: str) -> None:
    print(msg)
    print(msg, file=console)
    console.flush()


def env_delta(base_env: dict[str, str], overrides: dict[str, object]) -> dict[str, str]:
    delta: dict[str, str] = {}
    for key, value in overrides.items():
        value_str = str(value)
        if base_env.get(str(key)) != value_str:
            delta[str(key)] = value_str
    return delta


def build_record_from_log(
    experiment: dict[str, object],
    base_env: dict[str, str],
    log_path: Path,
    wall_ms: float,
    exit_code: int | None,
    status: str,
) -> dict[str, object]:
    metrics = parse_run_metrics(log_path)
    return {
        "id": experiment["id"],
        "label": experiment["label"],
        "analogue_of": experiment.get("analogue_of", ""),
        "notes": experiment.get("notes", ""),
        "run_id": experiment["run_id"],
        "log_path": str(log_path),
        "wall_ms": round(wall_ms, 1),
        "exit_code": exit_code,
        "status": status if metrics["postquant_val_bpb"] is not None else "partial_" + status,
        "env": env_delta(base_env, experiment.get("env", {})),
        **metrics,
    }


def parse_run_metrics(log_path: Path) -> dict[str, float | int | None]:
    text = log_path.read_text(encoding="utf-8")
    prequant_matches = PREQUANT_RE.findall(text)
    roundtrip_matches = ROUNDTRIP_RE.findall(text)
    serialized_matches = SERIALIZED_RE.findall(text)
    train_matches = TRAIN_STEP_RE.findall(text)
    if not prequant_matches or not train_matches:
        raise ValueError(f"Missing expected metrics in {log_path}")
    _, _, pre_loss, pre_bpb = prequant_matches[-1]
    _, _, _, train_time_ms, step_avg_ms, tok_s = train_matches[-1]
    post_loss = float(roundtrip_matches[-1][0]) if roundtrip_matches else None
    post_bpb = float(roundtrip_matches[-1][1]) if roundtrip_matches else None
    artifact = int(serialized_matches[-1]) if serialized_matches else 0
    plot_bpb = post_bpb if post_bpb is not None else float(pre_bpb)
    return {
        "prequant_val_loss": float(pre_loss),
        "prequant_val_bpb": float(pre_bpb),
        "postquant_val_loss": post_loss,
        "postquant_val_bpb": post_bpb,
        "plot_val_bpb": plot_bpb,
        "artifact_int8_bytes": artifact,
        "train_time_ms": float(train_time_ms),
        "step_avg_ms": float(step_avg_ms),
        "tok_s": float(tok_s),
    }


def write_results_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def write_results_tsv(path: Path, records: list[dict[str, object]]) -> None:
    columns = [
        "id",
        "label",
        "analogue_of",
        "prequant_val_bpb",
        "postquant_val_bpb",
        "delta_postquant_bpb",
        "artifact_int8_bytes",
        "delta_artifact_bytes",
        "tok_s",
        "status",
        "run_id",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for record in records:
            f.write("\t".join(str(record[col]) for col in columns) + "\n")


def write_summary_markdown(path: Path, records: list[dict[str, object]], manifest: dict[str, object], chart_path: Path) -> None:
    baseline = records[0]
    lines = [
        f"# {manifest.get('title', 'Top-5 Learning Suite')}",
        "",
        manifest.get("description", ""),
        "",
        f"- chart: `{chart_path}`",
        f"- baseline analogue: `{baseline['label']}`",
        f"- baseline plot bpb: `{baseline['plot_val_bpb']:.8f}`",
        f"- baseline artifact: `{baseline['artifact_int8_bytes']}`",
        "",
        "## Results",
        "",
        "| Experiment | Analogue | Plot BPB | Delta vs Base | Artifact | Delta Bytes | tok/s | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in sorted(records, key=lambda row: float(row["plot_val_bpb"])):
        lines.append(
            f"| {record['label']} | {record['analogue_of']} | "
            f"{record['plot_val_bpb']:.8f} | {record['delta_postquant_bpb']:+.8f} | "
            f"{record['artifact_int8_bytes']} | {record['delta_artifact_bytes']:+d} | {record['tok_s']:.0f} | {record['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `plot_bpb` uses post-quant BPB when available; if a run crashes after pre-quant eval, it falls back to the pre-quant BPB so the suite can still visualize the direction of the architecture change.",
            "- TTT is intentionally omitted in this MLX suite. The QK-gain and parallel-residual probes are training-time analogues, not legal-eval reproductions.",
            "- SDClip and GPTQ-embedding effects are approximated with standard-deviation-based int8 clipping because this MLX starter script does not implement GPTQ or Hessian collection.",
            "- Depth recurrence here means reusing a contiguous block window for one extra pass inside the forward graph. It is a local analogue, not a faithful PR port.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_svg(path: Path, records: list[dict[str, object]]) -> None:
    width = 1200
    height = 720
    margin = 70
    plot_w = 500
    bar_h = 36
    gap = 14
    top = 80
    font = "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"

    sorted_records = sorted(records, key=lambda row: float(row["delta_postquant_bpb"]))
    max_abs_delta = max(abs(float(r["delta_postquant_bpb"])) for r in records) or 1e-6
    max_artifact = max(int(r["artifact_int8_bytes"]) for r in records)
    min_artifact = min(int(r["artifact_int8_bytes"]) for r in records)
    max_bpb = max(float(r["plot_val_bpb"]) for r in records)
    min_bpb = min(float(r["plot_val_bpb"]) for r in records)

    def bar_x(delta: float) -> tuple[float, float]:
        zero_x = margin + plot_w / 2
        scale = (plot_w / 2 - 20) / max_abs_delta
        if delta >= 0.0:
            return zero_x, delta * scale
        return zero_x + delta * scale, -delta * scale

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f4ee"/>',
        f'<text x="{margin}" y="42" font-family="{font}" font-size="28" font-weight="700" fill="#1f2937">Top-5 Submission Learning Suite</text>',
        f'<text x="{margin}" y="64" font-family="{font}" font-size="14" fill="#475569">Left: post-quant BPB delta vs local base. Right: artifact size vs post-quant BPB.</text>',
        f'<line x1="{margin + plot_w/2}" y1="{top-20}" x2="{margin + plot_w/2}" y2="{top + len(sorted_records)*(bar_h+gap)}" stroke="#94a3b8" stroke-width="1.5"/>',
    ]
    for idx, record in enumerate(sorted_records):
        y = top + idx * (bar_h + gap)
        delta = float(record["delta_postquant_bpb"])
        x, w = bar_x(delta)
        color = "#0f766e" if delta < 0 else "#b45309" if delta > 0 else "#64748b"
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h}" rx="6" fill="{color}" opacity="0.85"/>')
        svg.append(f'<text x="{margin}" y="{y+23}" font-family="{font}" font-size="13" fill="#0f172a">{escape_xml(record["label"])}</text>')
        svg.append(f'<text x="{margin + plot_w - 10}" y="{y+23}" text-anchor="end" font-family="{font}" font-size="12" fill="#0f172a">{delta:+.4f} bpb</text>')

    scatter_x0 = 660
    scatter_y0 = 110
    scatter_w = 470
    scatter_h = 500
    svg.extend(
        [
            f'<rect x="{scatter_x0}" y="{scatter_y0}" width="{scatter_w}" height="{scatter_h}" fill="#fffdf8" stroke="#cbd5e1"/>',
            f'<text x="{scatter_x0}" y="{scatter_y0-18}" font-family="{font}" font-size="18" font-weight="700" fill="#1f2937">Artifact Size vs Post-Quant BPB</text>',
        ]
    )
    for tick in range(5):
        t = tick / 4
        x = scatter_x0 + t * scatter_w
        y = scatter_y0 + (1.0 - t) * scatter_h
        svg.append(f'<line x1="{x:.1f}" y1="{scatter_y0}" x2="{x:.1f}" y2="{scatter_y0+scatter_h}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<line x1="{scatter_x0}" y1="{y:.1f}" x2="{scatter_x0+scatter_w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        artifact_tick = min_artifact + t * (max_artifact - min_artifact)
        bpb_tick = max_bpb - t * (max_bpb - min_bpb)
        svg.append(f'<text x="{x:.1f}" y="{scatter_y0+scatter_h+20}" text-anchor="middle" font-family="{font}" font-size="12" fill="#334155">{artifact_tick:.0f}</text>')
        svg.append(f'<text x="{scatter_x0-10}" y="{y+4:.1f}" text-anchor="end" font-family="{font}" font-size="12" fill="#334155">{bpb_tick:.4f}</text>')
    for record in records:
        art = int(record["artifact_int8_bytes"])
        bpb = float(record["plot_val_bpb"])
        x = scatter_x0 + (art - min_artifact) / max(max_artifact - min_artifact, 1) * scatter_w
        y = scatter_y0 + (max_bpb - bpb) / max(max_bpb - min_bpb, 1e-9) * scatter_h
        if str(record["status"]).startswith("partial"):
            fill = "#7c3aed"
            svg.append(f'<rect x="{x-6:.1f}" y="{y-6:.1f}" width="12" height="12" fill="{fill}" opacity="0.9" transform="rotate(45 {x:.1f} {y:.1f})"/>')
        else:
            fill = "#0f766e" if float(record["delta_postquant_bpb"]) < 0 else "#b45309" if float(record["delta_postquant_bpb"]) > 0 else "#64748b"
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{fill}" opacity="0.9"/>')
        svg.append(f'<text x="{x+10:.1f}" y="{y-10:.1f}" font-family="{font}" font-size="12" fill="#0f172a">{escape_xml(record["id"])}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def escape_xml(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
