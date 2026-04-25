#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_train_log import parse_train_log


CAMPAIGN_DIR = Path(os.environ.get("CAMPAIGN_DIR", ROOT / "logs" / "h100_campaign_20260421"))
JSONL_PATH = CAMPAIGN_DIR / "experiments_master.jsonl"
CSV_PATH = CAMPAIGN_DIR / "experiments_master.csv"
HTML_PATH = CAMPAIGN_DIR / "experiment_dashboard.html"
STATE_PATH = ROOT / ".research" / "runpod_h100_campaign_state.json"
DEFAULT_TRAIN_BATCH_TOKENS = 524_288
WORKSPACE_PREFIX = "/workspace/parameter-golf/"


@dataclass
class Row:
    raw: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.raw.get("run_id", ""))

    @property
    def category(self) -> str:
        return str(self.raw.get("category", ""))

    @property
    def decision(self) -> str:
        return str(self.raw.get("decision", ""))

    @property
    def status(self) -> str:
        return str(self.raw.get("status", ""))

    @property
    def description(self) -> str:
        return str(self.raw.get("description", ""))

    @property
    def env_overrides(self) -> dict[str, Any]:
        env = self.raw.get("env_overrides")
        return env if isinstance(env, dict) else {}

    @property
    def telemetry(self) -> dict[str, Any]:
        telemetry = self.raw.get("telemetry")
        return telemetry if isinstance(telemetry, dict) else {}

    @property
    def val_bpb(self) -> float | None:
        return as_float(self.raw.get("val_bpb"))

    @property
    def val_loss(self) -> float | None:
        return as_float(self.raw.get("val_loss"))

    @property
    def artifact_bytes(self) -> int | None:
        return as_int(self.raw.get("bytes_total_int8_zlib"))

    @property
    def tok_s(self) -> float | None:
        return as_float(self.raw.get("tok_s"))

    @property
    def tok_s_per_gpu(self) -> float | None:
        return as_float(self.raw.get("tok_s_per_gpu"))

    @property
    def step_avg_ms(self) -> float | None:
        return as_float(self.raw.get("step_avg_ms"))

    @property
    def delta_vs_baseline(self) -> float | None:
        return as_float(self.raw.get("delta_vs_baseline"))

    @property
    def delta_vs_parent(self) -> float | None:
        return as_float(self.raw.get("delta_vs_parent"))

    @property
    def seed(self) -> int | None:
        return as_int(self.raw.get("seed"))

    @property
    def train_batch_tokens(self) -> int | None:
        return as_int(self.raw.get("train_batch_tokens"))

    @property
    def world_size(self) -> int | None:
        return as_int(self.raw.get("world_size"))

    @property
    def recipe_signature(self) -> str:
        if not self.env_overrides:
            return "baseline defaults"
        parts = [f"{key}={self.env_overrides[key]}" for key in sorted(self.env_overrides)]
        return ", ".join(parts)


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def as_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}%"


def fmt_int(value: int | None) -> str:
    if value is None:
        return ""
    return f"{value:,}"


def html_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def localize_repo_path(value: Any) -> Path | None:
    if not value:
        return None
    raw = str(value)
    path = Path(raw)
    if path.exists():
        return path
    if raw.startswith(WORKSPACE_PREFIX):
        candidate = ROOT / raw[len(WORKSPACE_PREFIX):]
        return candidate
    if raw.startswith(str(ROOT)):
        return Path(raw)
    return None


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def load_rows() -> list[Row]:
    rows: list[Row] = []
    if not JSONL_PATH.exists():
        return rows
    for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(Row(json.loads(line)))
    return rows


def enrich_rows(rows: list[Row]) -> None:
    for row in rows:
        raw = row.raw

        log_path = localize_repo_path(raw.get("path_log"))
        if log_path and log_path.exists():
            parsed = parse_train_log(log_path)
            for key in (
                "world_size",
                "grad_accum_steps",
                "train_batch_tokens",
                "train_seq_len",
                "warmup_steps",
                "max_wallclock_seconds",
                "train_microbatch_tokens_per_gpu",
                "peak_memory_allocated_mib",
                "peak_memory_reserved_mib",
                "final_eval_time_ms",
                "seed",
            ):
                if raw.get(key) in ("", None) and parsed.get(key) is not None:
                    raw[key] = parsed[key]

        if raw.get("train_batch_tokens") in ("", None):
            raw["train_batch_tokens"] = as_int(row.env_overrides.get("TRAIN_BATCH_TOKENS")) or DEFAULT_TRAIN_BATCH_TOKENS

        if raw.get("world_size") in ("", None):
            raw["world_size"] = 1

        if raw.get("grad_accum_steps") in ("", None):
            override = as_int(row.env_overrides.get("GRAD_ACCUM_STEPS"))
            world_size = as_int(raw.get("world_size")) or 1
            if override is not None:
                raw["grad_accum_steps"] = override
            elif 8 % world_size == 0:
                raw["grad_accum_steps"] = 8 // world_size

        if raw.get("train_microbatch_tokens_per_gpu") in ("", None):
            train_batch_tokens = as_int(raw.get("train_batch_tokens"))
            world_size = as_int(raw.get("world_size")) or 1
            grad_accum_steps = as_int(raw.get("grad_accum_steps")) or 1
            if train_batch_tokens is not None:
                raw["train_microbatch_tokens_per_gpu"] = train_batch_tokens // max(world_size * grad_accum_steps, 1)

        if raw.get("seed") in ("", None):
            raw["seed"] = as_int(row.env_overrides.get("SEED")) or 1337

        if raw.get("tok_s") in ("", None):
            train_batch_tokens = as_float(raw.get("train_batch_tokens"))
            step_avg_ms = as_float(raw.get("step_avg_ms"))
            if train_batch_tokens is not None and step_avg_ms not in (None, 0):
                raw["tok_s"] = round(train_batch_tokens * 1000.0 / step_avg_ms, 2)

        if raw.get("tok_s_per_gpu") in ("", None):
            tok_s = as_float(raw.get("tok_s"))
            world_size = as_int(raw.get("world_size")) or 1
            if tok_s is not None:
                raw["tok_s_per_gpu"] = round(tok_s / world_size, 2)

        telemetry = row.telemetry
        gpu_csv_path = localize_repo_path(telemetry.get("gpu_csv_path"))
        if gpu_csv_path and gpu_csv_path.exists():
            with gpu_csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                gpu_rows = list(reader)
            if gpu_rows:
                util_gpu = [float(item["utilization_gpu_pct"]) for item in gpu_rows]
                util_mem = [float(item["utilization_memory_pct"]) for item in gpu_rows]
                mem_used = [float(item["memory_used_mb"]) for item in gpu_rows]
                mem_total = [float(item["memory_total_mb"]) for item in gpu_rows]
                power = [float(item["power_w"]) for item in gpu_rows]
                temperature = [float(item["temperature_c"]) for item in gpu_rows]
                by_gpu_index: dict[str, list[dict[str, str]]] = defaultdict(list)
                for item in gpu_rows:
                    by_gpu_index[str(item["index"])].append(item)
                vram_pcts = [
                    (used / total) * 100.0
                    for used, total in zip(mem_used, mem_total)
                    if total > 0
                ]
                telemetry["gpu_util_mean_pct"] = round(sum(util_gpu) / len(util_gpu), 2)
                telemetry["gpu_util_max_pct"] = round(max(util_gpu), 2)
                telemetry["memory_controller_util_mean_pct"] = round(sum(util_mem) / len(util_mem), 2)
                telemetry["memory_controller_util_max_pct"] = round(max(util_mem), 2)
                telemetry["gpu_mem_used_mean_mb"] = round(sum(mem_used) / len(mem_used), 2)
                telemetry["gpu_mem_used_max_mb"] = round(max(mem_used), 2)
                telemetry["gpu_mem_total_mb"] = round(max(mem_total), 2)
                telemetry["power_mean_w"] = round(sum(power) / len(power), 2)
                telemetry["power_max_w"] = round(max(power), 2)
                telemetry["temperature_max_c"] = round(max(temperature), 2)
                telemetry["telemetry_samples"] = len(gpu_rows)
                if vram_pcts:
                    telemetry["vram_used_mean_pct"] = round(sum(vram_pcts) / len(vram_pcts), 2)
                    telemetry["vram_used_max_pct"] = round(max(vram_pcts), 2)
                if by_gpu_index:
                    per_gpu_util_means = [
                        sum(float(entry["utilization_gpu_pct"]) for entry in items) / len(items)
                        for items in by_gpu_index.values()
                    ]
                    per_gpu_vram_means = []
                    for items in by_gpu_index.values():
                        valid = [
                            (float(entry["memory_used_mb"]) / float(entry["memory_total_mb"])) * 100.0
                            for entry in items
                            if float(entry["memory_total_mb"]) > 0
                        ]
                        if valid:
                            per_gpu_vram_means.append(sum(valid) / len(valid))
                    telemetry["gpu_count"] = len(by_gpu_index)
                    telemetry["gpu_util_device_mean_min_pct"] = round(min(per_gpu_util_means), 2)
                    telemetry["gpu_util_device_mean_max_pct"] = round(max(per_gpu_util_means), 2)
                    if per_gpu_vram_means:
                        telemetry["vram_used_device_mean_min_pct"] = round(min(per_gpu_vram_means), 2)
                        telemetry["vram_used_device_mean_max_pct"] = round(max(per_gpu_vram_means), 2)
                raw["telemetry"] = telemetry

    baseline = next((row for row in rows if row.raw.get("parent_recipe") == "baseline_anchor" and row.val_bpb is not None), None)
    by_run_id = {row.run_id: row for row in rows}
    for row in rows:
        if row.raw.get("delta_vs_baseline") in ("", None) and baseline and row.val_bpb is not None and baseline.val_bpb is not None:
            row.raw["delta_vs_baseline"] = round(row.val_bpb - baseline.val_bpb, 8)
        parent = by_run_id.get(str(row.raw.get("parent_recipe", "")))
        if row.raw.get("delta_vs_parent") in ("", None) and parent and row.val_bpb is not None and parent.val_bpb is not None:
            row.raw["delta_vs_parent"] = round(row.val_bpb - parent.val_bpb, 8)


def write_csv(rows: list[Row]) -> None:
    fieldnames = [
        "run_id",
        "batch",
        "category",
        "priority",
        "description",
        "status",
        "decision",
        "parent_recipe",
        "seed",
        "started_at",
        "finished_at",
        "duration_seconds",
        "command",
        "env_overrides",
        "path_log",
        "exit_code",
        "iterations",
        "step",
        "step_avg_ms",
        "train_time_ms",
        "tok_s",
        "tok_s_per_gpu",
        "val_loss",
        "val_bpb",
        "model_params",
        "world_size",
        "grad_accum_steps",
        "train_batch_tokens",
        "train_seq_len",
        "train_microbatch_tokens_per_gpu",
        "warmup_steps",
        "max_wallclock_seconds",
        "peak_memory_allocated_mib",
        "peak_memory_reserved_mib",
        "final_eval_time_ms",
        "bytes_model_int8_zlib",
        "bytes_total_int8_zlib",
        "delta_vs_baseline",
        "delta_vs_parent",
        "telemetry",
        "notes",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            raw = dict(row.raw)
            raw["env_overrides"] = json.dumps(row.env_overrides, sort_keys=True)
            raw["telemetry"] = json.dumps(row.telemetry, sort_keys=True)
            writer.writerow({key: raw.get(key, "") for key in fieldnames})


def build_simple_table(columns: list[str], rows: list[list[Any]]) -> str:
    parts = ["<div class='table-wrap'><table><thead><tr>"]
    for column in columns:
        parts.append(f"<th>{html_escape(column)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for value in row:
            parts.append(f"<td><span class='cell-text'>{html_escape(value)}</span></td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def scatter_svg(rows: list[Row], x_key: str, y_key: str, x_label: str, y_label: str) -> str:
    points: list[tuple[float, float, str]] = []
    for row in rows:
        x_val = as_float(row.raw.get(x_key))
        y_val = as_float(row.raw.get(y_key))
        if x_val is None or y_val is None:
            continue
        points.append((x_val, y_val, row.run_id))
    if not points:
        return "<p>No data yet.</p>"

    width = 520
    height = 260
    margin = 36
    x_min = min(x for x, _, _ in points)
    x_max = max(x for x, _, _ in points)
    y_min = min(y for _, y, _ in points)
    y_max = max(y for _, y, _ in points)
    if math.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    def sx(value: float) -> float:
        return margin + (value - x_min) / (x_max - x_min) * (width - 2 * margin)

    def sy(value: float) -> float:
        return height - margin - (value - y_min) / (y_max - y_min) * (height - 2 * margin)

    labels = [
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="#0f766e" opacity="0.78"><title>{html_escape(run_id)} ({x:.2f}, {y:.6f})</title></circle>'
        for x, y, run_id in points
    ]
    return f"""
<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html_escape(x_label)} vs {html_escape(y_label)}">
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#b7ab97" />
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#b7ab97" />
  <text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" fill="#655f57">{html_escape(x_label)}</text>
  <text x="14" y="{height / 2:.1f}" text-anchor="middle" fill="#655f57" transform="rotate(-90 14 {height / 2:.1f})">{html_escape(y_label)}</text>
  <text x="{margin}" y="{height - margin + 18}" fill="#655f57">{html_escape(fmt_float(x_min, 2))}</text>
  <text x="{width - margin}" y="{height - margin + 18}" text-anchor="end" fill="#655f57">{html_escape(fmt_float(x_max, 2))}</text>
  <text x="{margin - 8}" y="{margin + 4}" text-anchor="end" fill="#655f57">{html_escape(fmt_float(y_max, 4))}</text>
  <text x="{margin - 8}" y="{height - margin + 4}" text-anchor="end" fill="#655f57">{html_escape(fmt_float(y_min, 4))}</text>
  {''.join(labels)}
</svg>
"""


def grouped_bar_svg(groups: list[tuple[str, list[tuple[str, float]]]], baseline_bpb: float | None) -> str:
    filtered = [(name, values[:4]) for name, values in groups if values]
    if not filtered:
        return "<p>No sweep data yet.</p>"

    width = 600
    height = 260
    margin = 36
    bar_width = 24
    gap = 12
    group_gap = 28
    deltas = [value - baseline_bpb for _, values in filtered for _, value in values if baseline_bpb is not None]
    if not deltas:
        return "<p>No sweep data yet.</p>"
    min_delta = min(deltas + [0.0])
    max_delta = max(deltas + [0.0])
    span = max(max_delta - min_delta, 1e-9)
    baseline_y = margin + (max_delta / span) * (height - 2 * margin)
    x = margin
    bars: list[str] = []
    labels: list[str] = []
    group_labels: list[str] = []
    for name, values in filtered:
        group_start = x
        for value_name, value in values:
            delta = value - baseline_bpb
            top = margin + (max_delta - max(delta, 0.0)) / span * (height - 2 * margin)
            bottom = margin + (max_delta - min(delta, 0.0)) / span * (height - 2 * margin)
            color = "#0f766e" if delta <= 0 else "#b45309"
            bars.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_width}" height="{max(bottom - top, 1):.1f}" fill="{color}" opacity="0.82">'
                f"<title>{html_escape(name)} {html_escape(value_name)} delta {delta:.6f}</title></rect>"
            )
            labels.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{height - 12}" text-anchor="middle" fill="#655f57" font-size="10">{html_escape(value_name)}</text>'
            )
            x += bar_width + gap
        group_labels.append(
            f'<text x="{group_start + (x - group_start - gap) / 2:.1f}" y="18" text-anchor="middle" fill="#655f57">{html_escape(name)}</text>'
        )
        x += group_gap

    return f"""
<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="Sweep deltas vs baseline">
  <line x1="{margin}" y1="{baseline_y:.1f}" x2="{width - margin}" y2="{baseline_y:.1f}" stroke="#655f57" stroke-dasharray="4 4" />
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#b7ab97" />
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#b7ab97" />
  <text x="{width / 2:.1f}" y="{height - 2}" text-anchor="middle" fill="#655f57">best val_bpb for tested values</text>
  {''.join(group_labels)}
  {''.join(bars)}
  {''.join(labels)}
</svg>
"""


def pareto_rows(rows: list[Row]) -> list[Row]:
    candidates = [row for row in rows if row.val_bpb is not None and row.artifact_bytes is not None]
    kept: list[Row] = []
    for row in sorted(candidates, key=lambda item: (item.artifact_bytes, item.val_bpb)):
        if not kept or row.val_bpb < min(existing.val_bpb for existing in kept if existing.val_bpb is not None):
            kept.append(row)
    return kept


def recipe_groups(rows: list[Row]) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row.recipe_signature].append(row)
    return groups


def axis_groups(rows: list[Row]) -> dict[str, dict[str, list[Row]]]:
    groups: dict[str, dict[str, list[Row]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.env_overrides.items():
            groups[str(key)][str(value)].append(row)
    return groups


def noise_stats(rows: list[Row]) -> tuple[float | None, float | None]:
    baseline_rows = [row for row in rows if row.recipe_signature == "baseline defaults" and row.val_bpb is not None]
    if len(baseline_rows) < 2:
        return None, None
    values = [row.val_bpb for row in baseline_rows if row.val_bpb is not None]
    noise = max(abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :])
    threshold = max(0.002, 2 * noise)
    return noise, threshold


def axis_recommendation(rows: list[Row], baseline_bpb: float | None, baseline_tok_s: float | None, noise: float | None, threshold: float | None) -> str:
    if baseline_bpb is None:
        return "inconclusive"
    best_quality = min((row for row in rows if row.val_bpb is not None), key=lambda row: row.val_bpb, default=None)
    if best_quality is None or best_quality.delta_vs_baseline is None:
        return "inconclusive"
    delta = best_quality.delta_vs_baseline
    if threshold is not None and delta <= -threshold:
        return "confirmed improvement"
    if noise is not None and delta < -noise:
        return "promising, needs rerun"
    if all(row.delta_vs_baseline is not None and row.delta_vs_baseline > (noise or 0.0) for row in rows if row.delta_vs_baseline is not None):
        return "likely harmful"
    best_tok = max((row.tok_s for row in rows if row.tok_s is not None), default=None)
    if best_tok is not None and baseline_tok_s is not None and best_tok > baseline_tok_s * 1.05 and delta >= -(noise or 0.0):
        return "throughput-only gain"
    return "inconclusive"


def main() -> None:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    enrich_rows(rows)
    write_csv(rows)

    state = load_state()
    complete = [row for row in rows if row.status not in {"planned", "running"}]
    failures = [row for row in rows if row.status in {"crash", "contaminated"}]
    best = min((row for row in rows if row.val_bpb is not None), key=lambda row: row.val_bpb, default=None)
    compact = min(
        (row for row in rows if row.val_bpb is not None and row.artifact_bytes is not None and row.artifact_bytes < 10_000_000),
        key=lambda row: row.val_bpb,
        default=None,
    )
    promoted = min(
        (row for row in rows if row.decision == "promote" and row.val_bpb is not None),
        key=lambda row: row.val_bpb,
        default=None,
    )
    baseline = next((row for row in rows if row.raw.get("parent_recipe") == "baseline_anchor"), None)
    baseline_bpb = baseline.val_bpb if baseline else None
    baseline_tok_s = baseline.tok_s if baseline else None
    noise, threshold = noise_stats(rows)

    top_quality = sorted((row for row in rows if row.val_bpb is not None), key=lambda row: row.val_bpb)[:5]
    best_delta = sorted(
        (row for row in rows if row.delta_vs_baseline is not None),
        key=lambda row: row.delta_vs_baseline,
    )[:5]
    pareto = pareto_rows(rows)

    replications: list[list[Any]] = []
    for signature, members in recipe_groups(rows).items():
        values = [row.val_bpb for row in members if row.val_bpb is not None]
        if len(values) < 2:
            continue
        replications.append(
            [
                signature,
                len(values),
                fmt_float(statistics.mean(values)),
                fmt_float(statistics.pstdev(values)),
                fmt_float(min(values)),
                fmt_float(max(values)),
                ", ".join(row.run_id for row in members),
            ]
        )
    replications.sort(key=lambda item: item[2])

    category_rows: list[list[Any]] = []
    category_groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        category_groups[row.category].append(row)
    for category, members in sorted(category_groups.items()):
        best_member = min((row for row in members if row.val_bpb is not None), key=lambda row: row.val_bpb, default=None)
        category_rows.append(
            [
                category,
                len(members),
                best_member.run_id if best_member else "",
                fmt_float(best_member.val_bpb if best_member else None),
                fmt_float(max((row.tok_s for row in members if row.tok_s is not None), default=None), 2),
            ]
        )

    axis_rows: list[list[Any]] = []
    sweep_groups: list[tuple[str, list[tuple[str, float]]]] = []
    for axis, values in sorted(axis_groups(rows).items()):
        value_best = []
        all_rows = [row for members in values.values() for row in members]
        recommendation = axis_recommendation(all_rows, baseline_bpb, baseline_tok_s, noise, threshold)
        for value, members in sorted(values.items()):
            best_member = min((row for row in members if row.val_bpb is not None), key=lambda row: row.val_bpb, default=None)
            if best_member is None:
                continue
            axis_rows.append(
                [
                    axis,
                    value,
                    best_member.run_id,
                    fmt_float(best_member.val_bpb),
                    fmt_float(best_member.delta_vs_baseline),
                    fmt_float(best_member.tok_s, 2),
                    recommendation,
                ]
            )
            value_best.append((value, best_member.val_bpb))
        if value_best:
            sweep_groups.append((axis, value_best))

    non_hparam_rows: list[list[Any]] = []
    for category in sorted({"throughput_engineering", "architecture_compact", "architecture_variant", "data_ordering", "data_sampling"}):
        members = [row for row in rows if row.category == category]
        if not members:
            continue
        best_member = min((row for row in members if row.val_bpb is not None), key=lambda row: row.val_bpb, default=None)
        fastest = max((row for row in members if row.tok_s is not None), key=lambda row: row.tok_s, default=None)
        non_hparam_rows.append(
            [
                category,
                best_member.run_id if best_member else "",
                fmt_float(best_member.val_bpb if best_member else None),
                fastest.run_id if fastest else "",
                fmt_float(fastest.tok_s if fastest else None, 2),
            ]
        )

    budget_note = state.get("budget_note") or state.get("remaining_credit_note") or "budget note not synced in state file"
    promoted_group_size = len(recipe_groups(rows).get(promoted.recipe_signature, [])) if promoted else 0
    promoted_replicated = promoted_group_size >= 2
    current_recipe_text = promoted.run_id if promoted else (best.run_id if best else "pending")
    stop_guidance = (
        "Screening should stop soon if the promoted recipe confirms on a rerun."
        if promoted_replicated
        else "Do not move to 8xH100 yet without one more confirmation of the promoted recipe."
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>H100 Campaign Dashboard</title>
  <style>
    :root {{
      --bg: #f6f2e9;
      --ink: #1d1d1b;
      --muted: #655f57;
      --card: #fffdf8;
      --line: #d8cfc1;
      --accent: #0f766e;
      --warn: #92400e;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: linear-gradient(180deg, #efe7d8 0%, var(--bg) 42%, #f9f6ef 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      line-height: 1.05;
    }}
    p {{
      color: var(--muted);
      margin: 0 0 14px;
    }}
    .cards, .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 22px 0 28px;
    }}
    .card, .section {{
      min-width: 0;
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(29, 29, 27, 0.05);
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 26px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 250, 241, 0.78);
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      table-layout: fixed;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #fff9ef;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    tbody tr:nth-child(even) {{
      background: rgba(255, 251, 244, 0.74);
    }}
    .cell-text {{
      display: inline-block;
      max-width: 100%;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      padding-left: 12px;
      margin: 16px 0 0;
    }}
    .warning {{
      border-left-color: var(--warn);
    }}
    .chart {{
      display: block;
      width: 100%;
      height: auto;
      background: #fffaf1;
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 22px 14px 40px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      .card, .section {{
        padding: 15px;
      }}
      th, td {{
        font-size: 12px;
        padding: 9px 7px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>1xH100 Screening Campaign</h1>
    <p>Decision dashboard for the attached H100 screening plan. Throughput engineering remains an explicit first-class workstream here, not a side quest.</p>
    <div class="cards">
      <div class="card"><div class="label">Total runs</div><div class="value">{len(rows)}</div></div>
      <div class="card"><div class="label">Completed runs</div><div class="value">{len(complete)}</div></div>
      <div class="card"><div class="label">Failed runs</div><div class="value">{len(failures)}</div></div>
      <div class="card"><div class="label">Best val_bpb</div><div class="value">{fmt_float(best.val_bpb if best else None)}</div></div>
      <div class="card"><div class="label">Best compact run</div><div class="value">{html_escape(compact.run_id if compact else "pending")}</div></div>
      <div class="card"><div class="label">Promoted recipe</div><div class="value">{html_escape(current_recipe_text)}</div></div>
      <div class="card"><div class="label">Baseline anchor</div><div class="value">{html_escape(baseline.run_id if baseline else "pending")}</div></div>
      <div class="card"><div class="label">Budget note</div><div class="value">{html_escape(str(budget_note))}</div></div>
    </div>

    <section class="section">
      <h2>Narrative Summary</h2>
      <p>The current baseline anchor is <strong>{html_escape(baseline.run_id if baseline else "pending")}</strong> at <strong>{fmt_float(baseline_bpb)}</strong> `val_bpb`. The current promoted recipe is <strong>{html_escape(current_recipe_text)}</strong>{' and it has at least one same-recipe replication.' if promoted_replicated else ' but it still needs a clean confirmation rerun.'}</p>
      <p>Throughput engineering is still on-plan. The current best throughput-led candidate is <strong>{html_escape(promoted.run_id if promoted else (best.run_id if best else "pending"))}</strong> with <strong>{fmt_float(promoted.tok_s if promoted else (best.tok_s if best else None), 2)}</strong> global tokens per second and <strong>{fmt_float(promoted.val_bpb if promoted else (best.val_bpb if best else None))}</strong> `val_bpb`.</p>
      <p class="callout">Noise estimate from baseline-style replications: <strong>{fmt_float(noise)}</strong>. Automatic keep threshold from the plan: <strong>{fmt_float(threshold)}</strong>. {html_escape(stop_guidance)}</p>
    </section>

    <div class="grid">
      <section class="section">
        <h2>Best Raw Quality</h2>
        {build_simple_table(
            ["run_id", "category", "val_bpb", "delta_vs_baseline", "tok_s", "description"],
            [[row.run_id, row.category, fmt_float(row.val_bpb), fmt_float(row.delta_vs_baseline), fmt_float(row.tok_s, 2), row.description] for row in top_quality],
        )}
      </section>
      <section class="section">
        <h2>Best Delta Vs Baseline</h2>
        {build_simple_table(
            ["run_id", "category", "delta_vs_baseline", "val_bpb", "tok_s", "decision"],
            [[row.run_id, row.category, fmt_float(row.delta_vs_baseline), fmt_float(row.val_bpb), fmt_float(row.tok_s, 2), row.decision] for row in best_delta],
        )}
      </section>
    </div>

    <div class="grid">
      <section class="section">
        <h2>Pareto Frontier</h2>
        {build_simple_table(
            ["run_id", "artifact_bytes", "val_bpb", "tok_s", "category"],
            [[row.run_id, fmt_int(row.artifact_bytes), fmt_float(row.val_bpb), fmt_float(row.tok_s, 2), row.category] for row in pareto],
        )}
      </section>
      <section class="section">
        <h2>Replications</h2>
        {build_simple_table(
            ["recipe", "n", "mean", "std", "min", "max", "runs"],
            replications or [["no replicated recipe groups yet", "", "", "", "", "", ""]],
        )}
      </section>
    </div>

    <div class="grid">
      <section class="section">
        <h2>Bytes Vs Quality</h2>
        {scatter_svg(rows, "bytes_total_int8_zlib", "val_bpb", "bytes_total_int8_zlib", "val_bpb")}
      </section>
      <section class="section">
        <h2>Step Time Vs Quality</h2>
        {scatter_svg(rows, "step_avg_ms", "val_bpb", "step_avg_ms", "val_bpb")}
      </section>
    </div>

    <section class="section">
      <h2>Key Sweeps</h2>
      <p class="muted">Grouped bars show the best `val_bpb` reached for each tested value relative to the baseline anchor.</p>
      {grouped_bar_svg(sweep_groups[:5], baseline_bpb)}
    </section>

    <div class="grid">
      <section class="section">
        <h2>Category Summary</h2>
        {build_simple_table(
            ["category", "runs", "best_run", "best_val_bpb", "best_tok_s"],
            category_rows,
        )}
      </section>
      <section class="section">
        <h2>Decision Intelligence</h2>
        {build_simple_table(
            ["axis", "value", "best_run", "best_val_bpb", "delta_vs_baseline", "tok_s", "assessment"],
            axis_rows or [["no axis sweeps yet", "", "", "", "", "", ""]],
        )}
      </section>
    </div>

    <section class="section">
      <h2>Non-Hparam Interventions</h2>
      {build_simple_table(
          ["category", "best_quality_run", "best_quality_bpb", "fastest_run", "fastest_tok_s"],
          non_hparam_rows or [["no non-hparam interventions yet", "", "", "", ""]],
      )}
    </section>

    <section class="section">
      <h2>Run Table</h2>
      {build_simple_table(
          [
              "run_id",
              "category",
              "status",
              "decision",
              "val_bpb",
              "tok_s",
              "gpu_util_mean",
              "vram_used_mean",
              "peak_mem_reserved_mib",
              "description",
          ],
          [
              [
                  row.run_id,
                  row.category,
                  row.status,
                  row.decision,
                  fmt_float(row.val_bpb),
                  fmt_float(row.tok_s, 2),
                  fmt_pct(as_float(row.telemetry.get("gpu_util_mean_pct"))),
                  fmt_pct(as_float(row.telemetry.get("vram_used_mean_pct"))),
                  fmt_int(as_int(row.raw.get("peak_memory_reserved_mib"))),
                  row.description,
              ]
              for row in rows
          ],
      )}
      <p class="callout warning">Collection now covers raw quality, artifact size, derived throughput, GPU utilization, VRAM occupancy, power, temperature, and CUDA peak memory. The main remaining manual input is the live credit balance note.</p>
    </section>
  </main>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
