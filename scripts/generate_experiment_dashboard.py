#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "logs" / "experiment_dashboard.html"

EQ_RE = re.compile(r"\b([A-Z][A-Z0-9_]+)=([A-Za-z0-9._+-]+)")


@dataclass
class Experiment:
    source: str
    batch: str
    phase: str
    name: str
    status: str
    val_bpb: float | None
    val_loss: float | None
    artifact: int | None
    tok_s: float | None
    delta_bpb: float | None
    baseline_spread: float | None
    train_time_ms: float | None
    params: dict[str, str]
    notes: str


VALID_STATUSES = {"improve", "keep", "ok", "mixed", "discard", "maybe", "worse", "contaminated"}


def safe_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def parse_params(*chunks: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, value in EQ_RE.findall(chunk):
            params[key] = value
    return params


def add_experiment(experiments: list[Experiment], **kwargs: Any) -> None:
    experiments.append(Experiment(**kwargs))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def collect_results_jsonl(experiments: list[Experiment]) -> None:
    for row in load_jsonl(ROOT / "results.jsonl"):
        env = row.get("env") or {}
        add_experiment(
            experiments,
            source="results.jsonl",
            batch=str(row.get("phase", "unknown")),
            phase=str(row.get("phase", "unknown")),
            name=str(row.get("change_summary") or row.get("run_id") or "run"),
            status=str(row.get("status", "unknown")),
            val_bpb=safe_float(row.get("val_bpb")),
            val_loss=safe_float(row.get("val_loss")),
            artifact=safe_int(row.get("artifact_int8_bytes")),
            tok_s=safe_float(row.get("tok_s")),
            delta_bpb=None,
            baseline_spread=None,
            train_time_ms=safe_float(row.get("train_time_ms")),
            params={k.upper(): str(v) for k, v in env.items() if k not in {"python", "log_path", "sandbox"}},
            notes=str(row.get("reason", "")),
        )


def collect_ablation_jsonl(experiments: list[Experiment], relpath: str, batch: str) -> None:
    for row in load_jsonl(ROOT / relpath):
        params = parse_params(str(row.get("candidate", "")))
        add_experiment(
            experiments,
            source=relpath,
            batch=batch,
            phase=str(row.get("phase", batch)),
            name=str(row.get("candidate", "candidate")),
            status=str(row.get("status", "unknown")),
            val_bpb=safe_float(row.get("candidate_bpb")),
            val_loss=None,
            artifact=safe_int(row.get("candidate_artifact_int8_bytes")),
            tok_s=safe_float(row.get("candidate_tok_s")),
            delta_bpb=safe_float(row.get("delta_bpb")),
            baseline_spread=safe_float(row.get("baseline_spread")),
            train_time_ms=None,
            params=params,
            notes=f"baseline_mean_bpb={row.get('baseline_mean_bpb')}",
        )


def collect_grid_jsonl(experiments: list[Experiment], relpath: str, batch: str) -> None:
    for row in load_jsonl(ROOT / relpath):
        params = {k: str(v) for k, v in (row.get("env") or {}).items()}
        add_experiment(
            experiments,
            source=relpath,
            batch=batch,
            phase=batch,
            name=str(row.get("label", row.get("id", "candidate"))),
            status=str(row.get("status", "unknown")),
            val_bpb=safe_float(row.get("postquant_val_bpb") or row.get("plot_val_bpb")),
            val_loss=safe_float(row.get("postquant_val_loss")),
            artifact=safe_int(row.get("artifact_int8_bytes")),
            tok_s=safe_float(row.get("tok_s")),
            delta_bpb=safe_float(row.get("delta_postquant_bpb")),
            baseline_spread=None,
            train_time_ms=safe_float(row.get("train_time_ms")),
            params=params,
            notes=str(row.get("notes", "")),
        )


def collect_morning_runs(experiments: list[Experiment]) -> None:
    files = [
        ("morning_relu_10m", "relu2 10m", "relu2"),
        ("morning_silu_10m", "silu 10m", "silu"),
        ("morning_silu2_10m", "silu2 10m", "silu2"),
    ]
    for stem, label, act in files:
        path = ROOT / "logs" / f"{stem}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        val = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)", text)
        art = re.search(r"serialized_model_int8_zlib:(\d+) bytes", text)
        stop = re.search(r"stopping_early: wallclock_cap train_time:(\d+)ms step:(\d+)/(\d+)", text)
        add_experiment(
            experiments,
            source=str(path.relative_to(ROOT)),
            batch="morning_10m_activation",
            phase="morning_10m_activation",
            name=label,
            status="ok" if val else "missing_final",
            val_bpb=safe_float(val.group(2) if val else None),
            val_loss=safe_float(val.group(1) if val else None),
            artifact=safe_int(art.group(1) if art else None),
            tok_s=None,
            delta_bpb=None,
            baseline_spread=None,
            train_time_ms=safe_float(stop.group(1) if stop else None),
            params={"MLP_ACT": act, "MAX_WALLCLOCK_SECONDS": "600"},
            notes="Matched 10-minute activation comparison",
        )


def is_valid_metric_run(exp: Experiment) -> bool:
    return (
        exp.status in VALID_STATUSES
        and exp.val_bpb is not None
        and exp.val_bpb > 0
        and exp.artifact is not None
        and exp.artifact > 0
    )


def build_svg_scatter(points: list[Experiment], width: int = 860, height: int = 360) -> str:
    usable = [p for p in points if is_valid_metric_run(p)]
    if not usable:
        return "<p>No scatter data available.</p>"
    min_x = min(p.artifact for p in usable)
    max_x = max(p.artifact for p in usable)
    min_y = min(p.val_bpb for p in usable)
    max_y = max(p.val_bpb for p in usable)
    left, right, top, bottom = 58, 18, 16, 34
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        if max_y == min_y:
            return top + plot_h / 2
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    colors = {
        "improve": "#0f766e",
        "mixed": "#1d4ed8",
        "worse": "#b91c1c",
        "keep": "#0f766e",
        "maybe": "#7c3aed",
        "discard": "#b91c1c",
        "crash": "#6b7280",
        "contaminated": "#92400e",
        "ok": "#2563eb",
    }

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="BPB vs artifact scatter">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')
    for tick in range(5):
        x = left + plot_w * tick / 4
        val = min_x + (max_x - min_x) * tick / 4
        parts.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+5}" stroke="#374151"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-8}" font-size="11" text-anchor="middle" fill="#374151">{int(val):,}</text>')
    for tick in range(5):
        y = top + plot_h * tick / 4
        val = max_y - (max_y - min_y) * tick / 4
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#374151"/>')
        parts.append(f'<text x="{left-9}" y="{y+4:.1f}" font-size="11" text-anchor="end" fill="#374151">{val:.3f}</text>')
    parts.append(f'<text x="{left + plot_w/2:.1f}" y="{height-2}" font-size="12" text-anchor="middle" fill="#111827">Artifact bytes (int8+zlib)</text>')
    parts.append(f'<text x="14" y="{top + plot_h/2:.1f}" font-size="12" text-anchor="middle" transform="rotate(-90 14 {top + plot_h/2:.1f})" fill="#111827">val_bpb (lower is better)</text>')
    for p in usable:
        color = colors.get(p.status, "#374151")
        title = html.escape(f"{p.name} | {p.batch} | {p.status} | bpb={p.val_bpb:.6f} | bytes={p.artifact:,}")
        parts.append(
            f'<circle cx="{sx(p.artifact):.1f}" cy="{sy(p.val_bpb):.1f}" r="4.5" fill="{color}" fill-opacity="0.85">'
            f"<title>{title}</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def build_svg_bars(experiments: list[Experiment], width: int = 860, bar_h: int = 22) -> str:
    usable = [e for e in experiments if is_valid_metric_run(e)]
    usable.sort(key=lambda e: e.val_bpb)
    top = usable[:12]
    if not top:
        return "<p>No ranked experiments available.</p>"
    left = 240
    right = 70
    top_pad = 12
    height = top_pad + len(top) * bar_h + 28
    plot_w = width - left - right
    min_v = min(e.val_bpb for e in top)
    max_v = max(e.val_bpb for e in top)
    if math.isclose(min_v, max_v):
        max_v = min_v + 1e-6
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Best experiments by BPB">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>')
    for i, exp in enumerate(top):
        y = top_pad + i * bar_h
        frac = (exp.val_bpb - min_v) / (max_v - min_v)
        bw = max(6, plot_w * (1 - frac))
        parts.append(f'<text x="8" y="{y+15}" font-size="12" fill="#111827">{html.escape(exp.name[:42])}</text>')
        parts.append(f'<rect x="{left}" y="{y+3}" width="{bw:.1f}" height="14" rx="3" fill="#2563eb" fill-opacity="0.85"/>')
        parts.append(f'<text x="{left+bw+6:.1f}" y="{y+15}" font-size="11" fill="#111827">{exp.val_bpb:.6f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def html_table(rows: list[Experiment]) -> str:
    out = [
        "<div class='table-wrap'><table class='data-table'>",
        "<thead><tr><th>Batch</th><th>Phase</th><th>Name</th><th>Status</th><th>val_bpb</th><th>Artifact</th><th>tok/s</th><th>Delta</th><th>Params</th></tr></thead>",
        "<tbody>",
    ]
    for e in rows:
        params = ", ".join(f"{k}={v}" for k, v in sorted(e.params.items()))[:180]
        out.append(
            "<tr>"
            f"<td class='cell-batch'>{html.escape(e.batch)}</td>"
            f"<td class='cell-phase'>{html.escape(e.phase)}</td>"
            f"<td class='cell-name'>{html.escape(e.name)}</td>"
            f"<td class='cell-status'><span class='status-pill status-{html.escape(e.status)}'>{html.escape(e.status)}</span></td>"
            f"<td class='cell-metric'>{'' if e.val_bpb is None else f'{e.val_bpb:.8f}'}</td>"
            f"<td class='cell-metric'>{'' if e.artifact is None else f'{e.artifact:,}'}</td>"
            f"<td class='cell-metric'>{'' if e.tok_s is None else f'{e.tok_s:.0f}'}</td>"
            f"<td class='cell-metric'>{'' if e.delta_bpb is None else f'{e.delta_bpb:+.8f}'}</td>"
            f"<td class='cell-params'><code>{html.escape(params)}</code></td>"
            "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def main() -> None:
    experiments: list[Experiment] = []

    collect_results_jsonl(experiments)
    collect_ablation_jsonl(experiments, "logs/overnight_local_ab/ab_results.jsonl", "overnight_round1")
    for round_id in range(2, 9):
        collect_ablation_jsonl(experiments, f"logs/overnight_local_ab_round{round_id}/ab_results.jsonl", f"overnight_round{round_id}")
    collect_ablation_jsonl(experiments, "logs/one_hour_silu_embed_ab/results.jsonl", "one_hour_silu_embed")
    collect_grid_jsonl(experiments, "logs/top5_learning_suite/top5_learning_suite_results.jsonl", "top5_learning_suite")
    collect_grid_jsonl(experiments, "logs/local_grid_rope_muon/local_grid_rope_muon_results.jsonl", "local_grid_rope_muon")
    collect_morning_runs(experiments)

    experiments = [e for e in experiments if e.val_bpb is not None]
    experiments.sort(key=lambda e: (e.val_bpb if e.val_bpb is not None else 1e9, e.batch, e.name))
    valid_metric_runs = [e for e in experiments if is_valid_metric_run(e)]

    status_counts = Counter(e.status for e in experiments)
    batch_counts = Counter(e.batch for e in experiments)
    param_values: dict[str, set[str]] = defaultdict(set)
    for e in experiments:
        for key, value in e.params.items():
            param_values[key].add(value)

    top_clean = [e for e in valid_metric_runs if e.status in {"improve", "keep", "ok", "mixed"}]
    top_clean.sort(key=lambda e: e.val_bpb if e.val_bpb is not None else 1e9)
    top_clean = top_clean[:20]

    param_rows = []
    for key in sorted(param_values):
        values = sorted(param_values[key], key=lambda x: (len(x), x))
        sample = ", ".join(values[:10])
        if len(values) > 10:
            sample += f" … (+{len(values)-10} more)"
        param_rows.append((key, len(values), sample))

    batch_summary = []
    for batch in sorted(batch_counts):
        batch_exps = [e for e in experiments if e.batch == batch]
        valid_batch_exps = [e for e in batch_exps if is_valid_metric_run(e)]
        best_pool = valid_batch_exps or batch_exps
        best = min(best_pool, key=lambda e: e.val_bpb if e.val_bpb is not None else 1e9)
        batch_summary.append((batch, len(batch_exps), best.name, best.val_bpb, best.status))

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Parameter Golf Experiment Dashboard</title>",
        "<style>",
        "body{font-family:Inter,system-ui,Segoe UI,sans-serif;margin:0;background:#f5f7fb;color:#111827}",
        ":root{--bg:#f3f5f9;--surface:#ffffff;--surface-2:#f8fafc;--line:#e5e7eb;--text:#111827;--muted:#6b7280;--accent:#2563eb;--accent-soft:#dbeafe;--good:#0f766e;--good-soft:#ccfbf1;--warn:#92400e;--warn-soft:#fef3c7;--bad:#b91c1c;--bad-soft:#fee2e2;}",
        "*{box-sizing:border-box}",
        "body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;margin:0;background:linear-gradient(180deg,#f8fafc 0%,#eef2f7 100%);color:var(--text)}",
        ".wrap{max-width:1560px;margin:0 auto;padding:28px 20px 48px}",
        "h1{margin:0 0 8px;font-size:clamp(2rem,3.4vw,3.2rem);line-height:1;letter-spacing:-0.04em}",
        "h2{margin:0 0 10px;font-size:1.05rem;line-height:1.2}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0 24px}",
        ".card{min-width:0;background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:18px;padding:16px 16px 18px;box-shadow:0 10px 30px rgba(15,23,42,.06)}",
        ".muted{color:var(--muted);font-size:13px;line-height:1.45}",
        ".section{margin-top:24px}",
        ".two{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,.8fr);gap:16px;align-items:start}",
        ".small{font-size:12px;line-height:1.4;overflow-wrap:anywhere}",
        ".metric{font-size:32px;font-weight:700;line-height:1;letter-spacing:-0.03em;overflow-wrap:anywhere}",
        "svg{display:block;width:100%;height:auto;max-width:100%}",
        ".table-wrap{overflow:auto;border:1px solid var(--line);border-radius:18px;background:var(--surface)}",
        "table{width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed}",
        "th,td{padding:10px 12px;border-bottom:1px solid #eef2f7;font-size:13px;text-align:left;vertical-align:top;overflow-wrap:anywhere;word-break:break-word}",
        "th{background:var(--surface-2);position:sticky;top:0;z-index:1;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#475569}",
        "tbody tr:nth-child(even){background:rgba(248,250,252,.7)}",
        ".cell-batch{width:12%}",
        ".cell-phase{width:11%;color:#475569}",
        ".cell-name{width:23%;font-weight:600}",
        ".cell-status{width:9%}",
        ".cell-metric{width:8%;font-variant-numeric:tabular-nums;white-space:nowrap}",
        ".cell-params{width:21%}",
        ".status-pill{display:inline-flex;align-items:center;max-width:100%;padding:4px 8px;border-radius:999px;font-size:12px;font-weight:600;line-height:1.2;overflow-wrap:anywhere}",
        ".status-improve,.status-keep,.status-ok{background:var(--good-soft);color:var(--good)}",
        ".status-mixed,.status-maybe{background:#ede9fe;color:#6d28d9}",
        ".status-worse,.status-discard,.status-crash{background:var(--bad-soft);color:var(--bad)}",
        ".status-contaminated,.status-cached,.status-partial_cached,.status-partial_crash{background:var(--warn-soft);color:var(--warn)}",
        "code{background:#eef2ff;padding:2px 6px;border-radius:7px;font-size:12px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}",
        "@media (max-width: 1100px){.two{grid-template-columns:1fr}.wrap{padding-inline:14px}}",
        "@media (max-width: 720px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.card{padding:14px}.metric{font-size:26px}th,td{padding:9px 10px;font-size:12px}}",
        "</style></head><body><div class='wrap'>",
        "<h1>Parameter Golf Experiment Dashboard</h1>",
        "<p class='muted'>Aggregated from the structured local experiment ledgers, overnight A/B runs, top-5 learning suite, grid searches, and long activation runs.</p>",
        "<div class='grid'>",
        f"<div class='card'><div class='muted'>Experiments</div><div class='metric'>{len(experiments)}</div></div>",
        f"<div class='card'><div class='muted'>Batches</div><div class='metric'>{len(batch_counts)}</div></div>",
        f"<div class='card'><div class='muted'>Best val_bpb</div><div class='metric'>{valid_metric_runs[0].val_bpb:.6f}</div><div class='small'>{html.escape(valid_metric_runs[0].name)}</div></div>",
        f"<div class='card'><div class='muted'>Parameters touched</div><div class='metric'>{len(param_values)}</div></div>",
        "</div>",
        "<div class='grid'>",
    ]
    for status, count in sorted(status_counts.items()):
        html_parts.append(f"<div class='card'><div class='muted'>{html.escape(status)}</div><div class='metric'>{count}</div></div>")
    html_parts.append("</div>")

    html_parts.extend([
        "<div class='section two'>",
        f"<div class='card'><h2>BPB vs Artifact</h2><div class='muted'>Completed structured experiments with valid final metrics. Lower and further left is better.</div>{build_svg_scatter(valid_metric_runs)}</div>",
        f"<div class='card'><h2>Best Recorded Experiments</h2><div class='muted'>Sorted by lowest recorded <code>val_bpb</code>.</div>{build_svg_bars(top_clean)}</div>",
        "</div>",
        "<div class='section two'>",
        "<div class='card'><h2>Batch Winners</h2><table><thead><tr><th>Batch</th><th>Rows</th><th>Best</th><th>val_bpb</th><th>Status</th></tr></thead><tbody>",
    ])
    for batch, count, best_name, best_bpb, best_status in batch_summary:
        html_parts.append(
            f"<tr><td>{html.escape(batch)}</td><td>{count}</td><td>{html.escape(best_name)}</td><td>{best_bpb:.8f}</td><td>{html.escape(best_status)}</td></tr>"
        )
    html_parts.append("</tbody></table></div>")
    html_parts.append("<div class='card'><h2>Parameters Tested</h2><table><thead><tr><th>Parameter</th><th>Distinct values</th><th>Observed values</th></tr></thead><tbody>")
    for key, count, sample in param_rows:
        html_parts.append(f"<tr><td><code>{html.escape(key)}</code></td><td>{count}</td><td>{html.escape(sample)}</td></tr>")
    html_parts.append("</tbody></table></div></div>")

    html_parts.append("<div class='section card'><h2>All Experiments</h2><div class='muted'>Complete structured history loaded into this dashboard.</div>")
    html_parts.append(html_table(experiments))
    html_parts.append("</div>")
    html_parts.append("</div></body></html>")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(html_parts), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
