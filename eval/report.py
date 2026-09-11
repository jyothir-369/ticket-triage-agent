"""Report generation — console, JSON, and HTML outputs for an eval run.

Functions
---------
- ``print_console_report`` — pass/fail summary + ASCII confusion matrices.
- ``save_json_report``     — ``eval/results/latest.json`` (and custom paths).
- ``save_html_report``     — self-contained ``report.html`` dashboard with
  heatmap confusion matrices, big-number metric cards, and a failing-case list.

Nothing here talks to the agent or the network — it is pure rendering of the
:class:`eval.metrics.EvalMetrics` object plus the per-ticket comparison rows.
"""

from __future__ import annotations

import html as html_module
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics import EvalMetrics

# Default output directory.
RESULTS_DIR = Path(__file__).parent / "results"

# CI / CLI pass thresholds (mirrors .github/workflows/ci.yml).
PASS_THRESHOLDS = {
    "category_accuracy": 0.70,
    "escalation_precision": 0.60,
}

# Brand palette used by the HTML dashboard.
BRAND = "#6366f1"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════


def failure_reasons(comp: dict[str, Any]) -> list[str]:
    """Human-readable list of why a comparison row failed (or empty)."""
    reasons: list[str] = []
    if not comp.get("category_match"):
        reasons.append(
            f"category: expected {comp.get('expected_category')} · "
            f"got {comp.get('actual_category')}"
        )
    if not comp.get("urgency_match"):
        reasons.append(
            f"urgency: expected {comp.get('expected_urgency')} · got {comp.get('actual_urgency')}"
        )
    if not comp.get("escalation_correct"):
        reasons.append(
            f"escalation: expected {comp.get('expected_should_escalate')} · "
            f"got {comp.get('actual_should_escalate')}"
        )
    return reasons


def passed(metrics: EvalMetrics) -> dict[str, bool]:
    """Whether each pass threshold is met by *metrics*."""
    return {
        "category_accuracy": (metrics.category_accuracy >= PASS_THRESHOLDS["category_accuracy"]),
        "escalation_precision": (
            metrics.escalation_precision >= PASS_THRESHOLDS["escalation_precision"]
        ),
    }


def run_metadata(source: str | None, concurrency: int) -> dict[str, Any]:
    """Common run metadata embedded in every report."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "source": source or "all",
        "concurrency": concurrency,
        "thresholds": PASS_THRESHOLDS,
    }


def ensure_results_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Console report
# ═══════════════════════════════════════════════════════════════════════════════


def print_console_report(
    metrics: EvalMetrics,
    comparisons: list[dict[str, Any]],
    source: str | None = None,
    concurrency: int = 5,
) -> None:
    """Pretty-print a pass/fail summary plus both confusion matrices."""
    bar = "=" * 74
    thin = "-" * 74
    meta = run_metadata(source, concurrency)

    print()
    print(bar)
    print("  EVALUATION REPORT — Support-Ticket Triage Agent")
    print(bar)
    print(f"  Generated : {meta['generated_at']} UTC")
    print(f"  Source    : {meta['source']}")
    print(f"  Tickets   : {metrics.total}")
    print(f"  Concurrency: {concurrency}")
    print(thin)

    print("\n  AGGREGATE METRICS")
    print(
        f"    Category accuracy           : {metrics.category_accuracy:.1%}"
        f"   ({metrics.n_correct_category}/{metrics.total})"
    )
    print(
        f"    Urgency accuracy            : {metrics.urgency_accuracy:.1%}"
        f"   ({metrics.n_correct_urgency}/{metrics.total})"
    )
    print(f"    Mean draft ROUGE-L          : {metrics.mean_draft_rouge_l:.3f}")
    print(f"    Escalation precision        : {metrics.escalation_precision:.1%}")
    print(f"    Escalation recall           : {metrics.escalation_recall:.1%}")
    print(f"    Escalation F1               : {metrics.escalation_f1:.1%}")
    print(f"    Escalation accuracy         : {metrics.escalation_accuracy:.1%}")
    print(f"    False escalation rate       : {metrics.false_escalation_rate:.1%}")
    print(f"    Calibration error           : {metrics.calibration_error:.1%}")
    print(
        f"    Latency p50 / p95           : {metrics.duration_p50_ms:.0f} ms / "
        f"{metrics.duration_p95_ms:.0f} ms"
    )

    checks = passed(metrics)
    print("\n  PASS / FAIL")
    for name, ok in checks.items():
        marker = "PASS" if ok else "FAIL"
        print(f"    {name:24s} {marker}")
    overall = all(checks.values())
    print(
        f"    {'overall'.ljust(24)} {'PASS' if overall else 'FAIL'} "
        f"(category_accuracy ≥ {PASS_THRESHOLDS['category_accuracy']:.0%} AND "
        f"escalation_precision ≥ {PASS_THRESHOLDS['escalation_precision']:.0%})"
    )

    _print_confusion_matrix(
        metrics.category_confusion_matrix,
        metrics.category_labels,
        "CATEGORY CONFUSION MATRIX (rows = expected · cols = predicted)",
    )
    _print_confusion_matrix(
        metrics.urgency_confusion_matrix,
        metrics.urgency_labels,
        "URGENCY CONFUSION MATRIX (rows = expected · cols = predicted)",
    )

    print("\n  PER-SOURCE BREAKDOWN")
    print(
        f"    {'source':12s} {'count':>5s} {'cat_acc':>8s} {'urg_acc':>8s} "
        f"{'esc_acc':>8s} {'draft_r':>8s}"
    )
    for src, row in metrics.by_source.items():
        print(
            f"    {src:12s} {row['total']:5d} "
            f"{row['category_accuracy']:7.1%} {row['urgency_accuracy']:7.1%} "
            f"{row['escalation_accuracy']:7.1%} {row['mean_draft_rouge_l']:7.3f}"
        )

    failing = [c for c in comparisons if failure_reasons(c)]
    print(f"\n  FAILING CASES  ({len(failing)} of {len(comparisons)})")
    if not failing:
        print("    None — every ticket passed every check. 🎉")
    for comp in failing:
        reasons = "; ".join(failure_reasons(comp))
        print(
            f"    - {comp['id']} ({comp.get('source')}): {reasons}{' · ' if reasons else ''}"
            f"conf={comp.get('confidence')}"
        )
    print(thin)
    print()


def _print_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str) -> None:
    """Print an ASCII confusion matrix with aligned columns."""
    print(f"\n  {title}")
    col_w = max({len(label) for label in labels} | {3}) + 1
    header = " " * (col_w + 3) + "".join(f"{label:>{col_w}}" for label in labels)
    print(header)
    for row_label, row in zip(labels, matrix, strict=True):
        print(f"    {row_label:>{col_w - 2}} |" + "".join(f"{v:>{col_w}d}" for v in row) + "   ←")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON report
# ═══════════════════════════════════════════════════════════════════════════════


def build_run_report(
    metrics: EvalMetrics,
    comparisons: list[dict[str, Any]],
    source: str | None = None,
    concurrency: int = 5,
) -> dict[str, Any]:
    """Assemble the full JSON report dict (metrics + thresholds + failing cases)."""
    failing = [
        {
            "id": c["id"],
            "source": c.get("source"),
            "reasons": failure_reasons(c),
            "confidence": c.get("confidence"),
            "category": {
                "expected": c.get("expected_category"),
                "actual": c.get("actual_category"),
            },
            "urgency": {
                "expected": c.get("expected_urgency"),
                "actual": c.get("actual_urgency"),
            },
            "escalation": {
                "expected": c.get("expected_should_escalate"),
                "actual": c.get("actual_should_escalate"),
            },
            "notes": c.get("notes"),
        }
        for c in comparisons
        if failure_reasons(c)
    ]
    return {
        "run": run_metadata(source, concurrency),
        "metrics": metrics.to_dict(),
        "pass": passed(metrics),
        "failing_cases": failing,
        "results": comparisons,
    }


def save_json_report(
    metrics: EvalMetrics,
    comparisons: list[dict[str, Any]],
    output_path: str | Path | None = None,
    source: str | None = None,
    concurrency: int = 5,
) -> Path:
    """Write the JSON report.  Defaults to ``eval/results/latest.json``."""
    path = Path(output_path) if output_path else RESULTS_DIR / "latest.json"
    ensure_results_dir(path)
    report = build_run_report(metrics, comparisons, source=source, concurrency=concurrency)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# HTML report (self-contained — no external CDN, opens offline)
# ═══════════════════════════════════════════════════════════════════════════════


def _heat_color(value: int, mx: int) -> str:
    """RGBA heat color scaled 0 → 1 against the matrix max."""
    if mx <= 0 or value <= 0:
        return "rgba(130,140,170,0.08)"
    t = 0.12 + 0.78 * (value / mx)
    return f"rgba(99,102,241,{t:.2f})"


def _heatmap_html(matrix: list[list[int]], labels: list[str], mx: int) -> str:
    """Build an HTML heatmap table for a confusion matrix."""
    rows = []
    for i in range(len(labels)):
        cells = ""
        for j in range(len(labels)):
            value = matrix[i][j]
            is_diag = i == j
            cells += (
                f'<td class="cell{" diag" if is_diag else ""}" '
                f'style="background:{_heat_color(value, mx)}">{value}</td>'
            )
        rows.append(f'<tr><th class="row-label">{html_module.escape(labels[i])}</th>{cells}</tr>')
    header = (
        '<tr><th class="corner"></th>'
        + "".join(f"<th>{html_module.escape(label)}</th>" for label in labels)
        + "</tr>"
    )
    return (
        '<table class="heatmap">'
        "<caption>Rows = expected · columns = predicted</caption>"
        + header
        + "".join(rows)
        + "</table>"
    )


def _metric_card(label: str, value: str, ok: bool | None = None, hint: str = "") -> str:
    cls = "metric-card"
    if ok is True:
        cls += " good"
    elif ok is False:
        cls += " bad"
    hint_html = f'<div class="hint">{html_module.escape(hint)}</div>' if hint else ""
    return (
        f'<div class="{cls}"><div class="metric-label">{html_module.escape(label)}</div>'
        f'<div class="metric-value">{html_module.escape(value)}</div>{hint_html}</div>'
    )


def save_html_report(
    metrics: EvalMetrics,
    comparisons: list[dict[str, Any]],
    output_path: str | Path | None = None,
    source: str | None = None,
    concurrency: int = 5,
) -> Path:
    """Write the self-contained HTML dashboard.  Defaults to ``eval/results/report.html``."""
    path = Path(output_path) if output_path else RESULTS_DIR / "report.html"
    ensure_results_dir(path)

    meta = run_metadata(source, concurrency)
    checks = passed(metrics)

    cat_mx = max((max(row) for row in metrics.category_confusion_matrix), default=0)
    urg_mx = max((max(row) for row in metrics.urgency_confusion_matrix), default=0)

    pass_badge = (
        '<span class="badge pass">PASS</span>'
        if all(checks.values())
        else '<span class="badge fail">FAIL</span>'
    )

    failing_rows = ""
    for c in comparisons:
        reasons = failure_reasons(c)
        if not reasons:
            continue
        fail_id = html_module.escape(c["id"])
        source_label = html_module.escape(str(c.get("source", "")))
        reasons_html = html_module.escape("; ".join(reasons))
        conf = c.get("confidence")
        conf_s = f"conf {conf:.2f}" if isinstance(conf, (int, float)) else ""
        failing_rows += (
            f"<tr><td><code>{fail_id}</code></td><td>{source_label}</td>"
            f'<td class="text">{reasons_html}</td><td>{conf_s}</td></tr>'
        )

    by_source_rows = ""
    for src, row in metrics.by_source.items():
        by_source_rows += (
            f"<tr><td>{html_module.escape(src)}</td><td>{row['total']}</td>"
            f"<td>{row['category_accuracy']:.1%}</td>"
            f"<td>{row['urgency_accuracy']:.1%}</td>"
            f"<td>{row['escalation_accuracy']:.1%}</td>"
            f"<td>{row['mean_draft_rouge_l']:.3f}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><title>Eval Report</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #f5f6fa; --surface: #ffffff; --surface2: #eef0f6; --border: #dfe2ea;
    --text: #14161f; --muted: #6b7280;
    --good: #16a34a; --bad: #dc2626; --accent: #6366f1;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0f1117; --surface: #181b24; --surface2: #20242f; --border: #2a2f3b;
      --text: #e8eaf0; --muted: #9aa1b2;
      --good: #4ade80; --bad: #f87171; --accent: #818cf8;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
    padding: clamp(16px, 4vw, 48px); max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.35rem; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); font-size: .85rem; margin: 6px 0 22px; }}
  .badge {{ padding: 2px 10px; border-radius: 999px; font-weight: 700; font-size: .75rem; }}
  .badge.pass {{ background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good); }}
  .badge.fail {{ background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0; }}
  .metric-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
  .metric-label {{ color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }}
  .metric-value {{ font-size: 1.55rem; font-weight: 700; margin-top: 2px; }}
  .metric-card.good .metric-value {{ color: var(--good); }}
  .metric-card.bad .metric-value {{ color: var(--bad); }}
  .hint {{ color: var(--muted); font-size: .72rem; margin-top: 2px; }}
  .two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin: 24px 0; }}
  .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
  .panel h2 {{ font-size: .95rem; margin-bottom: 10px; }}
  table.heatmap {{ border-collapse: collapse; width: 100%; }}
  table.heatmap caption {{ text-align: left; color: var(--muted); font-size: .75rem; margin-bottom: 6px; }}
  table.heatmap th, table.heatmap td {{ text-align: center; padding: 7px 8px; font-size: .8rem; border: 1px solid var(--border); }}
  table.heatmap th {{ color: var(--muted); font-weight: 600; }}
  table.heatmap .row-label {{ text-align: right; color: var(--text); font-weight: 600; white-space: nowrap; }}
  table.heatmap .cell {{ border-radius: 4px; }}
  table.heatmap .cell.diag {{ outline: 1px solid var(--accent); outline-offset: -1px; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .82rem; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
  td.text {{ white-space: pre-wrap; }}
  code {{ background: var(--surface2); padding: 1px 6px; border-radius: 5px; font-size: .78rem; }}
  .scroll {{ overflow-x: auto; }}
</style>
</head>
<body>
  <h1>Support-Ticket Triage Agent — Evaluation Dashboard</h1>
  <p class="sub">
    {meta["generated_at"]} UTC &middot; source <code>{meta["source"]}</code> &middot;
    {metrics.total} tickets &middot; thresholds cat_acc&#8805;{
        PASS_THRESHOLDS["category_accuracy"]:.0%},
    esc_prec&#8805;{PASS_THRESHOLDS["escalation_precision"]:.0%}
    &nbsp;{pass_badge}
  </p>

  <div class="grid">
    {
        _metric_card(
            "Category accuracy",
            f"{metrics.category_accuracy:.1%}",
            ok=checks["category_accuracy"],
            hint=f"{metrics.n_correct_category}/{metrics.total} tickets",
        )
    }
    {
        _metric_card(
            "Urgency accuracy",
            f"{metrics.urgency_accuracy:.1%}",
            hint=f"{metrics.n_correct_urgency}/{metrics.total} tickets",
        )
    }
    {
        _metric_card(
            "Draft similarity",
            f"{metrics.mean_draft_rouge_l:.3f}",
            hint="mean ROUGE-L vs human draft",
        )
    }
    {
        _metric_card(
            "Escalation precision",
            f"{metrics.escalation_precision:.1%}",
            ok=checks["escalation_precision"],
        )
    }
    {_metric_card("Escalation recall", f"{metrics.escalation_recall:.1%}")}
    {_metric_card("Escalation F1", f"{metrics.escalation_f1:.1%}")}
    {_metric_card("False escalation rate", f"{metrics.false_escalation_rate:.1%}")}
    {
        _metric_card(
            "Calibration error",
            f"{metrics.calibration_error:.1%}",
            hint="confidence &#8816; correctness",
        )
    }
    {_metric_card("Latency p50", f"{metrics.duration_p50_ms:.0f} ms")}
    {_metric_card("Latency p95", f"{metrics.duration_p95_ms:.0f} ms")}
  </div>

  <div class="two-col">
    <div class="panel">
      <h2>Category confusion matrix</h2>
      <div class="scroll">{
        _heatmap_html(metrics.category_confusion_matrix, metrics.category_labels, cat_mx)
    }</div>
    </div>
    <div class="panel">
      <h2>Urgency confusion matrix</h2>
      <div class="scroll">{
        _heatmap_html(metrics.urgency_confusion_matrix, metrics.urgency_labels, urg_mx)
    }</div>
    </div>
  </div>

  <div class="panel" style="margin-bottom:16px">
    <h2>Escalation confusion</h2>
    <div class="scroll">
      <table>
        <tr><th>True positive</th><th>False positive</th><th>False negative</th><th>True negative</th></tr>
        <tr><td>{metrics.escalation_confusion["TP"]}</td><td>{
        metrics.escalation_confusion["FP"]
    }</td>
            <td>{metrics.escalation_confusion["FN"]}</td><td>{
        metrics.escalation_confusion["TN"]
    }</td></tr>
      </table>
    </div>
  </div>

  <div class="panel" style="margin-bottom:16px">
    <h2>Per-source breakdown</h2>
    <div class="scroll">
      <table>
        <tr><th>Source</th><th>Count</th><th>Category</th><th>Urgency</th><th>Escalation</th><th>Draft ROUGE-L</th></tr>
        {by_source_rows}
      </table>
    </div>
  </div>

  <div class="panel">
    <h2>Failing cases ({sum(1 for c in comparisons if failure_reasons(c))})</h2>
    <div class="scroll">
      <table>
        <tr><th>Ticket</th><th>Source</th><th>Reason</th><th>Confidence</th></tr>
        {
        failing_rows
        or '<tr><td colspan="4" class="text">None — every ticket passed every check.</td></tr>'
    }
      </table>
    </div>
  </div>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
