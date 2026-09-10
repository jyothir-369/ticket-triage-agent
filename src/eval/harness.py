"""Evaluation harness for the triage agent.

Runs the agent against labeled test tickets and produces detailed
reports with accuracy metrics, draft similarity scores, and escalation
correctness analysis.

Usage::

    # Run from the command line
    python -m src.eval.harness

    # Or use programmatically
    harness = EvalHarness()
    report = await harness.run_all()
    harness.print_report(report)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.eval.metrics import (
    AggregateMetrics,
    MetricResult,
    aggregate_metrics,
    compare_results,
)

logger = structlog.get_logger(__name__)

# Default path relative to project root
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tickets.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# EvalHarness
# ═══════════════════════════════════════════════════════════════════════════════


class EvalHarness:
    """Orchestrates evaluation runs against labeled test tickets.

    Parameters
    ----------
    fixture_path:
        Path to a JSONL file with test ticket entries.
    concurrency:
        Max number of tickets to evaluate in parallel.
    confidence_threshold:
        Threshold used to determine escalation in the agent.
    """

    def __init__(
        self,
        fixture_path: str | Path | None = None,
        concurrency: int = 5,
        confidence_threshold: float = 0.7,
    ):
        self.fixture_path = Path(fixture_path) if fixture_path else _FIXTURE_PATH
        self.concurrency = concurrency
        self.confidence_threshold = confidence_threshold
        self._semaphore = asyncio.Semaphore(concurrency)

    # ── Load ─────────────────────────────────────────────────────────────────

    def load_tickets(self) -> list[dict[str, Any]]:
        """Load labeled test tickets from the JSONL fixture file.

        Returns
        -------
        list[dict]
            List of ticket dicts, each containing: id, content,
            expected_category, expected_urgency, should_escalate,
            human_approved_draft, expected_draft_keywords.
        """
        if not self.fixture_path.exists():
            logger.warning("eval.fixture_missing", path=str(self.fixture_path))
            return []

        tickets: list[dict[str, Any]] = []
        with open(self.fixture_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ticket = json.loads(line)
                    tickets.append(ticket)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "eval.fixture_parse_error",
                        line=line_num,
                        error=str(exc),
                    )

        logger.info("eval.tickets_loaded", count=len(tickets))
        return tickets

    # ── Run single ───────────────────────────────────────────────────────────

    async def run_single(self, ticket: dict[str, Any]) -> dict[str, Any]:
        """Run the agent on a single ticket and return the result.

        This method invokes the full triage pipeline and captures the
        output without persisting to the database.

        Parameters
        ----------
        ticket:
            A ticket dict from the eval fixture.

        Returns
        -------
        dict
            Actual agent output with keys: ticket_id, category, urgency,
            confidence, draft_text, should_escalate.
        """
        ticket_id = ticket["id"]
        content = ticket["content"]

        # Parse subject and body from content
        lines = content.split("\n", 2)
        subject = ""
        body = content
        for i, line in enumerate(lines):
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
                body = lines[i + 1].strip() if i + 1 < len(lines) else ""
                break

        async with self._semaphore:
            start = time.monotonic()
            try:
                from src.agent.graph import run_triage

                outcome = await run_triage(
                    ticket_id=ticket_id,
                    subject=subject or f"Ticket {ticket_id}",
                    body=body or content,
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)

                result = {
                    "ticket_id": ticket_id,
                    "category": outcome.get("category"),
                    "urgency": outcome.get("urgency"),
                    "confidence": outcome.get("confidence", 0.0),
                    "draft_text": outcome.get("drafted_response", ""),
                    "should_escalate": outcome.get("decision") == "escalate",
                    "latency_ms": elapsed_ms,
                    "error": None,
                }

                logger.info(
                    "eval.single.complete",
                    ticket_id=ticket_id,
                    category=result["category"],
                    urgency=result["urgency"],
                    latency_ms=elapsed_ms,
                )
                return result

            except Exception as exc:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                logger.error(
                    "eval.single.failed",
                    ticket_id=ticket_id,
                    error=str(exc),
                    latency_ms=elapsed_ms,
                )
                return {
                    "ticket_id": ticket_id,
                    "category": None,
                    "urgency": None,
                    "confidence": 0.0,
                    "draft_text": "",
                    "should_escalate": True,  # Default to escalate on error
                    "latency_ms": elapsed_ms,
                    "error": str(exc),
                }

    # ── Run all ──────────────────────────────────────────────────────────────

    async def run_all(self) -> list[dict[str, Any]]:
        """Run the agent on all tickets with bounded concurrency.

        Returns
        -------
        list[dict]
            List of result dicts from run_single(), one per ticket.
        """
        tickets = self.load_tickets()
        if not tickets:
            logger.error("eval.no_tickets")
            return []

        logger.info(
            "eval.run_all.start",
            total=len(tickets),
            concurrency=self.concurrency,
        )

        start = time.monotonic()
        tasks = [self.run_single(ticket) for ticket in tickets]
        results = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start

        logger.info(
            "eval.run_all.complete",
            total=len(results),
            elapsed_s=round(elapsed, 2),
        )

        return list(results)

    # ── Compare ──────────────────────────────────────────────────────────────

    def compare_results(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
    ) -> MetricResult:
        """Compare a single ticket's actual output against expected values.

        Parameters
        ----------
        actual:
            Agent output from run_single().
        expected:
            Expected values from the eval fixture.

        Returns
        -------
        MetricResult
            Detailed comparison result.
        """
        return compare_results(
            actual=actual,
            expected=expected,
            confidence_threshold=self.confidence_threshold,
        )

    # ── Full evaluation ──────────────────────────────────────────────────────

    async def evaluate(self) -> AggregateMetrics:
        """Run the full evaluation pipeline: load, run, compare, aggregate.

        Returns
        -------
        AggregateMetrics
            Complete evaluation results with all metrics.
        """
        tickets = self.load_tickets()
        if not tickets:
            return AggregateMetrics()

        results_raw = await self.run_all()
        ticket_lookup = {t["id"]: t for t in tickets}

        metric_results: list[MetricResult] = []
        for raw in results_raw:
            ticket_id = raw["ticket_id"]
            expected = ticket_lookup.get(ticket_id, {})
            metric = self.compare_results(raw, expected)
            metric_results.append(metric)

        return aggregate_metrics(metric_results)

    # ── Reporting ────────────────────────────────────────────────────────────

    def print_report(self, metrics: AggregateMetrics) -> None:
        """Print a human-readable evaluation report to the console.

        Parameters
        ----------
        metrics:
            Aggregated metrics from evaluate().
        """
        print("\n" + "=" * 72)
        print("  EVALUATION REPORT — Support Ticket Triage Agent")
        print("=" * 72)
        print(f"  Total tickets evaluated: {metrics.total}")
        print(f"  Run timestamp: {datetime.now(timezone.utc).isoformat()}")
        print("-" * 72)

        # Overall metrics
        print("\n  AGGREGATE METRICS")
        print(f"    Category Accuracy:        {metrics.category_accuracy:.1%}")
        print(f"    Urgency Accuracy:         {metrics.urgency_accuracy:.1%}")
        print(f"    Escalation Accuracy:      {metrics.escalation_accuracy:.1%}")
        print(f"    Confidence Threshold Acc: {metrics.confidence_threshold_accuracy:.1%}")
        print(f"    Avg Draft ROUGE-L:        {metrics.avg_draft_rouge_l:.4f}")
        print(f"    Avg Keyword Overlap:      {metrics.avg_draft_keyword_overlap:.1%}")
        print(f"    Overall Pass Rate:        {metrics.pass_rate:.1%}")

        # Confusion matrix
        cm = metrics.confusion_matrix.get("escalation", {})
        print("\n  ESCALATION CONFUSION MATRIX")
        print(f"                    Predicted Positive  Predicted Negative")
        print(f"    Actual Pos:     {cm.get('TP', 0):>17}  {cm.get('FN', 0):>17}")
        print(f"    Actual Neg:     {cm.get('FP', 0):>17}  {cm.get('TN', 0):>17}")

        # Per-ticket details
        print("\n  PER-TICKET RESULTS")
        print("-" * 72)
        for r in metrics.results:
            status = "PASS" if (r.category_correct and r.urgency_correct and r.escalation_correct) else "FAIL"
            symbol = "✅" if status == "PASS" else "❌"
            print(f"\n  {symbol} {r.ticket_id} [{status}]")
            print(f"    Category: {r.details.get('actual_category', '?')} "
                  f"{'==' if r.category_correct else '!='} {r.details.get('expected_category', '?')}")
            print(f"    Urgency:  {r.details.get('actual_urgency', '?')} "
                  f"{'==' if r.urgency_correct else '!='} {r.details.get('expected_urgency', '?')}")
            print(f"    Escalation: actual={r.details.get('actual_should_escalate')} "
                  f"expected={r.details.get('expected_should_escalate')} "
                  f"{'correct' if r.escalation_correct else 'WRONG'}")
            print(f"    Draft ROUGE-L: {r.draft_rouge_l:.4f} | "
                  f"Keyword Overlap: {r.draft_keyword_overlap:.1%}")
            if r.details.get("confidence"):
                print(f"    Confidence: {r.details['confidence']:.4f} "
                      f"(above threshold: {r.confidence_above_threshold})")
            if r.details.get("error"):
                print(f"    ERROR: {r.details['error']}")

        print("\n" + "=" * 72)

    def generate_json_summary(self, metrics: AggregateMetrics) -> dict[str, Any]:
        """Generate a JSON-serialisable summary of all metrics.

        Parameters
        ----------
        metrics:
            Aggregated metrics from evaluate().

        Returns
        -------
        dict
            Complete evaluation summary.
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fixture_path": str(self.fixture_path),
            "confidence_threshold": self.confidence_threshold,
            "concurrency": self.concurrency,
            "metrics": metrics.to_dict(),
            "per_ticket": [r.to_dict() for r in metrics.results],
        }

    def save_json_report(
        self,
        metrics: AggregateMetrics,
        output_path: str | Path = "eval_report.json",
    ) -> Path:
        """Save the evaluation report as a JSON file.

        Parameters
        ----------
        metrics:
            Aggregated metrics from evaluate().
        output_path:
            Path to write the JSON report.

        Returns
        -------
        Path
            Path to the written report file.
        """
        path = Path(output_path)
        report = self.generate_json_summary(metrics)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("eval.json_report_saved", path=str(path))
        return path

    def generate_html_dashboard(self, metrics: AggregateMetrics) -> str:
        """Generate an HTML dashboard with visualizations.

        Parameters
        ----------
        metrics:
            Aggregated metrics from evaluate().

        Returns
        -------
        str
            Complete HTML string for the dashboard.
        """
        cm = metrics.confusion_matrix.get("escalation", {})

        # Build per-ticket rows
        ticket_rows = ""
        for r in metrics.results:
            status_class = "pass" if (r.category_correct and r.urgency_correct and r.escalation_correct) else "fail"
            status_text = "PASS" if status_class == "pass" else "FAIL"
            ticket_rows += f"""
            <tr class="{status_class}">
                <td>{r.ticket_id}</td>
                <td>{'&#10003;' if r.category_correct else '&#10007;'}</td>
                <td>{'&#10003;' if r.urgency_correct else '&#10007;'}</td>
                <td>{'&#10003;' if r.escalation_correct else '&#10007;'}</td>
                <td>{r.draft_rouge_l:.3f}</td>
                <td>{r.draft_keyword_overlap:.1%}</td>
                <td class="{status_class}-badge">{status_text}</td>
            </tr>"""

        # Category breakdown
        cat_correct = sum(1 for r in metrics.results if r.category_correct)
        cat_incorrect = metrics.total - cat_correct
        urg_correct = sum(1 for r in metrics.results if r.urgency_correct)
        urg_incorrect = metrics.total - urg_correct
        esc_correct = sum(1 for r in metrics.results if r.escalation_correct)
        esc_incorrect = metrics.total - esc_correct

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Triage Agent Evaluation Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #eab308;
            --blue: #3b82f6;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }}
        h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
        .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; font-size: 0.9rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
        }}
        .card-label {{ color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .card-value {{ font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; }}
        .card-value.green {{ color: var(--green); }}
        .card-value.yellow {{ color: var(--yellow); }}
        .card-value.red {{ color: var(--red); }}
        .section {{ margin-bottom: 2rem; }}
        .section-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; }}
        table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: rgba(255,255,255,0.05); font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); }}
        .pass-badge {{ background: rgba(34,197,94,0.15); color: var(--green); padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}
        .fail-badge {{ background: rgba(239,68,68,0.15); color: var(--red); padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}
        .confusion-grid {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 0; max-width: 400px; }}
        .confusion-grid > div {{ padding: 1rem; text-align: center; border: 1px solid var(--border); }}
        .confusion-grid .header {{ background: rgba(255,255,255,0.05); font-weight: 600; font-size: 0.8rem; }}
        .confusion-grid .tp {{ background: rgba(34,197,94,0.1); }}
        .confusion-grid .tn {{ background: rgba(59,130,246,0.1); }}
        .confusion-grid .fp {{ background: rgba(239,68,68,0.1); }}
        .confusion-grid .fn {{ background: rgba(234,179,8,0.1); }}
        .bar-container {{ height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }}
        .bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
        .bar.green {{ background: var(--green); }}
        .bar.yellow {{ background: var(--yellow); }}
        .bar.red {{ background: var(--red); }}
    </style>
</head>
<body>
    <h1>Triage Agent Evaluation Dashboard</h1>
    <p class="subtitle">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} &middot; {metrics.total} tickets &middot; Threshold: {self.confidence_threshold}</p>

    <!-- Metric cards -->
    <div class="grid">
        <div class="card">
            <div class="card-label">Category Accuracy</div>
            <div class="card-value {'green' if metrics.category_accuracy >= 0.8 else 'yellow' if metrics.category_accuracy >= 0.6 else 'red'}">{metrics.category_accuracy:.1%}</div>
            <div class="bar-container"><div class="bar {'green' if metrics.category_accuracy >= 0.8 else 'yellow' if metrics.category_accuracy >= 0.6 else 'red'}" style="width:{metrics.category_accuracy:.0%}"></div></div>
        </div>
        <div class="card">
            <div class="card-label">Urgency Accuracy</div>
            <div class="card-value {'green' if metrics.urgency_accuracy >= 0.8 else 'yellow' if metrics.urgency_accuracy >= 0.6 else 'red'}">{metrics.urgency_accuracy:.1%}</div>
            <div class="bar-container"><div class="bar {'green' if metrics.urgency_accuracy >= 0.8 else 'yellow' if metrics.urgency_accuracy >= 0.6 else 'red'}" style="width:{metrics.urgency_accuracy:.0%}"></div></div>
        </div>
        <div class="card">
            <div class="card-label">Escalation Accuracy</div>
            <div class="card-value {'green' if metrics.escalation_accuracy >= 0.8 else 'yellow' if metrics.escalation_accuracy >= 0.6 else 'red'}">{metrics.escalation_accuracy:.1%}</div>
            <div class="bar-container"><div class="bar {'green' if metrics.escalation_accuracy >= 0.8 else 'yellow' if metrics.escalation_accuracy >= 0.6 else 'red'}" style="width:{metrics.escalation_accuracy:.0%}"></div></div>
        </div>
        <div class="card">
            <div class="card-label">Draft ROUGE-L</div>
            <div class="card-value {'green' if metrics.avg_draft_rouge_l >= 0.3 else 'yellow' if metrics.avg_draft_rouge_l >= 0.15 else 'red'}">{metrics.avg_draft_rouge_l:.3f}</div>
            <div class="bar-container"><div class="bar {'green' if metrics.avg_draft_rouge_l >= 0.3 else 'yellow' if metrics.avg_draft_rouge_l >= 0.15 else 'red'}" style="width:{min(metrics.avg_draft_rouge_l * 100, 100):.0f}%"></div></div>
        </div>
        <div class="card">
            <div class="card-label">Keyword Overlap</div>
            <div class="card-value {'green' if metrics.avg_draft_keyword_overlap >= 0.6 else 'yellow' if metrics.avg_draft_keyword_overlap >= 0.3 else 'red'}">{metrics.avg_draft_keyword_overlap:.1%}</div>
            <div class="bar-container"><div class="bar {'green' if metrics.avg_draft_keyword_overlap >= 0.6 else 'yellow' if metrics.avg_draft_keyword_overlap >= 0.3 else 'red'}" style="width:{metrics.avg_draft_keyword_overlap:.0%}"></div></div>
        </div>
        <div class="card">
            <div class="card-label">Overall Pass Rate</div>
            <div class="card-value {'green' if metrics.pass_rate >= 0.8 else 'yellow' if metrics.pass_rate >= 0.6 else 'red'}">{metrics.pass_rate:.1%}</div>
            <div class="bar-container"><div class="bar {'green' if metrics.pass_rate >= 0.8 else 'yellow' if metrics.pass_rate >= 0.6 else 'red'}" style="width:{metrics.pass_rate:.0%}"></div></div>
        </div>
    </div>

    <!-- Confusion Matrix -->
    <div class="section">
        <div class="section-title">Escalation Confusion Matrix</div>
        <div class="confusion-grid">
            <div class="header"></div>
            <div class="header">Predicted: Escalate</div>
            <div class="header">Predicted: No Escalate</div>
            <div class="header">Actual: Escalate</div>
            <div class="tp"><strong>{cm.get('TP', 0)}</strong><br><small>True Positive</small></div>
            <div class="fn"><strong>{cm.get('FN', 0)}</strong><br><small>False Negative</small></div>
            <div class="header">Actual: No Escalate</div>
            <div class="fp"><strong>{cm.get('FP', 0)}</strong><br><small>False Positive</small></div>
            <div class="tn"><strong>{cm.get('TN', 0)}</strong><br><small>True Negative</small></div>
        </div>
    </div>

    <!-- Per-ticket results -->
    <div class="section">
        <div class="section-title">Per-Ticket Results</div>
        <table>
            <thead>
                <tr>
                    <th>Ticket ID</th>
                    <th>Category</th>
                    <th>Urgency</th>
                    <th>Escalation</th>
                    <th>ROUGE-L</th>
                    <th>Keywords</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {ticket_rows}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        return html

    def save_html_dashboard(
        self,
        metrics: AggregateMetrics,
        output_path: str | Path = "eval_dashboard.html",
    ) -> Path:
        """Save the HTML dashboard to a file.

        Parameters
        ----------
        metrics:
            Aggregated metrics from evaluate().
        output_path:
            Path to write the HTML file.

        Returns
        -------
        Path
            Path to the written HTML file.
        """
        path = Path(output_path)
        html = self.generate_html_dashboard(metrics)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("eval.html_dashboard_saved", path=str(path))
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    """Run the evaluation harness from the command line."""
    harness = EvalHarness()
    metrics = await harness.evaluate()
    harness.print_report(metrics)
    harness.save_json_report(metrics, "eval_report.json")
    harness.save_html_dashboard(metrics, "eval_dashboard.html")


if __name__ == "__main__":
    asyncio.run(main())
