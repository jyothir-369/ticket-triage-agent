"""Evaluation runner — execute the eval harness, generate HTML report,
and optionally upload to artifact storage.

Usage::

    python -m scripts.eval_run
    python -m scripts.eval_run --fixture data/eval_tickets.jsonl
    python -m scripts.eval_run --concurrency 10 --limit 20
    python -m scripts.eval_run --output-dir reports/ --upload
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog

from eval.harness import EvalHarness
from eval.metrics import AggregateMetrics

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_eval_report(
    metrics: AggregateMetrics,
    harness: EvalHarness,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate JSON and HTML evaluation reports.

    Returns dict mapping report type to file path.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # JSON report
    json_path = output_dir / f"eval_report_{timestamp}.json"
    harness.save_json_report(metrics, json_path)
    paths["json"] = json_path

    # HTML dashboard
    html_path = output_dir / f"eval_dashboard_{timestamp}.html"
    harness.save_html_dashboard(metrics, html_path)
    paths["html"] = html_path

    # Also save a "latest" symlink-style copy
    latest_json = output_dir / "eval_report_latest.json"
    latest_html = output_dir / "eval_dashboard_latest.html"
    harness.save_json_report(metrics, latest_json)
    harness.save_html_dashboard(metrics, latest_html)
    paths["latest_json"] = latest_json
    paths["latest_html"] = latest_html

    logger.info(
        "eval.reports_generated",
        json=str(json_path),
        html=str(html_path),
    )
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# Artifact Upload (Optional)
# ═══════════════════════════════════════════════════════════════════════════════


def upload_report(
    html_path: Path,
    bucket: str | None = None,
    prefix: str = "eval-reports",
) -> str | None:
    """Upload the HTML report to artifact storage (S3-compatible).

    Returns the URL of the uploaded report, or None if upload is skipped.
    Requires AWS credentials or compatible S3 configuration.
    """
    if not bucket:
        bucket = "triage-agent-eval-reports"

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        s3 = boto3.client("s3")
        key = f"{prefix}/{html_path.name}"

        s3.upload_file(
            str(html_path),
            bucket,
            key,
            ExtraArgs={"ContentType": "text/html"},
        )

        # Generate URL (works for public buckets or pre-signed URLs)
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
        logger.info("eval.upload_complete", url=url, bucket=bucket, key=key)
        return url

    except (NoCredentialsError, ClientError, ImportError) as exc:
        logger.warning("eval.upload_failed", error=str(exc))
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def run_evaluation(
    fixture_path: str | None = None,
    concurrency: int = 5,
    confidence_threshold: float = 0.7,
    output_dir: str = "reports",
    limit: int | None = None,
    upload: bool = False,
) -> dict[str, Any]:
    """Run the full evaluation pipeline.

    Returns a summary dict with metrics and report paths.
    """
    start = time.monotonic()
    logger.info(
        "eval.start",
        fixture=fixture_path,
        concurrency=concurrency,
        limit=limit,
    )

    # ── Initialize harness ─────────────────────────────────────────────────
    harness = EvalHarness(
        fixture_path=fixture_path,
        concurrency=concurrency,
        confidence_threshold=confidence_threshold,
    )

    # ── Load tickets (with optional limit) ─────────────────────────────────
    tickets = harness.load_tickets()
    if not tickets:
        logger.error("eval.no_tickets")
        return {"error": "no eval tickets found", "total": 0}

    if limit and limit < len(tickets):
        tickets = tickets[:limit]
        # Override harness fixture by writing temp file
        import tempfile
        import os

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        try:
            for t in tickets:
                tmp.write(json.dumps(t) + "\n")
            tmp.close()
            harness = EvalHarness(
                fixture_path=tmp.name,
                concurrency=concurrency,
                confidence_threshold=confidence_threshold,
            )
        finally:
            os.unlink(tmp.name)

    logger.info("eval.tickets_loaded", count=len(tickets))

    # ── Run evaluation ─────────────────────────────────────────────────────
    metrics = await harness.evaluate()
    elapsed = time.monotonic() - start

    # ── Print console report ───────────────────────────────────────────────
    harness.print_report(metrics)

    # ── Generate reports ───────────────────────────────────────────────────
    out_path = Path(output_dir)
    report_paths = generate_eval_report(metrics, harness, out_path)

    # ── Upload (optional) ──────────────────────────────────────────────────
    upload_url = None
    if upload and "html" in report_paths:
        upload_url = upload_report(report_paths["html"])

    # ── Summary ────────────────────────────────────────────────────────────
    summary = {
        "total": metrics.total,
        "category_accuracy": round(metrics.category_accuracy, 4),
        "urgency_accuracy": round(metrics.urgency_accuracy, 4),
        "escalation_accuracy": round(metrics.escalation_accuracy, 4),
        "confidence_threshold_accuracy": round(metrics.confidence_threshold_accuracy, 4),
        "avg_draft_rouge_l": round(metrics.avg_draft_rouge_l, 4),
        "avg_draft_keyword_overlap": round(metrics.avg_draft_keyword_overlap, 4),
        "pass_rate": round(metrics.pass_rate, 4),
        "elapsed_seconds": round(elapsed, 2),
        "reports": {k: str(v) for k, v in report_paths.items()},
        "upload_url": upload_url,
    }

    logger.info("eval.complete", **{k: v for k, v in summary.items() if k != "reports"})
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the evaluation harness and generate reports.",
    )
    parser.add_argument(
        "--fixture",
        type=str,
        default=None,
        help="Path to eval fixture file (JSONL). Defaults to eval/fixtures/tickets.jsonl.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent ticket evaluations (default: 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for escalation (default: 0.7).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output reports (default: reports/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tickets to evaluate.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload HTML report to S3 artifact storage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = asyncio.run(
        run_evaluation(
            fixture_path=args.fixture,
            concurrency=args.concurrency,
            confidence_threshold=args.threshold,
            output_dir=args.output_dir,
            limit=args.limit,
            upload=args.upload,
        )
    )

    if "error" in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in result.items():
        if key == "reports":
            print(f"  Reports:")
            for rtype, rpath in value.items():
                print(f"    {rtype:.<20} {rpath}")
        elif key == "upload_url" and value:
            print(f"  {'Upload URL':.<30} {value}")
        else:
            print(f"  {key:.<30} {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
