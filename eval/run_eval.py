"""CLI entry point for the evaluation harness.

Usage::

    python -m eval.run_eval                      # full 100-ticket run
    python -m eval.run_eval --source huggingface # only one fixture source
    python -m eval.run_eval --concurrency 8
    python -m eval.run_eval --output eval/results/run-2026-09-11.json
    python -m eval.run_eval --html eval/results/report.html

Exit codes:
    ``0`` — run completed and all pass thresholds met.
    ``1`` — run completed but a pass threshold (category accuracy ≥ 70%,
            escalation precision ≥ 60%) was not met.
    ``2`` — setup error (missing fixture / unknown flag).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from eval.harness import VALID_SOURCES, EvalHarness
from eval.report import (
    PASS_THRESHOLDS,
    RESULTS_DIR,
    passed,
    print_console_report,
    save_html_report,
    save_json_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.run_eval",
        description=(
            "Run the Support-Ticket Triage Agent evaluation harness over the "
            "hand-labeled fixture (eval/fixtures/tickets.jsonl)."
        ),
    )
    parser.add_argument(
        "--source",
        choices=[*VALID_SOURCES, "all"],
        default="all",
        help="Which fixture source subset to evaluate (default: all).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of tickets evaluated in parallel (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the JSON report (default: eval/results/latest.json).",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Path for the HTML dashboard (default: eval/results/report.html).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = None if args.source == "all" else args.source

    harness = EvalHarness(concurrency=args.concurrency)

    # Guard against a missing fixture before kicking off async work.
    try:
        tickets = harness.load_tickets(source=source)
    except FileNotFoundError as exc:
        print(f"[eval] ERROR: {exc}", file=sys.stderr)
        return 2
    if not tickets:
        print(f"[eval] ERROR: no tickets found for source={args.source!r}", file=sys.stderr)
        return 2

    print(
        f"[eval] Evaluating {len(tickets)} tickets "
        f"(source={args.source}, concurrency={args.concurrency}) …"
    )
    metrics, comparisons = asyncio.run(
        harness.evaluate(source=source, concurrency=args.concurrency)
    )

    print_console_report(metrics, comparisons, source=source, concurrency=args.concurrency)

    output_path = Path(args.output) if args.output else RESULTS_DIR / "latest.json"
    saved_json = save_json_report(
        metrics,
        comparisons,
        output_path=output_path,
        source=source,
        concurrency=args.concurrency,
    )
    html_path = Path(args.html) if args.html else RESULTS_DIR / "report.html"
    saved_html = save_html_report(
        metrics,
        comparisons,
        output_path=html_path,
        source=source,
        concurrency=args.concurrency,
    )
    print(f"[eval] JSON report written to: {saved_json}")
    print(f"[eval] HTML dashboard written to: {saved_html}")

    checks = passed(metrics)
    if all(checks.values()):
        print("[eval] PASS — all thresholds met.")
        return 0

    print(
        "[eval] FAIL — thresholds not met "
        f"(need category_accuracy≥{PASS_THRESHOLDS['category_accuracy']:.0%} and "
        f"escalation_precision≥{PASS_THRESHOLDS['escalation_precision']:.0%}).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
