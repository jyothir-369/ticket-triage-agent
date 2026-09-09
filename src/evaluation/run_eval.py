"""Run the evaluation harness over the labeled ticket fixture."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import structlog

from src.agent.graph import run_triage
from src.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


def load_eval_tickets() -> list[dict]:
    """Load the labeled eval fixture."""
    path = Path(settings.eval_tickets_path)
    if not path.exists():
        logger.warning("eval.fixture_missing", path=str(path))
        return []
    with open(path) as f:
        return json.load(f)


async def run_eval() -> dict:
    """Execute the eval set and return aggregate metrics."""
    tickets = load_eval_tickets()
    if not tickets:
        logger.error("eval.no_tickets")
        return {"error": "no eval tickets found"}

    results: list[dict] = []
    for ticket in tickets:
        start = time.monotonic()
        outcome = await run_triage(
            ticket_id=ticket["id"],
            subject=ticket["subject"],
            body=ticket["body"],
        )
        elapsed = time.monotonic() - start

        correct_category = outcome["category"] == ticket.get("expected_category")
        correct_urgency = outcome["urgency"] == ticket.get("expected_urgency")
        should_escalate = ticket.get("expected_escalate", False)
        did_escalate = outcome["decision"] == "escalate"

        results.append({
            "ticket_id": ticket["id"],
            "correct_category": correct_category,
            "correct_urgency": correct_urgency,
            "escalation_match": should_escalate == did_escalate,
            "latency_s": round(elapsed, 2),
            "decision": outcome["decision"],
        })

    total = len(results)
    correct_cat = sum(1 for r in results if r["correct_category"])
    correct_urg = sum(1 for r in results if r["correct_urgency"])
    esc_match = sum(1 for r in results if r["escalation_match"])
    avg_latency = sum(r["latency_s"] for r in results) / total

    metrics = {
        "total": total,
        "category_accuracy": round(correct_cat / total, 3),
        "urgency_accuracy": round(correct_urg / total, 3),
        "escalation_accuracy": round(esc_match / total, 3),
        "avg_latency_s": round(avg_latency, 2),
        "results": results,
    }

    logger.info("eval.complete", **{k: v for k, v in metrics.items() if k != "results"})
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_eval())
