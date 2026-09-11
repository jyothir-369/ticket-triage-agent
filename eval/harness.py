"""Evaluation harness — runs the real triage agent over labeled tickets.

``EvalHarness`` is the orchestrator.  It:
  1. ``load_tickets``  — reads the labeled JSONL fixture (optionally filtered
     by source: ``huggingface`` | ``github`` | ``manual``).
  2. ``run_single``    — pushes one ticket through the **actual LangGraph**
     pipeline (``AgentExecutor.run`` — the graph directly, not the HTTP API).
  3. ``run_all``       — runs every ticket with bounded concurrency.
  4. ``compare``       — scores one agent output against its ground-truth label.

The per-ticket scoring lives in :mod:`eval.metrics` (``compare_single`` →
``compute_metrics``), so the harness stays thin and the metrics stay unit-testable.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from eval.metrics import EvalMetrics, compare_single, compute_metrics

# Default fixture location — relative to this package.
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tickets.jsonl"

VALID_SOURCES = ("huggingface", "github", "manual")


class EvalHarness:
    """Orchestrates evaluation runs against the labeled ticket fixture.

    Parameters
    ----------
    fixture_path:
        Path to the JSONL fixture.  Defaults to ``eval/fixtures/tickets.jsonl``.
    concurrency:
        Maximum number of tickets evaluated in parallel (also capped by the
        agent's global concurrency limit).
    confidence_threshold:
        Confidence boundary used by ``compare`` for ``confidence_calibrated``.
    """

    def __init__(
        self,
        fixture_path: str | Path | None = None,
        concurrency: int = 5,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.fixture_path = Path(fixture_path) if fixture_path else FIXTURE_PATH
        self.concurrency = concurrency
        self.confidence_threshold = confidence_threshold
        self._semaphore = asyncio.Semaphore(concurrency)

    # ── Load ─────────────────────────────────────────────────────────────────

    def load_tickets(self, source: str | None = None) -> list[dict[str, Any]]:
        """Load labeled tickets from the JSONL fixture.

        Parameters
        ----------
        source:
            Optional filter — one of ``"huggingface"``, ``"github"``,
            ``"manual"``.  ``None`` loads every ticket.

        Returns
        -------
        list[dict]
            Each dict has the fixture schema: ``id``, ``content``, ``source``,
            ``expected_category``, ``expected_urgency``, ``should_escalate``,
            ``human_approved_draft``, and optional ``notes``.
        """
        if source is not None and source not in VALID_SOURCES:
            raise ValueError(f"Unknown source {source!r} — expected one of {VALID_SOURCES}")

        if not self.fixture_path.exists():
            raise FileNotFoundError(
                f"Fixture not found: {self.fixture_path} — run "
                "'python -m eval.build_fixture' first."
            )

        tickets: list[dict[str, Any]] = []
        with open(self.fixture_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ticket: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Fixture parse error on line {line_num}: {exc}") from exc
                if source is None or ticket.get("source") == source:
                    tickets.append(ticket)

        return tickets

    # ── Run single ───────────────────────────────────────────────────────────

    async def run_single(self, ticket: dict[str, Any]) -> dict[str, Any]:
        """Run the agent on one ticket and capture the triage output.

        The ticket is submitted to the **compiled LangGraph pipeline**
        (:class:`src.agent.graph.AgentExecutor`) — not through the HTTP API.
        A fresh checkpoint ``thread_id`` is used per run so repeated evals
        never reuse prior state.

        Returns
        -------
        dict
            Keys: ``id``, ``source``, ``category``, ``urgency``, ``confidence``,
            ``retrieval_count``, ``draft``, ``should_escalate``,
            ``escalation_reason``, ``latency_ms``, ``error``.
        """
        ticket_id = str(ticket["id"])

        async with self._semaphore:
            start = time.monotonic()
            try:
                from src.agent.graph import AgentExecutor
                from src.models.schemas import Ticket

                agent_ticket = Ticket(
                    id=ticket_id,
                    content=str(ticket["content"]),
                    source=str(ticket.get("source", "eval")),
                    source_id=ticket.get("source_id"),
                )

                executor = AgentExecutor()
                # Unique thread_id per run ⇒ the checkpointer never carries
                # state from a previous evaluation pass.
                config = {"configurable": {"thread_id": f"eval-{ticket_id}-{uuid.uuid4().hex[:8]}"}}
                final_state = await executor.run(agent_ticket, config=config)
                latency_ms = int((time.monotonic() - start) * 1000)

                classification = final_state.get("classification")
                draft = final_state.get("draft")

                return {
                    "id": ticket_id,
                    "source": ticket.get("source", "unknown"),
                    "category": (classification.category.value if classification else None),
                    "urgency": (classification.urgency.value if classification else None),
                    "confidence": (
                        round(float(classification.confidence), 4) if classification else 0.0
                    ),
                    "retrieval_count": len(final_state.get("retrieved_docs", [])),
                    "draft": draft.draft_text if draft else "",
                    "should_escalate": bool(final_state.get("should_escalate")),
                    "escalation_reason": final_state.get("escalation_reason", ""),
                    "latency_ms": latency_ms,
                    "error": None,
                }

            except Exception as exc:  # noqa: BLE001 — capture any failure
                latency_ms = int((time.monotonic() - start) * 1000)
                return {
                    "id": ticket_id,
                    "source": ticket.get("source", "unknown"),
                    "category": None,
                    "urgency": None,
                    "confidence": 0.0,
                    "retrieval_count": 0,
                    "draft": "",
                    "should_escalate": True,  # fail safe → flag for human review
                    "escalation_reason": f"Evaluation error: {exc}",
                    "latency_ms": latency_ms,
                    "error": str(exc),
                }

    # ── Run all ──────────────────────────────────────────────────────────────

    async def run_all(
        self, concurrency: int = 5, source: str | None = None
    ) -> list[dict[str, Any]]:
        """Run every (filtered) ticket with bounded concurrency.

        Parameters
        ----------
        concurrency:
            Max simultaneous ``run_single`` tasks.
        source:
            Optional fixture filter (see :meth:`load_tickets`).

        Returns
        -------
        list[dict]
            One output dict per ticket, in fixture order.
        """
        tickets = self.load_tickets(source=source)
        if not tickets:
            return []

        self._semaphore = asyncio.Semaphore(concurrency)
        self.concurrency = concurrency

        tasks = [self.run_single(t) for t in tickets]
        results = await asyncio.gather(*tasks)
        return list(results)

    # ── Compare ──────────────────────────────────────────────────────────────

    def compare(self, actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
        """Score one actual output against its ground-truth label.

        Returns
        -------
        dict
            A comparison row with ``category_match``, ``urgency_match``,
            ``draft_similarity`` (ROUGE-L), ``escalation_correct``,
            ``confidence_calibrated`` plus debugging context.
        """
        return compare_single(
            actual,
            expected,
            confidence_threshold=self.confidence_threshold,
        )

    # ── End-to-end ───────────────────────────────────────────────────────────

    async def evaluate(
        self,
        source: str | None = None,
        concurrency: int = 5,
    ) -> tuple[EvalMetrics, list[dict[str, Any]]]:
        """Full pipeline: load → run → compare → aggregate.

        Parameters
        ----------
        source:
            Optional source filter (huggingface / github / manual).
        concurrency:
            Max parallel tickets.

        Returns
        -------
        tuple[EvalMetrics, list[dict]]
            Aggregated metrics plus the per-ticket comparison rows (used by
            the report generator for the failing-case list).
        """
        actual_rows = await self.run_all(concurrency=concurrency, source=source)
        tickets = self.load_tickets(source=source)
        expected_lookup = {t["id"]: t for t in tickets}

        comparisons = [
            self.compare(actual, expected_lookup.get(actual["id"], {})) for actual in actual_rows
        ]
        metrics = compute_metrics(comparisons)
        return metrics, comparisons
