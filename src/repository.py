"""TicketRepository — async data-access layer for the triage agent.

All public methods are ``async`` and use the shared session context manager
from ``src.models.database``.  Transient database errors are retried
automatically via tenacity.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update

from src.models.database import get_session
from src.models.schemas import (
    DashboardMetrics,
    RecentActivity,
    Ticket,
    TraceStep,
)
from src.models.ticket import TicketModel, TicketStatus
from src.models.trace import TraceModel

logger = logging.getLogger(__name__)


class TicketRepository:
    """Async repository for ticket CRUD and triage-bookkeeping operations.

    Every method opens its own session via the ``get_session`` context manager,
    which handles commit / rollback and exposes tenacity retry on transient
    connection failures.
    """

    # ── Create ──────────────────────────────────────────────────────────────────

    async def create_ticket(self, ticket: Ticket) -> TicketModel:
        """Persist a new :class:`Ticket` and return the ORM model.

        **Idempotency**: if ``ticket.source_id`` is provided and a ticket
        with that ``source_id`` already exists, returns the existing model
        instead of creating a duplicate.
        """
        async with get_session() as session:
            # Idempotency check
            if ticket.source_id:
                existing = await session.execute(
                    select(TicketModel).where(TicketModel.source_id == ticket.source_id)
                )
                existing_model = existing.scalar_one_or_none()
                if existing_model is not None:
                    logger.info(
                        "Ticket already exists for source_id=%s: %s",
                        ticket.source_id,
                        existing_model.id,
                    )
                    return existing_model

            model = TicketModel(
                id=str(ticket.id),
                content=ticket.content,
                source=ticket.source,
                source_id=ticket.source_id,
                metadata_=ticket.metadata.model_dump_json(),
                status=ticket.status.value,
                created_at=ticket.created_at,
            )
            session.add(model)
            await session.flush()  # populate defaults / generated columns
            await session.refresh(model)
            logger.info("Created ticket %s (source=%s)", model.id, model.source)
            return model

    # ── Generic update ──────────────────────────────────────────────────────────

    async def update_ticket(self, ticket_id: str, updates: dict[str, Any]) -> TicketModel:
        """Apply *updates* dict to the ticket row and return the refreshed model.

        Keys in *updates* must match ``TicketModel`` column names.  JSON-serialisable
        values are handled automatically for ``metadata_``.
        """
        # Serialise nested dicts that live in Text columns
        serialisable = {}
        for key, value in updates.items():
            if key == "metadata_" and isinstance(value, dict):
                serialisable[key] = json.dumps(value)
            else:
                serialisable[key] = value

        async with get_session() as session:
            stmt = (
                update(TicketModel)
                .where(TicketModel.id == ticket_id)
                .values(**serialisable)
                .returning(TicketModel)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                raise ValueError(f"Ticket {ticket_id} not found")
            await session.flush()
            await session.refresh(row)
            logger.info("Updated ticket %s with keys: %s", ticket_id, list(serialisable.keys()))
            return row

    # ── Status update (+ optional trace) ────────────────────────────────────────

    async def update_ticket_status(
        self,
        ticket_id: str,
        status: str,
        trace: dict[str, Any] | None = None,
    ) -> None:
        """Set the ticket lifecycle status and optionally append a trace entry."""
        now = datetime.now(UTC)
        values: dict[str, Any] = {"status": status, "updated_at": now}

        if status in (TicketStatus.RESOLVED.value, TicketStatus.FAILED.value):
            values["processed_at"] = now

        async with get_session() as session:
            stmt = update(TicketModel).where(TicketModel.id == ticket_id).values(**values)
            await session.execute(stmt)

            if trace is not None:
                trace_row = TraceModel(
                    ticket_id=ticket_id,
                    step=trace.get("step", "status_change"),
                    status=trace.get("status", status),
                    timestamp=now,
                    duration_ms=trace.get("duration_ms"),
                    data=json.dumps(trace.get("data")) if trace.get("data") else None,
                    error=trace.get("error"),
                )
                session.add(trace_row)

            await session.flush()
            logger.info("Ticket %s → %s", ticket_id, status)

    # ── Save triage results ─────────────────────────────────────────────────────

    async def save_triage_result(
        self,
        ticket_id: str,
        classification: dict[str, Any],
        draft: dict[str, Any],
        escalation: dict[str, Any],
    ) -> None:
        """Persist classification, draft, and escalation results on the ticket.

        Also marks the ticket as ``awaiting_review`` or ``escalated``
        depending on the escalation decision.
        """
        now = datetime.now(UTC)
        should_escalate = escalation.get("should_escalate", False)

        values: dict[str, Any] = {
            "category": classification.get("category"),
            "urgency": classification.get("urgency"),
            "confidence": classification.get("confidence"),
            "draft_text": draft.get("draft_text"),
            "draft_citations": json.dumps(draft.get("citations", [])),
            "escalation_reason": escalation.get("reason"),
            "status": (
                TicketStatus.ESCALATED.value
                if should_escalate
                else TicketStatus.AWAITING_REVIEW.value
            ),
            "updated_at": now,
        }

        async with get_session() as session:
            stmt = update(TicketModel).where(TicketModel.id == ticket_id).values(**values)
            await session.execute(stmt)
            await session.flush()
            logger.info(
                "Saved triage result for ticket %s (escalated=%s)", ticket_id, should_escalate
            )

    # ── Trace retrieval ─────────────────────────────────────────────────────────

    async def get_ticket_trace(self, ticket_id: str) -> dict[str, Any] | None:
        """Return the full trace for a ticket as a JSON-safe dict, or None."""
        async with get_session() as session:
            stmt = (
                select(TraceModel)
                .where(TraceModel.ticket_id == ticket_id)
                .order_by(TraceModel.timestamp)
            )
            result = await session.execute(stmt)
            rows = list(result.scalars().all())

            if not rows:
                return None

            return {
                "ticket_id": ticket_id,
                "steps": [
                    {
                        "id": r.id,
                        "step": r.step,
                        "status": r.status,
                        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                        "duration_ms": r.duration_ms,
                        "data": json.loads(r.data) if r.data else None,
                        "error": r.error,
                    }
                    for r in rows
                ],
            }

    # ── Escalated tickets ───────────────────────────────────────────────────────

    async def get_escalated_tickets(self, limit: int = 50) -> list[TicketModel]:
        """Return up to *limit* tickets currently in ``escalated`` status."""
        async with get_session() as session:
            stmt = (
                select(TicketModel)
                .where(TicketModel.status == TicketStatus.ESCALATED.value)
                .order_by(TicketModel.updated_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # ── Dashboard metrics ───────────────────────────────────────────────────────

    async def get_dashboard_metrics(self) -> DashboardMetrics:
        """Aggregate triage metrics across all tickets."""
        async with get_session() as session:
            # Totals
            total_q = await session.execute(select(func.count(TicketModel.id)))
            total_tickets = total_q.scalar() or 0

            processed_q = await session.execute(
                select(func.count(TicketModel.id)).where(
                    TicketModel.status.in_([
                        TicketStatus.RESOLVED.value,
                        TicketStatus.ESCALATED.value,
                        TicketStatus.APPROVED.value,
                        TicketStatus.REJECTED.value,
                        TicketStatus.AWAITING_REVIEW.value,
                    ])
                )
            )
            processed_tickets = processed_q.scalar() or 0

            escalated_q = await session.execute(
                select(func.count(TicketModel.id)).where(
                    TicketModel.status == TicketStatus.ESCALATED.value
                )
            )
            escalated_tickets = escalated_q.scalar() or 0

            # Approval rate
            approved_q = await session.execute(
                select(func.count(TicketModel.id)).where(
                    TicketModel.review_decision == "approved"
                )
            )
            approved_count = approved_q.scalar() or 0
            approval_rate = (
                approved_count / escalated_tickets if escalated_tickets > 0 else 0.0
            )

            # Confidence average
            avg_conf_q = await session.execute(
                select(func.avg(TicketModel.confidence)).where(
                    TicketModel.confidence.isnot(None)
                )
            )
            avg_confidence = avg_conf_q.scalar() or 0.0

            # Category distribution
            cat_q = await session.execute(
                select(TicketModel.category, func.count(TicketModel.id))
                .where(TicketModel.category.isnot(None))
                .group_by(TicketModel.category)
            )
            category_distribution = {row[0]: row[1] for row in cat_q.all()}

            # Urgency distribution
            urg_q = await session.execute(
                select(TicketModel.urgency, func.count(TicketModel.id))
                .where(TicketModel.urgency.isnot(None))
                .group_by(TicketModel.urgency)
            )
            urgency_distribution = {row[0]: row[1] for row in urg_q.all()}

            # Recent activity (last 10)
            recent_q = await session.execute(
                select(TicketModel)
                .order_by(TicketModel.updated_at.desc())
                .limit(10)
            )
            recent_rows = recent_q.scalars().all()
            recent_activity = [
                RecentActivity(
                    ticket_id=r.id,
                    action=r.status,
                    timestamp=r.updated_at,
                    details=r.escalation_reason or "",
                )
                for r in recent_rows
            ]

            # Success rate (resolved / total)
            resolved_q = await session.execute(
                select(func.count(TicketModel.id)).where(
                    TicketModel.status == TicketStatus.RESOLVED.value
                )
            )
            resolved_count = resolved_q.scalar() or 0
            success_rate = resolved_count / total_tickets if total_tickets > 0 else 0.0

            # Escalation rate
            escalation_rate = (
                escalated_tickets / total_tickets if total_tickets > 0 else 0.0
            )

            # Trace-based step count average
            steps_q = await session.execute(
                select(func.count(TraceModel.id)).group_by(TraceModel.ticket_id)
            )
            step_counts = [row[0] for row in steps_q.all()]
            avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0.0

            return DashboardMetrics(
                total_tickets=total_tickets,
                processed_tickets=processed_tickets,
                escalated_tickets=escalated_tickets,
                approval_rate=round(approval_rate, 4),
                escalation_rate=round(escalation_rate, 4),
                avg_confidence=round(float(avg_confidence), 4),
                avg_steps=round(avg_steps, 2),
                success_rate=round(success_rate, 4),
                category_distribution=category_distribution,
                urgency_distribution=urgency_distribution,
                recent_activity=recent_activity,
            )

    # ── Internal helpers ─────────────────────────────────────────────────────────

    async def _get_ticket(self, ticket_id: str) -> TicketModel | None:
        """Fetch a single ticket by ID (used internally and in tests)."""
        async with get_session() as session:
            result = await session.execute(
                select(TicketModel).where(TicketModel.id == ticket_id)
            )
            return result.scalar_one_or_none()

    # ── Add trace step ──────────────────────────────────────────────────────────

    async def add_trace_step(self, ticket_id: str, step: TraceStep) -> None:
        """Append a single :class:`TraceStep` to the ticket's audit trail."""
        trace_row = TraceModel(
            ticket_id=ticket_id,
            step=step.step,
            status=step.status.value,
            timestamp=step.timestamp,
            duration_ms=step.duration_ms,
            data=json.dumps(step.data) if step.data else None,
            error=step.error,
        )
        async with get_session() as session:
            session.add(trace_row)
            await session.flush()
            logger.debug("Added trace step '%s' for ticket %s", step.step, ticket_id)
