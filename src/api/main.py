"""FastAPI entry point — exposes triage, trace, and metrics endpoints."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.graph import run_triage
from src.config import get_settings
from src.models.database import Base, get_db_session, get_engine
from src.models.schemas import (
    DashboardMetrics,
    TicketCategory,
    TicketStatus,
    UrgencyLevel,
)
from src.models.ticket import (
    TicketModel,
    TicketCategory as DBTicketCategory,
    TicketStatus as DBTicketStatus,
    TicketUrgency as DBTicketUrgency,
)
from src.models.trace import TraceModel

logger = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan — create tables on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("api.startup", db=settings.database_url.split("@")[-1])
    yield
    await engine.dispose()


app = FastAPI(
    title="Support-Ticket Triage Agent",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /tickets/{ticket_id}/triage
# ---------------------------------------------------------------------------

class TriageResponse(BaseModel):
    ticket_id: int
    category: str
    urgency: str
    confidence: float
    drafted_response: str
    decision: str
    escalation_reason: str
    latency_ms: int


@app.post("/tickets/{ticket_id}/triage", response_model=TriageResponse)
async def triage_ticket(
    ticket_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Run the full triage pipeline on a ticket."""
    # Fetch ticket from DB
    result = await db.execute(select(TicketModel).where(TicketModel.id == str(ticket_id)))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    # Run the agent
    outcome = await run_triage(
        ticket_id=ticket.id,
        subject=ticket.content[:100],
        body=ticket.content,
    )

    # Update ticket in DB
    ticket.category = outcome["category"]
    ticket.urgency = outcome["urgency"]
    ticket.confidence = outcome["confidence"]
    ticket.draft_text = outcome["drafted_response"]

    if outcome["decision"] == "escalate":
        ticket.status = DBTicketStatus.ESCALATED.value
    else:
        ticket.status = DBTicketStatus.RESOLVED.value

    # Persist trace step
    trace = TraceModel(
        ticket_id=ticket.id,
        step="triage",
        status=outcome["decision"],
        duration_ms=outcome["latency_ms"],
    )
    db.add(trace)
    await db.flush()

    return TriageResponse(**outcome)


# ---------------------------------------------------------------------------
# GET /tickets/{ticket_id}/trace
# ---------------------------------------------------------------------------

class TraceStepOut(BaseModel):
    step: str
    status: str
    duration_ms: int | None
    error: str | None


class TraceOut(BaseModel):
    ticket_id: str
    steps: list[TraceStepOut]


@app.get("/tickets/{ticket_id}/trace", response_model=TraceOut)
async def get_trace(
    ticket_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Retrieve the step-by-step decision trace for a ticket."""
    result = await db.execute(
        select(TraceModel)
        .where(TraceModel.ticket_id == str(ticket_id))
        .order_by(TraceModel.timestamp)
    )
    steps = result.scalars().all()

    if not steps:
        raise HTTPException(status_code=404, detail=f"No trace for ticket {ticket_id}")

    return TraceOut(
        ticket_id=str(ticket_id),
        steps=[
            TraceStepOut(
                step=s.step,
                status=s.status,
                duration_ms=s.duration_ms,
                error=s.error,
            )
            for s in steps
        ],
    )


# ---------------------------------------------------------------------------
# GET /dashboard/triage-metrics
# ---------------------------------------------------------------------------

@app.get("/dashboard/triage-metrics", response_model=DashboardMetrics)
async def triage_metrics(
    db: AsyncSession = Depends(get_db_session),
):
    """Aggregate triage metrics for the dashboard."""
    # Counts
    total = (await db.execute(select(func.count(TicketModel.id)))).scalar() or 0
    resolved = (
        await db.execute(
            select(func.count(TicketModel.id)).where(
                TicketModel.status == DBTicketStatus.RESOLVED.value
            )
        )
    ).scalar() or 0
    escalated = (
        await db.execute(
            select(func.count(TicketModel.id)).where(
                TicketModel.status == DBTicketStatus.ESCALATED.value
            )
        )
    ).scalar() or 0

    # Averages
    avg_conf = (
        await db.execute(
            select(func.avg(TicketModel.confidence)).where(TicketModel.confidence.isnot(None))
        )
    ).scalar()

    # Breakdowns
    cat_rows = (
        await db.execute(
            select(TicketModel.category, func.count(TicketModel.id))
            .where(TicketModel.category.isnot(None))
            .group_by(TicketModel.category)
        )
    ).all()
    urg_rows = (
        await db.execute(
            select(TicketModel.urgency, func.count(TicketModel.id))
            .where(TicketModel.urgency.isnot(None))
            .group_by(TicketModel.urgency)
        )
    ).all()

    category_breakdown = {row[0]: row[1] for row in cat_rows}
    urgency_breakdown = {row[0]: row[1] for row in urg_rows}

    escalation_rate = escalated / total if total > 0 else 0.0
    success_rate = resolved / total if total > 0 else 0.0

    return DashboardMetrics(
        total_tickets=total,
        processed_tickets=resolved + escalated,
        escalated_tickets=escalated,
        escalation_rate=round(escalation_rate, 4),
        avg_confidence=round(avg_conf, 4) if avg_conf else 0.0,
        success_rate=round(success_rate, 4),
        category_distribution=category_breakdown,
        urgency_distribution=urgency_breakdown,
    )
