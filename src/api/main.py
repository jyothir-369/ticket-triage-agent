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
from src.models import Base, Ticket, TicketStatus, TriageTrace, TraceStep, engine, get_db_session

logger = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan — create tables on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    # Run the agent
    outcome = await run_triage(
        ticket_id=ticket.id,
        subject=ticket.subject,
        body=ticket.body,
    )

    # Update ticket in DB
    ticket.category = outcome["category"]
    ticket.urgency = outcome["urgency"]
    ticket.confidence = outcome["confidence"]
    ticket.drafted_response = outcome["drafted_response"]

    if outcome["decision"] == "escalate":
        ticket.status = TicketStatus.ESCALATED
    else:
        ticket.status = TicketStatus.TRIAGED

    # Persist trace
    trace = TriageTrace(
        ticket_id=ticket.id,
        status=outcome["decision"],
        total_steps=4,
        final_decision=outcome["decision"],
        latency_ms=outcome["latency_ms"],
    )
    db.add(trace)
    await db.flush()

    # Add trace steps
    steps = [
        TraceStep(
            trace_id=trace.id,
            step_type="classify",
            order=1,
            input_summary=f"{ticket.subject[:100]}",
            output_summary=f"category={outcome['category']}, urgency={outcome['urgency']}",
            latency_ms=outcome["latency_ms"] // 4,
        ),
        TraceStep(
            trace_id=trace.id,
            step_type="retrieve",
            order=2,
            input_summary="vector search",
            output_summary="docs retrieved",
            latency_ms=outcome["latency_ms"] // 4,
        ),
        TraceStep(
            trace_id=trace.id,
            step_type="draft",
            order=3,
            input_summary="context assembled",
            output_summary=f"drafted {len(outcome['drafted_response'])} chars",
            latency_ms=outcome["latency_ms"] // 4,
        ),
        TraceStep(
            trace_id=trace.id,
            step_type="escalation_check",
            order=4,
            input_summary=f"confidence={outcome['confidence']}",
            output_summary=f"decision={outcome['decision']}",
            latency_ms=outcome["latency_ms"] // 4,
        ),
    ]
    db.add_all(steps)

    return TriageResponse(**outcome)


# ---------------------------------------------------------------------------
# GET /tickets/{ticket_id}/trace
# ---------------------------------------------------------------------------

class TraceStepOut(BaseModel):
    step_type: str
    order: int
    input_summary: str | None
    output_summary: str | None
    latency_ms: int | None


class TraceOut(BaseModel):
    ticket_id: int
    status: str
    total_steps: int
    final_decision: str | None
    latency_ms: int | None
    steps: list[TraceStepOut]


@app.get("/tickets/{ticket_id}/trace", response_model=TraceOut)
async def get_trace(
    ticket_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Retrieve the step-by-step decision trace for a ticket."""
    result = await db.execute(
        select(TriageTrace).where(TriageTrace.ticket_id == ticket_id)
    )
    trace = result.scalar_one_or_none()

    if trace is None:
        raise HTTPException(status_code=404, detail=f"No trace for ticket {ticket_id}")

    steps_result = await db.execute(
        select(TraceStep).where(TraceStep.trace_id == trace.id).order_by(TraceStep.order)
    )
    steps = steps_result.scalars().all()

    return TraceOut(
        ticket_id=trace.ticket_id,
        status=trace.status,
        total_steps=trace.total_steps,
        final_decision=trace.final_decision,
        latency_ms=trace.latency_ms,
        steps=[
            TraceStepOut(
                step_type=s.step_type.value,
                order=s.order,
                input_summary=s.input_summary,
                output_summary=s.output_summary,
                latency_ms=s.latency_ms,
            )
            for s in steps
        ],
    )


# ---------------------------------------------------------------------------
# GET /dashboard/triage-metrics
# ---------------------------------------------------------------------------

class MetricsOut(BaseModel):
    total_tickets: int
    triaged: int
    escalated: int
    avg_confidence: float | None
    avg_latency_ms: float | None
    category_breakdown: dict[str, int]
    urgency_breakdown: dict[str, int]


@app.get("/dashboard/triage-metrics", response_model=MetricsOut)
async def triage_metrics(
    db: AsyncSession = Depends(get_db_session),
):
    """Aggregate triage metrics for the dashboard."""
    # Counts
    total = (await db.execute(select(func.count(Ticket.id)))).scalar() or 0
    triaged = (
        await db.execute(
            select(func.count(Ticket.id)).where(Ticket.status == TicketStatus.TRIAGED)
        )
    ).scalar() or 0
    escalated = (
        await db.execute(
            select(func.count(Ticket.id)).where(Ticket.status == TicketStatus.ESCALATED)
        )
    ).scalar() or 0

    # Averages
    avg_conf = (
        await db.execute(select(func.avg(Ticket.confidence)).where(Ticket.confidence.isnot(None)))
    ).scalar()
    avg_lat = (
        await db.execute(select(func.avg(TriageTrace.latency_ms)))
    ).scalar()

    # Breakdowns
    cat_rows = (
        await db.execute(
            select(Ticket.category, func.count(Ticket.id)).group_by(Ticket.category)
        )
    ).all()
    urg_rows = (
        await db.execute(
            select(Ticket.urgency, func.count(Ticket.id)).group_by(Ticket.urgency)
        )
    ).all()

    return MetricsOut(
        total_tickets=total,
        triaged=triaged,
        escalated=escalated,
        avg_confidence=round(avg_conf, 3) if avg_conf else None,
        avg_latency_ms=round(avg_lat, 1) if avg_lat else None,
        category_breakdown={str(row[0].value if hasattr(row[0], "value") else row[0]): row[1] for row in cat_rows},
        urgency_breakdown={str(row[0].value if hasattr(row[0], "value") else row[0]): row[1] for row in urg_rows},
    )
