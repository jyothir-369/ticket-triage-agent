"""Ticket CRUD and triage endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.graph import AgentExecutor
from src.config import get_settings
from src.models.database import get_db_session
from src.models.schemas import (
    DashboardMetrics,
    Ticket,
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
from src.repository import TicketRepository

logger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/tickets", tags=["tickets"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateTicketRequest(BaseModel):
    """Request body for creating a new support ticket."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Full text body of the ticket.",
    )
    source: str = Field(
        default="api",
        min_length=1,
        max_length=100,
        description="Origin system (e.g. 'email', 'github', 'intercom', 'api').",
    )
    source_id: str | None = Field(
        default=None,
        max_length=500,
        description="Original ID in the source system (e.g. GitHub issue #).",
    )
    customer_email: str | None = Field(
        default=None,
        description="Customer email address.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorisation.",
    )


class CreateTicketResponse(BaseModel):
    """Response after creating a ticket."""

    ticket_id: str
    status: str
    created_at: str


class TriageResponse(BaseModel):
    """Response after triggering triage processing."""

    ticket_id: str
    status: str
    message: str


class TicketStatusResponse(BaseModel):
    """Current ticket status."""

    ticket_id: str
    status: str
    category: str | None = None
    urgency: str | None = None
    confidence: float | None = None
    created_at: str
    updated_at: str
    processed_at: str | None = None


class TraceStepOut(BaseModel):
    """One step in the triage trace."""

    step: str
    status: str
    timestamp: str | None = None
    duration_ms: int | None = None
    data: dict[str, Any] | None = None
    error: str | None = None


class TraceOut(BaseModel):
    """Full trace for a ticket."""

    ticket_id: str
    steps: list[TraceStepOut]


class ApproveRequest(BaseModel):
    """Request body for approving a draft response."""

    reviewer: str = Field(
        default="system",
        description="Identifier of the reviewer approving the draft.",
    )


class ApproveResponse(BaseModel):
    """Response after approving a draft."""

    ticket_id: str
    status: str
    message: str


class EscalateRequest(BaseModel):
    """Request body for manually escalating a ticket."""

    reason: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Reason for manual escalation.",
    )
    reviewer: str = Field(
        default="system",
        description="Identifier of the person escalating.",
    )


class EscalateResponse(BaseModel):
    """Response after escalating a ticket."""

    ticket_id: str
    status: str
    message: str


class EscalatedTicketOut(BaseModel):
    """Summary of an escalated ticket."""

    ticket_id: str
    content_preview: str
    category: str | None = None
    urgency: str | None = None
    escalation_reason: str | None = None
    created_at: str
    updated_at: str


class EscalatedListResponse(BaseModel):
    """Paginated list of escalated tickets."""

    tickets: list[EscalatedTicketOut]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Background task — runs the triage pipeline asynchronously
# ---------------------------------------------------------------------------


async def _run_triage_background(ticket_id: str) -> None:
    """Execute the full triage pipeline for a ticket in the background.

    Updates the ticket status to PROCESSING before starting, and sets it
    to RESOLVED or ESCALATED on completion.  On failure, status is set
    to FAILED.
    """
    repo = TicketRepository()
    try:
        await repo.update_ticket_status(ticket_id, "processing")

        # Fetch the ticket to get its content
        async with get_db_session() as db:
            result = await db.execute(
                select(TicketModel).where(TicketModel.id == ticket_id)
            )
            ticket_model = result.scalar_one_or_none()

        if ticket_model is None:
            logger.error("triage_bg.ticket_not_found", ticket_id=ticket_id)
            return

        ticket = Ticket(
            id=ticket_model.id,
            content=ticket_model.content,
            source=ticket_model.source,
            source_id=ticket_model.source_id,
        )

        executor = AgentExecutor()
        final_state = await executor.run(ticket)

        # Update ticket with results
        classification = final_state.get("classification")
        draft = final_state.get("draft")
        should_escalate = final_state.get("should_escalate", False)

        updates: dict[str, Any] = {}
        if classification:
            updates["category"] = classification.category.value
            updates["urgency"] = classification.urgency.value
            updates["confidence"] = classification.confidence
        if draft:
            updates["draft_text"] = draft.draft_text
            updates["draft_citations"] = json.dumps(
                [c.model_dump() for c in draft.citations]
            )
        if should_escalate:
            updates["status"] = DBTicketStatus.ESCALATED.value
            updates["escalation_reason"] = final_state.get("escalation_reason", "")
        else:
            updates["status"] = DBTicketStatus.RESOLVED.value

        updates["processed_at"] = datetime.now(timezone.utc)
        await repo.update_ticket(ticket_id, updates)

        logger.info(
            "triage_bg.complete",
            ticket_id=ticket_id,
            escalated=should_escalate,
        )

    except Exception as exc:
        logger.error(
            "triage_bg.failed",
            ticket_id=ticket_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        await repo.update_ticket_status(ticket_id, "failed")


# ---------------------------------------------------------------------------
# POST /tickets/ — Create new ticket
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=CreateTicketResponse,
    status_code=201,
    summary="Create a new support ticket",
    description="Stores a new ticket in the database with PENDING status and returns the ticket ID.",
)
async def create_ticket(
    request: CreateTicketRequest,
    db: AsyncSession = Depends(get_db_session),
):
    ticket_id = str(uuid.uuid4())

    model = TicketModel(
        id=ticket_id,
        content=request.content,
        source=request.source,
        source_id=request.source_id,
        metadata_=json.dumps({
            "customer_email": request.customer_email,
            "tags": request.tags,
        }),
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db.add(model)
    await db.flush()

    logger.info("ticket.created", ticket_id=ticket_id, source=request.source)

    return CreateTicketResponse(
        ticket_id=ticket_id,
        status="pending",
        created_at=model.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# POST /tickets/{id}/triage — Trigger async triage
# ---------------------------------------------------------------------------


@router.post(
    "/{ticket_id}/triage",
    response_model=TriageResponse,
    status_code=202,
    summary="Trigger async triage processing",
    description=(
        "Kicks off the full triage pipeline (classify → retrieve → draft → "
        "escalation gate) as a background task. Returns 202 Accepted immediately."
    ),
)
async def triage_ticket(
    ticket_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
):
    result = await db.execute(select(TicketModel).where(TicketModel.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    if ticket.status not in ("pending", "open"):
        raise HTTPException(
            status_code=409,
            detail=f"Ticket {ticket_id} is already in status '{ticket.status}'",
        )

    background_tasks.add_task(_run_triage_background, ticket_id)

    logger.info("triage.triggered", ticket_id=ticket_id)

    return TriageResponse(
        ticket_id=ticket_id,
        status="processing",
        message="Triage processing has been started in the background.",
    )


# ---------------------------------------------------------------------------
# GET /tickets/{id}/status — Return current status
# ---------------------------------------------------------------------------


@router.get(
    "/{ticket_id}/status",
    response_model=TicketStatusResponse,
    summary="Get ticket status",
    description="Returns the current lifecycle status of a ticket.",
)
async def get_ticket_status(
    ticket_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    result = await db.execute(select(TicketModel).where(TicketModel.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    return TicketStatusResponse(
        ticket_id=ticket.id,
        status=ticket.status,
        category=ticket.category,
        urgency=ticket.urgency,
        confidence=ticket.confidence,
        created_at=ticket.created_at.isoformat(),
        updated_at=ticket.updated_at.isoformat(),
        processed_at=ticket.processed_at.isoformat() if ticket.processed_at else None,
    )


# ---------------------------------------------------------------------------
# GET /tickets/{id}/trace — Return full trace
# ---------------------------------------------------------------------------


@router.get(
    "/{ticket_id}/trace",
    response_model=TraceOut,
    summary="Get triage trace",
    description="Returns the full trace with all pipeline steps and timestamps.",
)
async def get_trace(
    ticket_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    result = await db.execute(
        select(TraceModel)
        .where(TraceModel.ticket_id == ticket_id)
        .order_by(TraceModel.timestamp)
    )
    steps = result.scalars().all()

    if not steps:
        # Check if ticket exists at all
        ticket_result = await db.execute(
            select(TicketModel).where(TicketModel.id == ticket_id)
        )
        if ticket_result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
        raise HTTPException(
            status_code=404,
            detail=f"No trace data for ticket {ticket_id}",
        )

    return TraceOut(
        ticket_id=ticket_id,
        steps=[
            TraceStepOut(
                step=s.step,
                status=s.status,
                timestamp=s.timestamp.isoformat() if s.timestamp else None,
                duration_ms=s.duration_ms,
                data=json.loads(s.data) if s.data else None,
                error=s.error,
            )
            for s in steps
        ],
    )


# ---------------------------------------------------------------------------
# POST /tickets/{id}/approve — Human approves draft
# ---------------------------------------------------------------------------


@router.post(
    "/{ticket_id}/approve",
    response_model=ApproveResponse,
    summary="Approve a draft response",
    description=(
        "Human approves the draft response, updates status to RESOLVED, "
        "and mocks sending the reply to the customer."
    ),
)
async def approve_ticket(
    ticket_id: str,
    request: ApproveRequest,
    db: AsyncSession = Depends(get_db_session),
):
    result = await db.execute(select(TicketModel).where(TicketModel.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    if ticket.status not in ("awaiting_review", "escalated"):
        raise HTTPException(
            status_code=409,
            detail=f"Ticket {ticket_id} cannot be approved in status '{ticket.status}'",
        )

    now = datetime.now(timezone.utc)
    ticket.status = DBTicketStatus.RESOLVED.value
    ticket.review_decision = "approved"
    ticket.reviewed_at = now
    ticket.updated_at = now
    await db.flush()

    logger.info(
        "ticket.approved",
        ticket_id=ticket_id,
        reviewer=request.reviewer,
    )

    # Mock sending reply to customer
    logger.info(
        "ticket.reply_sent",
        ticket_id=ticket_id,
        customer=getattr(ticket, "metadata_", None),
    )

    return ApproveResponse(
        ticket_id=ticket_id,
        status="resolved",
        message="Draft approved and reply sent to customer.",
    )


# ---------------------------------------------------------------------------
# POST /tickets/{id}/escalate — Human manually escalates
# ---------------------------------------------------------------------------


@router.post(
    "/{ticket_id}/escalate",
    response_model=EscalateResponse,
    summary="Manually escalate a ticket",
    description="Human manually escalates a ticket with a reason.",
)
async def escalate_ticket(
    ticket_id: str,
    request: EscalateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    result = await db.execute(select(TicketModel).where(TicketModel.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    now = datetime.now(timezone.utc)
    ticket.status = DBTicketStatus.ESCALATED.value
    ticket.escalation_reason = request.reason
    ticket.reviewed_at = now
    ticket.updated_at = now

    # Add a trace step
    trace_row = TraceModel(
        ticket_id=ticket_id,
        step="manual_escalation",
        status="escalated",
        timestamp=now,
        data=json.dumps({
            "reason": request.reason,
            "reviewer": request.reviewer,
        }),
    )
    db.add(trace_row)
    await db.flush()

    logger.info(
        "ticket.escalated",
        ticket_id=ticket_id,
        reviewer=request.reviewer,
        reason=request.reason,
    )

    return EscalateResponse(
        ticket_id=ticket_id,
        status="escalated",
        message="Ticket has been escalated for human review.",
    )


# ---------------------------------------------------------------------------
# GET /tickets/escalated — List escalated tickets
# ---------------------------------------------------------------------------


@router.get(
    "/escalated/list",
    response_model=EscalatedListResponse,
    summary="List escalated tickets",
    description="Returns a paginated list of escalated tickets.",
)
async def list_escalated_tickets(
    limit: int = Query(default=50, ge=1, le=200, description="Max tickets to return."),
    offset: int = Query(default=0, ge=0, description="Number of tickets to skip."),
    db: AsyncSession = Depends(get_db_session),
):
    # Count total
    total_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            TicketModel.status == DBTicketStatus.ESCALATED.value
        )
    )
    total = total_q.scalar() or 0

    # Fetch page
    result = await db.execute(
        select(TicketModel)
        .where(TicketModel.status == DBTicketStatus.ESCALATED.value)
        .order_by(TicketModel.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    tickets = result.scalars().all()

    return EscalatedListResponse(
        tickets=[
            EscalatedTicketOut(
                ticket_id=t.id,
                content_preview=t.content[:200] + ("..." if len(t.content) > 200 else ""),
                category=t.category,
                urgency=t.urgency,
                escalation_reason=t.escalation_reason,
                created_at=t.created_at.isoformat(),
                updated_at=t.updated_at.isoformat(),
            )
            for t in tickets
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
