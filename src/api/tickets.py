"""Ticket CRUD and triage endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.graph import AgentExecutor
from src.config import get_settings
from src.models.database import get_db_session, get_session
from src.models.schemas import (
    Ticket,
)
from src.models.ticket import (
    TicketModel,
)
from src.models.ticket import (
    TicketStatus as DBTicketStatus,
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
    is_duplicate: bool = Field(
        default=False,
        description="True if a ticket with the same source_id already existed.",
    )


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
    final_status: str | None = Field(
        default=None,
        description="Lifecycle status of the ticket after the last triage run.",
    )
    total_duration_ms: int | None = Field(
        default=None,
        description="Total wall-clock time of the triage run, if recorded in the trace JSON.",
    )
    loop_count: int | None = Field(
        default=None,
        description="Agent loop-iteration count at the end of the run, if recorded.",
    )


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


class UpdateDraftRequest(BaseModel):
    """Request body for updating a ticket's draft text."""

    draft_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="New draft response text.",
    )


class UpdateDraftResponse(BaseModel):
    """Response after updating the draft."""

    ticket_id: str
    status: str
    message: str


class EscalatedListResponse(BaseModel):
    """Paginated list of escalated tickets."""

    tickets: list[EscalatedTicketOut]
    total: int
    limit: int
    offset: int


class TicketListItem(BaseModel):
    """Summary of a ticket for the queue list."""

    ticket_id: str
    content_preview: str
    source: str = "api"
    status: str = "open"
    category: str | None = None
    urgency: str | None = None
    confidence: float | None = None
    escalation_reason: str | None = None
    created_at: str
    updated_at: str


class TicketListResponse(BaseModel):
    """Paginated list of all tickets with filtering."""

    tickets: list[TicketListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class TicketDetailResponse(BaseModel):
    """Full detail of one ticket for the detail view."""

    ticket_id: str
    content: str
    source: str = "api"
    source_id: str | None = None
    status: str = "open"
    category: str | None = None
    urgency: str | None = None
    confidence: float | None = None
    escalation_reason: str | None = None
    draft_text: str | None = None
    draft_citations: str | None = None
    metadata: str | None = None
    created_at: str
    updated_at: str
    processed_at: str | None = None
    reviewed_at: str | None = None
    review_decision: str | None = None
    trace: str | None = None


# ---------------------------------------------------------------------------
# Background task — runs the triage pipeline asynchronously
# ---------------------------------------------------------------------------


async def _run_triage_background(ticket_id: str) -> None:
    """Execute the full triage pipeline for a ticket in the background.

    Updates the ticket status to PROCESSING before starting, and sets it
    to RESOLVED or ESCALATED on completion.  On failure, status is set
    to FAILED.

    The AgentExecutor handles concurrency limits (semaphore) and total
    timeout internally.
    """
    from src.agent.graph import is_shutting_down

    # Reject new tasks during shutdown
    if is_shutting_down():
        logger.warning("triage_bg.rejected_shutdown", ticket_id=ticket_id)
        return

    repo = TicketRepository()
    try:
        await repo.update_ticket_status(ticket_id, "processing")

        # Fetch the ticket to get its content
        async with get_session() as db:
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

        updates["processed_at"] = datetime.now(UTC)
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
    """Create a new support ticket.

    **Idempotency**: if ``source_id`` is provided and a ticket with that
    ``source_id`` already exists, returns the existing ticket (HTTP 200)
    instead of creating a duplicate (HTTP 201).
    """
    # ── Idempotency check ────────────────────────────────────────────────────
    if request.source_id:
        existing_result = await db.execute(
            select(TicketModel).where(TicketModel.source_id == request.source_id)
        )
        existing_ticket = existing_result.scalar_one_or_none()
        if existing_ticket is not None:
            logger.info(
                "ticket.duplicate_detected",
                source_id=request.source_id,
                existing_ticket_id=existing_ticket.id,
                source=request.source,
            )
            return CreateTicketResponse(
                ticket_id=existing_ticket.id,
                status=existing_ticket.status,
                created_at=existing_ticket.created_at.isoformat(),
                is_duplicate=True,
            )

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
        created_at=datetime.now(UTC),
    )
    db.add(model)
    await db.flush()

    logger.info("ticket.created", ticket_id=ticket_id, source=request.source)

    return CreateTicketResponse(
        ticket_id=ticket_id,
        status="pending",
        created_at=model.created_at.isoformat(),
        is_duplicate=False,
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

    # Allow triage on new tickets (pending/open) AND re-triage on finished
    # tickets (resolved/escalated/failed/awaiting_review), but block while
    # the pipeline is actively running on the ticket.
    if ticket.status not in (
        "pending",
        "open",
        "resolved",
        "escalated",
        "failed",
        "awaiting_review",
        "classified",
    ):
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
    """Return the full triage trace, merging the two trace sources.

    Pipeline steps (classify / retrieve / draft / escalate_check / finalize)
    are stored as a JSON serialisation of :class:`TriageTrace` on the ticket
    row's ``trace`` column.  A handful of rows (escalate, manual_escalation)
    are stored individually in the ``trace_steps`` table.

    This endpoint unifies both sources into a single ordered step list so the
    dashboard Trace Viewer shows the whole pipeline.  When a step name exists
    in both sources the ``trace_steps`` row is preferred.
    """
    # ── 1. Fetch the ticket row — its trace JSON column + lifecycle status ──
    ticket_result = await db.execute(
        select(TicketModel).where(TicketModel.id == ticket_id)
    )
    ticket = ticket_result.scalar_one_or_none()
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    # ── 2. Parse the ticket's trace JSON ────────────────────────────────────
    json_steps: list[dict] = []
    json_final_status: str | None = None
    json_total_duration_ms: int | None = None
    json_loop_count: int | None = None
    if ticket.trace:
        try:
            trace_data = json.loads(ticket.trace)
            if isinstance(trace_data, dict):
                json_steps = trace_data.get("steps") or []
                json_final_status = trace_data.get("final_status")
                json_total_duration_ms = trace_data.get("total_duration_ms")
                json_loop_count = trace_data.get("loop_count")
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("trace.invalid_ticket_json", ticket_id=ticket_id, error=str(exc))

    # ── 3. Fetch trace_steps rows ───────────────────────────────────────────
    db_result = await db.execute(
        select(TraceModel)
        .where(TraceModel.ticket_id == ticket_id)
        .order_by(TraceModel.timestamp)
    )
    db_steps = list(db_result.scalars().all())

    def _ts(value: Any) -> str | None:
        """Normalise a timestamp (datetime | ISO str | None) to ISO string."""
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    # ── 4. Merge — key by step name; prefer the trace_steps version ─────────
    merged: dict[str, TraceStepOut] = {}
    for step in json_steps:
        if not isinstance(step, dict) or not step.get("step"):
            continue
        name = str(step["step"])
        raw_data = step.get("data")
        merged[name] = TraceStepOut(
            step=name,
            status=str(step.get("status", "unknown")),
            timestamp=_ts(step.get("timestamp")),
            duration_ms=step.get("duration_ms"),
            data=raw_data if isinstance(raw_data, dict) else None,
            error=step.get("error"),
        )

    for s in db_steps:
        merged[s.step] = TraceStepOut(
            step=s.step,
            status=s.status,
            timestamp=s.timestamp.isoformat() if s.timestamp else None,
            duration_ms=s.duration_ms,
            data=json.loads(s.data) if s.data else None,
            error=s.error,
        )

    steps = sorted(merged.values(), key=lambda s: s.timestamp or "")

    # ── 5. Top-level metadata ───────────────────────────────────────────────
    # Prefer the final_status recorded on the trace; fall back to the ticket
    # row's lifecycle status (handles tickets triaged before finalize set it).
    final_status = json_final_status or ticket.status

    return TraceOut(
        ticket_id=ticket_id,
        steps=steps,
        final_status=final_status,
        total_duration_ms=json_total_duration_ms,
        loop_count=json_loop_count,
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

    now = datetime.now(UTC)
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

    now = datetime.now(UTC)
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


# ---------------------------------------------------------------------------
# GET /tickets/list — List all tickets with filtering, sorting, pagination
# ---------------------------------------------------------------------------


@router.get(
    "/list",
    response_model=TicketListResponse,
    summary="List all tickets",
    description="Returns a paginated, filterable, sortable list of all tickets.",
)
async def list_tickets(
    search: str = Query(default="", description="Search in content and ticket ID."),
    category: str = Query(default="", description="Comma-separated categories to filter."),
    urgency: str = Query(default="", description="Comma-separated urgency levels to filter."),
    status: str = Query(default="", description="Comma-separated statuses to filter."),
    sort_by: str = Query(default="created_at", description="Field to sort by."),
    sort_dir: str = Query(default="desc", description="Sort direction: asc or desc."),
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    created_after: str = Query(default="", description="ISO date: tickets created after this."),
    created_before: str = Query(default="", description="ISO date: tickets created before this."),
    db: AsyncSession = Depends(get_db_session),
):
    from datetime import datetime as _dt

    query = select(TicketModel)
    count_query = select(func.count(TicketModel.id))

    # ── Filters ─────────────────────────────────────────────────────────────
    if search.strip():
        like_pattern = f"%{search.strip()}%"
        search_filter = TicketModel.id.ilike(like_pattern) | TicketModel.content.ilike(like_pattern)
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)

    if category.strip():
        cats = [c.strip().lower() for c in category.split(",") if c.strip()]
        if cats:
            query = query.where(TicketModel.category.in_(cats))
            count_query = count_query.where(TicketModel.category.in_(cats))

    if urgency.strip():
        urg_vals = [u.strip().lower() for u in urgency.split(",") if u.strip()]
        if urg_vals:
            query = query.where(TicketModel.urgency.in_(urg_vals))
            count_query = count_query.where(TicketModel.urgency.in_(urg_vals))

    if status.strip():
        stat_vals = [s.strip().lower() for s in status.split(",") if s.strip()]
        if stat_vals:
            query = query.where(TicketModel.status.in_(stat_vals))
            count_query = count_query.where(TicketModel.status.in_(stat_vals))

    if created_after.strip():
        try:
            after_dt = _dt.fromisoformat(created_after.strip().replace("Z", "+00:00"))
            query = query.where(TicketModel.created_at >= after_dt)
            count_query = count_query.where(TicketModel.created_at >= after_dt)
        except ValueError:
            pass

    if created_before.strip():
        try:
            before_dt = _dt.fromisoformat(created_before.strip().replace("Z", "+00:00"))
            query = query.where(TicketModel.created_at <= before_dt)
            count_query = count_query.where(TicketModel.created_at <= before_dt)
        except ValueError:
            pass

    # ── Count ───────────────────────────────────────────────────────────────
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    total_pages = max(1, (total + page_size - 1) // page_size)

    # ── Sort ────────────────────────────────────────────────────────────────
    sort_column_map = {
        "created_at": TicketModel.created_at,
        "updated_at": TicketModel.updated_at,
        "confidence": TicketModel.confidence,
        "urgency": TicketModel.urgency,
        "status": TicketModel.status,
        "category": TicketModel.category,
    }
    sort_col = sort_column_map.get(sort_by, TicketModel.created_at)
    if sort_dir.lower() == "asc":
        query = query.order_by(sort_col.asc().nulls_last())
    else:
        query = query.order_by(sort_col.desc().nulls_last())

    # ── Paginate ────────────────────────────────────────────────────────────
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    tickets = result.scalars().all()

    return TicketListResponse(
        tickets=[
            TicketListItem(
                ticket_id=t.id,
                content_preview=t.content[:200] + ("..." if len(t.content) > 200 else ""),
                source=t.source,
                status=t.status,
                category=t.category,
                urgency=t.urgency,
                confidence=t.confidence,
                escalation_reason=t.escalation_reason,
                created_at=t.created_at.isoformat(),
                updated_at=t.updated_at.isoformat(),
            )
            for t in tickets
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# ---------------------------------------------------------------------------
# PATCH /tickets/{id}/draft — Update the draft response text
# ---------------------------------------------------------------------------


@router.patch(
    "/{ticket_id}/draft",
    response_model=UpdateDraftResponse,
    summary="Update draft response",
    description="Replaces the ticket's draft response text (used for edit-and-approve).",
)
async def update_ticket_draft(
    ticket_id: str,
    request: UpdateDraftRequest,
    db: AsyncSession = Depends(get_db_session),
):
    result = await db.execute(select(TicketModel).where(TicketModel.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    now = datetime.now(UTC)
    ticket.draft_text = request.draft_text
    ticket.review_decision = None  # reset any prior decision since draft changed
    ticket.updated_at = now
    await db.flush()

    logger.info("ticket.draft_updated", ticket_id=ticket_id)

    return UpdateDraftResponse(
        ticket_id=ticket_id,
        status=ticket.status,
        message="Draft updated successfully.",
    )


# ---------------------------------------------------------------------------
# GET /tickets/{id} — Full ticket detail
# ---------------------------------------------------------------------------


@router.get(
    "/{ticket_id}",
    response_model=TicketDetailResponse,
    summary="Get full ticket detail",
    description="Returns the full ticket row including content, draft, and trace.",
)
async def get_ticket_detail(
    ticket_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    result = await db.execute(select(TicketModel).where(TicketModel.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    return TicketDetailResponse(
        ticket_id=ticket.id,
        content=ticket.content,
        source=ticket.source,
        source_id=ticket.source_id,
        status=ticket.status,
        category=ticket.category,
        urgency=ticket.urgency,
        confidence=ticket.confidence,
        escalation_reason=ticket.escalation_reason,
        draft_text=ticket.draft_text,
        draft_citations=ticket.draft_citations,
        metadata=ticket.metadata_,
        created_at=ticket.created_at.isoformat(),
        updated_at=ticket.updated_at.isoformat(),
        processed_at=ticket.processed_at.isoformat() if ticket.processed_at else None,
        reviewed_at=ticket.reviewed_at.isoformat() if ticket.reviewed_at else None,
        review_decision=ticket.review_decision,
        trace=ticket.trace,
    )
