"""Dashboard and health-check endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_db_session
from src.models.schemas import DashboardMetrics, RecentActivity
from src.models.ticket import TicketModel
from src.models.ticket import TicketStatus as DBTicketStatus
from src.models.trace import TraceModel

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["dashboard"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ActivityResponse(BaseModel):
    """Recent activity feed."""

    activities: list[RecentActivity]


class CategoryDistribution(BaseModel):
    """Category distribution response."""

    categories: dict[str, int]


# ---------------------------------------------------------------------------
# GET /dashboard/metrics — Aggregate triage metrics
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard/metrics",
    response_model=DashboardMetrics,
    summary="Get dashboard metrics",
    description=(
        "Returns aggregated metrics including total tickets, escalation rate, "
        "average confidence, average steps, and success rate. Pass ``days`` to "
        "scope all counts to tickets created within that window."
    ),
)
async def get_metrics(
    days: int | None = Query(default=None, ge=1, le=3650, description="Look-back window in days."),
    db: AsyncSession = Depends(get_db_session),
):
    from datetime import UTC, datetime, timedelta

    created_filter = None
    if days:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        created_filter = TicketModel.created_at >= cutoff

    def scoped(*where) -> tuple:
        """Return a where-clause tuple with the created_at filter appended."""
        return where + (created_filter,) if created_filter is not None else where

    # Counts
    total_q = await db.execute(
        select(func.count(TicketModel.id)).where(*scoped())
    )
    total_tickets = total_q.scalar() or 0

    processed_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            *scoped(
                TicketModel.status.in_([
                    DBTicketStatus.RESOLVED.value,
                    DBTicketStatus.ESCALATED.value,
                    DBTicketStatus.APPROVED.value,
                    DBTicketStatus.REJECTED.value,
                    DBTicketStatus.AWAITING_REVIEW.value,
                ])
            )
        )
    )
    processed_tickets = processed_q.scalar() or 0

    escalated_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            *scoped(TicketModel.status == DBTicketStatus.ESCALATED.value)
        )
    )
    escalated_tickets = escalated_q.scalar() or 0

    # Approval rate
    approved_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            *scoped(TicketModel.review_decision == "approved")
        )
    )
    approved_count = approved_q.scalar() or 0
    approval_rate = approved_count / escalated_tickets if escalated_tickets > 0 else 0.0

    # Confidence average
    avg_conf_q = await db.execute(
        select(func.avg(TicketModel.confidence)).where(
            *scoped(TicketModel.confidence.isnot(None))
        )
    )
    avg_confidence = avg_conf_q.scalar() or 0.0

    # Category distribution
    cat_q = await db.execute(
        select(TicketModel.category, func.count(TicketModel.id))
        .where(*scoped(TicketModel.category.isnot(None)))
        .group_by(TicketModel.category)
    )
    category_distribution = {row[0]: row[1] for row in cat_q.all()}

    # Urgency distribution
    urg_q = await db.execute(
        select(TicketModel.urgency, func.count(TicketModel.id))
        .where(*scoped(TicketModel.urgency.isnot(None)))
        .group_by(TicketModel.urgency)
    )
    urgency_distribution = {row[0]: row[1] for row in urg_q.all()}

    # Recent activity (last 10)
    recent_q = await db.execute(
        select(TicketModel)
        .where(*scoped())
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
    resolved_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            *scoped(TicketModel.status == DBTicketStatus.RESOLVED.value)
        )
    )
    resolved_count = resolved_q.scalar() or 0
    success_rate = resolved_count / total_tickets if total_tickets > 0 else 0.0

    # Escalation rate
    escalation_rate = escalated_tickets / total_tickets if total_tickets > 0 else 0.0

    # Trace-based step count average
    steps_q = await db.execute(
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


# ---------------------------------------------------------------------------
# GET /dashboard/activity — Recent activity log
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard/activity",
    response_model=ActivityResponse,
    summary="Get recent activity",
    description="Returns the most recent triage activity events.",
)
async def get_activity(
    db: AsyncSession = Depends(get_db_session),
):
    recent_q = await db.execute(
        select(TicketModel)
        .order_by(TicketModel.updated_at.desc())
        .limit(20)
    )
    recent_rows = recent_q.scalars().all()

    activities = [
        RecentActivity(
            ticket_id=r.id,
            action=r.status,
            timestamp=r.updated_at,
            details=r.escalation_reason or "",
        )
        for r in recent_rows
    ]

    return ActivityResponse(activities=activities)


# ---------------------------------------------------------------------------
# GET /dashboard/categories — Category distribution
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard/categories",
    response_model=CategoryDistribution,
    summary="Get category distribution",
    description="Returns the count of tickets per category.",
)
async def get_categories(
    db: AsyncSession = Depends(get_db_session),
):
    cat_q = await db.execute(
        select(TicketModel.category, func.count(TicketModel.id))
        .where(TicketModel.category.isnot(None))
        .group_by(TicketModel.category)
    )
    distribution = {row[0]: row[1] for row in cat_q.all()}

    return CategoryDistribution(categories=distribution)


# ---------------------------------------------------------------------------
# GET /dashboard/timeseries — Daily ticket counts for time-series chart
# ---------------------------------------------------------------------------


class TimeSeriesPoint(BaseModel):
    date: str
    resolved: int = 0
    escalated: int = 0
    pending: int = 0
    total: int = 0
    avg_confidence: float | None = None


class TimeSeriesResponse(BaseModel):
    points: list[TimeSeriesPoint]
    period_days: int


@router.get(
    "/dashboard/timeseries",
    response_model=TimeSeriesResponse,
    summary="Get daily ticket time-series",
    description="Returns daily counts of resolved/escalated/pending tickets for the selected period.",
)
async def get_timeseries(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back."),
    db: AsyncSession = Depends(get_db_session),
):
    cutoff = datetime.now(UTC) - timedelta(days=days)

    # Fetch all tickets created within the period
    result = await db.execute(
        select(TicketModel)
        .where(TicketModel.created_at >= cutoff)
        .order_by(TicketModel.created_at)
    )
    tickets = result.scalars().all()

    # Bucket by date
    daily: dict[str, dict] = {}
    for i in range(days):
        d = (datetime.now(UTC) - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        daily[d] = {"resolved": 0, "escalated": 0, "pending": 0, "total": 0, "conf_sum": 0.0, "conf_count": 0}

    for t in tickets:
        d = t.created_at.strftime("%Y-%m-%d") if t.created_at else None
        if d is None or d not in daily:
            continue
        daily[d]["total"] += 1
        s = (t.status or "").lower()
        if s in ("resolved", "approved"):
            daily[d]["resolved"] += 1
        elif s == "escalated":
            daily[d]["escalated"] += 1
        else:
            daily[d]["pending"] += 1
        if t.confidence is not None:
            daily[d]["conf_sum"] += float(t.confidence)
            daily[d]["conf_count"] += 1

    points = [
        TimeSeriesPoint(
            date=d,
            resolved=v["resolved"],
            escalated=v["escalated"],
            pending=v["pending"],
            total=v["total"],
            avg_confidence=(v["conf_sum"] / v["conf_count"]) if v["conf_count"] else None,
        )
        for d, v in sorted(daily.items())
    ]

    return TimeSeriesResponse(points=points, period_days=days)


# ---------------------------------------------------------------------------
# GET /dashboard/confidence-histogram — Bucketed confidence distribution
# ---------------------------------------------------------------------------


class ConfidenceBucket(BaseModel):
    range_label: str
    count: int
    min_val: float
    max_val: float


class ConfidenceHistogramResponse(BaseModel):
    buckets: list[ConfidenceBucket]
    total_with_confidence: int


@router.get(
    "/dashboard/confidence-histogram",
    response_model=ConfidenceHistogramResponse,
    summary="Get confidence score histogram",
    description="Returns tickets bucketed by confidence score (0-0.2, 0.2-0.4, etc.).",
)
async def get_confidence_histogram(
    days: int | None = Query(default=None, ge=1, le=3650, description="Look-back window in days."),
    db: AsyncSession = Depends(get_db_session),
):
    from datetime import UTC, datetime, timedelta

    q = select(TicketModel.confidence).where(TicketModel.confidence.isnot(None))
    if days:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        q = q.where(TicketModel.created_at >= cutoff)
    result = await db.execute(q)
    scores = [row[0] for row in result.all()]

    buckets = []
    edges = [i / 5 for i in range(6)]  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f"{lo:.1f}–{hi:.1f}"
        count = sum(1 for s in scores if lo <= s < hi) if i < len(edges) - 2 else sum(1 for s in scores if lo <= s <= hi)
        buckets.append(ConfidenceBucket(range_label=label, count=count, min_val=lo, max_val=hi))

    return ConfidenceHistogramResponse(
        buckets=buckets,
        total_with_confidence=len(scores),
    )


# ---------------------------------------------------------------------------
# GET /dashboard/escalation-reasons — Top escalation reasons
# ---------------------------------------------------------------------------


class EscalationReasonItem(BaseModel):
    reason: str
    count: int


class EscalationReasonsResponse(BaseModel):
    reasons: list[EscalationReasonItem]


@router.get(
    "/dashboard/escalation-reasons",
    response_model=EscalationReasonsResponse,
    summary="Get top escalation reasons",
    description="Returns the top 10 escalation reasons by frequency.",
)
async def get_escalation_reasons(
    days: int | None = Query(default=None, ge=1, le=3650, description="Look-back window in days."),
    db: AsyncSession = Depends(get_db_session),
):
    from datetime import UTC, datetime, timedelta

    q = (
        select(TicketModel.escalation_reason, func.count(TicketModel.id))
        .where(TicketModel.escalation_reason.isnot(None))
        .where(TicketModel.escalation_reason != "")
        .group_by(TicketModel.escalation_reason)
        .order_by(func.count(TicketModel.id).desc())
        .limit(10)
    )
    if days:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        q = q.where(TicketModel.created_at >= cutoff)
    result = await db.execute(q)
    reasons = [EscalationReasonItem(reason=row[0], count=row[1]) for row in result.all()]
    return EscalationReasonsResponse(reasons=reasons)


