"""Dashboard and health-check endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import get_settings
from src.models.database import get_db_session
from src.models.schemas import DashboardMetrics, RecentActivity
from src.models.ticket import TicketModel, TicketStatus as DBTicketStatus
from src.models.trace import TraceModel

logger = structlog.get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["dashboard"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    db: str
    qdrant: str
    redis: str


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
        "average confidence, average steps, and success rate."
    ),
)
async def get_metrics(
    db: AsyncSession = Depends(get_db_session),
):
    # Counts
    total_q = await db.execute(select(func.count(TicketModel.id)))
    total_tickets = total_q.scalar() or 0

    processed_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            TicketModel.status.in_([
                DBTicketStatus.RESOLVED.value,
                DBTicketStatus.ESCALATED.value,
                DBTicketStatus.APPROVED.value,
                DBTicketStatus.REJECTED.value,
                DBTicketStatus.AWAITING_REVIEW.value,
            ])
        )
    )
    processed_tickets = processed_q.scalar() or 0

    escalated_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            TicketModel.status == DBTicketStatus.ESCALATED.value
        )
    )
    escalated_tickets = escalated_q.scalar() or 0

    # Approval rate
    approved_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            TicketModel.review_decision == "approved"
        )
    )
    approved_count = approved_q.scalar() or 0
    approval_rate = approved_count / escalated_tickets if escalated_tickets > 0 else 0.0

    # Confidence average
    avg_conf_q = await db.execute(
        select(func.avg(TicketModel.confidence)).where(
            TicketModel.confidence.isnot(None)
        )
    )
    avg_confidence = avg_conf_q.scalar() or 0.0

    # Category distribution
    cat_q = await db.execute(
        select(TicketModel.category, func.count(TicketModel.id))
        .where(TicketModel.category.isnot(None))
        .group_by(TicketModel.category)
    )
    category_distribution = {row[0]: row[1] for row in cat_q.all()}

    # Urgency distribution
    urg_q = await db.execute(
        select(TicketModel.urgency, func.count(TicketModel.id))
        .where(TicketModel.urgency.isnot(None))
        .group_by(TicketModel.urgency)
    )
    urgency_distribution = {row[0]: row[1] for row in urg_q.all()}

    # Recent activity (last 10)
    recent_q = await db.execute(
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
    resolved_q = await db.execute(
        select(func.count(TicketModel.id)).where(
            TicketModel.status == DBTicketStatus.RESOLVED.value
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
# GET /health — Health check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verifies connectivity to DB, Qdrant, and Redis.",
)
async def health_check():
    db_status = "ok"
    qdrant_status = "ok"
    redis_status = "ok"

    # Check DB connectivity
    try:
        from src.models.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("health.db_failed", error=str(exc))
        db_status = f"error: {type(exc).__name__}"

    # Check Qdrant connectivity
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz"
            )
            if resp.status_code != 200:
                qdrant_status = f"error: status {resp.status_code}"
    except ImportError:
        qdrant_status = "error: httpx not installed"
    except Exception as exc:
        logger.error("health.qdrant_failed", error=str(exc))
        qdrant_status = f"error: {type(exc).__name__}"

    # Check Redis connectivity
    try:
        import aioredis

        redis = aioredis.from_url(settings.redis_url, socket_timeout=5)
        await redis.ping()
        await redis.close()
    except ImportError:
        redis_status = "error: aioredis not installed"
    except Exception as exc:
        logger.error("health.redis_failed", error=str(exc))
        redis_status = f"error: {type(exc).__name__}"

    overall = "ok" if all(s == "ok" for s in [db_status, qdrant_status, redis_status]) else "degraded"

    return HealthResponse(
        status=overall,
        db=db_status,
        qdrant=qdrant_status,
        redis=redis_status,
    )
