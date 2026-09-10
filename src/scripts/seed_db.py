"""Seed the database with sample tickets for development and evaluation."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import structlog
from sqlalchemy import select

from src.config import get_settings
from src.models import Base, TicketCategory, TicketStatus, TicketUrgency, get_engine
from src.models.database import get_session_factory
from src.models.ticket import TicketModel

logger = structlog.get_logger(__name__)
settings = get_settings()

SAMPLE_TICKETS = [
    {
        "content": "After updating to v2.3.1 the mobile app crashes every time I try to log in. "
                   "I've tried reinstalling but the issue persists. This is blocking our entire team.",
        "source": "email",
        "source_id": "EMAIL-001",
        "expected_category": "bug",
        "expected_urgency": "high",
    },
    {
        "content": "It would be great if the dashboard supported a dark mode theme. "
                   "Many of our users work late hours and the bright theme is straining.",
        "source": "github",
        "source_id": "GH-002",
        "expected_category": "feature_request",
        "expected_urgency": "low",
    },
    {
        "content": "Our latest invoice shows $299/mo but we signed up for the $199/mo plan. "
                   "Can you please look into this? Our billing cycle ends Friday.",
        "source": "email",
        "source_id": "EMAIL-003",
        "expected_category": "billing",
        "expected_urgency": "medium",
    },
    {
        "content": "I reset my password via the email link but the new password isn't accepted. "
                   "I'm now locked out and need access for a client demo tomorrow.",
        "source": "intercom",
        "source_id": "IC-004",
        "expected_category": "account_issue",
        "expected_urgency": "high",
    },
    {
        "content": "When I export tickets to CSV, the custom fields we added are missing "
                   "from the export. Is this a known issue?",
        "source": "api",
        "source_id": "API-005",
        "expected_category": "bug",
        "expected_urgency": "medium",
    },
    {
        "content": "Our Jira integration was working fine until yesterday. Now tickets are not "
                   "syncing. This affects our entire engineering workflow. No changes on our end.",
        "source": "email",
        "source_id": "EMAIL-006",
        "expected_category": "bug",
        "expected_urgency": "critical",
    },
    {
        "content": "We'd like to enable SSO using Okta for our organization. "
                   "Could you share the setup documentation or walk us through it?",
        "source": "intercom",
        "source_id": "IC-007",
        "expected_category": "usage_help",
        "expected_urgency": "low",
    },
    {
        "content": "We noticed that the /api/v2/users endpoint is returning data from other "
                   "organizations in the response. This is a critical security issue that "
                   "needs immediate attention. We have screenshots.",
        "source": "email",
        "source_id": "EMAIL-008",
        "expected_category": "bug",
        "expected_urgency": "critical",
    },
    {
        "content": "Our automation scripts are hitting the 1000 req/min limit. "
                   "Could we get an increase to 5000? Happy to provide justification.",
        "source": "api",
        "source_id": "API-009",
        "expected_category": "feature_request",
        "expected_urgency": "low",
    },
    {
        "content": "The billing portal shows we're on the free tier, but we're paying for "
                   "enterprise. We need this fixed before our audit next week.",
        "source": "email",
        "source_id": "EMAIL-010",
        "expected_category": "billing",
        "expected_urgency": "high",
    },
]


async def seed() -> None:
    """Insert sample tickets into the database."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Check if already seeded
        existing = (await session.execute(select(TicketModel))).scalars().all()
        if existing:
            logger.info("seed.already_seeded", count=len(existing))
            return

        for t in SAMPLE_TICKETS:
            ticket = TicketModel(
                id=str(uuid.uuid4()),
                content=t["content"],
                source=t["source"],
                source_id=t["source_id"],
                category=t["expected_category"],
                urgency=t["expected_urgency"],
                status=TicketStatus.OPEN.value,
            )
            session.add(ticket)

        await session.commit()
        logger.info("seed.done", count=len(SAMPLE_TICKETS))

    # Also write eval fixture
    eval_path = Path(settings.eval_tickets_path)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_tickets = [
        {
            "id": i + 1,
            "content": t["content"],
            "source": t["source"],
            "source_id": t["source_id"],
            "expected_category": t["expected_category"],
            "expected_urgency": t["expected_urgency"],
        }
        for i, t in enumerate(SAMPLE_TICKETS)
    ]
    eval_path.write_text(json.dumps(eval_tickets, indent=2))
    logger.info("seed.eval_fixture_written", path=str(eval_path))


if __name__ == "__main__":
    asyncio.run(seed())
