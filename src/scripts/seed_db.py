"""Seed the database with sample tickets for development and evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
from sqlalchemy import select

from src.config import get_settings
from src.models import Base, Ticket, TicketCategory, TicketStatus, TicketUrgency, engine
from src.models.database import async_session_factory

logger = structlog.get_logger(__name__)
settings = get_settings()

SAMPLE_TICKETS = [
    {
        "subject": "App crashes on login after latest update",
        "body": "After updating to v2.3.1 the mobile app crashes every time I try to log in. "
                "I've tried reinstalling but the issue persists. This is blocking our entire team.",
        "expected_category": "bug",
        "expected_urgency": "P1",
        "expected_escalate": False,
    },
    {
        "subject": "Feature request: Dark mode support",
        "body": "It would be great if the dashboard supported a dark mode theme. "
                "Many of our users work late hours and the bright theme is straining.",
        "expected_category": "feature_request",
        "expected_urgency": "P3",
        "expected_escalate": False,
    },
    {
        "subject": "Invoice discrepancy on premium plan",
        "body": "Our latest invoice shows $299/mo but we signed up for the $199/mo plan. "
                "Can you please look into this? Our billing cycle ends Friday.",
        "expected_category": "billing",
        "expected_urgency": "P2",
        "expected_escalate": False,
    },
    {
        "subject": "Cannot access account after password reset",
        "body": "I reset my password via the email link but the new password isn't accepted. "
                "I'm now locked out and need access for a client demo tomorrow.",
        "expected_category": "account",
        "expected_urgency": "P1",
        "expected_escalate": False,
    },
    {
        "subject": "Data export not including custom fields",
        "body": "When I export tickets to CSV, the custom fields we added are missing "
                "from the export. Is this a known issue?",
        "expected_category": "bug",
        "expected_urgency": "P2",
        "expected_escalate": False,
    },
    {
        "subject": "Integration with Jira suddenly stopped working",
        "body": "Our Jira integration was working fine until yesterday. Now tickets are not "
                "syncing. This affects our entire engineering workflow. No changes on our end.",
        "expected_category": "bug",
        "expected_urgency": "P0",
        "expected_escalate": True,
    },
    {
        "subject": "How to set up SSO with Okta?",
        "body": "We'd like to enable SSO using Okta for our organization. "
                "Could you share the setup documentation or walk us through it?",
        "expected_category": "general",
        "expected_urgency": "P3",
        "expected_escalate": False,
    },
    {
        "subject": "SECURITY: Potential data leak in API response",
        "body": "We noticed that the /api/v2/users endpoint is returning data from other "
                "organizations in the response. This is a critical security issue that "
                "needs immediate attention. We have screenshots.",
        "expected_category": "bug",
        "expected_urgency": "P0",
        "expected_escalate": True,
    },
    {
        "subject": "Request to increase API rate limit",
        "body": "Our automation scripts are hitting the 1000 req/min limit. "
                "Could we get an increase to 5000? Happy to provide justification.",
        "expected_category": "feature_request",
        "expected_urgency": "P3",
        "expected_escalate": False,
    },
    {
        "subject": "Billing portal shows wrong subscription tier",
        "body": "The billing portal shows we're on the free tier, but we're paying for "
                "enterprise. We need this fixed before our audit next week.",
        "expected_category": "billing",
        "expected_urgency": "P1",
        "expected_escalate": False,
    },
]


async def seed() -> None:
    """Insert sample tickets into the database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as session:
        # Check if already seeded
        existing = (await session.execute(select(Ticket))).scalars().all()
        if existing:
            logger.info("seed.already_seeded", count=len(existing))
            return

        for t in SAMPLE_TICKETS:
            ticket = Ticket(
                subject=t["subject"],
                body=t["body"],
                category=TicketCategory(t["expected_category"]),
                urgency=TicketUrgency(t["expected_urgency"]),
                status=TicketStatus.OPEN,
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
            "subject": t["subject"],
            "body": t["body"],
            "expected_category": t["expected_category"],
            "expected_urgency": t["expected_urgency"],
            "expected_escalate": t["expected_escalate"],
        }
        for i, t in enumerate(SAMPLE_TICKETS)
    ]
    eval_path.write_text(json.dumps(eval_tickets, indent=2))
    logger.info("seed.eval_fixture_written", path=str(eval_path))


if __name__ == "__main__":
    asyncio.run(seed())
