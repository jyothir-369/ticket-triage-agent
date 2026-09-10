"""Seed database tables, Qdrant collection, Redis, and insert default test tickets.

Idempotent — safe to run multiple times.  Use --force to drop and recreate.

Usage::

    python -m scripts.seed_db
    python -m scripts.seed_db --force      # drop & recreate everything
    python -m scripts.seed_db --no-qdrant  # skip Qdrant collection creation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
from sqlalchemy import text

from src.config import get_settings
from src.models import Base, DBTicketCategory, DBTicketStatus, DBTicketUrgency, get_engine
from src.models.database import get_session_factory
from src.models.ticket import TicketModel

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Sample tickets ─────────────────────────────────────────────────────────────

SAMPLE_TICKETS: list[dict[str, str]] = [
    {
        "content": (
            "After updating to v2.3.1 the mobile app crashes every time I try to log in. "
            "I've tried reinstalling but the issue persists. This is blocking our entire team."
        ),
        "source": "email",
        "source_id": "EMAIL-001",
        "expected_category": "bug",
        "expected_urgency": "high",
    },
    {
        "content": (
            "It would be great if the dashboard supported a dark mode theme. "
            "Many of our users work late hours and the bright theme is straining."
        ),
        "source": "github",
        "source_id": "GH-002",
        "expected_category": "feature_request",
        "expected_urgency": "low",
    },
    {
        "content": (
            "Our latest invoice shows $299/mo but we signed up for the $199/mo plan. "
            "Can you please look into this? Our billing cycle ends Friday."
        ),
        "source": "email",
        "source_id": "EMAIL-003",
        "expected_category": "billing",
        "expected_urgency": "medium",
    },
    {
        "content": (
            "I reset my password via the email link but the new password isn't accepted. "
            "I'm now locked out and need access for a client demo tomorrow."
        ),
        "source": "intercom",
        "source_id": "IC-004",
        "expected_category": "account_issue",
        "expected_urgency": "high",
    },
    {
        "content": (
            "When I export tickets to CSV, the custom fields we added are missing "
            "from the export. Is this a known issue?"
        ),
        "source": "api",
        "source_id": "API-005",
        "expected_category": "bug",
        "expected_urgency": "medium",
    },
    {
        "content": (
            "Our Jira integration was working fine until yesterday. Now tickets are not "
            "syncing. This affects our entire engineering workflow. No changes on our end."
        ),
        "source": "email",
        "source_id": "EMAIL-006",
        "expected_category": "bug",
        "expected_urgency": "critical",
    },
    {
        "content": (
            "We'd like to enable SSO using Okta for our organization. "
            "Could you share the setup documentation or walk us through it?"
        ),
        "source": "intercom",
        "source_id": "IC-007",
        "expected_category": "usage_help",
        "expected_urgency": "low",
    },
    {
        "content": (
            "We noticed that the /api/v2/users endpoint is returning data from other "
            "organizations in the response. This is a critical security issue that "
            "needs immediate attention. We have screenshots."
        ),
        "source": "email",
        "source_id": "EMAIL-008",
        "expected_category": "bug",
        "expected_urgency": "critical",
    },
    {
        "content": (
            "Our automation scripts are hitting the 1000 req/min limit. "
            "Could we get an increase to 5000? Happy to provide justification."
        ),
        "source": "api",
        "source_id": "API-009",
        "expected_category": "feature_request",
        "expected_urgency": "low",
    },
    {
        "content": (
            "The billing portal shows we're on the free tier, but we're paying for "
            "enterprise. We need this fixed before our audit next week."
        ),
        "source": "email",
        "source_id": "EMAIL-010",
        "expected_category": "billing",
        "expected_urgency": "high",
    },
]


# ── PostgreSQL ─────────────────────────────────────────────────────────────────


async def ensure_tables(force: bool = False) -> None:
    """Create all ORM tables (idempotent).  Drops first when *force* is True."""
    engine = get_engine()
    async with engine.begin() as conn:
        if force:
            logger.warning("seed.drop_all_tables")
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    logger.info("seed.tables_created")


# ── Qdrant ─────────────────────────────────────────────────────────────────────


def ensure_qdrant_collection(force: bool = False) -> None:
    """Create the Qdrant collection with correct vector size (idempotent)."""
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=settings.qdrant_timeout,
    )

    existing = [c.name for c in client.get_collections().collections]
    collection_name = settings.qdrant_collection

    if collection_name in existing:
        if force:
            logger.warning("seed.qdrant_drop_collection", collection=collection_name)
            client.delete_collection(collection_name)
        else:
            logger.info("seed.qdrant_collection_exists", collection=collection_name)
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_vector_size,
            distance=Distance.COSINE,
        ),
    )

    # Create payload indexes for metadata filtering
    for field_name in ("ticket_id", "source", "category", "urgency", "status"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    logger.info(
        "seed.qdrant_created",
        collection=collection_name,
        vector_size=settings.embedding_vector_size,
    )


# ── Redis ──────────────────────────────────────────────────────────────────────


async def verify_redis() -> None:
    """Verify Redis is reachable and log the result."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        await client.close()
        logger.info("seed.redis_ok", url=settings.redis_url)
    except Exception as exc:
        logger.warning("seed.redis_unavailable", error=str(exc))


# ── Sample tickets ─────────────────────────────────────────────────────────────


async def insert_sample_tickets(force: bool = False) -> int:
    """Insert sample tickets into the DB if the table is empty (or if *force*).

    Returns the number of inserted tickets.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        if force:
            # Delete existing sample tickets
            from sqlalchemy import delete

            await session.execute(
                delete(TicketModel).where(
                    TicketModel.source_id.in_([t["source_id"] for t in SAMPLE_TICKETS])
                )
            )
            await session.flush()

        # Check count
        from sqlalchemy import select, func

        count_q = await session.execute(select(func.count(TicketModel.id)))
        existing_count = count_q.scalar() or 0

        if existing_count > 0 and not force:
            logger.info("seed.already_seeded", count=existing_count)
            return 0

        inserted = 0
        for t in SAMPLE_TICKETS:
            # Skip if already exists (idempotency)
            if not force:
                exists_q = await session.execute(
                    select(TicketModel).where(TicketModel.source_id == t["source_id"])
                )
                if exists_q.scalar_one_or_none() is not None:
                    continue

            ticket = TicketModel(
                id=str(uuid.uuid4()),
                content=t["content"],
                source=t["source"],
                source_id=t["source_id"],
                category=t["expected_category"],
                urgency=t["expected_urgency"],
                status=DBTicketStatus.OPEN.value,
            )
            session.add(ticket)
            inserted += 1

        await session.commit()
        logger.info("seed.tickets_inserted", count=inserted)
        return inserted


# ── Eval fixture ───────────────────────────────────────────────────────────────


def write_eval_fixture() -> None:
    """Write the labeled eval fixture JSON from sample tickets."""
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
            "should_escalate": t["expected_urgency"] in ("high", "critical"),
        }
        for i, t in enumerate(SAMPLE_TICKETS)
    ]
    eval_path.write_text(json.dumps(eval_tickets, indent=2), encoding="utf-8")
    logger.info("seed.eval_fixture_written", path=str(eval_path), count=len(eval_tickets))


# ── Main ───────────────────────────────────────────────────────────────────────


async def main(force: bool = False, skip_qdrant: bool = False) -> None:
    """Run the full seeding pipeline."""
    logger.info("seed.start", force=force, skip_qdrant=skip_qdrant)

    # 1. PostgreSQL tables
    await ensure_tables(force=force)

    # 2. Qdrant collection
    if not skip_qdrant:
        ensure_qdrant_collection(force=force)

    # 3. Redis connection check
    await verify_redis()

    # 4. Sample tickets
    await insert_sample_tickets(force=force)

    # 5. Eval fixture
    write_eval_fixture()

    logger.info("seed.complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed database, Qdrant, Redis, and sample tickets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate all resources (tables, collection, tickets).",
    )
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Skip Qdrant collection creation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(force=args.force, skip_qdrant=args.no_qdrant))
