"""Backfill existing ticket records into Qdrant.

Run once after deploying Qdrant indexing to populate the vector store
with already-resolved tickets. Idempotent — uses upsert semantics so
re-running is safe.

Usage:
    python -m src.scripts.backfill_qdrant
    python -m src.scripts.backfill_qdrant --limit 500
    python -m src.scripts.backfill_qdrant --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import structlog

logger = structlog.get_logger(__name__)


async def run(limit: int | None, dry_run: bool) -> dict[str, int]:
    """Backfill resolved and awaiting-review tickets into Qdrant.

    Returns a dict with ``indexed``, ``skipped``, and ``failed`` counts.
    """
    from sqlalchemy import select

    from src.models.database import get_session
    from src.models.ticket import TicketModel
    from src.services.retrieval import get_retriever

    retriever = get_retriever()

    indexed = 0
    skipped = 0
    failed = 0

    async with get_session() as session:
        query = select(TicketModel).where(
            TicketModel.status.in_(["resolved", "awaiting_review", "escalated"])
        )
        if limit is not None:
            query = query.limit(limit)
        result = await session.execute(query)
        tickets = result.scalars().all()

    total = len(tickets)
    logger.info("backfill.start", total=total, dry_run=dry_run, limit=limit)

    for ticket in tickets:
        if not ticket.content:
            skipped += 1
            continue

        metadata = {
            "ticket_id": ticket.id,
            "category": ticket.category or "other",
            "urgency": ticket.urgency or "medium",
            "source": ticket.source or "api",
            "status": ticket.status,
        }

        if dry_run:
            logger.info(
                "backfill.dry_run_would_index",
                ticket_id=ticket.id,
                category=metadata["category"],
                urgency=metadata["urgency"],
            )
            indexed += 1
            continue

        try:
            ok = await retriever.add_document(
                doc_id=ticket.id,
                content=ticket.content,
                metadata=metadata,
            )
            if ok:
                indexed += 1
                logger.debug("backfill.indexed", ticket_id=ticket.id)
            else:
                failed += 1
                logger.warning("backfill.index_returned_false", ticket_id=ticket.id)
        except Exception as exc:
            failed += 1
            logger.error("backfill.index_error", ticket_id=ticket.id, error=str(exc))

    await retriever.close()

    summary = {"indexed": indexed, "skipped": skipped, "failed": failed, "total": total}
    logger.info("backfill.complete", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill tickets into Qdrant")
    parser.add_argument("--limit", type=int, default=None, help="Max tickets to process")
    parser.add_argument("--dry-run", action="store_true", help="Log without indexing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    summary = asyncio.run(run(limit=args.limit, dry_run=args.dry_run))

    if summary["failed"] > 0:
        print(
            f"Backfill finished with {summary['failed']} failures. "
            f"Indexed: {summary['indexed']}, Skipped: {summary['skipped']}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Backfill complete. Indexed: {summary['indexed']}, "
        f"Skipped: {summary['skipped']}, Total: {summary['total']}"
    )


if __name__ == "__main__":
    main()
