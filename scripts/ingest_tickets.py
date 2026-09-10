"""Data ingestion script — load tickets from GitHub Issues or CSV, preprocess,
generate embeddings, and bulk-index into Qdrant + PostgreSQL.

Idempotent by default (skips already-indexed tickets).  Use --force to re-index.

Usage::

    # From CSV file
    python -m scripts.ingest_tickets --csv data/tickets.csv

    # From GitHub Issues
    python -m scripts.ingest_tickets --github-owner myorg --github-repo myrepo

    # Subset + force re-index
    python -m scripts.ingest_tickets --csv data/tickets.csv --limit 50 --force

    # Validate only (no ingestion)
    python -m scripts.ingest_tickets --csv data/tickets.csv --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy import select, func

from src.config import get_settings
from src.models import Base, get_engine
from src.models.database import get_session_factory
from src.models.schemas import Ticket, TicketMetadata, TicketStatus
from src.models.ticket import TicketModel
from src.services.embedding import get_embedding_provider

logger = structlog.get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Text Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════


def clean_markdown(text: str) -> str:
    """Strip common Markdown artefacts from ticket content.

    Handles: headers, bold/italic, links, images, code blocks, list markers,
    blockquotes, and horizontal rules.
    """
    if not text:
        return ""

    # Remove code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove images ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove headers (# ...)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def extract_metadata_from_content(text: str) -> dict[str, Any]:
    """Heuristically extract metadata from ticket content.

    Looks for: subject lines, email addresses, version numbers, URLs,
    and priority keywords.
    """
    metadata: dict[str, Any] = {}

    # Subject line
    subject_match = re.search(r"(?i)^subject:\s*(.+)$", text, re.MULTILINE)
    if subject_match:
        metadata["subject"] = subject_match.group(1).strip()

    # Email addresses
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if emails:
        metadata["emails_found"] = list(set(emails))

    # Version numbers (e.g. v2.3.1, 1.0.0-beta)
    versions = re.findall(r"(?i)\bv?(\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?)\b", text)
    if versions:
        metadata["versions_mentioned"] = list(set(versions))

    # URLs
    urls = re.findall(r"https?://[^\s<>\"']+", text)
    if urls:
        metadata["urls"] = list(set(urls))

    # Priority / urgency keywords
    urgent_patterns = [
        r"(?i)\b(critical|urgent|asap|immediately|emergency)\b",
        r"(?i)\b(blocking|down|outage|incident)\b",
    ]
    for pattern in urgent_patterns:
        matches = re.findall(pattern, text)
        if matches:
            metadata.setdefault("urgency_keywords", []).extend(matches)
    if "urgency_keywords" in metadata:
        metadata["urgency_keywords"] = list(set(metadata["urgency_keywords"]))

    return metadata


def preprocess_ticket(raw: dict[str, str]) -> dict[str, Any]:
    """Clean and enrich a raw ticket dict from CSV or GitHub.

    Returns a dict with: id, content, cleaned_content, source, source_id,
    metadata, created_at.
    """
    content = (raw.get("content") or raw.get("body") or "").strip()
    cleaned = clean_markdown(content)
    extracted_meta = extract_metadata_from_content(content)

    # Merge any explicit metadata fields
    source = (raw.get("source") or "csv").strip().lower()
    source_id = raw.get("source_id") or raw.get("id") or raw.get("number")

    # Parse created_at if present
    created_at = datetime.now(timezone.utc)
    if raw.get("created_at"):
        try:
            created_at = datetime.fromisoformat(
                raw["created_at"].replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            pass

    # Merge custom metadata
    custom_fields = {}
    for key in ("labels", "assignee", "milestone", "state"):
        if raw.get(key):
            custom_fields[key] = raw[key]

    return {
        "id": str(uuid.uuid4()),
        "content": cleaned if cleaned else content,
        "raw_content": content,
        "source": source,
        "source_id": str(source_id) if source_id else None,
        "metadata": {
            "tags": raw.get("labels", []) if isinstance(raw.get("labels"), list) else [],
            "customer_email": raw.get("customer_email"),
            "custom_fields": {**extracted_meta, **custom_fields},
        },
        "created_at": created_at.isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


def validate_ticket(raw: dict[str, str]) -> tuple[bool, str]:
    """Validate a raw ticket dict before ingestion.

    Returns (is_valid, error_message).
    """
    content = (raw.get("content") or raw.get("body") or "").strip()
    if not content:
        return False, "Empty content"

    if len(content) > 50_000:
        return False, f"Content too long ({len(content)} chars > 50000 max)"

    source_id = raw.get("source_id") or raw.get("id") or raw.get("number")
    if source_id is not None and len(str(source_id)) > 500:
        return False, f"source_id too long ({len(str(source_id))} chars > 500 max)"

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Data Source Loaders
# ═══════════════════════════════════════════════════════════════════════════════


def load_from_csv(csv_path: str, limit: int | None = None) -> list[dict[str, str]]:
    """Load raw tickets from a CSV file.

    Expected columns: content (or body), source, source_id, created_at.
    Extra columns are passed through as metadata.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    tickets: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            tickets.append(dict(row))

    logger.info("ingest.csv_loaded", path=csv_path, count=len(tickets))
    return tickets


def load_from_github_issues(
    owner: str,
    repo: str,
    token: str | None = None,
    state: str = "all",
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Fetch tickets from the GitHub Issues API.

    Parameters
    ----------
    owner : str
        GitHub repository owner.
    repo : str
        GitHub repository name.
    token : str, optional
        GitHub personal access token.  Falls back to GITHUB_TOKEN env var.
    state : str
        Issue state filter: 'open', 'closed', or 'all'.
    limit : int, optional
        Maximum number of issues to fetch.
    """
    import httpx

    token = token or ""
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params: dict[str, Any] = {"state": state, "per_page": 100, "page": 1}
    tickets: list[dict[str, str]] = []

    with httpx.Client(timeout=30.0) as client:
        while True:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            issues = response.json()

            if not issues:
                break

            for issue in issues:
                # Skip pull requests (they also appear in Issues API)
                if "pull_request" in issue:
                    continue

                tickets.append({
                    "content": f"{issue.get('title', '')}\n\n{issue.get('body', '')}",
                    "source": "github",
                    "source_id": str(issue["number"]),
                    "created_at": issue.get("created_at", ""),
                    "labels": [label["name"] for label in issue.get("labels", [])],
                    "state": issue.get("state", ""),
                    "assignee": (
                        issue["assignee"]["login"] if issue.get("assignee") else None
                    ),
                    "milestone": (
                        issue["milestone"]["title"] if issue.get("milestone") else None
                    ),
                })

                if limit and len(tickets) >= limit:
                    break

            if limit and len(tickets) >= limit:
                break

            # Check for next page
            if "next" not in response.headers.get("link", ""):
                break
            params["page"] += 1

    tickets = tickets[:limit] if limit else tickets
    logger.info(
        "ingest.github_loaded",
        owner=owner,
        repo=repo,
        count=len(tickets),
    )
    return tickets


# ═══════════════════════════════════════════════════════════════════════════════
# Ingestion Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def check_already_indexed(
    source_ids: list[str | None],
    force: bool,
) -> set[str]:
    """Return the set of source_ids already in PostgreSQL (skipped unless --force)."""
    if force:
        return set()

    valid_ids = [sid for sid in source_ids if sid]
    if not valid_ids:
        return set()

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(TicketModel.source_id).where(
                TicketModel.source_id.in_(valid_ids)
            )
        )
        return {row[0] for row in result.all()}


async def store_in_postgres(tickets: list[dict[str, Any]], force: bool) -> int:
    """Insert preprocessed tickets into PostgreSQL.

    Returns the number of newly inserted tickets.
    """
    session_factory = get_session_factory()
    inserted = 0

    async with session_factory() as session:
        for t in tickets:
            try:
                # Idempotency: skip if source_id already exists
                if not force and t.get("source_id"):
                    exists_q = await session.execute(
                        select(TicketModel).where(
                            TicketModel.source_id == t["source_id"]
                        )
                    )
                    if exists_q.scalar_one_or_none() is not None:
                        continue

                model = TicketModel(
                    id=t["id"],
                    content=t["content"],
                    source=t["source"],
                    source_id=t.get("source_id"),
                    metadata_=json.dumps(t.get("metadata", {})),
                    status=TicketStatus.OPEN.value,
                    created_at=datetime.fromisoformat(t["created_at"]),
                )
                session.add(model)
                inserted += 1
            except Exception as exc:
                logger.error(
                    "ingest.postgres_insert_failed",
                    source_id=t.get("source_id"),
                    error=str(exc),
                )

        await session.commit()

    logger.info("ingest.postgres_done", inserted=inserted)
    return inserted


async def index_into_qdrant(
    tickets: list[dict[str, Any]],
    embeddings: list[list[float]],
    force: bool,
) -> dict[str, int]:
    """Upsert ticket embeddings into Qdrant with metadata payloads.

    Returns {"indexed": N, "failed": M}.
    """
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=settings.qdrant_timeout,
    )

    collection = settings.qdrant_collection

    # Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        from qdrant_client.models import Distance, VectorParams

        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=settings.embedding_vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("ingest.qdrant_collection_created", collection=collection)

    indexed = 0
    failed = 0
    batch_size = 64

    for batch_start in range(0, len(tickets), batch_size):
        batch_end = min(batch_start + batch_size, len(tickets))
        points: list[PointStruct] = []

        for i in range(batch_start, batch_end):
            t = tickets[i]
            emb = embeddings[i]

            payload = {
                "ticket_id": t["id"],
                "source": t["source"],
                "content": t["content"],
                "created_at": t["created_at"],
                **t.get("metadata", {}),
            }

            # Add category/urgency if available (from expected or classification)
            if t.get("expected_category"):
                payload["category"] = t["expected_category"]
            if t.get("expected_urgency"):
                payload["urgency"] = t["expected_urgency"]

            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, t.get("source_id") or t["id"])),
                    vector=emb,
                    payload=payload,
                )
            )

        try:
            client.upsert(collection_name=collection, points=points)
            indexed += len(points)
        except Exception as exc:
            failed += len(points)
            logger.error(
                "ingest.qdrant_batch_failed",
                batch_start=batch_start,
                error=str(exc),
            )

    logger.info("ingest.qdrant_done", indexed=indexed, failed=failed)
    return {"indexed": indexed, "failed": failed}


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def ingest(
    tickets_raw: list[dict[str, str]],
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full ingestion pipeline: validate → preprocess → embed → index.

    Returns a summary dict with counts and timing.
    """
    start = datetime.now(timezone.utc)
    logger.info(
        "ingest.start",
        total_raw=len(tickets_raw),
        force=force,
        dry_run=dry_run,
    )

    # ── Step 1: Validate ───────────────────────────────────────────────────
    valid_tickets: list[dict[str, str]] = []
    validation_errors: list[dict[str, str]] = []

    for i, raw in enumerate(tickets_raw):
        is_valid, error = validate_ticket(raw)
        if is_valid:
            valid_tickets.append(raw)
        else:
            validation_errors.append({
                "index": str(i),
                "source_id": raw.get("source_id", raw.get("id", "unknown")),
                "error": error,
            })

    if validation_errors:
        logger.warning(
            "ingest.validation_errors",
            count=len(validation_errors),
            errors=validation_errors[:5],  # Log first 5
        )

    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    processed = [preprocess_ticket(t) for t in valid_tickets]
    logger.info("ingest.preprocessed", count=len(processed))

    # ── Step 3: Check for already-indexed ──────────────────────────────────
    source_ids = [t.get("source_id") for t in processed]
    already_indexed = await check_already_indexed(source_ids, force)

    if already_indexed and not force:
        processed = [
            t for t in processed if t.get("source_id") not in already_indexed
        ]
        logger.info(
            "ingest.skipped_existing",
            count=len(already_indexed),
            remaining=len(processed),
        )

    if not processed:
        logger.info("ingest.nothing_to_do")
        return {
            "total_raw": len(tickets_raw),
            "valid": len(valid_tickets),
            "indexed": 0,
            "skipped": len(already_indexed),
            "failed": len(validation_errors),
        }

    # ── Step 4: Generate embeddings ────────────────────────────────────────
    provider = get_embedding_provider(use_cache=True)
    texts_to_embed = [t["content"] for t in processed]

    logger.info("ingest.embeddings_start", count=len(texts_to_embed))
    embeddings = await provider.embed(texts_to_embed)
    logger.info("ingest.embeddings_done", count=len(embeddings))

    # ── Step 5: Store in PostgreSQL ────────────────────────────────────────
    if not dry_run:
        pg_inserted = await store_in_postgres(processed, force)
    else:
        pg_inserted = len(processed)
        logger.info("ingest.dry_run_skip_postgres", count=pg_inserted)

    # ── Step 6: Index into Qdrant ──────────────────────────────────────────
    if not dry_run:
        qdrant_result = await index_into_qdrant(processed, embeddings, force)
    else:
        qdrant_result = {"indexed": len(processed), "failed": 0}
        logger.info("ingest.dry_run_skip_qdrant", count=qdrant_result["indexed"])

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    summary = {
        "total_raw": len(tickets_raw),
        "valid": len(valid_tickets),
        "preprocessed": len(processed),
        "postgres_inserted": pg_inserted,
        "qdrant_indexed": qdrant_result["indexed"],
        "qdrant_failed": qdrant_result["failed"],
        "skipped_existing": len(already_indexed),
        "validation_errors": len(validation_errors),
        "elapsed_seconds": round(elapsed, 2),
    }

    logger.info("ingest.complete", **summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest tickets from GitHub Issues or CSV into Qdrant + PostgreSQL.",
    )

    # Source options (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with ticket data.",
    )
    source.add_argument(
        "--github-owner",
        type=str,
        help="GitHub repository owner (use with --github-repo).",
    )

    # GitHub options
    parser.add_argument(
        "--github-repo",
        type=str,
        help="GitHub repository name (required with --github-owner).",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub token. Falls back to GITHUB_TOKEN env var.",
    )
    parser.add_argument(
        "--github-state",
        choices=["open", "closed", "all"],
        default="all",
        help="Issue state filter (default: all).",
    )

    # General options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tickets to ingest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing (delete existing and re-insert).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preprocess only — skip DB and Qdrant writes.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load tickets from the chosen source
    if args.csv:
        tickets_raw = load_from_csv(args.csv, limit=args.limit)
    elif args.github_owner:
        if not args.github_repo:
            logger.error("ingest.github_repo_required")
            sys.exit(1)
        token = args.github_token or ""
        tickets_raw = load_from_github_issues(
            owner=args.github_owner,
            repo=args.github_repo,
            token=token,
            state=args.github_state,
            limit=args.limit,
        )
    else:
        logger.error("ingest.no_source")
        sys.exit(1)

    if not tickets_raw:
        logger.warning("ingest.no_tickets_found")
        sys.exit(0)

    # Run the ingestion pipeline
    result = asyncio.run(
        ingest(tickets_raw, force=args.force, dry_run=args.dry_run)
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  INGESTION SUMMARY")
    print("=" * 60)
    for key, value in result.items():
        print(f"  {key:.<30} {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
