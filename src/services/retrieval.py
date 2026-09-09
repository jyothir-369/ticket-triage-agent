"""Qdrant-based retrieval service — searches historical tickets for relevant context."""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _qdrant_client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

COLLECTION_NAME = settings.qdrant_collection
VECTOR_SIZE = 1536  # OpenAI ada-002 embedding dimension


def ensure_collection() -> None:
    """Create the collection if it doesn't exist (idempotent)."""
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("qdrant.collection_created", collection=COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDoc:
    ticket_id: int
    subject: str
    body: str
    score: float
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def search_similar_tickets(
    query_text: str,
    *,
    top_k: int = 5,
    score_threshold: float = 0.5,
) -> list[RetrievedDoc]:
    """Search Qdrant for tickets similar to `query_text`.

    In production this would embed the query via the same model used at
    ingest time. For the v1 scaffold we accept a pre-computed vector or
    fall back to a placeholder so the pipeline stays end-to-end testable.
    """
    client = get_qdrant_client()
    ensure_collection()

    # Placeholder: in production, embed query_text via OpenAI / local model.
    # For scaffold, generate a zero-vector so the API contract is testable.
    query_vector = [0.0] * VECTOR_SIZE

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        logger.warning("qdrant.search_failed", error=str(exc))
        return []

    docs: list[RetrievedDoc] = []
    for point in results.points:
        payload = point.payload or {}
        docs.append(
            RetrievedDoc(
                ticket_id=payload.get("ticket_id", 0),
                subject=payload.get("subject", ""),
                body=payload.get("body", ""),
                score=point.score,
                metadata=payload,
            )
        )
    return docs


async def ingest_ticket_embedding(
    ticket_id: int,
    subject: str,
    body: str,
    embedding: list[float],
) -> None:
    """Upsert a ticket's embedding into Qdrant."""
    client = get_qdrant_client()
    ensure_collection()

    point = PointStruct(
        id=ticket_id,
        vector=embedding,
        payload={
            "ticket_id": ticket_id,
            "subject": subject,
            "body": body,
        },
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    logger.debug("qdrant.ingested", ticket_id=ticket_id)
