"""Qdrant-based retrieval service — TicketRetriever with full Qdrant integration.

Provides a class-based retriever wrapping a Qdrant vector store with:
- Vector search with embedding generation and min-score filtering
- Single and bulk document indexing with progress tracking
- Automatic collection creation with correct vector size and payload indexes
- Connection health checks
- Retry with exponential backoff (max 3 attempts)
- Circuit breaker (5 failures in 60 s)
- Redis-backed query-result caching with configurable TTL
- LangGraph tool registration
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SearchParams,
    VectorParams,
)

from src.config import get_settings
from src.models.schemas import RetrievedDocument
from src.utils.circuit_breaker import CircuitBreakerOpen, get_circuit_breaker

logger = structlog.get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Qdrant Client Factory
# ═══════════════════════════════════════════════════════════════════════════════


def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient configured for either local or Cloud deployment.

    When settings.qdrant_url is set, connects to Qdrant Cloud using URL + API key.
    Otherwise connects to local Qdrant using host/port settings.
    """
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout,
        )
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        timeout=settings.qdrant_timeout,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker — shared from utils/circuit_breaker.py
# 5 failures in 60s → OPEN for 30s
# ═══════════════════════════════════════════════════════════════════════════════

_qdrant_breaker = get_circuit_breaker(
    name="qdrant_service",
    failure_threshold=5,
    timeout=30.0,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Redis Query Cache
# ═══════════════════════════════════════════════════════════════════════════════


class QueryCache:
    """Redis-backed cache for retriever query results.

    Uses SHA-256 hashes of normalised query keys.  Supports async Redis
    with configurable TTL (default 5 minutes for retriever results).
    """

    KEY_PREFIX = "ret:"

    def __init__(self, redis_url: str, ttl_seconds: int = 300) -> None:
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            connect_kwargs: dict[str, Any] = {
                "decode_responses": True,
                "max_connections": settings.redis_max_connections,
                "socket_connect_timeout": settings.redis_socket_timeout,
            }
            # Detect TLS for Upstash (rediss://)
            if self._redis_url.startswith("rediss://"):
                connect_kwargs["ssl_cert_reqs"] = "required"

            self._client = aioredis.from_url(
                self._redis_url,
                **connect_kwargs,
            )
            logger.info("retrieval.cache.connected", redis_url=self._redis_url)
        return self._client

    @staticmethod
    def _make_key(query: str, limit: int, filters: dict | None, min_score: float) -> str:
        """Deterministic cache key from query parameters."""
        payload = json.dumps(
            {"q": query, "l": limit, "f": filters, "s": min_score},
            sort_keys=True,
            default=str,
        )
        h = hashlib.sha256(payload.encode()).hexdigest()
        return f"{QueryCache.KEY_PREFIX}{h}"

    async def get(
        self, query: str, limit: int, filters: dict | None, min_score: float
    ) -> list[dict] | None:
        try:
            client = await self._get_client()
            key = self._make_key(query, limit, filters, min_score)
            raw = await client.get(key)
            if raw is not None:
                logger.debug("retrieval.cache.hit", key=key[:16])
                return json.loads(raw)
        except Exception as exc:
            logger.warning("retrieval.cache.get_failed", error=str(exc))
        return None

    async def set(
        self,
        query: str,
        limit: int,
        filters: dict | None,
        min_score: float,
        results: list[dict],
    ) -> None:
        try:
            client = await self._get_client()
            key = self._make_key(query, limit, filters, min_score)
            await client.set(key, json.dumps(results, default=str), ex=self._ttl)
            logger.debug("retrieval.cache.set", key=key[:16], ttl=self._ttl)
        except Exception as exc:
            logger.warning("retrieval.cache.set_failed", error=str(exc))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ═══════════════════════════════════════════════════════════════════════════════
# TicketRetriever
# ═══════════════════════════════════════════════════════════════════════════════


class TicketRetriever:
    """Qdrant-backed vector retriever for support tickets.

    Features
    --------
    * Vector search with automatic query embedding via the configured provider.
    * ``min_score`` post-filter and optional Qdrant payload filters.
    * Single-document ``add_document`` and batch ``bulk_index`` with progress.
    * Automatic collection creation with correct vector size + payload indexes.
    * Connection health checks.
    * Retry with exponential back-off (max 3 attempts) via tenacity.
    * Circuit breaker (5 failures in 60 s).
    * Redis-backed query-result cache with configurable TTL.
    """

    COLLECTION_NAME: str = settings.qdrant_collection
    VECTOR_SIZE: int = settings.embedding_vector_size
    PAYLOAD_INDEX_FIELDS: list[str] = [
        "ticket_id",
        "source",
        "category",
        "urgency",
        "status",
    ]

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        grpc_port: int | None = None,
        timeout: int | None = None,
        *,
        use_cache: bool = True,
    ) -> None:
        self._host = host or settings.qdrant_host
        self._port = port or settings.qdrant_port
        self._grpc_port = grpc_port or settings.qdrant_grpc_port
        self._timeout = timeout or settings.qdrant_timeout

        self._client: QdrantClient | None = None
        self._circuit_breaker = _qdrant_breaker
        self._cache: QueryCache | None = (
            QueryCache(
                redis_url=settings.redis_url,
                ttl_seconds=settings.retriever_cache_ttl_seconds,
            )
            if use_cache
            else None
        )

        self._collection_ensured: bool = False

    # ── Client lifecycle ────────────────────────────────────────────────────

    @property
    def client(self) -> QdrantClient:
        """Lazy-initialise and return the Qdrant client singleton."""
        if self._client is None:
            self._client = get_qdrant_client()
            logger.info(
                "retrieval.client_created",
                url=settings.qdrant_url or f"{self._host}:{self._port}",
                timeout=self._timeout,
            )
        return self._client

    # ── Health check ────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if Qdrant is reachable and responsive."""
        try:
            collections = self.client.get_collections()
            logger.info(
                "retrieval.health_ok",
                collections=len(collections.collections),
            )
            return True
        except Exception as exc:
            logger.error("retrieval.health_failed", error=str(exc))
            return False

    # ── Collection management ───────────────────────────────────────────────

    def ensure_collection_exists(self) -> None:
        """Create the collection with correct vector size if it doesn't exist.

        Also creates payload indexes on metadata fields used for filtering.
        """
        if self._collection_ensured:
            return

        existing = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE, distance=Distance.COSINE
                ),
            )
            logger.info(
                "retrieval.collection_created",
                collection=self.COLLECTION_NAME,
                vector_size=self.VECTOR_SIZE,
            )

        # Create payload indexes for metadata filtering (idempotent)
        for field_name in self.PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Index already exists — safe to ignore
                pass

        self._collection_ensured = True
        logger.debug("retrieval.collection_ready", collection=self.COLLECTION_NAME)

    # ── Retry-wrapped helpers ───────────────────────────────────────────────

    def _retry_operation(self, operation: str, fn, *args, **kwargs):
        """Execute *fn* with up to 3 retry attempts and exponential back-off.

        Uses multiplier=2, min=1s, max=10s with jitter.
        Also handles HTTP 429 (rate limit) from Qdrant Cloud.
        """
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
        )

        from src.utils.retry import wait_exponential_with_jitter

        # Wrap to detect 429 rate limits
        def _wrapped(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "rate" in exc_str:
                    logger.warning(
                        "retrieval.qdrant_rate_limited",
                        operation=operation,
                        error=str(exc)[:200],
                    )
                    raise ConnectionError(f"Qdrant rate limit (429): {exc}") from exc
                raise

        retrier = retry(
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            stop=stop_after_attempt(3),
            wait=wait_exponential_with_jitter(
                multiplier=2.0,
                min_delay=1.0,
                max_delay=10.0,
                jitter_range=0.3,
            ),
            reraise=True,
        )
        return retrier(_wrapped)(*args, **kwargs)

    # ── Search ──────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        min_score: float = 0.5,
    ) -> list[RetrievedDocument]:
        """Search Qdrant for documents similar to *query*.

        Steps:
        1. Check circuit breaker — return empty list if open.
        2. Check Redis query cache.
        3. Generate query embedding and search Qdrant.
        4. Convert Qdrant points to ``RetrievedDocument`` objects.
        5. Apply ``min_score`` post-filter and sort descending by score.

        Returns an empty list when the circuit breaker is open or on errors.
        Logs degradation events for monitoring.
        """
        # Check circuit breaker
        if self._circuit_breaker.state.value != "closed":
            logger.warning(
                "retrieval.degraded.circuit_breaker_open",
                state=self._circuit_breaker.state.value,
                failure_count=self._circuit_breaker._failure_count,
            )
            return []

        # Check query cache
        if self._cache is not None:
            cached = await self._cache.get(query, limit, filters, min_score)
            if cached is not None:
                return [RetrievedDocument(**d) for d in cached]

        try:
            results = await asyncio.wait_for(
                self._do_search(query, limit, filters, min_score),
                timeout=10.0,
            )
            self._circuit_breaker.record_success()

            # Populate cache
            if self._cache is not None:
                await self._cache.set(
                    query, limit, filters, min_score,
                    [r.model_dump(mode="json") for r in results],
                )

            return results
        except CircuitBreakerOpen:
            logger.warning(
                "retrieval.degraded.circuit_breaker_rejected",
                query_len=len(query),
            )
            return []
        except TimeoutError:
            self._circuit_breaker.record_failure()
            logger.warning(
                "retrieval.degraded.timeout",
                query_len=len(query),
                timeout_seconds=10.0,
            )
            return []
        except Exception as exc:
            self._circuit_breaker.record_failure()
            logger.error(
                "retrieval.degraded.search_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

    async def _do_search(
        self,
        query: str,
        limit: int,
        filters: dict | None,
        min_score: float,
    ) -> list[RetrievedDocument]:
        """Internal search implementation (may raise)."""
        self.ensure_collection_exists()

        # Generate query embedding
        from src.services.embedding import get_embedding_provider

        provider = get_embedding_provider(use_cache=True)
        query_embeddings = await provider.embed([query])
        query_vector = query_embeddings[0]

        # Build optional Qdrant filter
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        # Execute vector search
        search_params = SearchParams(exact=False, hnsw_ef=128)

        results = self._retry_operation(
            "qdrant_search",
            self.client.query_points,
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            search_params=search_params,
            score_threshold=min_score,
        )

        # Convert to RetrievedDocument list
        docs: list[RetrievedDocument] = []
        for point in results.points:
            payload = point.payload or {}
            doc = RetrievedDocument(
                id=str(point.id),
                content=payload.get("content", payload.get("body", "")),
                metadata={k: v for k, v in payload.items() if k not in ("content", "body")},
                similarity_score=round(point.score, 4),
                source=payload.get("source", "past_ticket"),
            )
            docs.append(doc)

        # Sort by score descending (Qdrant usually returns sorted, but be safe)
        docs.sort(key=lambda d: d.similarity_score, reverse=True)

        logger.info(
            "retrieval.search_complete",
            query_len=len(query),
            results=len(docs),
            min_score=min_score,
            limit=limit,
        )
        return docs

    # ── Single document indexing ────────────────────────────────────────────

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> bool:
        """Index a single document into Qdrant.

        If *embedding* is not provided, it is generated from *content*
        using the configured embedding provider.

        Returns True on success, False on failure.
        Logs degradation when circuit breaker is open.
        """
        if self._circuit_breaker.state.value != "closed":
            logger.warning(
                "retrieval.degraded.circuit_open_skip_index",
                doc_id=doc_id,
                state=self._circuit_breaker.state.value,
            )
            return False

        try:
            self.ensure_collection_exists()

            # Generate embedding if not provided
            if embedding is None:
                from src.services.embedding import get_embedding_provider

                provider = get_embedding_provider(use_cache=True)
                embeddings = await provider.embed([content])
                embedding = embeddings[0]

            # Merge content into payload for retrieval
            payload = {**metadata, "content": content}

            point = PointStruct(
                id=str(doc_id),
                vector=embedding,
                payload=payload,
            )
            self._retry_operation(
                "qdrant_upsert",
                self.client.upsert,
                collection_name=self.COLLECTION_NAME,
                points=[point],
            )
            self._circuit_breaker.record_success()
            logger.debug("retrieval.document_indexed", doc_id=doc_id)
            return True
        except CircuitBreakerOpen:
            logger.warning("retrieval.degraded.circuit_open_index", doc_id=doc_id)
            return False
        except Exception as exc:
            self._circuit_breaker.record_failure()
            logger.error("retrieval.index_failed", doc_id=doc_id, error=str(exc))
            return False

    # ── Bulk indexing ───────────────────────────────────────────────────────

    async def bulk_index(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Batch-index multiple documents into Qdrant with progress tracking.

        Each dict in *documents* must contain at least ``doc_id`` and ``content``.
        Optional keys: ``metadata`` (dict), ``embedding`` (list[float]).

        Returns ``{"indexed": N, "failed": M}`` counts.
        """
        if not self._circuit_breaker.allow_request():
            logger.warning("retrieval.circuit_open_skip_bulk")
            return {"indexed": 0, "failed": len(documents)}

        self.ensure_collection_exists()

        indexed = 0
        failed = 0
        batch_size = 64
        total = len(documents)

        # Generate embeddings in bulk for efficiency
        from src.services.embedding import get_embedding_provider

        provider = get_embedding_provider(use_cache=True)

        # Separate docs that need embedding generation from those that have one
        docs_needing_embedding: list[tuple[int, str]] = []
        for i, doc in enumerate(documents):
            if doc.get("embedding") is None:
                docs_needing_embedding.append((i, doc.get("content", "")))

        # Batch-generate missing embeddings
        if docs_needing_embedding:
            texts = [text for _, text in docs_needing_embedding]
            generated = await provider.embed(texts)
            for (idx, _), emb in zip(docs_needing_embedding, generated):
                documents[idx]["embedding"] = emb

        # Index in batches with progress tracking
        for batch_start in range(0, total, batch_size):
            batch = documents[batch_start : batch_start + batch_size]
            points: list[PointStruct] = []

            for doc in batch:
                doc_id = doc.get("doc_id")
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                embedding = doc.get("embedding", [])

                if not doc_id or not content or not embedding:
                    failed += 1
                    continue

                payload = {**metadata, "content": content}
                points.append(
                    PointStruct(
                        id=str(doc_id),
                        vector=embedding,
                        payload=payload,
                    )
                )

            if points:
                try:
                    self._retry_operation(
                        "qdrant_bulk_upsert",
                        self.client.upsert,
                        collection_name=self.COLLECTION_NAME,
                        points=points,
                    )
                    indexed += len(points)
                    self._circuit_breaker.record_success()
                except Exception as exc:
                    failed += len(points)
                    self._circuit_breaker.record_failure()
                    logger.error(
                        "retrieval.bulk_batch_failed",
                        batch_start=batch_start,
                        batch_size=len(points),
                        error=str(exc),
                    )

            progress = min(batch_start + batch_size, total)
            logger.debug(
                "retrieval.bulk_progress",
                indexed=indexed,
                failed=failed,
                progress=f"{progress}/{total}",
            )

        logger.info(
            "retrieval.bulk_complete",
            total=total,
            indexed=indexed,
            failed=failed,
        )
        return {"indexed": indexed, "failed": failed}

    # ── Cleanup ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close Redis cache and Qdrant client connections."""
        if self._cache is not None:
            await self._cache.close()
        if self._client is not None:
            self._client.close()
            self._client = None


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy dataclass (kept for backward compatibility with existing code)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RetrievedDoc:
    """Legacy retrieval result (used by drafting service)."""

    ticket_id: int
    subject: str
    body: str
    score: float
    metadata: dict | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level retriever singleton
# ═══════════════════════════════════════════════════════════════════════════════

_retriever: TicketRetriever | None = None


def get_retriever(use_cache: bool = True) -> TicketRetriever:
    """Return (and lazily create) the module-level TicketRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = TicketRetriever(use_cache=use_cache)
    return _retriever


# ═══════════════════════════════════════════════════════════════════════════════
# Backward-compatible async functions (delegate to the singleton)
# ═══════════════════════════════════════════════════════════════════════════════


async def search_similar_tickets(
    query_text: str,
    *,
    top_k: int = 5,
    score_threshold: float = 0.5,
) -> list[RetrievedDoc]:
    """Search Qdrant for tickets similar to ``query_text`` (legacy wrapper)."""
    retriever = get_retriever()
    docs = await retriever.search(
        query_text, limit=top_k, min_score=score_threshold
    )
    return [
        RetrievedDoc(
            ticket_id=int(d.metadata.get("ticket_id", d.id)),
            subject=d.metadata.get("subject", ""),
            body=d.content,
            score=d.similarity_score,
            metadata=d.metadata,
        )
        for d in docs
    ]


async def ingest_ticket_embedding(
    ticket_id: int,
    subject: str,
    body: str,
    embedding: list[float],
) -> None:
    """Upsert a ticket's embedding into Qdrant (legacy wrapper)."""
    retriever = get_retriever()
    await retriever.add_document(
        doc_id=str(ticket_id),
        content=f"{subject} {body}",
        metadata={"ticket_id": ticket_id, "subject": subject, "body": body},
        embedding=embedding,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph Tool
# ═══════════════════════════════════════════════════════════════════════════════


class SearchTicketsInput(BaseModel):
    """Input schema for the ``search_tickets`` LangGraph tool."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language search query for similar support tickets.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return.",
    )
    min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0 to 1.0).",
    )
    category: str | None = Field(
        default=None,
        description="Optional filter by ticket category (e.g. 'bug', 'feature_request').",
    )
    urgency: str | None = Field(
        default=None,
        description="Optional filter by urgency level (e.g. 'low', 'medium', 'high', 'critical').",
    )


@tool
async def search_tickets(
    query: str,
    limit: int = 5,
    min_score: float = 0.5,
    category: str | None = None,
    urgency: str | None = None,
) -> list[dict]:
    """Search for similar historical support tickets using vector similarity.

    Finds past tickets that are semantically similar to the given query,
    returning their content, metadata, and similarity scores. Useful for
    understanding how similar issues were resolved previously.

    Args:
        query: Natural-language search query describing the issue.
        limit: Maximum number of results (1-20, default 5).
        min_score: Minimum similarity score threshold (0.0-1.0, default 0.5).
        category: Optional category filter (e.g. 'bug', 'feature_request').
        urgency: Optional urgency filter (e.g. 'low', 'medium', 'high', 'critical').

    Returns:
        List of dicts, each containing: id, content, metadata, similarity_score, source.
    """
    retriever = get_retriever()

    # Build optional payload filters
    filters: dict[str, Any] = {}
    if category:
        filters["category"] = category
    if urgency:
        filters["urgency"] = urgency

    docs = await retriever.search(
        query,
        limit=limit,
        filters=filters if filters else None,
        min_score=min_score,
    )

    return [
        {
            "id": d.id,
            "content": d.content[:500],  # Truncate for LLM context
            "metadata": d.metadata,
            "similarity_score": d.similarity_score,
            "source": d.source,
        }
        for d in docs
    ]


# Export tool for agent registration
RETRIEVER_TOOLS = [search_tickets]
