"""Tests for the TicketRetriever class and retrieval module."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.retrieval import (
    RetrievedDoc,
    TicketRetriever,
    get_retriever,
    search_tickets,
)
from src.utils.circuit_breaker import CircuitBreaker

# ═══════════════════════════════════════════════════════════════════════════════
# CircuitBreaker
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3, failure_window_seconds=60.0)
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, failure_window_seconds=60.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, failure_window_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(
            failure_threshold=3, failure_window_seconds=60.0, recovery_timeout_seconds=0.1
        )
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

        import time
        time.sleep(0.15)
        assert cb.state == "half_open"
        assert cb.allow_request() is True

    def test_reopens_on_failure_in_half_open(self):
        cb = CircuitBreaker(
            failure_threshold=3, failure_window_seconds=60.0, recovery_timeout_seconds=0.1
        )
        for _ in range(3):
            cb.record_failure()

        import time
        time.sleep(0.15)
        assert cb.state == "half_open"

        cb.record_failure()
        assert cb.state == "open"

    def test_closes_on_success_in_half_open(self):
        cb = CircuitBreaker(
            failure_threshold=3, failure_window_seconds=60.0, recovery_timeout_seconds=0.1
        )
        for _ in range(3):
            cb.record_failure()

        import time
        time.sleep(0.15)
        assert cb.state == "half_open"

        cb.record_success()
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_prunes_old_failures(self):
        cb = CircuitBreaker(failure_threshold=3, failure_window_seconds=0.05)
        cb.record_failure()

        import time
        time.sleep(0.1)

        cb.record_failure()
        cb.record_failure()
        # Only 2 failures in the window (first one was pruned)
        assert cb.state == "closed"


# ═══════════════════════════════════════════════════════════════════════════════
# TicketRetriever
# ═══════════════════════════════════════════════════════════════════════════════


class TestTicketRetrieverInit:
    def test_default_init(self):
        retriever = TicketRetriever(use_cache=False)
        assert retriever._host == "localhost"
        assert retriever._port == 6333
        assert retriever._grpc_port == 6334
        assert retriever._timeout == 60
        assert retriever._client is None
        assert retriever._cache is None

    def test_custom_init(self):
        retriever = TicketRetriever(
            host="custom-host",
            port=9999,
            grpc_port=9998,
            timeout=120,
            use_cache=False,
        )
        assert retriever._host == "custom-host"
        assert retriever._port == 9999
        assert retriever._grpc_port == 9998
        assert retriever._timeout == 120

    def test_cache_enabled(self):
        retriever = TicketRetriever(use_cache=True)
        assert retriever._cache is not None

    def test_cache_disabled(self):
        retriever = TicketRetriever(use_cache=False)
        assert retriever._cache is None


class TestTicketRetrieverHealthCheck:
    def test_health_check_success(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        assert retriever.health_check() is True

    def test_health_check_failure(self):
        retriever = TicketRetriever(use_cache=False)
        retriever._client = MagicMock()
        retriever._client.get_collections.side_effect = ConnectionError("refused")

        assert retriever.health_check() is False


class TestTicketRetrieverEnsureCollection:
    def test_creates_collection_when_missing(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        retriever.ensure_collection_exists()

        mock_client.create_collection.assert_called_once()
        mock_client.create_payload_index.assert_called()

    def test_skips_creation_when_exists(self):
        retriever = TicketRetriever(use_cache=False)
        mock_collection = MagicMock()
        mock_collection.name = "tickets"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [mock_collection]
        retriever._client = mock_client

        retriever.ensure_collection_exists()

        mock_client.create_collection.assert_not_called()

    def test_ensures_only_once(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        retriever.ensure_collection_exists()
        retriever.ensure_collection_exists()

        # create_collection called only once (second call short-circuits)
        mock_client.create_collection.assert_called_once()


class TestTicketRetrieverAddDocument:
    @pytest.mark.asyncio
    async def test_add_document_success(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        mock_provider = MagicMock()
        mock_provider.embed = AsyncMock(return_value=[[0.1] * 1536])

        with patch("src.services.embedding.get_embedding_provider", return_value=mock_provider):
            result = await retriever.add_document(
                doc_id="doc-1",
                content="Test ticket content",
                metadata={"ticket_id": 1, "subject": "Test"},
            )

        assert result is True
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_with_existing_embedding(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        embedding = [0.1] * 1536
        result = await retriever.add_document(
            doc_id="doc-1",
            content="Test ticket content",
            metadata={"ticket_id": 1},
            embedding=embedding,
        )

        assert result is True
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_circuit_breaker_open(self):
        retriever = TicketRetriever(use_cache=False)
        retriever._circuit_breaker._state = "open"
        retriever._circuit_breaker._opened_at = time.monotonic()

        result = await retriever.add_document(
            doc_id="doc-1",
            content="Test",
            metadata={},
        )

        assert result is False


class TestTicketRetrieverBulkIndex:
    @pytest.mark.asyncio
    async def test_bulk_index_empty_list(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        result = await retriever.bulk_index([])
        assert result == {"indexed": 0, "failed": 0}

    @pytest.mark.asyncio
    async def test_bulk_index_with_embeddings(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        docs = [
            {
                "doc_id": f"doc-{i}",
                "content": f"Content {i}",
                "metadata": {"ticket_id": i},
                "embedding": [0.1] * 1536,
            }
            for i in range(3)
        ]

        result = await retriever.bulk_index(docs)
        assert result["indexed"] == 3
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_index_skips_incomplete_docs(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        docs = [
            {"doc_id": "doc-1", "content": "Content", "embedding": [0.1] * 1536},
            {"doc_id": "", "content": "Content", "embedding": [0.1] * 1536},  # empty id
            {"content": "Content", "embedding": [0.1] * 1536},  # no doc_id
        ]

        result = await retriever.bulk_index(docs)
        assert result["indexed"] == 1
        assert result["failed"] == 2

    @pytest.mark.asyncio
    async def test_bulk_index_generates_embeddings(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client

        mock_provider = MagicMock()
        mock_provider.embed = AsyncMock(return_value=[[0.1] * 1536] * 2)

        docs = [
            {"doc_id": "doc-1", "content": "Content 1"},
            {"doc_id": "doc-2", "content": "Content 2"},
        ]

        with patch("src.services.embedding.get_embedding_provider", return_value=mock_provider):
            result = await retriever.bulk_index(docs)

        assert result["indexed"] == 2
        assert result["failed"] == 0
        mock_provider.embed.assert_called_once_with(["Content 1", "Content 2"])

    @pytest.mark.asyncio
    async def test_bulk_index_circuit_breaker_open(self):
        retriever = TicketRetriever(use_cache=False)
        retriever._circuit_breaker._state = "open"
        retriever._circuit_breaker._opened_at = time.monotonic()

        result = await retriever.bulk_index(
            [{"doc_id": "1", "content": "c", "embedding": [0.1] * 1536}]
        )
        assert result == {"indexed": 0, "failed": 1}


class TestTicketRetrieverSearch:
    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.query_points.return_value.points = []
        retriever._client = mock_client

        mock_provider = MagicMock()
        mock_provider.embed = AsyncMock(return_value=[[0.1] * 1536])

        with patch("src.services.embedding.get_embedding_provider", return_value=mock_provider):
            results = await retriever.search("test query", limit=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_circuit_breaker_open(self):
        retriever = TicketRetriever(use_cache=False)
        retriever._circuit_breaker._state = "open"
        retriever._circuit_breaker._opened_at = time.monotonic()

        results = await retriever.search("test query")
        assert results == []


class TestGetRetriever:
    def test_singleton(self):
        import src.services.retrieval as mod

        mod._retriever = None
        r1 = get_retriever(use_cache=False)
        r2 = get_retriever(use_cache=False)
        assert r1 is r2

    def teardown_method(self):
        import src.services.retrieval as mod

        mod._retriever = None


class TestRetrievedDoc:
    def test_legacy_dataclass(self):
        doc = RetrievedDoc(
            ticket_id=1,
            subject="Test",
            body="Body text",
            score=0.85,
            metadata={"key": "value"},
        )
        assert doc.ticket_id == 1
        assert doc.subject == "Test"
        assert doc.score == 0.85
        assert doc.metadata == {"key": "value"}

    def test_legacy_dataclass_minimal(self):
        doc = RetrievedDoc(
            ticket_id=1,
            subject="Test",
            body="Body",
            score=0.85,
        )
        assert doc.metadata is None


class TestSearchTicketsTool:
    @pytest.mark.asyncio
    async def test_search_tickets_returns_dicts(self):
        """Verify the search_tickets tool returns the expected dict format."""
        from unittest.mock import AsyncMock

        from src.models.schemas import RetrievedDocument

        mock_docs = [
            RetrievedDocument(
                id="doc-1",
                content="Test content",
                metadata={"ticket_id": 1},
                similarity_score=0.9,
                source="past_ticket",
            )
        ]

        with patch("src.services.retrieval.get_retriever") as mock_get:
            mock_retriever = MagicMock()
            mock_retriever.search = AsyncMock(return_value=mock_docs)
            mock_get.return_value = mock_retriever

            results = await search_tickets.ainvoke(
                {"query": "test", "limit": 5, "min_score": 0.5}
            )

        assert len(results) == 1
        assert results[0]["id"] == "doc-1"
        assert results[0]["similarity_score"] == 0.9
