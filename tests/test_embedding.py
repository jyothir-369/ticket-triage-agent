"""Tests for the embedding service — providers, cache, circuit breaker."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.embedding import (
    CachedEmbeddingProvider,
    CircuitBreaker,
    CircuitBreakerOpenError,
    EmbeddingCache,
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
    get_embedding_provider,
)
from src.services.retrieval import TicketRetriever

# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, failure_window_seconds=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(
            failure_threshold=2,
            failure_window_seconds=60,
            recovery_timeout_seconds=0.1,
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        time.sleep(0.15)
        assert cb.state == "half_open"
        assert cb.allow_request() is True

    def test_reopens_on_failure_in_half_open(self):
        cb = CircuitBreaker(
            failure_threshold=2,
            failure_window_seconds=60,
            recovery_timeout_seconds=0.1,
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"
        cb.record_failure()
        assert cb.state == "open"

    def test_closes_on_success_in_half_open(self):
        cb = CircuitBreaker(
            failure_threshold=2,
            failure_window_seconds=60,
            recovery_timeout_seconds=0.1,
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"
        cb.record_success()
        assert cb.state == "closed"

    def test_prunes_old_failures(self):
        cb = CircuitBreaker(failure_threshold=3, failure_window_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.record_failure()
        # Old failures should be pruned, only 1 recent failure
        assert cb.state == "closed"


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI Provider
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpenAIEmbeddingProvider:
    def test_get_dimension(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert provider.get_dimension() == 1536

    def test_empty_texts(self):
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        result = asyncio.run(provider.embed([]))
        assert result == []

    def test_circuit_breaker_blocks_requests(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()

        provider = OpenAIEmbeddingProvider(api_key="test-key", circuit_breaker=cb)
        with pytest.raises(CircuitBreakerOpenError):
            asyncio.run(provider.embed(["test"]))

    def test_embed_single_batch(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            provider = OpenAIEmbeddingProvider(api_key="test-key")
            result = asyncio.run(provider.embed(["hello", "world"]))

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    def test_embed_multiple_batches(self):
        texts = [f"text_{i}" for i in range(150)]  # > 100 = 2 batches

        # First batch returns 100 items, second returns 50
        batch_responses = [
            {"data": [{"index": i, "embedding": [float(i)]} for i in range(100)]},
            {"data": [{"index": i, "embedding": [float(100 + i)]} for i in range(50)]},
        ]

        mock_responses = []
        for resp_data in batch_responses:
            mock_resp = MagicMock()
            mock_resp.json.return_value = resp_data
            mock_resp.raise_for_status = MagicMock()
            mock_responses.append(mock_resp)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_responses)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            provider = OpenAIEmbeddingProvider(api_key="test-key")
            result = asyncio.run(provider.embed(texts))

        assert len(result) == 150
        assert mock_client.post.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Sentence Transformer Provider
# ═══════════════════════════════════════════════════════════════════════════════


class TestSentenceTransformerProvider:
    def test_get_dimension(self):
        provider = SentenceTransformerProvider()
        assert provider.get_dimension() == 384

    def test_empty_texts(self):
        provider = SentenceTransformerProvider()
        result = asyncio.run(provider.embed([]))
        assert result == []

    def test_embed_calls_model(self):
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [[0.1] * 384, [0.2] * 384]

        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = mock_model

            provider = SentenceTransformerProvider(model_name="test-model")
            result = asyncio.run(provider.embed(["hello", "world"]))

        assert len(result) == 2
        assert len(result[0]) == 384
        mock_model.encode.assert_called_once_with(
            ["hello", "world"], show_progress_bar=False
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Cache
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmbeddingCache:
    def test_make_key_deterministic(self):
        key1 = EmbeddingCache._make_key("hello world")
        key2 = EmbeddingCache._make_key("hello world")
        assert key1 == key2
        assert key1.startswith("emb:")

    def test_make_key_different_for_different_texts(self):
        key1 = EmbeddingCache._make_key("hello")
        key2 = EmbeddingCache._make_key("world")
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self):
        cache = EmbeddingCache(redis_url="redis://localhost:6379/0")
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)
        cache._client = mock_client

        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_embedding_on_hit(self):
        cache = EmbeddingCache(redis_url="redis://localhost:6379/0")
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=json.dumps([0.1, 0.2, 0.3]))
        cache._client = mock_client

        result = await cache.get("hello")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_set_stores_embedding(self):
        cache = EmbeddingCache(redis_url="redis://localhost:6379/0")
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        cache._client = mock_client

        await cache.set("hello", [0.1, 0.2])
        mock_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_batch(self):
        cache = EmbeddingCache(redis_url="redis://localhost:6379/0")
        mock_client = AsyncMock()
        mock_client.mget = AsyncMock(
            return_value=[json.dumps([0.1]), None, json.dumps([0.3])]
        )
        cache._client = mock_client

        result = await cache.get_batch(["a", "b", "c"])
        assert 0 in result
        assert 2 in result
        assert 1 not in result


# ═══════════════════════════════════════════════════════════════════════════════
# Cached Embedding Provider
# ═══════════════════════════════════════════════════════════════════════════════


class TestCachedEmbeddingProvider:
    def test_empty_texts(self):
        mock_provider = MagicMock(spec=EmbeddingProvider)
        mock_cache = MagicMock(spec=EmbeddingCache)
        cached = CachedEmbeddingProvider(mock_provider, mock_cache)

        result = asyncio.run(cached.embed([]))
        assert result == []

    def test_uses_cache_hit(self):
        mock_provider = MagicMock(spec=EmbeddingProvider)
        mock_cache = MagicMock(spec=EmbeddingCache)
        mock_cache.get_batch = AsyncMock(return_value={0: [0.1, 0.2]})

        cached = CachedEmbeddingProvider(mock_provider, mock_cache)
        result = asyncio.run(cached.embed(["hello"]))

        assert result == [[0.1, 0.2]]
        mock_provider.embed.assert_not_called()

    def test_calls_provider_for_miss(self):
        mock_provider = MagicMock(spec=EmbeddingProvider)
        mock_provider.embed = AsyncMock(return_value=[[0.3, 0.4]])
        mock_cache = MagicMock(spec=EmbeddingCache)
        mock_cache.get_batch = AsyncMock(return_value={})
        mock_cache.set_batch = AsyncMock()

        cached = CachedEmbeddingProvider(mock_provider, mock_cache)
        result = asyncio.run(cached.embed(["hello"]))

        assert result == [[0.3, 0.4]]
        mock_provider.embed.assert_called_once_with(["hello"])


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetEmbeddingProvider:
    @patch("src.services.embedding.settings")
    def test_auto_selects_sentence_transformer_without_key(self, mock_settings):
        mock_settings.openai_api_key = ""
        mock_settings.embedding_provider = None
        mock_settings.redis_url = "redis://localhost"
        mock_settings.gemini_api_key = ""
        mock_settings.embedding_vector_size = 1536
        mock_settings.redis_max_connections = 10
        mock_settings.redis_socket_timeout = 5
        provider = get_embedding_provider(use_cache=False)
        assert isinstance(provider, SentenceTransformerProvider)

    @patch("src.services.embedding.settings")
    def test_selects_openai_when_key_available(self, mock_settings):
        mock_settings.openai_api_key = "sk-test"
        mock_settings.embedding_provider = None
        mock_settings.redis_url = "redis://localhost"
        mock_settings.gemini_api_key = ""
        mock_settings.embedding_vector_size = 1536
        mock_settings.redis_max_connections = 10
        mock_settings.redis_socket_timeout = 5
        provider = get_embedding_provider(use_cache=False)
        assert isinstance(provider, OpenAIEmbeddingProvider)

    @patch("src.services.embedding.settings")
    def test_explicit_model_override(self, mock_settings):
        mock_settings.openai_api_key = "sk-test"
        mock_settings.embedding_provider = "sentence_transformer"
        mock_settings.redis_url = "redis://localhost"
        provider = get_embedding_provider(use_cache=False)
        assert isinstance(provider, SentenceTransformerProvider)


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Dimension Compatibility (regression — B1)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmbeddingDimensionCompatibility:
    """Regression suite for the silent 384/1536 mismatch (PROJECT_AUDIT §7)."""

    # --- get_dimension across all providers ---

    def test_openai_get_dimension_1536(self):
        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        assert provider.get_dimension() == 1536

    def test_sentence_transformer_get_dimension_384(self):
        provider = SentenceTransformerProvider()
        assert provider.get_dimension() == 384

    def test_gemini_get_dimension_1536(self):
        provider = GeminiEmbeddingProvider(api_key="gem-test-key")
        assert provider.get_dimension() == 1536

    def test_gemini_custom_dimension(self):
        provider = GeminiEmbeddingProvider(api_key="gem-test-key", dimension=768)
        assert provider.get_dimension() == 768

    # --- factory auto-selection picks the right provider dimension ---

    @patch("src.services.embedding.settings")
    def test_auto_selects_sentence_transformer_384_without_keys(self, mock_settings):
        mock_settings.openai_api_key = ""
        mock_settings.embedding_provider = None
        mock_settings.gemini_api_key = ""
        mock_settings.redis_url = "redis://localhost"
        mock_settings.redis_max_connections = 10
        mock_settings.redis_socket_timeout = 5
        provider = get_embedding_provider(use_cache=False)
        assert isinstance(provider, SentenceTransformerProvider)
        assert provider.get_dimension() == 384

    @patch("src.services.embedding.settings")
    def test_auto_selects_openai_1536_when_key_available(self, mock_settings):
        mock_settings.openai_api_key = "sk-test"
        mock_settings.embedding_provider = None
        mock_settings.gemini_api_key = ""
        mock_settings.redis_url = "redis://localhost"
        mock_settings.redis_max_connections = 10
        mock_settings.redis_socket_timeout = 5
        provider = get_embedding_provider(use_cache=False)
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.get_dimension() == 1536

    @patch("src.services.embedding.settings")
    def test_auto_selects_gemini_1536_when_gemini_key_available(self, mock_settings):
        mock_settings.gemini_api_key = "gem-test-key"
        mock_settings.embedding_provider = None
        mock_settings.openai_api_key = ""
        mock_settings.redis_url = "redis://localhost"
        mock_settings.redis_max_connections = 10
        mock_settings.redis_socket_timeout = 5
        mock_settings.embedding_vector_size = 1536
        provider = get_embedding_provider(use_cache=False)
        assert isinstance(provider, GeminiEmbeddingProvider)
        assert provider.get_dimension() == 1536

    # --- ensure_collection_exists mismatch (no silent breakage) ---

    def test_mismatch_raises_valueerror(self):
        """ST emits 384; collection forced 1536 → ValueError loud."""
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client
        TicketRetriever.VECTOR_SIZE = 1536

        with patch(
            "src.services.embedding.get_embedding_provider",
            return_value=SentenceTransformerProvider(),
        ):
            with pytest.raises(ValueError, match="Embedding dimension mismatch"):
                retriever.ensure_collection_exists()

    def test_matching_dimensions_no_raise(self):
        """384 collection and ST provider agree → no exception."""
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client
        TicketRetriever.VECTOR_SIZE = 384

        with patch(
            "src.services.embedding.get_embedding_provider",
            return_value=SentenceTransformerProvider(),
        ):
            retriever.ensure_collection_exists()  # should not raise

    def test_openai_matches_default_collection(self):
        """OpenAI (1536) vs default collection (1536) → no raise."""
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client
        TicketRetriever.VECTOR_SIZE = 1536

        with patch(
            "src.services.embedding.get_embedding_provider",
            return_value=OpenAIEmbeddingProvider(api_key="sk-test"),
        ):
            retriever.ensure_collection_exists()  # should not raise

    def test_gemini_matches_default_collection(self):
        """Gemini (1536) vs default collection (1536) → no raise."""
        retriever = TicketRetriever(use_cache=False)
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        retriever._client = mock_client
        TicketRetriever.VECTOR_SIZE = 1536

        with patch(
            "src.services.embedding.get_embedding_provider",
            return_value=GeminiEmbeddingProvider(api_key="gem-test-key"),
        ):
            retriever.ensure_collection_exists()  # should not raise

    def test_cached_provider_inherits_active_dimension(self):
        """Wrapping with cache keeps the underlying provider's dimension."""
        provider = CachedEmbeddingProvider(
            provider=SentenceTransformerProvider(),
            cache=MagicMock(spec=EmbeddingCache),
        )
        assert provider.get_dimension() == 384
