"""Embedding service — provider abstraction, caching, and circuit breaker.

Provides a unified interface for text embedding with support for:
- OpenAI ada-002 (remote, async, batched with retry)
- Sentence Transformers (local fallback)
- Redis caching with 24-hour TTL
- Circuit breaker pattern for external API resilience
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = structlog.get_logger(__name__)

settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is in OPEN state."""


@dataclass
class CircuitBreaker:
    """Circuit breaker for external API calls.

    Tracks failures within a rolling time window. Opens the circuit
    after *failure_threshold* failures within *failure_window_seconds*,
    then resets after *recovery_timeout_seconds*.
    """

    failure_threshold: int = 5
    failure_window_seconds: float = 60.0
    recovery_timeout_seconds: float = 30.0

    _failures: deque[float] = field(default_factory=deque, repr=False)
    _state: Literal["closed", "open", "half_open"] = field(
        default="closed", repr=False
    )
    _opened_at: float = field(default=0.0, repr=False)

    @property
    def state(self) -> str:
        if self._state == "open":
            if time.monotonic() - self._opened_at >= self.recovery_timeout_seconds:
                self._state = "half_open"
                logger.info("circuit_breaker.half_open")
        return self._state

    def record_success(self) -> None:
        """Record a successful call — resets failure count in half-open."""
        if self._state == "half_open":
            self._state = "closed"
            self._failures.clear()
            logger.info("circuit_breaker.closed_after_recovery")
        elif self._state == "closed":
            self._failures.clear()

    def record_failure(self) -> None:
        """Record a failed call — may trip the circuit breaker."""
        now = time.monotonic()
        self._failures.append(now)

        # Prune old failures outside the window
        cutoff = now - self.failure_window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

        if self._state == "half_open":
            self._state = "open"
            self._opened_at = now
            logger.warning(
                "circuit_breaker.reopened",
                failures=len(self._failures),
            )
        elif len(self._failures) >= self.failure_threshold:
            self._state = "open"
            self._opened_at = now
            logger.warning(
                "circuit_breaker.opened",
                failures=len(self._failures),
                threshold=self.failure_threshold,
            )

    def allow_request(self) -> bool:
        """Return True if a request is allowed through."""
        return self.state != "open"


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract Provider
# ═══════════════════════════════════════════════════════════════════════════════


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI Provider
# ═══════════════════════════════════════════════════════════════════════════════


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-ada-002.

    Features:
    - Async API calls
    - Batched processing (100 texts per batch)
    - Retry logic with tenacity
    - Circuit breaker protection
    """

    BATCH_SIZE = 100

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._dimension = 1536  # ada-002 output dimension

    def get_dimension(self) -> int:
        return self._dimension

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_openai_api(self, batch: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API for a single batch."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": batch, "model": self._model},
                timeout=30.0,
            )
            response.raise_for_status()

        data = response.json()
        # Sort by index to maintain order
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with batching, retry, and circuit breaker protection."""
        if not texts:
            return []

        if not self._circuit_breaker.allow_request():
            raise CircuitBreakerOpenError(
                "Circuit breaker is open — too many recent failures. "
                f"Recover after {self._circuit_breaker.recovery_timeout_seconds}s."
            )

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            try:
                batch_embeddings = await self._call_openai_api(batch)
                all_embeddings.extend(batch_embeddings)
                self._circuit_breaker.record_success()
            except CircuitBreakerOpenError:
                raise
            except Exception as exc:
                self._circuit_breaker.record_failure()
                logger.error(
                    "embedding.openai.batch_failed",
                    batch_start=i,
                    batch_size=len(batch),
                    error=str(exc),
                )
                raise

        logger.debug(
            "embedding.openai.complete",
            total_texts=len(texts),
            batches=(len(texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE,
        )
        return all_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Sentence Transformer Provider (local fallback)
# ═══════════════════════════════════════════════════════════════════════════════


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    Uses the all-MiniLM-L6-v2 model as a lightweight local fallback
    that requires no API keys or network access.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dimension = 384  # all-MiniLM-L6-v2 output dimension

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("embedding.st.loading_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("embedding.st.model_loaded", model=self._model_name)
        return self._model

    def get_dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the local SentenceTransformer model.

        Runs the synchronous model in a thread executor to avoid blocking.
        """
        if not texts:
            return []

        model = self._load_model()

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, show_progress_bar=False).tolist(),
        )

        logger.debug(
            "embedding.st.complete",
            total_texts=len(texts),
            model=self._model_name,
        )
        return embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Redis Cache
# ═══════════════════════════════════════════════════════════════════════════════


class EmbeddingCache:
    """Redis-backed cache for embedding vectors.

    Uses SHA-256 hashes of input texts as cache keys.
    Supports async Redis client with configurable TTL.
    """

    DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours
    KEY_PREFIX = "emb:"

    def __init__(self, redis_url: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._client = None

    async def _get_client(self):
        """Lazy-initialize the async Redis client."""
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            logger.info("embedding.cache.connected", redis_url=self._redis_url)
        return self._client

    @staticmethod
    def _make_key(text: str) -> str:
        """Generate a deterministic cache key for a text string."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{EmbeddingCache.KEY_PREFIX}{text_hash}"

    async def get(self, text: str) -> list[float] | None:
        """Retrieve a cached embedding for the given text."""
        try:
            client = await self._get_client()
            key = self._make_key(text)
            raw = await client.get(key)
            if raw is not None:
                logger.debug("embedding.cache.hit", key=key[:16])
                return json.loads(raw)
        except Exception as exc:
            logger.warning("embedding.cache.get_failed", error=str(exc))
        return None

    async def set(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in the cache with TTL."""
        try:
            client = await self._get_client()
            key = self._make_key(text)
            await client.set(key, json.dumps(embedding), ex=self._ttl)
            logger.debug("embedding.cache.set", key=key[:16], ttl=self._ttl)
        except Exception as exc:
            logger.warning("embedding.cache.set_failed", error=str(exc))

    async def get_batch(self, texts: list[str]) -> dict[int, list[float]]:
        """Retrieve cached embeddings for a batch of texts.

        Returns a dict mapping original index to cached embedding.
        """
        cache_map: dict[int, list[float]] = {}
        try:
            client = await self._get_client()
            keys = [self._make_key(t) for t in texts]
            raw_values = await client.mget(keys)
            for idx, raw in enumerate(raw_values):
                if raw is not None:
                    cache_map[idx] = json.loads(raw)
            logger.debug(
                "embedding.cache.batch_get",
                requested=len(texts),
                hits=len(cache_map),
            )
        except Exception as exc:
            logger.warning("embedding.cache.batch_get_failed", error=str(exc))
        return cache_map

    async def set_batch(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Store a batch of embeddings in the cache."""
        try:
            client = await self._get_client()
            pipe = client.pipeline()
            for text, embedding in zip(texts, embeddings):
                key = self._make_key(text)
                pipe.set(key, json.dumps(embedding), ex=self._ttl)
            await pipe.execute()
            logger.debug("embedding.cache.batch_set", count=len(texts))
        except Exception as exc:
            logger.warning("embedding.cache.batch_set_failed", error=str(exc))

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# ═══════════════════════════════════════════════════════════════════════════════
# Cached Embedding Provider (decorator)
# ═══════════════════════════════════════════════════════════════════════════════


class CachedEmbeddingProvider(EmbeddingProvider):
    """Decorator that adds Redis caching to any EmbeddingProvider.

    Checks the cache before calling the underlying provider, and stores
    results for future lookups.
    """

    def __init__(self, provider: EmbeddingProvider, cache: EmbeddingCache) -> None:
        self._provider = provider
        self._cache = cache

    def get_dimension(self) -> int:
        return self._provider.get_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed with cache-first strategy."""
        if not texts:
            return []

        # Check cache for existing embeddings
        cached = await self._cache.get_batch(texts)

        # Identify texts that need fresh embeddings
        uncached_indices = [i for i in range(len(texts)) if i not in cached]
        uncached_texts = [texts[i] for i in uncached_indices]

        # Fetch embeddings for uncached texts
        new_embeddings: list[list[float]] = []
        if uncached_texts:
            new_embeddings = await self._provider.embed(uncached_texts)
            # Store new embeddings in cache
            await self._cache.set_batch(uncached_texts, new_embeddings)

        # Reassemble results in original order
        results: list[list[float]] = []
        new_emb_idx = 0
        for i in range(len(texts)):
            if i in cached:
                results.append(cached[i])
            else:
                results.append(new_embeddings[new_emb_idx])
                new_emb_idx += 1

        logger.debug(
            "embedding.cached.complete",
            total=len(texts),
            cache_hits=len(cached),
            fresh=len(uncached_texts),
        )
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════


def get_embedding_provider(
    *,
    use_cache: bool = True,
    embedding_model: Literal["openai", "sentence_transformer"] | None = None,
) -> EmbeddingProvider:
    """Factory function that returns the configured embedding provider.

    Reads configuration from environment variables via ``get_settings()``.
    Returns a CachedEmbeddingProvider wrapping the underlying provider
    when caching is enabled.

    Args:
        use_cache: Whether to wrap the provider with Redis caching.
        embedding_model: Override the embedding model selection.
            If None, uses ``settings.embedding_provider`` or defaults
            to "openai" when an API key is available, else "sentence_transformer".

    Returns:
        An EmbeddingProvider instance.
    """
    model = embedding_model or getattr(settings, "embedding_provider", None)

    # Auto-detect based on available API keys
    if model is None:
        if settings.openai_api_key:
            model = "openai"
        else:
            model = "sentence_transformer"

    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        failure_window_seconds=60.0,
    )

    if model == "openai":
        if not settings.openai_api_key:
            logger.warning(
                "embedding.openai.no_key_falling_back",
                fallback="sentence_transformer",
            )
            model = "sentence_transformer"
        else:
            provider: EmbeddingProvider = OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key,
                circuit_breaker=circuit_breaker,
            )

    if model == "sentence_transformer":
        provider = SentenceTransformerProvider()

    # Wrap with cache if enabled
    if use_cache:
        cache = EmbeddingCache(redis_url=settings.redis_url)
        provider = CachedEmbeddingProvider(provider=provider, cache=cache)

    logger.info(
        "embedding.provider_created",
        model=model,
        use_cache=use_cache,
        dimension=provider.get_dimension(),
    )
    return provider
