"""Services layer — LLM adapters, retrieval, classification, embeddings."""

from src.services.embedding import (
    CachedEmbeddingProvider,
    CircuitBreaker,
    CircuitBreakerOpenError,
    EmbeddingCache,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
    get_embedding_provider,
)

__all__ = [
    "CachedEmbeddingProvider",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "EmbeddingCache",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "get_embedding_provider",
]
