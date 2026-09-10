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
from src.services.retrieval import (
    QueryCache,
    TicketRetriever,
    get_retriever,
    search_tickets,
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
    "QueryCache",
    "TicketRetriever",
    "get_retriever",
    "search_tickets",
]
