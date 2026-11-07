"""Services layer — LLM adapters, retrieval, classification, embeddings, drafting."""

from src.services.classification import (
    TicketClassifier,
    get_classifier,
)
from src.services.drafting import (
    DraftGenerator,
    DraftQualityScore,
    SafetyCheckResult,
    get_draft_generator,
)
from src.services.delivery import (
    DeliveryResult,
    deliver_reply,
)
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
    "DeliveryResult",
    "DraftGenerator",
    "DraftQualityScore",
    "EmbeddingCache",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SafetyCheckResult",
    "SentenceTransformerProvider",
    "TicketClassifier",
    "deliver_reply",
    "get_classifier",
    "get_draft_generator",
    "get_embedding_provider",
    "QueryCache",
    "TicketRetriever",
    "get_retriever",
    "search_tickets",
]
