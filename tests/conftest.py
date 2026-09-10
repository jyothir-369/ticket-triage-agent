"""Shared fixtures for the test suite.

Provides:
  - In-memory SQLite test database with schema creation/teardown
  - Async HTTP test client wired to the FastAPI app with DB override
  - Mock LLM, classifier, retriever, and draft generator fixtures
  - Factories for creating test tickets, classifications, and drafts
  - Environment overrides (TESTING=True) for safe test execution
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.config import Settings, get_settings
from src.models import Base, get_db_session
from src.models.schemas import (
    Citation,
    DraftResponse,
    EscalationDecision,
    RetrievedDocument,
    Ticket,
    TicketCategory,
    TicketClassification,
    TicketMetadata,
    TicketStatus,
    TriageTrace,
    UrgencyLevel,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Environment setup
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True, scope="session")
def _set_testing_env() -> Generator[None, None, None]:
    """Set TESTING=True for the entire test session."""
    old = os.environ.get("TESTING")
    os.environ["TESTING"] = "True"
    yield
    if old is None:
        os.environ.pop("TESTING", None)
    else:
        os.environ["TESTING"] = old


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear the LRU-cached Settings singleton before each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Event loop
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Test database engine + session
# ═══════════════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """In-memory SQLite engine with StaticPool for consistent async connections."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional session that rolls back after each test."""
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP test client
# ═══════════════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client wired to the FastAPI app with test DB override."""

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Test data factories
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def make_ticket():
    """Factory fixture for creating test Ticket objects."""

    def _make(
        ticket_id: int | str = "test-001",
        content: str = "Subject: Test ticket\n\nBody:\nThis is a test ticket.",
        source: str = "api",
        source_id: str | None = None,
        **kwargs: Any,
    ) -> Ticket:
        return Ticket(
            id=ticket_id,
            content=content,
            source=source,
            source_id=source_id,
            **kwargs,
        )

    return _make


@pytest.fixture
def make_classification():
    """Factory fixture for creating test TicketClassification objects."""

    def _make(
        category: TicketCategory = TicketCategory.BUG,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
        confidence: float = 0.85,
        reasoning: str = "Test classification.",
        **kwargs: Any,
    ) -> TicketClassification:
        return TicketClassification(
            category=category,
            urgency=urgency,
            confidence=confidence,
            reasoning=reasoning,
            **kwargs,
        )

    return _make


@pytest.fixture
def make_draft():
    """Factory fixture for creating test DraftResponse objects."""

    def _make(
        draft_text: str = "Thank you for contacting us. We are looking into your issue.",
        citations: list[Citation] | None = None,
        confidence: float = 0.80,
        reasoning: str = "Test draft.",
        **kwargs: Any,
    ) -> DraftResponse:
        return DraftResponse(
            draft_text=draft_text,
            citations=citations or [],
            confidence=confidence,
            reasoning=reasoning,
            **kwargs,
        )

    return _make


@pytest.fixture
def make_retrieved_doc():
    """Factory fixture for creating test RetrievedDocument objects."""

    def _make(
        doc_id: str = "doc-001",
        content: str = "This is a retrieved document with helpful information.",
        similarity_score: float = 0.85,
        source: str = "past_ticket",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RetrievedDocument:
        return RetrievedDocument(
            id=doc_id,
            content=content,
            similarity_score=similarity_score,
            source=source,
            metadata=metadata or {},
            **kwargs,
        )

    return _make


@pytest.fixture
def make_escalation():
    """Factory fixture for creating test EscalationDecision objects."""

    def _make(
        should_escalate: bool = False,
        reason: str = "Confidence meets threshold",
        confidence_score: float = 0.85,
        threshold_used: float = 0.70,
        **kwargs: Any,
    ) -> EscalationDecision:
        return EscalationDecision(
            should_escalate=should_escalate,
            reason=reason,
            confidence_score=confidence_score,
            threshold_used=threshold_used,
            **kwargs,
        )

    return _make


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-built test data fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_ticket() -> Ticket:
    """A typical support ticket for bug reports."""
    return Ticket(
        id="test-bug-001",
        content=(
            "Subject: Login page returns 500 error\n\n"
            "Body:\nI'm unable to log in. Every time I enter my credentials "
            "I get a 500 Internal Server Error. This is blocking my work."
        ),
        source="api",
    )


@pytest.fixture
def sample_feature_request_ticket() -> Ticket:
    """A feature request ticket."""
    return Ticket(
        id="test-feat-001",
        content=(
            "Subject: Request for dark mode\n\n"
            "Body:\nPlease add dark mode support to the web application. "
            "It would be a great quality-of-life improvement."
        ),
        source="email",
    )


@pytest.fixture
def sample_billing_ticket() -> Ticket:
    """A billing issue ticket."""
    return Ticket(
        id="test-bill-001",
        content=(
            "Subject: Unauthorized charge on credit card\n\n"
            "Body:\nI noticed an unauthorized charge of $99.99 on my credit card. "
            "I need this investigated immediately and want a refund."
        ),
        source="api",
    )


@pytest.fixture
def sample_critical_ticket() -> Ticket:
    """A critical production-down ticket."""
    return Ticket(
        id="test-crit-001",
        content=(
            "Subject: CRITICAL - Production system down\n\n"
            "Body:\nOur entire production system is down. All users are affected. "
            "We are losing revenue every minute. Need immediate assistance."
        ),
        source="api",
    )


@pytest.fixture
def sample_classification() -> TicketClassification:
    """A high-confidence bug classification."""
    return TicketClassification(
        category=TicketCategory.BUG,
        urgency=UrgencyLevel.HIGH,
        confidence=0.92,
        reasoning="Clear bug report with 500 error on login page.",
    )


@pytest.fixture
def sample_draft() -> DraftResponse:
    """A typical draft response."""
    return DraftResponse(
        draft_text=(
            "Thank you for reporting this issue. We've identified a server-side "
            "error affecting the login page. Our engineering team is actively "
            "investigating and we expect a fix within the next 2 hours."
        ),
        citations=[],
        confidence=0.80,
        reasoning="Bug report with clear error indication.",
    )


@pytest.fixture
def sample_retrieved_docs() -> list[RetrievedDocument]:
    """A list of retrieved documents for testing."""
    return [
        RetrievedDocument(
            id="doc-001",
            content="Previous login issue was resolved by clearing session cache.",
            similarity_score=0.88,
            source="past_ticket",
            metadata={"ticket_id": 123, "category": "bug"},
        ),
        RetrievedDocument(
            id="doc-002",
            content="Known issue with Redis session store causing 500 errors.",
            similarity_score=0.75,
            source="knowledge_base",
            metadata={"topic": "redis"},
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Mock LLM fixture
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_llm_response():
    """Factory fixture for creating mock LLM JSON responses."""

    def _make(
        category: str = "BUG",
        urgency: str = "HIGH",
        confidence: float = 0.90,
        reasoning: str = "Test classification.",
    ) -> str:
        import json

        return json.dumps({
            "category": category,
            "urgency": urgency,
            "confidence": confidence,
            "reasoning": reasoning,
        })

    return _make


@pytest.fixture
def mock_draft_llm_response():
    """Factory fixture for creating mock draft LLM JSON responses."""

    def _make(
        draft_text: str = "Thank you for contacting us. We are investigating.",
        reasoning: str = "Standard support response.",
    ) -> str:
        import json

        return json.dumps({
            "draft_text": draft_text,
            "citations": [],
            "reasoning": reasoning,
        })

    return _make


# ═══════════════════════════════════════════════════════════════════════════════
# Mock service fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_classifier():
    """Mock TicketClassifier that returns a fixed classification."""
    classifier = MagicMock()
    classifier.classify = AsyncMock(
        return_value=TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.90,
            reasoning="Mock classification.",
        )
    )
    classifier.count_tokens = MagicMock(return_value=100)
    classifier.estimate_cost = MagicMock(return_value=0.001)
    classifier.get_token_usage = MagicMock(
        return_value=MagicMock(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.001,
            model="mock-model",
        )
    )
    classifier.get_cost_summary = MagicMock(
        return_value={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "estimated_cost_usd": 0.001,
            "model": "mock-model",
        }
    )
    classifier.reset_usage = MagicMock()
    return classifier


@pytest.fixture
def mock_retriever():
    """Mock TicketRetriever that returns fixed retrieved documents."""
    retriever = AsyncMock()
    retriever.search = AsyncMock(
        return_value=[
            RetrievedDocument(
                id="doc-001",
                content="Previous similar issue was resolved by restarting the service.",
                similarity_score=0.88,
                source="past_ticket",
                metadata={"ticket_id": 123},
            ),
        ]
    )
    retriever.add_document = AsyncMock(return_value=True)
    retriever.bulk_index = AsyncMock(return_value={"indexed": 1, "failed": 0})
    retriever.health_check = MagicMock(return_value=True)
    retriever.close = AsyncMock()
    return retriever


@pytest.fixture
def mock_draft_generator():
    """Mock DraftGenerator that returns a fixed draft response."""
    generator = AsyncMock()
    generator.generate = AsyncMock(
        return_value=DraftResponse(
            draft_text="Thank you for your report. We are investigating the issue.",
            citations=[],
            confidence=0.80,
            reasoning="Mock draft generation.",
            citations_validated=False,
        )
    )
    generator.count_tokens = MagicMock(return_value=200)
    generator.estimate_cost = MagicMock(return_value=0.002)
    generator.get_token_usage = MagicMock(
        return_value=MagicMock(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            estimated_cost_usd=0.002,
            model="mock-model",
        )
    )
    generator.get_cost_summary = MagicMock(
        return_value={
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
            "estimated_cost_usd": 0.002,
            "model": "mock-model",
        }
    )
    generator.reset_usage = MagicMock()
    return generator


@pytest.fixture
def mock_embedding_provider():
    """Mock EmbeddingProvider that returns fixed embeddings."""
    provider = AsyncMock()
    provider.embed = AsyncMock(
        return_value=[[0.1] * 384]  # 384-dim dummy vector
    )
    provider.get_dimension = MagicMock(return_value=384)
    return provider


# ═══════════════════════════════════════════════════════════════════════════════
# Patched service singletons (auto-use for tests needing mocked services)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def patched_classifier(mock_classifier):
    """Patch get_classifier to return the mock classifier."""
    with patch("src.services.classification.get_classifier", return_value=mock_classifier):
        yield mock_classifier


@pytest.fixture
def patched_retriever(mock_retriever):
    """Patch get_retriever to return the mock retriever."""
    with patch("src.services.retrieval.get_retriever", return_value=mock_retriever):
        yield mock_retriever


@pytest.fixture
def patched_draft_generator(mock_draft_generator):
    """Patch get_draft_generator to return the mock draft generator."""
    with patch("src.services.drafting.get_draft_generator", return_value=mock_draft_generator):
        yield mock_draft_generator


@pytest.fixture
def patched_call_llm(mock_llm_response):
    """Patch call_llm to return a mock LLM classification response."""
    with patch("src.services.llm.call_llm", new_callable=AsyncMock) as mock:
        mock.return_value = mock_llm_response()
        yield mock


# ═══════════════════════════════════════════════════════════════════════════════
# Full mock agent state
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_agent_state(
    sample_ticket,
    sample_classification,
    sample_draft,
    sample_retrieved_docs,
):
    """A fully populated AgentState for testing node functions."""
    from src.models.schemas import TriageTrace

    return {
        "ticket": sample_ticket,
        "classification": sample_classification,
        "retrieved_docs": [
            {
                "id": d.id,
                "content": d.content,
                "metadata": d.metadata,
                "similarity_score": d.similarity_score,
                "source": d.source,
            }
            for d in sample_retrieved_docs
        ],
        "draft": sample_draft,
        "escalation": None,
        "should_escalate": False,
        "escalation_reason": "",
        "trace": TriageTrace(ticket_id=str(sample_ticket.id)),
        "loop_count": 0,
        "tool_call_count": 0,
        "error_message": "",
        "messages": [],
    }


@pytest.fixture
def sample_agent_state_escalated(
    sample_ticket,
    sample_classification,
    sample_draft,
    make_escalation,
):
    """An AgentState that should trigger escalation."""
    from src.models.schemas import TriageTrace

    low_conf_class = TicketClassification(
        category=TicketCategory.BUG,
        urgency=UrgencyLevel.HIGH,
        confidence=0.30,
        reasoning="Low confidence.",
    )
    return {
        "ticket": sample_ticket,
        "classification": low_conf_class,
        "retrieved_docs": [],
        "draft": DraftResponse(
            draft_text="We are looking into this.",
            confidence=0.30,
            reasoning="Low confidence draft.",
        ),
        "escalation": make_escalation(
            should_escalate=True,
            reason="Low confidence",
            confidence_score=0.30,
        ),
        "should_escalate": True,
        "escalation_reason": "Low confidence",
        "trace": TriageTrace(ticket_id=str(sample_ticket.id)),
        "loop_count": 0,
        "tool_call_count": 0,
        "error_message": "",
        "messages": [],
    }
