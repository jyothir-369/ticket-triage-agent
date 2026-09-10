"""Tests for TicketRepository — async CRUD, status updates, triage results, and metrics."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.models import Base
from src.models.schemas import (
    DashboardMetrics,
    Ticket,
    TicketMetadata,
    TicketStatus,
    TraceStep,
)
from src.repository import TicketRepository

# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def test_engine():
    """In-memory SQLite engine with StaticPool for consistent connections."""
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
async def repo(test_engine):
    """TicketRepository wired to the in-memory test database."""
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    import src.models.database as db_mod
    original_factory = db_mod._session_factory

    db_mod._session_factory = session_factory

    yield TicketRepository()

    db_mod._session_factory = original_factory


def _make_ticket(**overrides) -> Ticket:
    """Create a Ticket with sensible defaults."""
    defaults = {
        "id": "test-001",
        "content": "My app crashes on login",
        "source": "email",
        "source_id": "EM-12345",
        "metadata": TicketMetadata(customer_email="user@test.com", tags=["urgent"]),
    }
    defaults.update(overrides)
    return Ticket(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# create_ticket
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestCreateTicket:
    async def test_creates_row(self, repo: TicketRepository):
        ticket = _make_ticket()
        model = await repo.create_ticket(ticket)

        assert model.id == "test-001"
        assert model.content == "My app crashes on login"
        assert model.source == "email"
        assert model.source_id == "EM-12345"
        assert model.status == TicketStatus.OPEN.value

    async def test_metadata_stored_as_json(self, repo: TicketRepository):
        ticket = _make_ticket()
        model = await repo.create_ticket(ticket)

        assert model.metadata_ is not None
        parsed = json.loads(model.metadata_)
        assert parsed["customer_email"] == "user@test.com"
        assert "urgent" in parsed["tags"]

    async def test_default_status_is_open(self, repo: TicketRepository):
        model = await repo.create_ticket(_make_ticket())
        assert model.status == "open"

    async def test_created_at_is_set(self, repo: TicketRepository):
        model = await repo.create_ticket(_make_ticket())
        assert model.created_at is not None


# ═══════════════════════════════════════════════════════════════════════════════
# update_ticket
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestUpdateTicket:
    async def test_updates_fields(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        updated = await repo.update_ticket("test-001", {"content": "Updated content"})

        assert updated.content == "Updated content"

    async def test_nonexistent_raises(self, repo: TicketRepository):
        with pytest.raises(ValueError, match="not found"):
            await repo.update_ticket("does-not-exist", {"content": "X"})

    async def test_updates_status(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        updated = await repo.update_ticket("test-001", {"status": "classified"})
        assert updated.status == "classified"

    async def test_metadata_dict_serialised(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        new_meta = {"customer_email": "new@test.com", "tags": ["vip"]}
        updated = await repo.update_ticket("test-001", {"metadata_": new_meta})
        parsed = json.loads(updated.metadata_)
        assert parsed["customer_email"] == "new@test.com"


# ═══════════════════════════════════════════════════════════════════════════════
# update_ticket_status
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestUpdateTicketStatus:
    async def test_sets_status(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        await repo.update_ticket_status("test-001", "classified")

        model = await repo._get_ticket("test-001")
        assert model.status == "classified"

    async def test_sets_processed_at_on_resolved(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        await repo.update_ticket_status("test-001", "resolved")

        model = await repo._get_ticket("test-001")
        assert model.processed_at is not None

    async def test_sets_processed_at_on_failed(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        await repo.update_ticket_status("test-001", "failed")

        model = await repo._get_ticket("test-001")
        assert model.processed_at is not None

    async def test_appends_trace(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        trace_data = {"step": "classify", "status": "completed", "duration_ms": 150}
        await repo.update_ticket_status("test-001", "classified", trace=trace_data)

        trace = await repo.get_ticket_trace("test-001")
        assert trace is not None
        assert len(trace["steps"]) == 1
        assert trace["steps"][0]["step"] == "classify"

    async def test_no_trace_when_none(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        await repo.update_ticket_status("test-001", "classified", trace=None)

        trace = await repo.get_ticket_trace("test-001")
        assert trace is None


# ═══════════════════════════════════════════════════════════════════════════════
# save_triage_result
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestSaveTriageResult:
    async def test_saves_classification_and_draft(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())

        classification = {
            "category": "bug",
            "urgency": "high",
            "confidence": 0.85,
        }
        draft = {
            "draft_text": "We are investigating the login issue.",
            "citations": [{"claim": "known issue", "source_doc_id": "doc-1"}],
        }
        escalation = {
            "should_escalate": False,
            "reason": "",
        }

        await repo.save_triage_result("test-001", classification, draft, escalation)
        model = await repo._get_ticket("test-001")

        assert model.category == "bug"
        assert model.urgency == "high"
        assert model.confidence == 0.85
        assert model.draft_text == "We are investigating the login issue."
        assert model.status == TicketStatus.AWAITING_REVIEW.value

        citations = json.loads(model.draft_citations)
        assert len(citations) == 1

    async def test_escalated_status_when_needed(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())

        classification = {"category": "billing", "urgency": "low", "confidence": 0.3}
        draft = {"draft_text": "Please clarify.", "citations": []}
        escalation = {"should_escalate": True, "reason": "Low confidence"}

        await repo.save_triage_result("test-001", classification, draft, escalation)
        model = await repo._get_ticket("test-001")

        assert model.status == TicketStatus.ESCALATED.value
        assert model.escalation_reason == "Low confidence"


# ═══════════════════════════════════════════════════════════════════════════════
# get_ticket_trace
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestGetTicketTrace:
    async def test_returns_none_when_empty(self, repo: TicketRepository):
        result = await repo.get_ticket_trace("nonexistent")
        assert result is None

    async def test_returns_ordered_steps(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        now = datetime.now(UTC)

        step_a = TraceStep(
            step="draft",
            status="completed",
            timestamp=now,
            duration_ms=200,
            data={"response": "text"},
        )
        step_b = TraceStep(
            step="classify",
            status="completed",
            timestamp=now,
            duration_ms=100,
        )
        await repo.add_trace_step("test-001", step_a)
        await repo.add_trace_step("test-001", step_b)

        trace = await repo.get_ticket_trace("test-001")
        assert trace is not None
        assert trace["ticket_id"] == "test-001"
        assert len(trace["steps"]) == 2
        step_names = [s["step"] for s in trace["steps"]]
        assert "classify" in step_names
        assert "draft" in step_names

    async def test_data_parsed_from_json(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        step = TraceStep(
            step="retrieve",
            status="completed",
            duration_ms=50,
            data={"docs": ["doc-1", "doc-2"]},
        )
        await repo.add_trace_step("test-001", step)

        trace = await repo.get_ticket_trace("test-001")
        assert trace["steps"][0]["data"] == {"docs": ["doc-1", "doc-2"]}


# ═══════════════════════════════════════════════════════════════════════════════
# get_escalated_tickets
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestGetEscalatedTickets:
    async def test_returns_escalated_only(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket(id="t1", source_id="S1"))
        await repo.create_ticket(_make_ticket(id="t2", source_id="S2"))

        await repo.update_ticket_status("t1", "escalated")
        await repo.update_ticket_status("t2", "resolved")

        escalated = await repo.get_escalated_tickets()
        assert len(escalated) == 1
        assert escalated[0].id == "t1"

    async def test_respects_limit(self, repo: TicketRepository):
        for i in range(5):
            await repo.create_ticket(_make_ticket(id=f"e{i}", source_id=f"ES{i}"))
            await repo.update_ticket_status(f"e{i}", "escalated")

        result = await repo.get_escalated_tickets(limit=3)
        assert len(result) == 3

    async def test_empty_when_none_escalated(self, repo: TicketRepository):
        result = await repo.get_escalated_tickets()
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# get_dashboard_metrics
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestGetDashboardMetrics:
    async def test_empty_database(self, repo: TicketRepository):
        metrics = await repo.get_dashboard_metrics()
        assert metrics.total_tickets == 0
        assert metrics.processed_tickets == 0
        assert metrics.escalated_tickets == 0
        assert metrics.approval_rate == 0.0

    async def test_populated_metrics(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket(id="m1", source_id="MS1"))
        await repo.create_ticket(_make_ticket(id="m2", source_id="MS2"))
        await repo.create_ticket(_make_ticket(id="m3", source_id="MS3"))

        classification = {"category": "bug", "urgency": "high", "confidence": 0.9}
        draft = {"draft_text": "Response", "citations": []}

        await repo.save_triage_result(
            "m1", classification, draft, {"should_escalate": False, "reason": ""}
        )
        await repo.save_triage_result(
            "m2", classification, draft,
            {"should_escalate": True, "reason": "low confidence"},
        )
        await repo.save_triage_result(
            "m3", classification, draft,
            {"should_escalate": True, "reason": "complex issue"},
        )

        await repo.update_ticket_status("m1", "resolved")
        await repo.update_ticket_status("m2", "escalated")

        step = TraceStep(step="classify", status="completed", duration_ms=100)
        await repo.add_trace_step("m1", step)
        await repo.add_trace_step("m2", step)
        await repo.add_trace_step("m3", step)

        metrics = await repo.get_dashboard_metrics()

        assert metrics.total_tickets == 3
        assert metrics.escalated_tickets == 2  # m2 and m3 (should_escalate=True)
        assert metrics.processed_tickets == 3  # all three are classified/escalated/resolved
        assert metrics.success_rate == pytest.approx(1 / 3, abs=0.01)
        assert metrics.escalation_rate == pytest.approx(2 / 3, abs=0.01)
        assert metrics.avg_confidence == pytest.approx(0.9, abs=0.01)
        assert metrics.avg_steps == 1.0
        assert "bug" in metrics.category_distribution
        assert "high" in metrics.urgency_distribution
        assert len(metrics.recent_activity) > 0

    async def test_metrics_returns_dashboard_metrics_schema(self, repo: TicketRepository):
        metrics = await repo.get_dashboard_metrics()
        assert isinstance(metrics, DashboardMetrics)


# ═══════════════════════════════════════════════════════════════════════════════
# add_trace_step
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAddTraceStep:
    async def test_adds_step(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        step = TraceStep(
            step="classify",
            status="completed",
            duration_ms=120,
            data={"category": "bug"},
        )
        await repo.add_trace_step("test-001", step)

        trace = await repo.get_ticket_trace("test-001")
        assert trace is not None
        assert len(trace["steps"]) == 1
        assert trace["steps"][0]["step"] == "classify"
        assert trace["steps"][0]["duration_ms"] == 120

    async def test_multiple_steps(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        for s in ["classify", "retrieve", "draft"]:
            step = TraceStep(step=s, status="completed")
            await repo.add_trace_step("test-001", step)

        trace = await repo.get_ticket_trace("test-001")
        assert len(trace["steps"]) == 3
        names = [s["step"] for s in trace["steps"]]
        assert names == ["classify", "retrieve", "draft"]

    async def test_error_stored(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        step = TraceStep(
            step="retrieve",
            status="failed",
            error="Connection timeout to vector store",
        )
        await repo.add_trace_step("test-001", step)

        trace = await repo.get_ticket_trace("test-001")
        assert trace["steps"][0]["error"] == "Connection timeout to vector store"

    async def test_status_stored(self, repo: TicketRepository):
        await repo.create_ticket(_make_ticket())
        step = TraceStep(step="draft", status="running")
        await repo.add_trace_step("test-001", step)

        trace = await repo.get_ticket_trace("test-001")
        assert trace["steps"][0]["status"] == "running"
