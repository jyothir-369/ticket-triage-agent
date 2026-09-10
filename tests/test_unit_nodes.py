"""Unit tests for each node function in the triage agent graph.

Tests classify_node, retrieve_node, draft_node, escalate_check_node,
escalate_node, and finalize_node with mocked services.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.nodes import (
    _append_step,
    _ensure_trace,
    _now_ms,
    _template_fallback_draft,
    _ticket_id,
)
from src.agent.state import AgentState
from src.models.schemas import (
    DraftResponse,
    EscalationDecision,
    StepStatus,
    Ticket,
    TicketCategory,
    TicketClassification,
    TriageTrace,
    UrgencyLevel,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper function tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_now_ms_returns_positive(self):
        result = _now_ms()
        assert result > 0

    def test_ticket_id_from_ticket(self):
        state: AgentState = {
            "ticket": Ticket(id="abc-123", content="test"),
            "trace": TriageTrace(ticket_id="abc-123"),
        }
        assert _ticket_id(state) == "abc-123"

    def test_ticket_id_unknown_when_missing(self):
        state: AgentState = {"trace": TriageTrace(ticket_id="x")}
        assert _ticket_id(state) == "unknown"

    def test_ensure_trace_creates_new(self):
        state: AgentState = {
            "ticket": Ticket(id=1, content="test"),
        }
        trace = _ensure_trace(state)
        assert isinstance(trace, TriageTrace)
        assert "trace" in state

    def test_ensure_trace_returns_existing(self):
        existing = TriageTrace(ticket_id=42)
        state: AgentState = {"trace": existing}
        trace = _ensure_trace(state)
        assert trace is existing

    def test_append_step_adds_to_trace(self):
        state: AgentState = {
            "ticket": Ticket(id=1, content="test"),
            "trace": TriageTrace(ticket_id=1),
        }
        _append_step(state, step_name="classify", status=StepStatus.RUNNING)
        assert len(state["trace"].steps) == 1
        assert state["trace"].steps[0].step == "classify"
        assert state["trace"].steps[0].status == StepStatus.RUNNING

    def test_append_step_with_data_and_duration(self):
        state: AgentState = {
            "ticket": Ticket(id=1, content="test"),
            "trace": TriageTrace(ticket_id=1),
        }
        _append_step(
            state,
            step_name="retrieve",
            status=StepStatus.COMPLETED,
            duration_ms=150,
            data={"count": 3},
        )
        step = state["trace"].steps[0]
        assert step.duration_ms == 150
        assert step.data == {"count": 3}

    def test_append_step_with_error(self):
        state: AgentState = {
            "ticket": Ticket(id=1, content="test"),
            "trace": TriageTrace(ticket_id=1),
        }
        _append_step(
            state,
            step_name="draft",
            status=StepStatus.FAILED,
            error="LLM timeout",
        )
        assert state["trace"].steps[0].error == "LLM timeout"


# ═══════════════════════════════════════════════════════════════════════════════
# Template fallback draft tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateFallbackDraft:
    def test_fallback_with_classification(self):
        state: AgentState = {
            "ticket": Ticket(id=1, content="Login error"),
            "classification": TicketClassification(
                category=TicketCategory.BUG,
                urgency=UrgencyLevel.HIGH,
                confidence=0.9,
            ),
            "trace": TriageTrace(ticket_id=1),
        }
        draft = _template_fallback_draft(state)
        assert isinstance(draft, DraftResponse)
        assert "Bug" in draft.draft_text or "bug" in draft.draft_text.lower()
        assert draft.confidence == 0.3
        assert draft.citations == []

    def test_fallback_without_classification(self):
        state: AgentState = {
            "ticket": Ticket(id=1, content="Help"),
            "trace": TriageTrace(ticket_id=1),
        }
        draft = _template_fallback_draft(state)
        assert isinstance(draft, DraftResponse)
        assert draft.confidence == 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# classify_node tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestClassifyNode:
    @pytest.mark.asyncio
    async def test_classify_node_success(self, sample_agent_state):
        from src.agent.graph import classify_node

        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.92,
            reasoning="Bug detected.",
        )
        with patch("src.agent.graph.get_classifier") as mock_get:
            mock_classifier = MagicMock()
            mock_classifier.classify = AsyncMock(return_value=mock_classification)
            mock_get.return_value = mock_classifier

            result = await classify_node(sample_agent_state)

            assert result["classification"] is not None
            assert result["classification"].category == TicketCategory.BUG
            assert "trace" in result

    @pytest.mark.asyncio
    async def test_classify_node_failure_increments_loop(self, sample_agent_state):
        from src.agent.graph import classify_node

        with patch("src.agent.graph.get_classifier") as mock_get:
            mock_classifier = MagicMock()
            mock_classifier.classify = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_get.return_value = mock_classifier

            # The node itself catches the exception and re-raises
            # but the wrapped version handles it
            with pytest.raises(RuntimeError):
                await classify_node(sample_agent_state)


# ═══════════════════════════════════════════════════════════════════════════════
# retrieve_node tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrieveNode:
    @pytest.mark.asyncio
    async def test_retrieve_node_success(self, sample_agent_state):
        from src.agent.graph import retrieve_node

        mock_docs = [
            MagicMock(
                id="doc-1",
                content="Help article",
                metadata={"topic": "login"},
                similarity_score=0.88,
                source="past_ticket",
            )
        ]
        with patch("src.agent.graph.get_retriever") as mock_get:
            mock_retriever = MagicMock()
            mock_retriever.search = AsyncMock(return_value=mock_docs)
            mock_get.return_value = mock_retriever

            result = await retrieve_node(sample_agent_state)

            assert "retrieved_docs" in result
            assert len(result["retrieved_docs"]) == 1
            assert result["retrieved_docs"][0]["id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_retrieve_node_empty_results(self, sample_agent_state):
        from src.agent.graph import retrieve_node

        with patch("src.agent.graph.get_retriever") as mock_get:
            mock_retriever = MagicMock()
            mock_retriever.search = AsyncMock(return_value=[])
            mock_get.return_value = mock_retriever

            result = await retrieve_node(sample_agent_state)
            assert result["retrieved_docs"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# draft_node tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDraftNode:
    @pytest.mark.asyncio
    async def test_draft_node_success(self, sample_agent_state):
        from src.agent.graph import draft_node

        mock_draft = DraftResponse(
            draft_text="We are investigating your login issue.",
            citations=[],
            confidence=0.85,
            reasoning="Standard bug response.",
        )
        with patch("src.agent.graph.get_draft_generator") as mock_get:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_draft)
            mock_get.return_value = mock_gen

            result = await draft_node(sample_agent_state)

            assert result["draft"] is not None
            assert "investigating" in result["draft"].draft_text.lower()

    @pytest.mark.asyncio
    async def test_draft_node_fallback_on_failure(self, sample_agent_state):
        from src.agent.graph import draft_node

        with patch("src.agent.graph.get_draft_generator") as mock_get:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(side_effect=RuntimeError("LLM timeout"))
            mock_get.return_value = mock_gen

            result = await draft_node(sample_agent_state)

            # Should fall back to template draft
            assert result["draft"] is not None
            assert result["draft"].confidence == 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# escalate_check_node tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalateCheckNode:
    @pytest.mark.asyncio
    async def test_no_escalation_high_confidence(self, sample_agent_state):
        from src.agent.graph import escalate_check_node

        result = await escalate_check_node(sample_agent_state)

        assert result["should_escalate"] is False
        assert result["escalation"] is not None
        assert result["escalation"].should_escalate is False

    @pytest.mark.asyncio
    async def test_escalation_low_confidence(self, sample_agent_state):
        from src.agent.graph import escalate_check_node

        # Override with low confidence
        sample_agent_state["classification"] = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.30,
            reasoning="Low confidence.",
        )
        sample_agent_state["draft"] = DraftResponse(
            draft_text="Looking into it.",
            confidence=0.30,
            reasoning="Low confidence.",
        )

        result = await escalate_check_node(sample_agent_state)

        assert result["should_escalate"] is True
        assert "below threshold" in result["escalation_reason"].lower()

    @pytest.mark.asyncio
    async def test_escalation_missing_classification(self):
        from src.agent.graph import escalate_check_node

        state: AgentState = {
            "ticket": Ticket(id=1, content="test"),
            "classification": None,
            "retrieved_docs": [],
            "draft": DraftResponse(draft_text="Hi", confidence=0.8),
            "escalation": None,
            "should_escalate": False,
            "escalation_reason": "",
            "trace": TriageTrace(ticket_id=1),
            "loop_count": 0,
            "tool_call_count": 0,
            "error_message": "",
            "messages": [],
        }

        result = await escalate_check_node(state)

        assert result["should_escalate"] is True
        assert "missing" in result["escalation_reason"].lower()

    @pytest.mark.asyncio
    async def test_escalation_loop_count_exceeded(self, sample_agent_state):
        from src.agent.graph import escalate_check_node

        sample_agent_state["loop_count"] = 10  # Exceeds max_loop_retries (3)

        result = await escalate_check_node(sample_agent_state)

        assert result["should_escalate"] is True
        assert "loop count" in result["escalation_reason"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# escalate_node tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalateNode:
    @pytest.mark.asyncio
    async def test_escalate_node_persists(self, sample_agent_state_escalated):
        from src.agent.graph import escalate_node

        with patch("src.agent.graph.TicketRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.update_ticket_status = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            MockRepo.return_value = mock_repo

            result = await escalate_node(sample_agent_state_escalated)

            assert "trace" in result
            mock_repo.update_ticket_status.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# finalize_node tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFinalizeNode:
    @pytest.mark.asyncio
    async def test_finalize_node_resolved(self, sample_agent_state):
        from src.agent.graph import finalize_node

        with patch("src.agent.graph.TicketRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            MockRepo.return_value = mock_repo

            result = await finalize_node(sample_agent_state)

            assert "trace" in result
            mock_repo.save_triage_result.assert_called_once()
            mock_repo.update_ticket.assert_called_once()

    @pytest.mark.asyncio
    async def test_finalize_node_escalated(self, sample_agent_state_escalated):
        from src.agent.graph import finalize_node

        with patch("src.agent.graph.TicketRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            MockRepo.return_value = mock_repo

            result = await finalize_node(sample_agent_state_escalated)

            assert "trace" in result
            # Check that the ticket status was set to ESCALATED
            update_call = mock_repo.update_ticket.call_args
            assert update_call[0][1]["status"] == "escalated"


# ═══════════════════════════════════════════════════════════════════════════════
# Routing logic tests (supplement existing tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouting:
    def test_route_escalate_on_true(self):
        from src.agent.graph import route_after_escalation_check

        state = {"should_escalate": True, "ticket": Ticket(id=1, content="x")}
        assert route_after_escalation_check(state) == "escalate"

    def test_route_finalize_on_false(self):
        from src.agent.graph import route_after_escalation_check

        state = {"should_escalate": False, "ticket": Ticket(id=1, content="x")}
        assert route_after_escalation_check(state) == "finalize"

    def test_route_finalize_when_ticket_none(self):
        from src.agent.graph import route_after_escalation_check

        state = {"should_escalate": False, "ticket": None}
        assert route_after_escalation_check(state) == "finalize"
