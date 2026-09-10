"""Integration tests — spin up mocked services, run full agent on test tickets,
assert status and trace.

These tests mock external APIs (LLM, Qdrant) but exercise the full agent graph
pipeline end-to-end with real database operations.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from src.agent.graph import AgentExecutor, run_triage
from src.models.schemas import (
    DraftResponse,
    Ticket,
    TicketCategory,
    TicketClassification,
    UrgencyLevel,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Full agent pipeline with mocked LLM/Qdrant
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentPipelineIntegration:
    """Run the full agent graph with mocked external services."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_pipeline_resolves_ticket(self):
        """Full pipeline: classify → retrieve → draft → escalate_check → finalize."""
        ticket = Ticket(
            id="integ-001",
            content="Subject: Login error\n\nBody:\nPage crashes on login.",
            source="api",
        )

        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.92,
            reasoning="Clear bug.",
        )
        mock_docs = [
            MagicMock(
                id="doc-1",
                content="Fix for login crash",
                metadata={"topic": "login"},
                similarity_score=0.88,
                source="past_ticket",
            )
        ]
        mock_draft = DraftResponse(
            draft_text="We are investigating the login issue.",
            confidence=0.85,
            reasoning="Bug response.",
        )

        with (
            patch("src.agent.graph.get_classifier") as mock_class,
            patch("src.agent.graph.get_retriever") as mock_ret,
            patch("src.agent.graph.get_draft_generator") as mock_draft_gen,
            patch("src.agent.graph.TicketRepository") as MockRepo,
        ):
            mock_classifier = MagicMock()
            mock_classifier.classify = AsyncMock(return_value=mock_classification)
            mock_class.return_value = mock_classifier

            mock_retriever = MagicMock()
            mock_retriever.search = AsyncMock(return_value=mock_docs)
            mock_ret.return_value = mock_retriever

            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_draft)
            mock_draft_gen.return_value = mock_gen

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            mock_repo.update_ticket_status = AsyncMock()
            MockRepo.return_value = mock_repo

            executor = AgentExecutor()
            final_state = await executor.run(ticket)

            # Assert final state
            assert final_state["classification"] is not None
            assert final_state["classification"].category == TicketCategory.BUG
            assert final_state["draft"] is not None
            assert final_state["should_escalate"] is False
            assert final_state["trace"] is not None
            assert len(final_state["trace"].steps) > 0

            # Assert trace has all expected steps
            step_names = [s.step for s in final_state["trace"].steps]
            assert "classify" in step_names
            assert "retrieve" in step_names
            assert "draft" in step_names
            assert "escalate_check" in step_names
            assert "finalize" in step_names

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_pipeline_escalates_ticket(self):
        """Full pipeline when low confidence triggers escalation."""
        ticket = Ticket(
            id="integ-002",
            content="Subject: Something weird\n\nBody:\nNot sure what this is about.",
            source="api",
        )

        low_conf_class = TicketClassification(
            category=TicketCategory.OTHER,
            urgency=UrgencyLevel.LOW,
            confidence=0.30,
            reasoning="Very unclear ticket.",
        )
        low_draft = DraftResponse(
            draft_text="We are looking into your request.",
            confidence=0.30,
            reasoning="Low confidence.",
        )

        with (
            patch("src.agent.graph.get_classifier") as mock_class,
            patch("src.agent.graph.get_retriever") as mock_ret,
            patch("src.agent.graph.get_draft_generator") as mock_draft_gen,
            patch("src.agent.graph.TicketRepository") as MockRepo,
        ):
            mock_classifier = MagicMock()
            mock_classifier.classify = AsyncMock(return_value=low_conf_class)
            mock_class.return_value = mock_classifier

            mock_retriever = MagicMock()
            mock_retriever.search = AsyncMock(return_value=[])
            mock_ret.return_value = mock_retriever

            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=low_draft)
            mock_draft_gen.return_value = mock_gen

            mock_repo = MagicMock()
            mock_repo.update_ticket_status = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            MockRepo.return_value = mock_repo

            executor = AgentExecutor()
            final_state = await executor.run(ticket)

            assert final_state["should_escalate"] is True
            assert final_state["escalation"] is not None
            assert final_state["escalation"].should_escalate is True

            # Verify escalate node was called (trace should have it)
            step_names = [s.step for s in final_state["trace"].steps]
            assert "escalate" in step_names

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_with_no_retrieved_docs(self):
        """Pipeline handles empty retrieval results gracefully."""
        ticket = Ticket(
            id="integ-003",
            content="Subject: Unique issue\n\nBody:\nSomething never seen before.",
            source="api",
        )

        mock_classification = TicketClassification(
            category=TicketCategory.OTHER,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.75,
            reasoning="Unclear issue.",
        )
        mock_draft = DraftResponse(
            draft_text="Thank you for your report. We'll look into it.",
            confidence=0.70,
            reasoning="No context available.",
        )

        with (
            patch("src.agent.graph.get_classifier") as mock_class,
            patch("src.agent.graph.get_retriever") as mock_ret,
            patch("src.agent.graph.get_draft_generator") as mock_draft_gen,
            patch("src.agent.graph.TicketRepository") as MockRepo,
        ):
            mock_classifier = MagicMock()
            mock_classifier.classify = AsyncMock(return_value=mock_classification)
            mock_class.return_value = mock_classifier

            mock_retriever = MagicMock()
            mock_retriever.search = AsyncMock(return_value=[])
            mock_ret.return_value = mock_retriever

            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_draft)
            mock_draft_gen.return_value = mock_gen

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            MockRepo.return_value = mock_repo

            executor = AgentExecutor()
            final_state = await executor.run(ticket)

            assert final_state["retrieved_docs"] == []
            assert final_state["draft"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: run_triage convenience function
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunTriageIntegration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_triage_returns_summary(self):
        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.90,
            reasoning="Bug.",
        )
        mock_draft = DraftResponse(
            draft_text="We are investigating.",
            confidence=0.85,
            reasoning="Standard.",
        )

        with (
            patch("src.agent.graph.get_classifier") as mock_class,
            patch("src.agent.graph.get_retriever") as mock_ret,
            patch("src.agent.graph.get_draft_generator") as mock_draft_gen,
            patch("src.agent.graph.TicketRepository") as MockRepo,
        ):
            mock_classifier = MagicMock()
            mock_classifier.classify = AsyncMock(return_value=mock_classification)
            mock_class.return_value = mock_classifier

            mock_retriever = MagicMock()
            mock_retriever.search = AsyncMock(return_value=[])
            mock_ret.return_value = mock_retriever

            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_draft)
            mock_draft_gen.return_value = mock_gen

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            MockRepo.return_value = mock_repo

            result = await run_triage(
                ticket_id="rt-001",
                subject="Login error",
                body="Page crashes on login.",
            )

            assert result["ticket_id"] == "rt-001"
            assert result["category"] == "bug"
            assert result["urgency"] == "high"
            assert result["decision"] == "complete"
            assert result["latency_ms"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: API → Agent → DB flow
# ═══════════════════════════════════════════════════════════════════════════════


class TestAPIAgentDBFlow:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_and_triage_ticket(self, client: AsyncClient):
        """POST ticket → verify DB row created."""
        resp = await client.post(
            "/tickets/",
            json={"content": "Integration test ticket", "source": "api"},
        )
        assert resp.status_code == 201
        ticket_id = resp.json()["ticket_id"]

        # Verify ticket in DB via status endpoint
        status_resp = await client.get(f"/tickets/{ticket_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["ticket_id"] == ticket_id
        assert status_resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_escalate_and_list(self, client: AsyncClient):
        """Create ticket → escalate → verify it appears in escalated list."""
        # Create
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Integration escalation test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Escalate
        esc_resp = await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": "Manual escalation for integration test"},
        )
        assert esc_resp.status_code == 200

        # List escalated
        list_resp = await client.get("/tickets/escalated/list")
        assert list_resp.status_code == 200
        escalated_ids = [t["ticket_id"] for t in list_resp.json()["tickets"]]
        assert ticket_id in escalated_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Agent state transitions
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentStateTransitions:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_carries_through_pipeline(self):
        """Verify state is correctly accumulated across nodes."""
        ticket = Ticket(id="state-001", content="Test state flow", source="api")

        mock_class = TicketClassification(
            category=TicketCategory.BILLING,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.80,
            reasoning="Billing issue.",
        )
        mock_doc = MagicMock(
            id="doc-1",
            content="Billing help",
            metadata={},
            similarity_score=0.75,
            source="knowledge_base",
        )
        mock_draft = DraftResponse(
            draft_text="Regarding your billing question...",
            confidence=0.80,
            reasoning="Billing response.",
        )

        with (
            patch("src.agent.graph.get_classifier") as mc,
            patch("src.agent.graph.get_retriever") as mr,
            patch("src.agent.graph.get_draft_generator") as md,
            patch("src.agent.graph.TicketRepository") as MockRepo,
        ):
            mc_val = MagicMock()
            mc_val.classify = AsyncMock(return_value=mock_class)
            mc.return_value = mc_val

            mr_val = MagicMock()
            mr_val.search = AsyncMock(return_value=[mock_doc])
            mr.return_value = mr_val

            md_val = MagicMock()
            md_val.generate = AsyncMock(return_value=mock_draft)
            md.return_value = md_val

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            MockRepo.return_value = mock_repo

            executor = AgentExecutor()
            state = await executor.run(ticket)

            # Verify classification carried through
            assert state["classification"].category == TicketCategory.BILLING
            assert state["classification"].urgency == UrgencyLevel.MEDIUM

            # Verify retrieved docs carried through
            assert len(state["retrieved_docs"]) == 1
            assert state["retrieved_docs"][0]["id"] == "doc-1"

            # Verify draft carried through
            assert "billing" in state["draft"].draft_text.lower()

            # Verify trace has all steps
            step_names = [s.step for s in state["trace"].steps]
            assert "classify" in step_names
            assert "retrieve" in step_names
            assert "draft" in step_names
            assert "escalate_check" in step_names
            assert "finalize" in step_names
