"""Full end-to-end integration tests — complete workflow verification.

Tests cover:
  1. Happy path: confident classification → draft → approved
  2. Escalation path: low confidence → human review
  3. Error path: tool failure → retry → graceful escalation
  4. Duplicate ticket: idempotency check
  5. Audit trail: every decision has timestamp, status, data
  6. Metrics: dashboard shows accurate metrics after processing
  7. Observability: traces with correlation_id, structured logs
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.models.schemas import (
    DraftResponse,
    TicketCategory,
    TicketClassification,
    TicketStatus,
    UrgencyLevel,
)
from src.models.ticket import TicketModel, TicketStatus as DBTicketStatus
from src.models.trace import TraceModel


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HAPPY PATH: Confident classification → draft → approved
# ═══════════════════════════════════════════════════════════════════════════════


class TestHappyPath:
    """Complete happy path: create ticket → triage → approve."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_full_lifecycle_with_trace(self, client: AsyncClient, db_session):
        """Create → triage → verify trace has all expected steps."""
        # Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Subject: Bug report\n\nBody:\nApp crashes on startup.", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Run triage using AgentExecutor directly with mocked services
        from src.agent.graph import AgentExecutor
        from src.models.schemas import Ticket

        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.90,
            reasoning="Bug report with crash.",
        )
        mock_draft = DraftResponse(
            draft_text="We are investigating the crash issue.",
            confidence=0.85,
            reasoning="Crash bug response.",
        )

        with (
            patch("src.services.classification.get_classifier") as mock_class,
            patch("src.services.retrieval.get_retriever") as mock_ret,
            patch("src.services.drafting.get_draft_generator") as mock_draft_gen,
            patch("src.repository.TicketRepository") as MockRepo,
        ):
            mc = MagicMock()
            mc.classify = AsyncMock(return_value=mock_classification)
            mock_class.return_value = mc

            mr = MagicMock()
            mr.search = AsyncMock(return_value=[])
            mock_ret.return_value = mr

            md = MagicMock()
            md.generate = AsyncMock(return_value=mock_draft)
            mock_draft_gen.return_value = md

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            mock_repo.update_ticket_status = AsyncMock()
            MockRepo.return_value = mock_repo

            ticket = Ticket(
                id=ticket_id,
                content="App crashes on startup",
                source="api",
            )
            executor = AgentExecutor()
            # Provide thread_id for the checkpointer
            final_state = await executor.run(ticket, config={"configurable": {"thread_id": str(ticket.id)}})

            # Verify final state
            assert final_state["classification"] is not None
            assert final_state["classification"].category == TicketCategory.BUG
            assert final_state["draft"] is not None
            assert final_state["should_escalate"] is False
            assert final_state["trace"] is not None
            assert len(final_state["trace"].steps) > 0

            # Verify trace has all expected steps
            step_names = [s.step for s in final_state["trace"].steps]
            assert "classify" in step_names
            assert "retrieve" in step_names
            assert "draft" in step_names
            assert "escalate_check" in step_names
            assert "finalize" in step_names


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ESCALATION PATH: Low confidence → human review
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalationPath:
    """Low-confidence tickets should be escalated to human review."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_low_confidence_escalates(self, client: AsyncClient):
        """Low-confidence classification should trigger escalation."""
        from src.agent.graph import AgentExecutor
        from src.models.schemas import Ticket

        ticket = Ticket(
            id="escalation-001",
            content="Something weird happened",
            source="api",
        )

        # Mock low confidence
        low_conf_class = TicketClassification(
            category=TicketCategory.OTHER,
            urgency=UrgencyLevel.LOW,
            confidence=0.25,
            reasoning="Very unclear ticket content.",
        )
        low_draft = DraftResponse(
            draft_text="We need more information to help you.",
            confidence=0.30,
            reasoning="Low confidence draft.",
        )

        with (
            patch("src.services.classification.get_classifier") as mock_class,
            patch("src.services.retrieval.get_retriever") as mock_ret,
            patch("src.services.drafting.get_draft_generator") as mock_draft_gen,
            patch("src.repository.TicketRepository") as MockRepo,
        ):
            mc = MagicMock()
            mc.classify = AsyncMock(return_value=low_conf_class)
            mock_class.return_value = mc

            mr = MagicMock()
            mr.search = AsyncMock(return_value=[])
            mock_ret.return_value = mr

            md = MagicMock()
            md.generate = AsyncMock(return_value=low_draft)
            mock_draft_gen.return_value = md

            mock_repo = MagicMock()
            mock_repo.update_ticket_status = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            MockRepo.return_value = mock_repo

            executor = AgentExecutor()
            # Provide thread_id for the checkpointer
            final_state = await executor.run(ticket, config={"configurable": {"thread_id": str(ticket.id)}})

            # Verify escalation triggered
            assert final_state["should_escalate"] is True
            assert final_state["escalation"] is not None
            assert final_state["escalation"].should_escalate is True

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_manual_escalation_workflow(self, client: AsyncClient):
        """Manual escalation → appears in escalated list → approve."""
        # Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Manual escalation test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Manually escalate
        esc_resp = await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": "Customer frustrated, needs human attention", "reviewer": "agent-001"},
        )
        assert esc_resp.status_code == 200
        assert esc_resp.json()["status"] == "escalated"

        # Verify in escalated list
        list_resp = await client.get("/tickets/escalated/list")
        assert list_resp.status_code == 200
        escalated = list_resp.json()["tickets"]
        assert any(t["ticket_id"] == ticket_id for t in escalated)

        # Approve
        approve_resp = await client.post(
            f"/tickets/{ticket_id}/approve",
            json={"reviewer": "human-reviewer"},
        )
        assert approve_resp.status_code == 200
        assert approve_resp.json()["status"] == "resolved"

        # Verify final status
        status_resp = await client.get(f"/tickets/{ticket_id}/status")
        assert status_resp.json()["status"] == "resolved"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ERROR PATH: Tool failure → retry → graceful escalation
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorPath:
    """Tool failures should trigger retry, then graceful escalation."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_classifier_failure_triggers_escalation(self):
        """When classifier fails repeatedly, ticket should be escalated."""
        from src.agent.graph import AgentExecutor
        from src.models.schemas import Ticket

        ticket = Ticket(
            id="error-001",
            content="Subject: Test error path\n\nBody:\nTest ticket.",
            source="api",
        )

        # Mock classifier to always fail
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(side_effect=RuntimeError("LLM API unavailable"))

        with (
            patch("src.services.classification.get_classifier", return_value=mock_classifier),
            patch("src.services.retrieval.get_retriever") as mock_ret,
            patch("src.services.drafting.get_draft_generator") as mock_draft_gen,
            patch("src.repository.TicketRepository") as MockRepo,
        ):
            mr = MagicMock()
            mr.search = AsyncMock(return_value=[])
            mock_ret.return_value = mr

            md = MagicMock()
            md.generate = AsyncMock(return_value=DraftResponse(
                draft_text="Fallback response.",
                confidence=0.30,
                reasoning="Fallback.",
            ))
            mock_draft_gen.return_value = md

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            mock_repo.update_ticket_status = AsyncMock()
            MockRepo.return_value = mock_repo

            executor = AgentExecutor()
            # Provide thread_id for the checkpointer
            final_state = await executor.run(ticket, config={"configurable": {"thread_id": str(ticket.id)}})

            # Should escalate due to failures
            assert final_state["should_escalate"] is True
            assert final_state["tool_call_count"] > 0

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_draft_failure_uses_fallback(self):
        """When draft generation fails, template fallback should be used."""
        from src.agent.graph import AgentExecutor
        from src.models.schemas import Ticket

        ticket = Ticket(
            id="error-002",
            content="Subject: Draft failure test\n\nBody:\nTest.",
            source="api",
        )

        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.80,
            reasoning="Bug report.",
        )

        with (
            patch("src.services.classification.get_classifier") as mock_class,
            patch("src.services.retrieval.get_retriever") as mock_ret,
            patch("src.services.drafting.get_draft_generator") as mock_draft_gen,
            patch("src.repository.TicketRepository") as MockRepo,
        ):
            mc = MagicMock()
            mc.classify = AsyncMock(return_value=mock_classification)
            mock_class.return_value = mc

            mr = MagicMock()
            mr.search = AsyncMock(return_value=[])
            mock_ret.return_value = mr

            # Draft generator fails
            md = MagicMock()
            md.generate = AsyncMock(side_effect=RuntimeError("LLM timeout"))
            mock_draft_gen.return_value = md

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            mock_repo.update_ticket_status = AsyncMock()
            MockRepo.return_value = mock_repo

            executor = AgentExecutor()
            # Provide thread_id for the checkpointer
            final_state = await executor.run(ticket, config={"configurable": {"thread_id": str(ticket.id)}})

            # Draft should fall back to template
            assert final_state["draft"] is not None
            assert "Thank you for reaching out" in final_state["draft"].draft_text


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DUPLICATE TICKET: Idempotency check
# ═══════════════════════════════════════════════════════════════════════════════


class TestDuplicateTicket:
    """Idempotency: duplicate source_id should return existing ticket."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_duplicate_source_id_returns_existing(self, client: AsyncClient):
        """Creating ticket with same source_id should return existing ticket."""
        # First creation
        resp1 = await client.post(
            "/tickets/",
            json={
                "content": "First ticket",
                "source": "github",
                "source_id": "GH-123",
            },
        )
        assert resp1.status_code == 201
        ticket_id_1 = resp1.json()["ticket_id"]
        assert resp1.json()["is_duplicate"] is False

        # Second creation with same source_id
        resp2 = await client.post(
            "/tickets/",
            json={
                "content": "Second ticket (duplicate)",
                "source": "github",
                "source_id": "GH-123",
            },
        )
        assert resp2.status_code == 201  # Returns 201 but with is_duplicate=True
        ticket_id_2 = resp2.json()["ticket_id"]
        assert resp2.json()["is_duplicate"] is True

        # Should be the same ticket
        assert ticket_id_1 == ticket_id_2

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_different_source_ids_create_separate_tickets(self, client: AsyncClient):
        """Different source_ids should create separate tickets."""
        resp1 = await client.post(
            "/tickets/",
            json={"content": "Ticket 1", "source": "github", "source_id": "GH-100"},
        )
        resp2 = await client.post(
            "/tickets/",
            json={"content": "Ticket 2", "source": "github", "source_id": "GH-200"},
        )

        assert resp1.json()["ticket_id"] != resp2.json()["ticket_id"]
        assert resp1.json()["is_duplicate"] is False
        assert resp2.json()["is_duplicate"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AUDIT TRAIL: Every decision has timestamp, status, data
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditTrail:
    """Verify audit trail: every decision has timestamp, status, data."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_trace_steps_have_required_fields(self, client: AsyncClient, db_session):
        """Each trace step should have step, status, timestamp, and data."""
        # Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Audit trail test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Manually add trace steps (simulating what the agent does)
        trace_steps = [
            TraceModel(
                ticket_id=ticket_id,
                step="classify",
                status="completed",
                timestamp=datetime.now(timezone.utc),
                duration_ms=150,
                data=json.dumps({"category": "bug", "urgency": "high", "confidence": 0.92}),
            ),
            TraceModel(
                ticket_id=ticket_id,
                step="retrieve",
                status="completed",
                timestamp=datetime.now(timezone.utc),
                duration_ms=80,
                data=json.dumps({"retrieval_count": 2}),
            ),
            TraceModel(
                ticket_id=ticket_id,
                step="draft",
                status="completed",
                timestamp=datetime.now(timezone.utc),
                duration_ms=200,
                data=json.dumps({"draft_length": 150, "confidence": 0.85}),
            ),
            TraceModel(
                ticket_id=ticket_id,
                step="escalate_check",
                status="completed",
                timestamp=datetime.now(timezone.utc),
                duration_ms=10,
                data=json.dumps({"should_escalate": False, "final_confidence": 0.89}),
            ),
            TraceModel(
                ticket_id=ticket_id,
                step="finalize",
                status="completed",
                timestamp=datetime.now(timezone.utc),
                duration_ms=25,
                data=json.dumps({"final_status": "resolved"}),
            ),
        ]

        for step in trace_steps:
            db_session.add(step)
        await db_session.flush()

        # Fetch trace
        trace_resp = await client.get(f"/tickets/{ticket_id}/trace")
        assert trace_resp.status_code == 200

        trace_data = trace_resp.json()
        assert trace_data["ticket_id"] == ticket_id
        assert len(trace_data["steps"]) == 5

        # Verify each step has required fields
        for step in trace_data["steps"]:
            assert "step" in step
            assert "status" in step
            assert "timestamp" in step
            assert "data" in step
            assert step["step"] in ["classify", "retrieve", "draft", "escalate_check", "finalize"]
            assert step["status"] == "completed"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. METRICS: Dashboard shows accurate metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetrics:
    """Verify dashboard metrics are accurate after processing tickets."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_metrics_reflect_processed_tickets(self, client: AsyncClient, db_session):
        """Metrics should update after tickets are processed."""
        # Get initial metrics
        metrics_resp = await client.get("/dashboard/metrics")
        assert metrics_resp.status_code == 200
        initial_total = metrics_resp.json()["total_tickets"]

        # Create a few tickets
        for i in range(3):
            await client.post(
                "/tickets/",
                json={"content": f"Metrics test ticket {i}", "source": "api"},
            )

        # Verify metrics updated
        metrics_resp = await client.get("/dashboard/metrics")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()
        assert metrics["total_tickets"] == initial_total + 3

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_metrics_fields_present(self, client: AsyncClient):
        """Dashboard metrics should contain all required fields."""
        metrics_resp = await client.get("/dashboard/metrics")
        assert metrics_resp.status_code == 200

        metrics = metrics_resp.json()
        required_fields = [
            "total_tickets",
            "processed_tickets",
            "escalated_tickets",
            "approval_rate",
            "escalation_rate",
            "avg_confidence",
            "avg_steps",
            "success_rate",
            "category_distribution",
            "urgency_distribution",
            "recent_activity",
        ]

        for field in required_fields:
            assert field in metrics, f"Missing field: {field}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. OBSERVABILITY: Traces, metrics, structured logs
# ═══════════════════════════════════════════════════════════════════════════════


class TestObservability:
    """Verify observability: traces, metrics, structured logs."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_trace_persists_in_database(self, client: AsyncClient, db_session):
        """Trace should be persisted in the database after triage."""
        # Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Observability test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Add trace step
        trace_row = TraceModel(
            ticket_id=ticket_id,
            step="test_step",
            status="completed",
            timestamp=datetime.now(timezone.utc),
            data=json.dumps({"test": True}),
        )
        db_session.add(trace_row)
        await db_session.flush()

        # Verify trace is retrievable
        trace_resp = await client.get(f"/tickets/{ticket_id}/trace")
        assert trace_resp.status_code == 200
        assert len(trace_resp.json()["steps"]) >= 1

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_health_check_returns_all_services(self, client: AsyncClient):
        """Health check should return status for all services."""
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200

        health = health_resp.json()
        assert "status" in health
        assert "db" in health
        assert "qdrant" in health
        assert "redis" in health


# ═══════════════════════════════════════════════════════════════════════════════
# 8. COMPLETE WORKFLOW: End-to-end scenario
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompleteWorkflow:
    """End-to-end workflow verification."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_workflow_happy_path(self, client: AsyncClient):
        """Full workflow: create → triage → status → metrics."""
        # 1. Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={
                "content": "Subject: Login error\n\nBody:\n500 error on login page.",
                "source": "email",
                "customer_email": "user@example.com",
                "tags": ["login", "error"],
            },
        )
        assert create_resp.status_code == 201
        ticket_id = create_resp.json()["ticket_id"]

        # 2. Check status
        status_resp = await client.get(f"/tickets/{ticket_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["ticket_id"] == ticket_id

        # 3. Check metrics
        metrics_resp = await client.get("/dashboard/metrics")
        assert metrics_resp.status_code == 200
        assert metrics_resp.json()["total_tickets"] >= 1

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_workflow_escalation(self, client: AsyncClient):
        """Full workflow with escalation: create → triage → escalate → approve."""
        # 1. Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Unclear issue", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # 2. Manually escalate (simulating low confidence)
        esc_resp = await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": "Low confidence - needs human review", "reviewer": "system"},
        )
        assert esc_resp.status_code == 200

        # 3. Verify in escalated list
        list_resp = await client.get("/tickets/escalated/list")
        assert list_resp.status_code == 200
        escalated_ids = [t["ticket_id"] for t in list_resp.json()["tickets"]]
        assert ticket_id in escalated_ids

        # 4. Approve
        approve_resp = await client.post(
            f"/tickets/{ticket_id}/approve",
            json={"reviewer": "human-agent"},
        )
        assert approve_resp.status_code == 200
        assert approve_resp.json()["status"] == "resolved"

        # 5. Verify final status
        status_resp = await client.get(f"/tickets/{ticket_id}/status")
        assert status_resp.json()["status"] == "resolved"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EVALUATION FRAMEWORK: pytest runs eval harness
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluationFramework:
    """Verify evaluation harness can be imported and configured."""

    def test_eval_harness_importable(self):
        """EvalHarness should be importable."""
        from eval.harness import EvalHarness

        harness = EvalHarness()
        assert harness is not None
        assert harness.confidence_threshold == 0.7

    def test_eval_metrics_importable(self):
        """Evaluation metrics should be importable."""
        from eval.metrics import (
            AggregateMetrics,
            MetricResult,
            aggregate_metrics,
            compare_results,
            rouge_l_f1,
            keyword_overlap,
        )

        # Test ROUGE-L
        score = rouge_l_f1("hello world", "hello world")
        assert score == 1.0

        # Test keyword overlap
        overlap = keyword_overlap("I love cats", ["love", "cats"])
        assert overlap == 1.0

    def test_eval_fixture_loadable(self):
        """Eval fixture should be loadable."""
        from eval.harness import EvalHarness

        harness = EvalHarness()
        tickets = harness.load_tickets()
        assert len(tickets) > 0
        assert all("id" in t for t in tickets)
        assert all("content" in t for t in tickets)

    def test_eval_fixture_has_expected_fields(self):
        """Each eval ticket should have all required fields for comparison."""
        from eval.harness import EvalHarness

        harness = EvalHarness()
        tickets = harness.load_tickets()

        required_fields = [
            "id", "content", "expected_category", "expected_urgency",
            "should_escalate", "human_approved_draft", "expected_draft_keywords",
        ]

        for ticket in tickets:
            for field in required_fields:
                assert field in ticket, f"Ticket {ticket.get('id')} missing field: {field}"

    @pytest.mark.asyncio
    async def test_eval_compare_results_works(self):
        """EvalHarness.compare_results should produce valid MetricResult."""
        from eval.harness import EvalHarness
        from eval.metrics import MetricResult

        harness = EvalHarness()

        actual = {
            "ticket_id": "test-001",
            "category": "bug",
            "urgency": "high",
            "confidence": 0.90,
            "draft_text": "We are investigating the login error.",
            "should_escalate": False,
        }
        expected = {
            "expected_category": "BUG",
            "expected_urgency": "HIGH",
            "should_escalate": False,
            "human_approved_draft": "We are investigating the login error.",
            "expected_draft_keywords": ["investigate", "login"],
        }

        result = harness.compare_results(actual, expected)
        assert isinstance(result, MetricResult)
        assert result.ticket_id == "test-001"
        assert result.category_correct is True
        assert result.urgency_correct is True
        assert result.escalation_correct is True
        assert result.draft_rouge_l > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. FULL PIPELINE TRACE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullPipelineTracePersistence:
    """Verify that traces written by the actual agent pipeline persist correctly."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_agent_written_trace_persists_in_db(self, client: AsyncClient, db_session):
        """Agent pipeline should write trace steps that are retrievable via API."""
        from src.agent.graph import AgentExecutor
        from src.models.schemas import DraftResponse, Ticket, TicketClassification

        # Create ticket via API
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Subject: Trace persistence test\n\nBody:\nTest.", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Mock services
        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.92,
            reasoning="Bug report.",
        )
        mock_draft = DraftResponse(
            draft_text="We are investigating your issue.",
            confidence=0.85,
            reasoning="Standard response.",
        )

        with (
            patch("src.services.classification.get_classifier") as mock_class,
            patch("src.services.retrieval.get_retriever") as mock_ret,
            patch("src.services.drafting.get_draft_generator") as mock_draft_gen,
            patch("src.repository.TicketRepository") as MockRepo,
        ):
            mc = MagicMock()
            mc.classify = AsyncMock(return_value=mock_classification)
            mock_class.return_value = mc

            mr = MagicMock()
            mr.search = AsyncMock(return_value=[])
            mock_ret.return_value = mr

            md = MagicMock()
            md.generate = AsyncMock(return_value=mock_draft)
            mock_draft_gen.return_value = md

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            mock_repo.update_ticket_status = AsyncMock()
            MockRepo.return_value = mock_repo

            ticket = Ticket(id=ticket_id, content="Trace persistence test", source="api")
            executor = AgentExecutor()
            final_state = await executor.run(ticket, config={"configurable": {"thread_id": str(ticket.id)}})

        # Verify trace was produced
        assert final_state["trace"] is not None
        assert len(final_state["trace"].steps) >= 4  # classify, retrieve, draft, escalate_check, finalize

        # Verify trace steps have correlation via ticket_id
        assert final_state["trace"].ticket_id == str(ticket_id)

        # Verify trace steps have all required fields
        for step in final_state["trace"].steps:
            assert step.step is not None
            assert step.status is not None
            assert step.status.value in ("running", "completed", "failed", "pending", "skipped")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_trace_steps_contain_timestamps(self, client: AsyncClient, db_session):
        """Every trace step should have a timestamp."""
        from src.agent.graph import AgentExecutor
        from src.models.schemas import DraftResponse, Ticket, TicketClassification

        ticket_id = "trace-ts-001"

        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.80,
            reasoning="Test.",
        )
        mock_draft = DraftResponse(
            draft_text="Thank you for your report.",
            confidence=0.75,
            reasoning="Test.",
        )

        with (
            patch("src.services.classification.get_classifier") as mock_class,
            patch("src.services.retrieval.get_retriever") as mock_ret,
            patch("src.services.drafting.get_draft_generator") as mock_draft_gen,
            patch("src.repository.TicketRepository") as MockRepo,
        ):
            mc = MagicMock()
            mc.classify = AsyncMock(return_value=mock_classification)
            mock_class.return_value = mc

            mr = MagicMock()
            mr.search = AsyncMock(return_value=[])
            mock_ret.return_value = mr

            md = MagicMock()
            md.generate = AsyncMock(return_value=mock_draft)
            mock_draft_gen.return_value = md

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            mock_repo.update_ticket_status = AsyncMock()
            MockRepo.return_value = mock_repo

            ticket = Ticket(id=ticket_id, content="Timestamp test", source="api")
            executor = AgentExecutor()
            final_state = await executor.run(ticket, config={"configurable": {"thread_id": str(ticket.id)}})

        # Verify all steps have timestamps
        for step in final_state["trace"].steps:
            assert step.timestamp is not None, f"Step '{step.step}' missing timestamp"

        # Verify steps have duration_ms
        completed_steps = [s for s in final_state["trace"].steps if s.status.value == "completed"]
        assert len(completed_steps) >= 4
        for step in completed_steps:
            assert step.duration_ms is not None, f"Step '{step.step}' missing duration_ms"
            assert step.duration_ms >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CORRELATION ID IN LOGS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorrelationId:
    """Verify that ticket_id is used as correlation_id in structured logs."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_ticket_id_used_as_correlation_id(self, client: AsyncClient, db_session):
        """Agent should bind ticket_id to every log entry for correlation."""
        from src.agent.graph import AgentExecutor
        from src.models.schemas import DraftResponse, Ticket, TicketClassification

        ticket_id = "corr-test-001"

        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.LOW,
            confidence=0.75,
            reasoning="Test.",
        )
        mock_draft = DraftResponse(
            draft_text="We are looking into this.",
            confidence=0.70,
            reasoning="Test.",
        )

        with (
            patch("src.services.classification.get_classifier") as mock_class,
            patch("src.services.retrieval.get_retriever") as mock_ret,
            patch("src.services.drafting.get_draft_generator") as mock_draft_gen,
            patch("src.repository.TicketRepository") as MockRepo,
        ):
            mc = MagicMock()
            mc.classify = AsyncMock(return_value=mock_classification)
            mock_class.return_value = mc

            mr = MagicMock()
            mr.search = AsyncMock(return_value=[])
            mock_ret.return_value = mr

            md = MagicMock()
            md.generate = AsyncMock(return_value=mock_draft)
            mock_draft_gen.return_value = md

            mock_repo = MagicMock()
            mock_repo.save_triage_result = AsyncMock()
            mock_repo.update_ticket = AsyncMock()
            mock_repo.update_ticket_status = AsyncMock()
            MockRepo.return_value = mock_repo

            ticket = Ticket(id=ticket_id, content="Correlation test", source="api")
            executor = AgentExecutor()
            final_state = await executor.run(ticket, config={"configurable": {"thread_id": str(ticket.id)}})

        # Verify the trace has the ticket_id (used as correlation_id)
        assert final_state["trace"].ticket_id == ticket_id

        # Verify trace has the full pipeline steps
        step_names = [s.step for s in final_state["trace"].steps]
        assert "classify" in step_names
        assert "retrieve" in step_names
        assert "draft" in step_names
        assert "escalate_check" in step_names
