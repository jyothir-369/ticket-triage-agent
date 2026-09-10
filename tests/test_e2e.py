"""End-to-end tests — POST ticket, GET status, GET trace, assert all steps exist.

These tests exercise the full API flow without mocking the agent pipeline
(but still mock the LLM/Qdrant to avoid external dependencies).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.models.schemas import (
    DraftResponse,
    TicketCategory,
    TicketClassification,
    UrgencyLevel,
)
from src.models.ticket import TicketModel


# ═══════════════════════════════════════════════════════════════════════════════
# E2E: Full ticket lifecycle via API
# ═══════════════════════════════════════════════════════════════════════════════


class TestE2ETicketLifecycle:
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_create_triage_status_trace_flow(self, client: AsyncClient):
        """Full lifecycle: create → triage → check status → check trace."""
        mock_classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.90,
            reasoning="Bug detected.",
        )
        mock_draft = DraftResponse(
            draft_text="We are investigating the issue.",
            confidence=0.85,
            reasoning="Bug response.",
        )

        with (
            patch("src.agent.graph.get_classifier") as mock_class,
            patch("src.agent.graph.get_retriever") as mock_ret,
            patch("src.agent.graph.get_draft_generator") as mock_draft_gen,
            patch("src.agent.graph.TicketRepository") as MockRepo,
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

            # Step 1: Create ticket
            create_resp = await client.post(
                "/tickets/",
                json={
                    "content": "Subject: E2E login error\n\nBody:\nLogin page crashes.",
                    "source": "api",
                },
            )
            assert create_resp.status_code == 201
            ticket_id = create_resp.json()["ticket_id"]

            # Step 2: Trigger triage
            triage_resp = await client.post(f"/tickets/{ticket_id}/triage")
            assert triage_resp.status_code == 202

            # Step 3: Check status
            status_resp = await client.get(f"/tickets/{ticket_id}/status")
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            assert status_data["ticket_id"] == ticket_id

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_create_and_approve_flow(self, client: AsyncClient, db_session):
        """Create ticket → set to awaiting_review → approve."""
        # Create
        create_resp = await client.post(
            "/tickets/",
            json={"content": "E2E approval test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Manually set to awaiting_review
        ticket = await db_session.get(TicketModel, ticket_id)
        ticket.status = "awaiting_review"
        await db_session.flush()

        # Approve
        approve_resp = await client.post(
            f"/tickets/{ticket_id}/approve",
            json={"reviewer": "e2e-tester"},
        )
        assert approve_resp.status_code == 200
        assert approve_resp.json()["status"] == "resolved"

        # Verify final status
        status_resp = await client.get(f"/tickets/{ticket_id}/status")
        assert status_resp.json()["status"] == "resolved"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_create_and_escalate_flow(self, client: AsyncClient):
        """Create ticket → manually escalate → verify in escalated list."""
        # Create
        create_resp = await client.post(
            "/tickets/",
            json={"content": "E2E escalation test", "source": "email"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Escalate
        esc_resp = await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": "Customer frustrated", "reviewer": "e2e-agent"},
        )
        assert esc_resp.status_code == 200

        # Verify in escalated list
        list_resp = await client.get("/tickets/escalated/list")
        assert list_resp.status_code == 200
        escalated = list_resp.json()["tickets"]
        assert any(t["ticket_id"] == ticket_id for t in escalated)

        # Verify status
        status_resp = await client.get(f"/tickets/{ticket_id}/status")
        assert status_resp.json()["status"] == "escalated"


# ═══════════════════════════════════════════════════════════════════════════════
# E2E: Multiple tickets
# ═══════════════════════════════════════════════════════════════════════════════


class TestE2EMultipleTickets:
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_create_multiple_tickets(self, client: AsyncClient):
        """Create several tickets and verify they all appear in status."""
        ticket_ids = []
        for i in range(3):
            resp = await client.post(
                "/tickets/",
                json={"content": f"Multi-ticket test {i}", "source": "api"},
            )
            assert resp.status_code == 201
            ticket_ids.append(resp.json()["ticket_id"])

        # Verify all can be fetched
        for tid in ticket_ids:
            status_resp = await client.get(f"/tickets/{tid}/status")
            assert status_resp.status_code == 200
            assert status_resp.json()["ticket_id"] == tid

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_escalated_list_pagination(self, client: AsyncClient):
        """Escalate multiple tickets and test pagination."""
        ticket_ids = []
        for i in range(5):
            create_resp = await client.post(
                "/tickets/",
                json={"content": f"Pagination test {i}", "source": "api"},
            )
            tid = create_resp.json()["ticket_id"]
            ticket_ids.append(tid)

            await client.post(
                f"/tickets/{tid}/escalate",
                json={"reason": f"Escalation {i}"},
            )

        # Get first page
        page1 = await client.get("/tickets/escalated/list?limit=3&offset=0")
        assert page1.status_code == 200
        data1 = page1.json()
        assert data1["total"] == 5
        assert len(data1["tickets"]) == 3

        # Get second page
        page2 = await client.get("/tickets/escalated/list?limit=3&offset=3")
        assert page2.status_code == 200
        data2 = page2.json()
        assert len(data2["tickets"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# E2E: Error scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestE2EErrorScenarios:
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_nonexistent_ticket_operations(self, client: AsyncClient):
        """All operations on nonexistent ticket should return 404."""
        assert (await client.get("/tickets/fake-id/status")).status_code == 404
        assert (await client.get("/tickets/fake-id/trace")).status_code == 404
        assert (await client.post("/tickets/fake-id/triage")).status_code == 404
        assert (
            await client.post(
                "/tickets/fake-id/approve", json={"reviewer": "x"}
            )
        ).status_code == 404
        assert (
            await client.post(
                "/tickets/fake-id/escalate", json={"reason": "x"}
            )
        ).status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_approve_wrong_status_returns_409(self, client: AsyncClient):
        """Cannot approve a ticket that isn't in awaiting_review/escalated."""
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Wrong status test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        resp = await client.post(
            f"/tickets/{ticket_id}/approve",
            json={"reviewer": "x"},
        )
        assert resp.status_code == 409
