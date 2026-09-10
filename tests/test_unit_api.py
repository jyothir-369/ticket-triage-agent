"""Unit tests for API endpoints using TestClient.

Tests all ticket CRUD endpoints, triage triggering, status checks,
trace retrieval, approval, escalation, and dashboard metrics.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.models.ticket import TicketModel

# ═══════════════════════════════════════════════════════════════════════════════
# Health endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_liveness(self, client: AsyncClient):
        resp = await client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    @pytest.mark.asyncio
    async def test_readiness_success(self, client: AsyncClient):
        resp = await client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


# ═══════════════════════════════════════════════════════════════════════════════
# Ticket creation
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateTicket:
    @pytest.mark.asyncio
    async def test_create_ticket_success(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/",
            json={
                "content": "I have a login issue with my account",
                "source": "api",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "ticket_id" in data
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_ticket_with_metadata(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/",
            json={
                "content": "Billing question about my subscription",
                "source": "email",
                "customer_email": "test@example.com",
                "tags": ["billing", "urgent"],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_ticket_empty_content_rejected(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/",
            json={"content": "", "source": "api"},
        )
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_ticket_long_content(self, client: AsyncClient):
        long_content = "This is a detailed bug report. " * 1000
        resp = await client.post(
            "/tickets/",
            json={"content": long_content, "source": "api"},
        )
        assert resp.status_code == 201


# ═══════════════════════════════════════════════════════════════════════════════
# Triage triggering
# ═══════════════════════════════════════════════════════════════════════════════


class TestTriageTicket:
    @pytest.mark.asyncio
    async def test_triage_ticket_success(self, client: AsyncClient, db_session):
        # Create a ticket first
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Login error", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Trigger triage
        resp = await client.post(f"/tickets/{ticket_id}/triage")
        assert resp.status_code == 202
        assert resp.json()["status"] == "processing"

    @pytest.mark.asyncio
    async def test_triage_ticket_not_found(self, client: AsyncClient):
        resp = await client.post("/tickets/nonexistent/triage")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_triage_ticket_already_processing(self, client: AsyncClient, db_session):
        # Create a ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Set it to processing status
        ticket = await db_session.get(TicketModel, ticket_id)
        ticket.status = "processing"
        await db_session.flush()

        # Try to triage again
        resp = await client.post(f"/tickets/{ticket_id}/triage")
        assert resp.status_code == 409


# ═══════════════════════════════════════════════════════════════════════════════
# Ticket status
# ═══════════════════════════════════════════════════════════════════════════════


class TestTicketStatus:
    @pytest.mark.asyncio
    async def test_get_status_success(self, client: AsyncClient):
        # Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Test status", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        resp = await client.get(f"/tickets/{ticket_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticket_id"] == ticket_id
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, client: AsyncClient):
        resp = await client.get("/tickets/nonexistent/status")
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Trace retrieval
# ═══════════════════════════════════════════════════════════════════════════════


class TestTraceRetrieval:
    @pytest.mark.asyncio
    async def test_get_trace_not_found(self, client: AsyncClient):
        resp = await client.get("/tickets/nonexistent/trace")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_trace_no_steps(self, client: AsyncClient):
        # Create ticket but no triage yet
        create_resp = await client.post(
            "/tickets/",
            json={"content": "No triage yet", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        resp = await client.get(f"/tickets/{ticket_id}/trace")
        assert resp.status_code == 404
        assert "No trace data" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# Approval
# ═══════════════════════════════════════════════════════════════════════════════


class TestApproval:
    @pytest.mark.asyncio
    async def test_approve_ticket_success(self, client: AsyncClient, db_session):
        # Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Approval test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # Set status to awaiting_review
        ticket = await db_session.get(TicketModel, ticket_id)
        ticket.status = "awaiting_review"
        await db_session.flush()

        resp = await client.post(
            f"/tickets/{ticket_id}/approve",
            json={"reviewer": "admin"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_approve_ticket_wrong_status(self, client: AsyncClient, db_session):
        # Create ticket with pending status
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Wrong status", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        resp = await client.post(
            f"/tickets/{ticket_id}/approve",
            json={"reviewer": "admin"},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_approve_ticket_not_found(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/nonexistent/approve",
            json={"reviewer": "admin"},
        )
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Manual escalation
# ═══════════════════════════════════════════════════════════════════════════════


class TestManualEscalation:
    @pytest.mark.asyncio
    async def test_escalate_ticket_success(self, client: AsyncClient):
        # Create ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Escalation test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        resp = await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": "Customer is very unhappy", "reviewer": "agent1"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "escalated"

    @pytest.mark.asyncio
    async def test_escalate_ticket_not_found(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/nonexistent/escalate",
            json={"reason": "Test"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_escalate_ticket_empty_reason(self, client: AsyncClient):
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Empty reason", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        resp = await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": ""},
        )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# Escalated list
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalatedList:
    @pytest.mark.asyncio
    async def test_list_escalated_empty(self, client: AsyncClient):
        resp = await client.get("/tickets/escalated/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["tickets"] == []

    @pytest.mark.asyncio
    async def test_list_escalated_with_ticket(self, client: AsyncClient):
        # Create and escalate a ticket
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Escalated ticket", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": "Needs human review"},
        )

        resp = await client.get("/tickets/escalated/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_list_escalated_pagination(self, client: AsyncClient):
        resp = await client.get("/tickets/escalated/list?limit=5&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 5
        assert data["offset"] == 0
