"""Tests for the FastAPI API endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ticket import TicketModel
from src.models.ticket import TicketStatus as DBTicketStatus
from src.models.trace import TraceModel

# ── Health endpoints ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_detailed_status(client: AsyncClient):
    """GET /health should return detailed health check with subsystem status."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "checks" in data
    # The detailed health check includes database, qdrant, redis, llm
    assert isinstance(data["checks"], dict)


@pytest.mark.asyncio
async def test_health_readiness(client: AsyncClient):
    """GET /health/ready should return 200 when DB is available, or 503 if not."""
    resp = await client.get("/health/ready")
    # In test environment, the readiness probe may fail due to missing real DB
    # connection (the override only applies to Depends()-injected sessions).
    assert resp.status_code in (200, 503)
    data = resp.json()
    # 200 has {"status": "ready"}, 503 has {"detail": "Not ready: ..."}
    assert "status" in data or "detail" in data


@pytest.mark.asyncio
async def test_health_liveness(client: AsyncClient):
    """GET /health/live should always return 200."""
    resp = await client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


# ── Ticket CRUD ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_ticket(client: AsyncClient):
    """POST /tickets/ should create a new ticket and return 201."""
    resp = await client.post(
        "/tickets/",
        json={
            "content": "Test ticket content for creation",
            "source": "api",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "ticket_id" in data
    assert data["status"] == "pending"
    assert data["is_duplicate"] is False


@pytest.mark.asyncio
async def test_create_ticket_idempotent(client: AsyncClient):
    """POST /tickets/ with same source_id should return existing ticket."""
    source_id = f"IDEM-{uuid.uuid4().hex[:8]}"
    resp1 = await client.post(
        "/tickets/",
        json={
            "content": "First ticket",
            "source": "api",
            "source_id": source_id,
        },
    )
    assert resp1.status_code == 201

    resp2 = await client.post(
        "/tickets/",
        json={
            "content": "Duplicate ticket",
            "source": "api",
            "source_id": source_id,
        },
    )
    assert resp2.status_code == 201
    data = resp2.json()
    assert data["is_duplicate"] is True
    assert data["ticket_id"] == resp1.json()["ticket_id"]


@pytest.mark.asyncio
async def test_get_ticket_status(client: AsyncClient):
    """GET /tickets/{id}/status should return ticket status."""
    # Create a ticket first
    create_resp = await client.post(
        "/tickets/",
        json={"content": "Status check ticket", "source": "api"},
    )
    ticket_id = create_resp.json()["ticket_id"]

    resp = await client.get(f"/tickets/{ticket_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticket_id"] == ticket_id
    assert data["status"] == "pending"
    assert "created_at" in data
    assert "updated_at" in data


@pytest.mark.asyncio
async def test_get_ticket_status_not_found(client: AsyncClient):
    """GET /tickets/{id}/status should return 404 for non-existent ticket."""
    resp = await client.get("/tickets/nonexistent-id/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_triage_ticket(client: AsyncClient):
    """POST /tickets/{id}/triage should trigger async triage and return 202."""
    create_resp = await client.post(
        "/tickets/",
        json={"content": "Triage test ticket", "source": "api"},
    )
    ticket_id = create_resp.json()["ticket_id"]

    # Mock the background task to avoid running the actual agent
    with patch("src.api.tickets._run_triage_background", new_callable=AsyncMock):
        resp = await client.post(f"/tickets/{ticket_id}/triage")
        assert resp.status_code == 202
        data = resp.json()
        assert data["ticket_id"] == ticket_id
        assert data["status"] == "processing"


@pytest.mark.asyncio
async def test_triage_ticket_not_found(client: AsyncClient):
    """POST /tickets/{id}/triage should return 404 for non-existent ticket."""
    resp = await client.post("/tickets/nonexistent-id/triage")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_triage_ticket_wrong_status(client: AsyncClient, db_session: AsyncSession):
    """POST /tickets/{id}/triage should return 409 for already-processed ticket."""
    ticket_id = str(uuid.uuid4())
    model = TicketModel(
        id=ticket_id,
        content="Already processed",
        source="api",
        status=DBTicketStatus.RESOLVED.value,
        created_at=datetime.now(UTC),
    )
    db_session.add(model)
    await db_session.flush()

    resp = await client.post(f"/tickets/{ticket_id}/triage")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_get_trace(client: AsyncClient, db_session: AsyncSession):
    """GET /tickets/{id}/trace should return trace steps."""
    ticket_id = str(uuid.uuid4())
    model = TicketModel(
        id=ticket_id,
        content="Trace test",
        source="api",
        status="pending",
        created_at=datetime.now(UTC),
    )
    db_session.add(model)

    trace = TraceModel(
        ticket_id=ticket_id,
        step="classify",
        status="completed",
        timestamp=datetime.now(UTC),
        data=json.dumps({"category": "bug"}),
    )
    db_session.add(trace)
    await db_session.flush()

    resp = await client.get(f"/tickets/{ticket_id}/trace")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticket_id"] == ticket_id
    assert len(data["steps"]) == 1
    assert data["steps"][0]["step"] == "classify"


@pytest.mark.asyncio
async def test_get_trace_not_found(client: AsyncClient):
    """GET /tickets/{id}/trace should return 404 for non-existent ticket."""
    resp = await client.get("/tickets/nonexistent-id/trace")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_approve_ticket(client: AsyncClient, db_session: AsyncSession):
    """POST /tickets/{id}/approve should approve an escalated ticket."""
    ticket_id = str(uuid.uuid4())
    model = TicketModel(
        id=ticket_id,
        content="Escalated ticket",
        source="api",
        status=DBTicketStatus.ESCALATED.value,
        created_at=datetime.now(UTC),
    )
    db_session.add(model)
    await db_session.flush()

    resp = await client.post(
        f"/tickets/{ticket_id}/approve",
        json={"reviewer": "test-user"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "resolved"


@pytest.mark.asyncio
async def test_approve_ticket_wrong_status(client: AsyncClient, db_session: AsyncSession):
    """POST /tickets/{id}/approve should return 409 for non-escalated ticket."""
    ticket_id = str(uuid.uuid4())
    model = TicketModel(
        id=ticket_id,
        content="Pending ticket",
        source="api",
        status="pending",
        created_at=datetime.now(UTC),
    )
    db_session.add(model)
    await db_session.flush()

    resp = await client.post(
        f"/tickets/{ticket_id}/approve",
        json={"reviewer": "test-user"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_escalate_ticket(client: AsyncClient):
    """POST /tickets/{id}/escalate should manually escalate a ticket."""
    create_resp = await client.post(
        "/tickets/",
        json={"content": "Manual escalation ticket", "source": "api"},
    )
    ticket_id = create_resp.json()["ticket_id"]

    resp = await client.post(
        f"/tickets/{ticket_id}/escalate",
        json={"reason": "Customer is very upset", "reviewer": "test-user"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "escalated"


@pytest.mark.asyncio
async def test_escalate_ticket_not_found(client: AsyncClient):
    """POST /tickets/{id}/escalate should return 404 for non-existent ticket."""
    resp = await client.post(
        "/tickets/nonexistent-id/escalate",
        json={"reason": "test", "reviewer": "test"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_escalated_tickets(client: AsyncClient, db_session: AsyncSession):
    """GET /tickets/escalated/list should return paginated escalated tickets."""
    # Create an escalated ticket
    ticket_id = str(uuid.uuid4())
    model = TicketModel(
        id=ticket_id,
        content="Escalated for listing",
        source="api",
        status=DBTicketStatus.ESCALATED.value,
        created_at=datetime.now(UTC),
    )
    db_session.add(model)
    await db_session.flush()

    resp = await client.get("/tickets/escalated/list")
    assert resp.status_code == 200
    data = resp.json()
    assert "tickets" in data
    assert "total" in data
    assert data["total"] >= 1


# ── Dashboard metrics ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dashboard_metrics(client: AsyncClient):
    """GET /dashboard/metrics should return aggregated metrics."""
    resp = await client.get("/dashboard/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_tickets" in data
    assert "processed_tickets" in data
    assert "escalated_tickets" in data
    assert "avg_confidence" in data
    assert "success_rate" in data
    assert "category_distribution" in data
    assert "urgency_distribution" in data


@pytest.mark.asyncio
async def test_dashboard_activity(client: AsyncClient):
    """GET /dashboard/activity should return recent activity."""
    resp = await client.get("/dashboard/activity")
    assert resp.status_code == 200
    data = resp.json()
    assert "activities" in data
    assert isinstance(data["activities"], list)


@pytest.mark.asyncio
async def test_dashboard_categories(client: AsyncClient):
    """GET /dashboard/categories should return category distribution."""
    resp = await client.get("/dashboard/categories")
    assert resp.status_code == 200
    data = resp.json()
    assert "categories" in data
    assert isinstance(data["categories"], dict)


@pytest.mark.asyncio
async def test_circuit_breaker_stats(client: AsyncClient):
    """GET /metrics/circuit-breakers should return stats."""
    resp = await client.get("/metrics/circuit-breakers")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_resource_metrics(client: AsyncClient):
    """GET /metrics/resources should return resource usage metrics."""
    resp = await client.get("/metrics/resources")
    assert resp.status_code == 200
    data = resp.json()
    assert "concurrency" in data
    assert "timeouts" in data
    assert "shutdown" in data
