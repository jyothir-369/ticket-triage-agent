"""Tests for the FastAPI API endpoints."""

import pytest
from httpx import AsyncClient

from src.models import Ticket, TicketCategory, TicketStatus, TicketUrgency


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_triage_ticket_not_found(client: AsyncClient):
    resp = await client.post("/tickets/9999/triage")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_trace_ticket_not_found(client: AsyncClient):
    resp = await client.get("/tickets/9999/trace")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_triage_metrics_empty(client: AsyncClient):
    resp = await client.get("/dashboard/triage-metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tickets"] == 0
    assert data["triaged"] == 0
    assert data["escalated"] == 0
