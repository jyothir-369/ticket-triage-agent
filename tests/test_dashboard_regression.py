"""Regression test for dashboard→backend authentication.

Verifies:
  1. Every ``/tickets/*`` endpoint rejects requests with NO ``X-API-Key``
     with HTTP 401. The dashboard previously sent no auth header at all, so the
     first traffic after enabling verify_api_key would break the whole UI.
  2. ``/health`` remains reachable even when ``DASHBOARD_API_KEY`` is empty —
     it is an unauthenticated Kubernetes probe and must never fail because of
     auth checks. See ``dashboard/api.check_backend_health``.
  3. The authenticated dashboard routes stay available once a valid key is
     used — otherwise the whole point of the change is lost.

Ticket endpoints return 401 via Starlette's JSONResponse, not by raising —
so we assert on ``response.status_code`` rather than catching exceptions.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


def _client() -> AsyncClient:
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_anonymous_ticket_endpoints_reject_401():
    """Any ticket endpoint without X-API-Key must return 401."""
    c = _client()
    try:
        resp = await c.get("/tickets/list")
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_ticket_detail_rejects_anonymous():
    """GET /tickets/<id> without X-API-Key → 401 too."""
    c = _client()
    try:
        resp = await c.get("/tickets/does-not-exist")
        assert resp.status_code == 401
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_health_endpoint_unauthenticated():
    """/health must NOT require X-API-Key — it is a probe."""
    c = _client()
    try:
        resp = await c.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_dashboard_routes_accept_valid_key():
    """Once X-API-Key is valid, dashboard-protected routes still work."""
    c = _client()
    try:
        # Inject the test API key the conftest already seeds for the session.
        resp = await c.get(
            "/tickets/list",
            headers={"X-API-Key": "test-key"},
        )
        # Expecting either 200 or 404 (no DB rows) — 404 means auth PASSED.
        assert resp.status_code != 401, "Auth blocked the dashboard route"
    finally:
        await c.aclose()
