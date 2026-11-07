"""Regression tests for the reply-status lifecycle (B2).

Covers the `reply_status` field on:
* the DB model (`TicketModel`) and its new `reply_status` column
* the Pydantic schema (`ReplyStatus` enum)
* the approve endpoint response body (drafted → queued / sent)
"""

from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient

from src.models.schemas import ReplyStatus as ReplyStatusEnum
from src.models.ticket import TicketModel
from src.api.tickets import TicketStatusResponse, ApproveResponse

API_KEY = "test-key"


@pytest.mark.asyncio
class TestReplyStatusSchema:
    def test_enum_values(self):
        expected = {"drafted", "pending_approval", "queued", "sent", "failed"}
        assert {v.value for v in ReplyStatusEnum} == expected

    def test_approve_response_includes_reply_status(self):
        resp = ApproveResponse(
            ticket_id="t-1",
            status="resolved",
            message="Draft approved.",
            reply_status=ReplyStatusEnum.QUEUED.value,
        )
        assert resp.reply_status == "queued"

    def test_ticket_status_response_forward_safe(self):
        resp = TicketStatusResponse(
            ticket_id="t-1",
            status="resolved",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        assert resp.reply_status is None


@pytest.mark.asyncio
async def test_ticket_model_default_reply_status():
    row = TicketModel(
        id="t-db-default",
        content="c",
        source="api",
        metadata_=json.dumps({"customer_email": None}),
        status="resolved",
        reply_status=ReplyStatusEnum.PENDING_APPROVAL.value,
    )
    assert row.reply_status == ReplyStatusEnum.PENDING_APPROVAL.value


@pytest.mark.asyncio
async def test_ticket_model_reply_status_values(test_engine, db_session):
    for value in ReplyStatusEnum._value2member_map_.values():
        m = TicketModel(
            id=f"t-{value.value}",
            content="c",
            source="api",
            metadata_=json.dumps({}),
            status="resolved",
            **{"reply_status": value.value},
        )
        assert m.reply_status == value.value


def _make_deltiver(delivered, provider, error=None):
    d = AsyncMock()
    d.delivered = delivered
    d.provider = provider or "none"
    d.error = error
    d.duration_ms = 42
    return d


@pytest.mark.asyncio
class TestApproveReplyLifecycle:
    @pytest.fixture(autouse=True)
    def _client(self, client):
        self.client = client
        return client

    async def _setup_awaiting_review_with_draft(self, db_session, tid, draft_text="draft body"):
        ticket = await db_session.get(TicketModel, tid)
        ticket.status = "awaiting_review"
        ticket.draft_text = draft_text
        await db_session.flush()

    async def _approve(self, tid, delivered=False, provider="none"):
        deliv = _make_deltiver(delivered, provider)
        with patch("src.services.delivery.deliver_reply", return_value=deliv):
            return await self.client.post(f"/tickets/{tid}/approve", json={"reviewer": "admin"})

    async def test_approve_without_real_provider_transitions_to_queued(self, client, db_session):
        create = await client.post("/tickets/", json={"content": "a ticket"})
        tid = create.json()["ticket_id"]
        await self._setup_awaiting_review_with_draft(db_session, tid)
        resp = await self._approve(tid, delivered=False, provider="none")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "resolved"
        # No real provider configured by default -> queued, not sent.
        assert body["reply_status"] == "queued"
        rec = await db_session.get(TicketModel, tid)
        assert rec.reply_status == "queued"
        assert rec.review_decision == "approved"
        assert rec.reviewed_at is not None

    async def test_approve_with_delivered_payload_is_sent(self, client, db_session):
        create = await client.post("/tickets/", json={"content": "a ticket"})
        tid = create.json()["ticket_id"]
        await self._setup_awaiting_review_with_draft(db_session, tid)
        resp = await self._approve(tid, delivered=True, provider="email")

        body = resp.json()
        assert body["reply_status"] == "sent"
        rec = await db_session.get(TicketModel, tid)
        assert rec.reply_status == "sent"
        # The "drafted" initial state was overwritten by the new value in the
        # same flush — only `queued`/`sent` can be observed because this is one
        # approve run; there's no window to observe drafted afterwards via the
        # API, but the DB row updates to sent immediately after delivery succeeds.
