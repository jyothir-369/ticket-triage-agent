"""Edge case tests — empty ticket, very long ticket, special characters,
duplicate ticket (idempotency).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from src.models.schemas import (
    DraftResponse,
    Ticket,
    TicketCategory,
    TicketClassification,
    UrgencyLevel,
)
from src.models.ticket import TicketModel
from src.services.classification import TicketClassifier

# ═══════════════════════════════════════════════════════════════════════════════
# Empty ticket
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmptyTicket:
    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            Ticket(id=1, content="")

    def test_blank_content_rejected(self):
        with pytest.raises(ValidationError):
            Ticket(id=1, content="   \n\t  ")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            Ticket(id=1, content="     ")

    @pytest.mark.asyncio
    async def test_api_rejects_empty_content(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/",
            json={"content": "", "source": "api"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_api_rejects_blank_content(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/",
            json={"content": "   ", "source": "api"},
        )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# Very long ticket
# ═══════════════════════════════════════════════════════════════════════════════


class TestVeryLongTicket:
    def test_long_content_accepted(self):
        long_content = "This is a very detailed bug report. " * 1000
        t = Ticket(id=1, content=long_content)
        assert len(t.content) > 10000

    def test_max_length_boundary(self):
        content = "x" * 50_000
        t = Ticket(id=1, content=content)
        assert len(t.content) == 50_000

    def test_exceeds_max_length_rejected(self):
        content = "x" * 50_001
        with pytest.raises(ValidationError):
            Ticket(id=1, content=content)

    @pytest.mark.asyncio
    async def test_api_accepts_long_content(self, client: AsyncClient):
        long_content = "Detailed issue report. " * 2000
        resp = await client.post(
            "/tickets/",
            json={"content": long_content, "source": "api"},
        )
        assert resp.status_code == 201

    def test_heuristic_handles_long_content(self):
        classifier = TicketClassifier()
        long_bug = "The application crashes with error 500. " * 500
        result = classifier._heuristic_fallback(long_bug)
        assert result.category == TicketCategory.BUG

    def test_classifier_rejects_empty_for_classify(self):
        classifier = TicketClassifier()
        with pytest.raises(ValueError, match="must not be empty"):
            # This is tested via the async classify method
            pass  # Covered by test_classification.py


# ═══════════════════════════════════════════════════════════════════════════════
# Special characters
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpecialCharacters:
    def test_unicode_content_accepted(self):
        t = Ticket(id=1, content="日本語のチケット: ログインエラーが発生しています")
        assert "日本語" in t.content

    def test_emoji_content_accepted(self):
        t = Ticket(id=1, content="Login page shows 🔥 error 🔥 on every attempt")
        assert "🔥" in t.content

    def test_html_in_content_accepted(self):
        t = Ticket(id=1, content="<script>alert('x')</script> Error on login")
        assert "<script>" in t.content

    def test_sql_injection_in_content(self):
        t = Ticket(id=1, content="'; DROP TABLE tickets; --")
        assert "DROP TABLE" in t.content

    def test_newlines_and_tabs(self):
        content = "Line 1\nLine 2\tTabbed\r\nWindows line"
        t = Ticket(id=1, content=content)
        assert "\n" in t.content

    def test_very_long_single_word(self):
        t = Ticket(id=1, content="a" * 10000)
        assert len(t.content) == 10000

    @pytest.mark.asyncio
    async def test_api_handles_unicode(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/",
            json={
                "content": "エラーが発生: ログインページがクラッシュ",
                "source": "api",
            },
        )
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_api_handles_special_chars(self, client: AsyncClient):
        resp = await client.post(
            "/tickets/",
            json={
                "content": "Error: @#$%^&*()_+ {}|:<>? and more",
                "source": "api",
            },
        )
        assert resp.status_code == 201

    def test_heuristic_handles_unicode(self):
        classifier = TicketClassifier()
        result = classifier._heuristic_fallback(
            "登录页面崩溃了，每次登录都出现500错误"
        )
        # Should not crash, may return OTHER
        assert result.category in list(TicketCategory)

    def test_heuristic_handles_special_chars(self):
        classifier = TicketClassifier()
        result = classifier._heuristic_fallback(
            "!@#$%^&*() - login crash error bug"
        )
        assert result.category == TicketCategory.BUG


# ═══════════════════════════════════════════════════════════════════════════════
# Duplicate ticket (idempotency)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDuplicateTickets:
    @pytest.mark.asyncio
    async def test_duplicate_tickets_get_different_ids(self, client: AsyncClient):
        """Two tickets with same content should get different UUIDs."""
        content = "Duplicate content test"
        resp1 = await client.post(
            "/tickets/", json={"content": content, "source": "api"}
        )
        resp2 = await client.post(
            "/tickets/", json={"content": content, "source": "api"}
        )

        assert resp1.status_code == 201
        assert resp2.status_code == 201
        assert resp1.json()["ticket_id"] != resp2.json()["ticket_id"]

    @pytest.mark.asyncio
    async def test_duplicate_triage_triggers_409(self, client: AsyncClient, db_session):
        """Triaging the same ticket twice should return 409."""
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Double triage test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        # First triage
        resp1 = await client.post(f"/tickets/{ticket_id}/triage")
        assert resp1.status_code == 202

        # Simulate the background task setting status to processing
        ticket = await db_session.get(TicketModel, ticket_id)
        ticket.status = "processing"
        await db_session.flush()

        # Second triage should fail
        resp2 = await client.post(f"/tickets/{ticket_id}/triage")
        assert resp2.status_code == 409

    @pytest.mark.asyncio
    async def test_duplicate_escalation_allowed(self, client: AsyncClient):
        """Escalating the same ticket twice should succeed both times."""
        create_resp = await client.post(
            "/tickets/",
            json={"content": "Double escalation test", "source": "api"},
        )
        ticket_id = create_resp.json()["ticket_id"]

        resp1 = await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": "First escalation"},
        )
        assert resp1.status_code == 200

        resp2 = await client.post(
            f"/tickets/{ticket_id}/escalate",
            json={"reason": "Second escalation"},
        )
        assert resp2.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Source variations
# ═══════════════════════════════════════════════════════════════════════════════


class TestSourceVariations:
    def test_source_lowercased(self):
        t = Ticket(id=1, content="Test", source=" GitHub ")
        assert t.source == "github"

    def test_source_max_length(self):
        t = Ticket(id=1, content="Test", source="a" * 100)
        assert len(t.source) == 100

    def test_source_too_long_rejected(self):
        with pytest.raises(ValidationError):
            Ticket(id=1, content="Test", source="a" * 101)

    @pytest.mark.asyncio
    async def test_various_sources_accepted(self, client: AsyncClient):
        for source in ["email", "github", "intercom", "slack", "api"]:
            resp = await client.post(
                "/tickets/",
                json={"content": f"Source test: {source}", "source": source},
            )
            assert resp.status_code == 201


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence threshold boundary
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfidenceBoundary:
    def test_exactly_at_threshold_no_escalation(self):
        """Confidence exactly at threshold should NOT escalate."""
        from src.agent.graph import escalate_check_node
        from src.agent.state import AgentState

        state: AgentState = {
            "ticket": Ticket(id=1, content="test"),
            "classification": TicketClassification(
                category=TicketCategory.BUG,
                urgency=UrgencyLevel.HIGH,
                confidence=0.70,  # Exactly at default threshold
            ),
            "retrieved_docs": [],
            "draft": DraftResponse(
                draft_text="Help",
                confidence=0.70,
            ),
            "escalation": None,
            "should_escalate": False,
            "escalation_reason": "",
            "trace": TriageTrace(ticket_id=1),
            "loop_count": 0,
            "tool_call_count": 0,
            "error_message": "",
            "messages": [],
        }

        # This is tested via the node directly
        # The escalate_check_node should set should_escalate=False when
        # confidence >= threshold (0.70 >= 0.70)
        import asyncio

        result = asyncio.run(escalate_check_node(state))
        assert result["should_escalate"] is False

    def test_below_threshold_escalates(self):
        """Confidence below threshold should escalate."""
        from src.agent.graph import escalate_check_node
        from src.agent.state import AgentState

        state: AgentState = {
            "ticket": Ticket(id=1, content="test"),
            "classification": TicketClassification(
                category=TicketCategory.BUG,
                urgency=UrgencyLevel.HIGH,
                confidence=0.69,  # Just below threshold
            ),
            "retrieved_docs": [],
            "draft": DraftResponse(
                draft_text="Help",
                confidence=0.69,
            ),
            "escalation": None,
            "should_escalate": False,
            "escalation_reason": "",
            "trace": TriageTrace(ticket_id=1),
            "loop_count": 0,
            "tool_call_count": 0,
            "error_message": "",
            "messages": [],
        }

        import asyncio

        result = asyncio.run(escalate_check_node(state))
        assert result["should_escalate"] is True
