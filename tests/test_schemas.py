"""Tests for the Pydantic schemas — validation, enums, and model methods."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    Citation,
    DashboardMetrics,
    DraftResponse,
    EscalationDecision,
    RetrievedDocument,
    StepStatus,
    Ticket,
    TicketCategory,
    TicketClassification,
    TicketMetadata,
    TicketStatus,
    TraceStep,
    TriageTrace,
    UrgencyLevel,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnums:
    def test_ticket_category_values(self):
        assert TicketCategory.BUG.value == "bug"
        assert TicketCategory.FEATURE_REQUEST.value == "feature_request"
        assert TicketCategory.ACCOUNT_ISSUE.value == "account_issue"
        assert TicketCategory.BILLING.value == "billing"
        assert TicketCategory.USAGE_HELP.value == "usage_help"
        assert TicketCategory.OTHER.value == "other"

    def test_urgency_level_values(self):
        assert UrgencyLevel.LOW.value == "low"
        assert UrgencyLevel.MEDIUM.value == "medium"
        assert UrgencyLevel.HIGH.value == "high"
        assert UrgencyLevel.CRITICAL.value == "critical"

    def test_ticket_status_values(self):
        assert TicketStatus.OPEN.value == "open"
        assert TicketStatus.ESCALATED.value == "escalated"
        assert TicketStatus.RESOLVED.value == "resolved"

    def test_step_status_values(self):
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"


# ═══════════════════════════════════════════════════════════════════════════════
# Ticket
# ═══════════════════════════════════════════════════════════════════════════════


class TestTicket:
    def test_valid_ticket(self):
        t = Ticket(id=1, content="My app is broken")
        assert t.id == 1
        assert t.source == "api"  # default
        assert t.status == TicketStatus.OPEN

    def test_blank_content_rejected(self):
        with pytest.raises(ValidationError, match="content must not be blank"):
            Ticket(id=1, content="   ")

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            Ticket(id=1, content="")

    def test_source_lowercased(self):
        t = Ticket(id=1, content="Help", source=" GitHub ")
        assert t.source == "github"

    def test_created_at_ensure_utc(self):
        naive = datetime(2025, 1, 1, 12, 0, 0)
        t = Ticket(id=1, content="Help", created_at=naive)
        assert t.created_at.tzinfo is not None

    def test_with_metadata(self):
        meta = TicketMetadata(customer_email="a@b.com", tags=["urgent"])
        t = Ticket(id=1, content="Help", metadata=meta)
        assert t.metadata.customer_email == "a@b.com"
        assert "urgent" in t.metadata.tags


# ═══════════════════════════════════════════════════════════════════════════════
# TicketClassification
# ═══════════════════════════════════════════════════════════════════════════════


class TestTicketClassification:
    def test_valid_classification(self):
        c = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.85,
        )
        assert c.is_high_confidence()
        assert not c.is_high_confidence(threshold=0.9)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            TicketClassification(
                category=TicketCategory.BUG,
                urgency=UrgencyLevel.LOW,
                confidence=1.5,
            )

    def test_negative_confidence(self):
        with pytest.raises(ValidationError):
            TicketClassification(
                category=TicketCategory.BUG,
                urgency=UrgencyLevel.LOW,
                confidence=-0.1,
            )

    def test_to_dict(self):
        c = TicketClassification(
            category=TicketCategory.BILLING,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.6,
        )
        d = c.to_dict()
        assert d["category"] == "billing"
        assert d["urgency"] == "medium"
        assert isinstance(d, dict)

    def test_reasoning_stripped(self):
        c = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.LOW,
            confidence=0.5,
            reasoning="  seems like a bug  ",
        )
        assert c.reasoning == "seems like a bug"


# ═══════════════════════════════════════════════════════════════════════════════
# RetrievedDocument
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrievedDocument:
    def test_valid_doc(self):
        d = RetrievedDocument(
            id="doc-1",
            content="Some help article",
            similarity_score=0.92,
        )
        assert d.source == "unknown"  # default

    def test_blank_content_rejected(self):
        with pytest.raises(ValidationError, match="content must not be blank"):
            RetrievedDocument(id="1", content="  ", similarity_score=0.5)

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            RetrievedDocument(id="1", content="Hi", similarity_score=1.5)

    def test_source_lowercased(self):
        d = RetrievedDocument(id="1", content="Hi", similarity_score=0.5, source=" Past_Ticket ")
        assert d.source == "past_ticket"


# ═══════════════════════════════════════════════════════════════════════════════
# DraftResponse
# ═══════════════════════════════════════════════════════════════════════════════


class TestDraftResponse:
    def test_valid_draft(self):
        dr = DraftResponse(
            draft_text="We can help with that.",
            confidence=0.8,
        )
        assert dr.is_high_confidence()

    def test_blank_draft_rejected(self):
        with pytest.raises(ValidationError, match="draft_text must not be blank"):
            DraftResponse(draft_text="  ", confidence=0.5)

    def test_citations_validated_without_citations_rejected(self):
        with pytest.raises(ValueError, match="citations_validated is True"):
            DraftResponse(
                draft_text="Hello",
                confidence=0.8,
                citations_validated=True,
                citations=[],
            )

    def test_citations_validated_with_citations_ok(self):
        citation = Citation(claim="It works", source_doc_id="d1")
        dr = DraftResponse(
            draft_text="It works.",
            confidence=0.9,
            citations=[citation],
            citations_validated=True,
        )
        assert len(dr.citations) == 1

    def test_to_dict(self):
        dr = DraftResponse(draft_text="Hi", confidence=0.5)
        d = dr.to_dict()
        assert d["draft_text"] == "Hi"


# ═══════════════════════════════════════════════════════════════════════════════
# TraceStep
# ═══════════════════════════════════════════════════════════════════════════════


class TestTraceStep:
    def test_valid_step(self):
        s = TraceStep(step="classify", status=StepStatus.COMPLETED, duration_ms=120)
        assert s.status == StepStatus.COMPLETED

    def test_empty_step_name_rejected(self):
        with pytest.raises(ValidationError):
            TraceStep(step="")

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            TraceStep(step="classify", duration_ms=-1)

    def test_naive_timestamp_gets_utc(self):
        naive = datetime(2025, 6, 1, 10, 0)
        s = TraceStep(step="classify", timestamp=naive)
        assert s.timestamp.tzinfo is not None


# ═══════════════════════════════════════════════════════════════════════════════
# EscalationDecision
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalationDecision:
    def test_escalate(self):
        e = EscalationDecision(
            should_escalate=True,
            reason="Low confidence",
            confidence_score=0.4,
            threshold_used=0.7,
        )
        assert not e.is_high_confidence()

    def test_no_escalate(self):
        e = EscalationDecision(
            should_escalate=False,
            confidence_score=0.9,
            threshold_used=0.7,
        )
        assert e.is_high_confidence()

    def test_to_dict(self):
        e = EscalationDecision(should_escalate=True, reason="test")
        d = e.to_dict()
        assert d["should_escalate"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# TriageTrace
# ═══════════════════════════════════════════════════════════════════════════════


class TestTriageTrace:
    def test_add_step_orders_by_timestamp(self):
        trace = TriageTrace(ticket_id=1)
        s1 = TraceStep(step="a", timestamp=datetime(2025, 1, 1, 12, 0, 1, tzinfo=UTC))
        s2 = TraceStep(step="b", timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC))
        trace.add_step(s1)
        trace.add_step(s2)
        assert trace.steps[0].step == "b"
        assert trace.steps[1].step == "a"

    def test_is_high_confidence_no_classification(self):
        trace = TriageTrace(ticket_id=1)
        assert not trace.is_high_confidence()

    def test_is_high_confidence_with_classification(self):
        trace = TriageTrace(
            ticket_id=1,
            classification=TicketClassification(
                category=TicketCategory.BUG,
                urgency=UrgencyLevel.HIGH,
                confidence=0.9,
            ),
        )
        assert trace.is_high_confidence()

    def test_to_dict(self):
        trace = TriageTrace(ticket_id=42)
        d = trace.to_dict()
        assert d["ticket_id"] == 42


# ═══════════════════════════════════════════════════════════════════════════════
# DashboardMetrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestDashboardMetrics:
    def test_defaults(self):
        m = DashboardMetrics()
        assert m.total_tickets == 0
        assert m.processed_tickets == 0

    def test_consistency_check(self):
        with pytest.raises(ValueError, match="processed_tickets cannot exceed"):
            DashboardMetrics(total_tickets=5, processed_tickets=10)

    def test_escalated_exceeds_processed(self):
        with pytest.raises(ValueError, match="escalated_tickets cannot exceed"):
            DashboardMetrics(total_tickets=10, processed_tickets=5, escalated_tickets=6)

    def test_to_dict(self):
        m = DashboardMetrics(total_tickets=100, processed_tickets=80)
        d = m.to_dict()
        assert d["total_tickets"] == 100
