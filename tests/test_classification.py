"""Tests for the classification service."""

import pytest

from src.models.ticket import TicketCategory, TicketUrgency


def test_ticket_category_enum():
    assert TicketCategory.BUG.value == "bug"
    assert TicketCategory.FEATURE_REQUEST.value == "feature_request"
    assert TicketCategory.BILLING.value == "billing"
    assert TicketCategory.ACCOUNT.value == "account"
    assert TicketCategory.GENERAL.value == "general"
    assert TicketCategory.UNKNOWN.value == "unknown"


def test_ticket_urgency_enum():
    assert TicketUrgency.P0.value == "P0"
    assert TicketUrgency.P1.value == "P1"
    assert TicketUrgency.P2.value == "P2"
    assert TicketUrgency.P3.value == "P3"


def test_ticket_status_enum():
    assert TicketStatus.OPEN.value == "open"
    assert TicketStatus.TRIAGED.value == "triaged"
    assert TicketStatus.ESCALATED.value == "escalated"
    assert TicketStatus.RESOLVED.value == "resolved"
