"""Tests for the classification service — enums + schema integration."""

import pytest

from src.models.schemas import TicketCategory, TicketStatus, UrgencyLevel


def test_ticket_category_enum():
    assert TicketCategory.BUG.value == "bug"
    assert TicketCategory.FEATURE_REQUEST.value == "feature_request"
    assert TicketCategory.BILLING.value == "billing"
    assert TicketCategory.ACCOUNT_ISSUE.value == "account_issue"
    assert TicketCategory.USAGE_HELP.value == "usage_help"
    assert TicketCategory.OTHER.value == "other"


def test_urgency_level_enum():
    assert UrgencyLevel.LOW.value == "low"
    assert UrgencyLevel.MEDIUM.value == "medium"
    assert UrgencyLevel.HIGH.value == "high"
    assert UrgencyLevel.CRITICAL.value == "critical"


def test_ticket_status_enum():
    assert TicketStatus.OPEN.value == "open"
    assert TicketStatus.CLASSIFIED.value == "classified"
    assert TicketStatus.ESCALATED.value == "escalated"
    assert TicketStatus.RESOLVED.value == "resolved"
