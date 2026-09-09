"""Tests for the evaluation harness."""

import json
import tempfile
from pathlib import Path

import pytest

from src.config import get_settings


def test_eval_fixture_format():
    """Verify the eval fixture has the expected schema."""
    path = Path(get_settings().eval_tickets_path)
    if not path.exists():
        pytest.skip("Eval fixture not found")

    tickets = json.loads(path.read_text())
    assert isinstance(tickets, list)
    assert len(tickets) > 0

    required_keys = {"id", "subject", "body", "expected_category", "expected_urgency"}
    for t in tickets:
        assert required_keys.issubset(t.keys()), f"Missing keys in ticket: {t.get('id')}"
