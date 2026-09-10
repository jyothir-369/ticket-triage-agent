"""Tests for the evaluation harness and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import get_settings

# ── Eval fixture format ────────────────────────────────────────────────────────


def test_eval_fixture_format():
    """Verify the eval fixture has the expected schema."""
    path = Path(get_settings().eval_tickets_path)
    if not path.exists():
        pytest.skip("Eval fixture not found")

    tickets = json.loads(path.read_text())
    assert isinstance(tickets, list)
    assert len(tickets) > 0

    required_keys = {"id", "content", "expected_category", "expected_urgency"}
    for t in tickets:
        assert required_keys.issubset(t.keys()), f"Missing keys in ticket: {t.get('id')}"


# ── EvalHarness import and config ──────────────────────────────────────────────


def test_eval_harness_importable():
    """EvalHarness should be importable from src.eval.harness."""
    from src.eval.harness import EvalHarness

    harness = EvalHarness()
    assert harness is not None
    assert harness.confidence_threshold == 0.7


def test_eval_metrics_importable():
    """Evaluation metrics should be importable."""
    from src.eval.metrics import (
        keyword_overlap,
        rouge_l_f1,
    )

    # Test ROUGE-L
    score = rouge_l_f1("hello world", "hello world")
    assert score == 1.0

    # Test keyword overlap
    overlap = keyword_overlap("I love cats", ["love", "cats"])
    assert overlap == 1.0


def test_eval_fixture_loadable():
    """Eval fixture should be loadable via EvalHarness."""
    from src.eval.harness import EvalHarness

    harness = EvalHarness()
    tickets = harness.load_tickets()
    assert len(tickets) > 0
    assert all("id" in t for t in tickets)
    assert all("content" in t for t in tickets)


def test_eval_fixture_has_expected_fields():
    """Each eval ticket should have all required fields for comparison."""
    from src.eval.harness import EvalHarness

    harness = EvalHarness()
    tickets = harness.load_tickets()

    for ticket in tickets:
        assert "id" in ticket, "Ticket missing 'id'"
        assert "content" in ticket, f"Ticket {ticket.get('id')} missing 'content'"
        assert "expected_category" in ticket, f"Ticket {ticket.get('id')} missing 'expected_category'"
        assert "expected_urgency" in ticket, f"Ticket {ticket.get('id')} missing 'expected_urgency'"


@pytest.mark.asyncio
async def test_eval_compare_results_works():
    """EvalHarness.compare_results should produce valid MetricResult."""
    from src.eval.harness import EvalHarness
    from src.eval.metrics import MetricResult

    harness = EvalHarness()

    actual = {
        "ticket_id": "test-001",
        "category": "bug",
        "urgency": "high",
        "confidence": 0.85,
        "draft_text": "We are investigating the login issue.",
        "should_escalate": False,
    }
    expected = {
        "id": "test-001",
        "expected_category": "bug",
        "expected_urgency": "high",
        "should_escalate": False,
    }

    result = harness.compare_results(actual, expected)
    assert isinstance(result, MetricResult)
    assert result.ticket_id == "test-001"
    assert result.category_correct is True
    assert result.urgency_correct is True
    assert result.escalation_correct is True


def test_rouge_l_identical():
    """ROUGE-L should return 1.0 for identical strings."""
    from src.eval.metrics import rouge_l_f1

    assert rouge_l_f1("hello world", "hello world") == 1.0


def test_rouge_l_empty():
    """ROUGE-L should handle empty strings gracefully."""
    from src.eval.metrics import rouge_l_f1

    score = rouge_l_f1("", "hello")
    assert 0.0 <= score <= 1.0


def test_keyword_overlap_perfect():
    """Keyword overlap should return 1.0 when all keywords are present."""
    from src.eval.metrics import keyword_overlap

    assert keyword_overlap("I love cats and dogs", ["love", "cats", "dogs"]) == 1.0


def test_keyword_overlap_partial():
    """Keyword overlap should return a fraction for partial matches."""
    from src.eval.metrics import keyword_overlap

    score = keyword_overlap("I love cats", ["love", "cats", "dogs"])
    assert 0.0 < score < 1.0


def test_keyword_overlap_empty():
    """Keyword overlap with empty text should return 0.0."""
    from src.eval.metrics import keyword_overlap

    score = keyword_overlap("", ["hello"])
    assert score == 0.0


def test_aggregate_metrics():
    """aggregate_metrics should compute averages from a list of MetricResults."""
    from src.eval.metrics import MetricResult, aggregate_metrics

    results = [
        MetricResult(
            ticket_id="1",
            category_correct=True,
            urgency_correct=True,
            escalation_correct=True,
            confidence_above_threshold=True,
            draft_rouge_l=0.8,
            draft_keyword_overlap=0.7,
        ),
        MetricResult(
            ticket_id="2",
            category_correct=False,
            urgency_correct=True,
            escalation_correct=True,
            confidence_above_threshold=True,
            draft_rouge_l=0.6,
            draft_keyword_overlap=0.5,
        ),
    ]

    agg = aggregate_metrics(results)
    assert agg.total == 2
    assert agg.category_accuracy == 0.5
    assert agg.urgency_accuracy == 1.0
    assert agg.escalation_accuracy == 1.0
    assert agg.pass_rate >= 0.0
