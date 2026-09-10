"""Pytest integration for the evaluation harness.

Run with::

    pytest tests/test_eval.py -v
    pytest tests/test_eval.py -v --eval-threshold 0.8
    pytest tests/test_eval.py -v -k "not slow"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.harness import EvalHarness
from src.eval.metrics import (
    AggregateMetrics,
    MetricResult,
    aggregate_metrics,
    compare_results,
    keyword_overlap,
    rouge_l_f1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def eval_fixture_path() -> Path:
    """Path to the evaluation fixture file."""
    return Path(__file__).parent.parent / "eval" / "fixtures" / "tickets.jsonl"


@pytest.fixture(scope="session")
def eval_harness(eval_fixture_path: Path) -> EvalHarness:
    """Shared EvalHarness instance for the test session."""
    return EvalHarness(
        fixture_path=eval_fixture_path,
        concurrency=5,
        confidence_threshold=0.7,
    )


@pytest.fixture(scope="session")
def eval_tickets(eval_fixture_path: Path) -> list[dict]:
    """Load all eval tickets once for the session."""
    tickets = []
    with open(eval_fixture_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tickets.append(json.loads(line))
    return tickets


@pytest.fixture(scope="session")
def eval_metrics(eval_harness: EvalHarness) -> AggregateMetrics:
    """Run the full evaluation once and cache results for all tests.

    This fixture is session-scoped so the expensive agent run happens
    only once, and all metric tests run against the cached results.
    """
    import asyncio

    return asyncio.run(eval_harness.evaluate())


# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRougeL:
    """Test the ROUGE-L implementation."""

    def test_identical_texts(self):
        """Identical texts should score 1.0."""
        assert rouge_l_f1("hello world", "hello world") == 1.0

    def test_empty_texts(self):
        """Two empty texts should score 1.0."""
        assert rouge_l_f1("", "") == 1.0

    def test_one_empty(self):
        """One empty text should score 0.0."""
        assert rouge_l_f1("hello world", "") == 0.0
        assert rouge_l_f1("", "hello world") == 0.0

    def test_partial_overlap(self):
        """Partial overlap should yield a score between 0 and 1."""
        score = rouge_l_f1("the cat sat on the mat", "the cat is on the mat")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        """Completely different texts should score 0.0."""
        assert rouge_l_f1("apple banana", "xyz qwerty") == 0.0

    def test_subset(self):
        """A subset should have lower recall but some precision."""
        score = rouge_l_f1("the cat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_symmetry(self):
        """ROUGE-L F1 should be symmetric."""
        a = "the quick brown fox"
        b = "the quick red fox"
        assert abs(rouge_l_f1(a, b) - rouge_l_f1(b, a)) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword overlap unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKeywordOverlap:
    """Test keyword overlap scoring."""

    def test_all_found(self):
        """All keywords present should return 1.0."""
        assert keyword_overlap("fix the bug with version and patch", ["fix", "version", "patch"]) == 1.0

    def test_none_found(self):
        """No keywords present should return 0.0."""
        assert keyword_overlap("hello world", ["fix", "version"]) == 0.0

    def test_partial(self):
        """Partial match should return the fraction found."""
        score = keyword_overlap("fix the version", ["fix", "version", "patch"])
        assert abs(score - 2 / 3) < 1e-10

    def test_empty_keywords(self):
        """Empty keyword list should return 1.0."""
        assert keyword_overlap("anything", []) == 1.0

    def test_case_insensitive(self):
        """Keywords should match case-insensitively."""
        assert keyword_overlap("FIX the BUG", ["fix", "bug"]) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# compare_results unit tests (no agent needed)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompareResults:
    """Test the compare_results function with mock data."""

    def test_perfect_match(self):
        """Perfect match on all dimensions."""
        actual = {
            "ticket_id": "t1",
            "category": "bug",
            "urgency": "high",
            "confidence": 0.9,
            "draft_text": "We are fixing the bug",
            "should_escalate": False,
        }
        expected = {
            "expected_category": "BUG",
            "expected_urgency": "HIGH",
            "should_escalate": False,
            "human_approved_draft": "We are fixing the bug",
            "expected_draft_keywords": ["fixing", "bug"],
        }
        result = compare_results(actual, expected)
        assert result.category_correct is True
        assert result.urgency_correct is True
        assert result.escalation_correct is True
        assert result.draft_rouge_l > 0.5

    def test_wrong_category(self):
        """Wrong category should be flagged."""
        actual = {
            "ticket_id": "t2",
            "category": "billing",
            "urgency": "high",
            "confidence": 0.9,
            "draft_text": "test",
            "should_escalate": False,
        }
        expected = {
            "expected_category": "BUG",
            "expected_urgency": "HIGH",
            "should_escalate": False,
            "human_approved_draft": "test",
            "expected_draft_keywords": [],
        }
        result = compare_results(actual, expected)
        assert result.category_correct is False
        assert result.urgency_correct is True

    def test_wrong_escalation(self):
        """Wrong escalation decision should be flagged."""
        actual = {
            "ticket_id": "t3",
            "category": "bug",
            "urgency": "critical",
            "confidence": 0.3,
            "draft_text": "test",
            "should_escalate": False,  # Agent said no escalate
        }
        expected = {
            "expected_category": "BUG",
            "expected_urgency": "CRITICAL",
            "should_escalate": True,  # Expected escalate
            "human_approved_draft": "test",
            "expected_draft_keywords": [],
        }
        result = compare_results(actual, expected)
        assert result.escalation_correct is False

    def test_confidence_threshold_above_no_escalation(self):
        """High confidence with no expected escalation is correct."""
        actual = {
            "ticket_id": "t4",
            "category": "bug",
            "urgency": "low",
            "confidence": 0.9,
            "draft_text": "test",
            "should_escalate": False,
        }
        expected = {
            "expected_category": "BUG",
            "expected_urgency": "LOW",
            "should_escalate": False,
            "human_approved_draft": "test",
            "expected_draft_keywords": [],
        }
        result = compare_results(actual, expected, confidence_threshold=0.7)
        assert result.confidence_above_threshold is True
        assert result.confidence_correctly_predicts_escalation is True

    def test_confidence_threshold_below_expected_escalation(self):
        """Low confidence with expected escalation is correct."""
        actual = {
            "ticket_id": "t5",
            "category": "bug",
            "urgency": "high",
            "confidence": 0.4,
            "draft_text": "test",
            "should_escalate": True,
        }
        expected = {
            "expected_category": "BUG",
            "expected_urgency": "HIGH",
            "should_escalate": True,
            "human_approved_draft": "test",
            "expected_draft_keywords": [],
        }
        result = compare_results(actual, expected, confidence_threshold=0.7)
        assert result.confidence_above_threshold is False
        assert result.confidence_correctly_predicts_escalation is True

    def test_confidence_threshold_above_but_expected_escalation(self):
        """High confidence but escalation expected is a miss."""
        actual = {
            "ticket_id": "t6",
            "category": "bug",
            "urgency": "high",
            "confidence": 0.9,
            "draft_text": "test",
            "should_escalate": False,
        }
        expected = {
            "expected_category": "BUG",
            "expected_urgency": "HIGH",
            "should_escalate": True,  # Should have escalated
            "human_approved_draft": "test",
            "expected_draft_keywords": [],
        }
        result = compare_results(actual, expected, confidence_threshold=0.7)
        assert result.confidence_correctly_predicts_escalation is False


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate metrics unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAggregateMetrics:
    """Test aggregate_metrics with mock MetricResult lists."""

    def test_empty_results(self):
        """Empty results should produce zero metrics."""
        agg = aggregate_metrics([])
        assert agg.total == 0
        assert agg.category_accuracy == 0.0

    def test_all_correct(self):
        """All correct results should yield 100% accuracy."""
        results = [
            MetricResult(
                ticket_id=f"t{i}",
                category_correct=True,
                urgency_correct=True,
                escalation_correct=True,
                draft_rouge_l=0.8,
                draft_keyword_overlap=0.9,
                confidence_correctly_predicts_escalation=True,
                details={
                    "expected_should_escalate": i % 2 == 0,
                    "actual_should_escalate": i % 2 == 0,
                },
            )
            for i in range(5)
        ]
        agg = aggregate_metrics(results)
        assert agg.category_accuracy == 1.0
        assert agg.urgency_accuracy == 1.0
        assert agg.escalation_accuracy == 1.0
        assert agg.pass_rate == 1.0

    def test_confusion_matrix(self):
        """Confusion matrix should count TP, TN, FP, FN correctly."""
        results = [
            MetricResult(
                ticket_id="tp1",
                details={"expected_should_escalate": True, "actual_should_escalate": True},
            ),
            MetricResult(
                ticket_id="tn1",
                details={"expected_should_escalate": False, "actual_should_escalate": False},
            ),
            MetricResult(
                ticket_id="fp1",
                details={"expected_should_escalate": False, "actual_should_escalate": True},
            ),
            MetricResult(
                ticket_id="fn1",
                details={"expected_should_escalate": True, "actual_should_escalate": False},
            ),
        ]
        agg = aggregate_metrics(results)
        cm = agg.confusion_matrix["escalation"]
        assert cm["TP"] == 1
        assert cm["TN"] == 1
        assert cm["FP"] == 1
        assert cm["FN"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# EvalHarness unit tests (no agent run)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvalHarness:
    """Test EvalHarness methods that don't require running the agent."""

    def test_load_tickets(self, eval_harness: EvalHarness, eval_fixture_path: Path):
        """Should load all tickets from the JSONL fixture."""
        tickets = eval_harness.load_tickets()
        assert len(tickets) == 8
        assert tickets[0]["id"] == "test-001"
        assert tickets[0]["expected_category"] == "BUG"

    def test_load_tickets_missing_file(self):
        """Should return empty list for missing fixture file."""
        harness = EvalHarness(fixture_path="/nonexistent/path.jsonl")
        assert harness.load_tickets() == []

    def test_compare_results_delegates(self, eval_harness: EvalHarness):
        """compare_results should produce a MetricResult."""
        actual = {
            "ticket_id": "t1",
            "category": "bug",
            "urgency": "high",
            "confidence": 0.9,
            "draft_text": "test draft",
            "should_escalate": False,
        }
        expected = {
            "expected_category": "BUG",
            "expected_urgency": "HIGH",
            "should_escalate": False,
            "human_approved_draft": "test draft",
            "expected_draft_keywords": ["test"],
        }
        result = eval_harness.compare_results(actual, expected)
        assert isinstance(result, MetricResult)
        assert result.ticket_id == "t1"
        assert result.category_correct is True


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests (run agent against eval set)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestEvalIntegration:
    """Integration tests that run the full agent against the eval set.

    These are marked @pytest.mark.slow because they invoke the full
    triage pipeline (LLM calls, vector search, etc.).  Run with::

        pytest tests/test_eval.py -v -m slow
    """

    def test_category_accuracy_threshold(self, eval_metrics: AggregateMetrics):
        """Category accuracy should meet minimum threshold."""
        assert eval_metrics.category_accuracy >= 0.5, (
            f"Category accuracy {eval_metrics.category_accuracy:.1%} "
            f"is below 50% threshold"
        )

    def test_urgency_accuracy_threshold(self, eval_metrics: AggregateMetrics):
        """Urgency accuracy should meet minimum threshold."""
        assert eval_metrics.urgency_accuracy >= 0.5, (
            f"Urgency accuracy {eval_metrics.urgency_accuracy:.1%} "
            f"is below 50% threshold"
        )

    def test_escalation_accuracy_threshold(self, eval_metrics: AggregateMetrics):
        """Escalation accuracy should meet minimum threshold."""
        assert eval_metrics.escalation_accuracy >= 0.5, (
            f"Escalation accuracy {eval_metrics.escalation_accuracy:.1%} "
            f"is below 50% threshold"
        )

    def test_draft_rouge_l_threshold(self, eval_metrics: AggregateMetrics):
        """Average draft ROUGE-L should be above minimum."""
        assert eval_metrics.avg_draft_rouge_l >= 0.05, (
            f"Average ROUGE-L {eval_metrics.avg_draft_rouge_l:.4f} "
            f"is below 0.05 threshold"
        )

    def test_pass_rate_threshold(self, eval_metrics: AggregateMetrics):
        """Overall pass rate should meet minimum threshold."""
        assert eval_metrics.pass_rate >= 0.3, (
            f"Pass rate {eval_metrics.pass_rate:.1%} "
            f"is below 30% threshold"
        )

    def test_all_tickets_evaluated(self, eval_metrics: AggregateMetrics):
        """All 8 tickets should have been evaluated."""
        assert eval_metrics.total == 8, (
            f"Expected 8 tickets evaluated, got {eval_metrics.total}"
        )

    def test_no_errors_in_results(self, eval_metrics: AggregateMetrics):
        """No ticket should have produced an error."""
        for r in eval_metrics.results:
            assert r.details.get("error") is None, (
                f"Ticket {r.ticket_id} produced an error: {r.details['error']}"
            )

    def test_report_generation(self, eval_harness: EvalHarness, eval_metrics: AggregateMetrics):
        """Should be able to generate and save all report formats."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = eval_harness.save_json_report(
                eval_metrics, Path(tmpdir) / "report.json"
            )
            assert json_path.exists()

            html_path = eval_harness.save_html_dashboard(
                eval_metrics, Path(tmpdir) / "dashboard.html"
            )
            assert html_path.exists()

            # Verify JSON is valid
            with open(json_path, encoding="utf-8") as f:
                report = json.load(f)
            assert "metrics" in report
            assert "per_ticket" in report

            # Verify HTML is non-empty
            html_content = html_path.read_text(encoding="utf-8")
            assert len(html_content) > 1000
