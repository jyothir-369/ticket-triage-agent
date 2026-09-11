"""Unit tests for the eval package (metrics, harness, reports).

These exercise the pure-Python scoring/aggregation/reporting logic only —
no agent, no network, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness import EvalHarness
from eval.metrics import (
    CATEGORIES,
    URGENCIES,
    compare_single,
    compute_metrics,
    percentile,
    rouge_l_f1,
)
from eval.report import build_run_report, save_html_report, save_json_report

# ═══════════════════════════════════════════════════════════════════════════════
# Restore a fixture default after patching (paths are unique per test).
# ═══════════════════════════════════════════════════════════════════════════════


def _make_expected(**overrides) -> dict:
    row = {
        "id": "test-001",
        "content": "Subject: 500 error\n\nBody:\nLogin page crashes with 500.",
        "source": "manual",
        "expected_category": "BUG",
        "expected_urgency": "HIGH",
        "should_escalate": False,
        "human_approved_draft": (
            "Thanks for reporting the 500 error. We are investigating and will "
            "follow up with a fix shortly."
        ),
    }
    row.update(overrides)
    return row


def _make_actual(**overrides) -> dict:
    row = {
        "id": "test-001",
        "category": "bug",
        "urgency": "high",
        "confidence": 0.92,
        "retrieval_count": 3,
        "draft": (
            "Thank you for the report. Our team is investigating the 500 error "
            "you are seeing on login and will update you soon."
        ),
        "should_escalate": False,
        "latency_ms": 1200,
        "error": None,
    }
    row.update(overrides)
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestRougeL:
    def test_exact_match(self):
        assert rouge_l_f1("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_empty_both(self):
        assert rouge_l_f1("", "") == 1.0

    def test_one_empty(self):
        assert rouge_l_f1("hello world", "") == 0.0

    def test_partial_overlap_bounded(self):
        score = rouge_l_f1("the fox jumps", "the dog runs")
        assert 0.0 <= score <= 1.0


class TestPercentile:
    def test_empty(self):
        assert percentile([], 50) == 0.0

    def test_known_values(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert percentile(values, 50) == 2.0  # nearest-rank
        assert percentile(values, 100) == 4.0
        assert percentile(values, 0) == 1.0


class TestCompareSingle:
    def test_perfect(self):
        comp = compare_single(_make_actual(), _make_expected())
        assert comp["category_match"] is True
        assert comp["urgency_match"] is True
        assert comp["escalation_correct"] is True
        assert comp["confidence_calibrated"] is True
        assert comp["draft_similarity"] > 0.0

    def test_wrong_category(self):
        expected = _make_expected(expected_category="BILLING")
        comp = compare_single(_make_actual(), expected)
        assert comp["category_match"] is False
        assert comp["confidence_calibrated"] is False  # confident but wrong

    def test_escalation_mismatch(self):
        expected = _make_expected(should_escalate=True)
        comp = compare_single(_make_actual(), expected)
        assert comp["escalation_correct"] is False

    def test_low_confidence_wrong_is_calibrated(self):
        # Low confidence + wrong answer ⇒ correctly signals uncertainty.
        actual = _make_actual(category="feature_request", confidence=0.3)
        expected = _make_expected(expected_category="BILLING")
        comp = compare_single(actual, expected)
        assert comp["category_match"] is False
        assert comp["confidence_calibrated"] is True


class TestComputeMetrics:
    def _rows(self):
        base_expected = _make_expected()
        base_actual = _make_actual()
        rows = []
        # 5 correct BUG/HIGH/no-escalate
        for _ in range(5):
            rows.append(compare_single(base_actual, base_expected))
        # 1 wrong category (actual FEATURE_REQUEST vs expected BILLING) — no escalation
        wrong = compare_single(
            _make_actual(category="feature_request"),
            _make_expected(expected_category="BILLING"),
        )
        # 1 false-positive escalation (expected no-escalate, model escalates)
        fp = compare_single(_make_actual(should_escalate=True), base_expected)
        # 1 correct escalation (expected escalate, model escalates) but wrong cat/urgency
        tp = compare_single(
            _make_actual(should_escalate=True),
            _make_expected(
                should_escalate=True,
                expected_category="ACCOUNT_ISSUE",
                expected_urgency="MEDIUM",
            ),
        )
        rows += [wrong, fp, tp]
        return rows

    def test_confusion_matrix_shape(self):
        metrics = compute_metrics(self._rows())
        assert len(metrics.category_confusion_matrix) == len(CATEGORIES) == 6
        assert len(metrics.urgency_confusion_matrix) == len(URGENCIES) == 4
        for row in metrics.category_confusion_matrix:
            assert len(row) == 6
        for row in metrics.urgency_confusion_matrix:
            assert len(row) == 4

    def test_accuracy_values(self):
        metrics = compute_metrics(self._rows())
        assert metrics.total == 8
        # category correct: 5 perfect + fp row = 6 (wrong & tp rows mismatched)
        assert metrics.category_accuracy == pytest.approx(6 / 8)
        # urgency correct: 5 perfect + wrong + fp = 7 (tp row mismatched)
        assert metrics.urgency_accuracy == pytest.approx(7 / 8)

    def test_escalation_ppv(self):
        metrics = compute_metrics(self._rows())
        # TP=1 (tp), FP=1 (fp), FN=0, TN=6 (5 perfect + wrong)
        assert metrics.escalation_precision == pytest.approx(1 / 2)
        assert metrics.escalation_recall == pytest.approx(1.0)
        assert metrics.escalation_f1 == pytest.approx(2 / 3)

    def test_false_escalation_rate(self):
        metrics = compute_metrics(self._rows())
        # auto-resolvable = FP + TN = 1 + 6, FP=1
        assert metrics.false_escalation_rate == pytest.approx(1 / 7)

    def test_latency_percentiles(self):
        metrics = compute_metrics(self._rows())
        assert metrics.duration_p50_ms > 0
        assert metrics.duration_p95_ms >= metrics.duration_p50_ms

    def test_by_source(self):
        metrics = compute_metrics(self._rows())
        assert "manual" in metrics.by_source
        assert metrics.by_source["manual"]["total"] == metrics.total


# ═══════════════════════════════════════════════════════════════════════════════
# harness (pure-logic paths, no network)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tiny_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "tickets.jsonl"
    rows = [
        _make_expected(id="t-1", source="huggingface"),
        _make_expected(
            id="t-2",
            source="github",
            expected_category="FEATURE_REQUEST",
            expected_urgency="LOW",
            should_escalate=True,
        ),
        _make_expected(id="t-3", source="manual"),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


class TestHarnessLoad:
    def test_load_all(self, tiny_fixture):
        harness = EvalHarness(fixture_path=tiny_fixture)
        tickets = harness.load_tickets()
        assert len(tickets) == 3

    def test_load_filtered(self, tiny_fixture):
        harness = EvalHarness(fixture_path=tiny_fixture)
        assert len(harness.load_tickets(source="github")) == 1
        assert len(harness.load_tickets(source="huggingface")) == 1
        assert len(harness.load_tickets(source="manual")) == 1

    def test_load_missing_fixture(self, tmp_path):
        harness = EvalHarness(fixture_path=tmp_path / "nope.jsonl")
        with pytest.raises(FileNotFoundError):
            harness.load_tickets()


class TestHarnessCompare:
    def test_compare_delegates(self, tiny_fixture):
        harness = EvalHarness(fixture_path=tiny_fixture)
        expected = harness.load_tickets()[0]
        actual = _make_actual()
        comp = harness.compare(actual, expected)
        assert comp["category_match"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# report
# ═══════════════════════════════════════════════════════════════════════════════


class TestReports:
    def test_build_run_report(self):
        expected = _make_expected()
        actual = _make_actual()
        metrics = compute_metrics([compare_single(actual, expected)])
        report = build_run_report(metrics, [], source="all", concurrency=5)
        assert report["metrics"]["category_accuracy"] > 0
        assert report["run"]["source"] == "all"
        assert "failing_cases" in report
        assert "thresholds" in report["run"]

    def test_save_json_and_html(self, tmp_path):
        expected = _make_expected()
        actual = _make_actual()
        comparisons = [compare_single(actual, expected)]
        metrics = compute_metrics(comparisons)

        json_path = save_json_report(metrics, comparisons, output_path=tmp_path / "latest.json")
        html_path = save_html_report(metrics, comparisons, output_path=tmp_path / "report.html")

        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["metrics"]["total"] == 1

        html = html_path.read_text(encoding="utf-8")
        assert "<table" in html
        assert "Category accuracy" in html
        assert "heatmap" in html
