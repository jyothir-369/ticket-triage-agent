"""Evaluation metrics for the triage agent.

Provides scoring functions that compare actual agent outputs against
expected (ground-truth) values from the labeled eval fixture.

Metrics:
  - category_accuracy: exact match of predicted category vs expected
  - urgency_accuracy: exact match of predicted urgency vs expected
  - draft_similarity: ROUGE-L F1 score vs human-approved draft
  - escalation_correctness: should_escalate vs actual escalation decision
  - confidence_threshold_accuracy: whether confidence > threshold correctly
    predicts escalation
  - retrieval_recall_at_1: fraction of expected doc IDs found in the top-1
    retrieved result (Recall@1)
  - retrieval_precision_at_5: fraction of top-5 retrieved docs that are
    expected (Precision@5)
  - retrieval_ndcg_at_10: normalized discounted cumulative gain@10 over the
    retrieved list (nDCG@10)
  - retrieval_pass_rate: fraction of tickets where recall@1 >= 0 AND
    precision@5 >= 0 (combined retrieval quality gate)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L implementation (pure Python, no external dependency)
# ═══════════════════════════════════════════════════════════════════════════════


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the Longest Common Subsequence.

    Uses the standard dynamic-programming approach with O(m*n) time
    and O(min(m,n)) space via a rolling array.
    """
    if not a or not b:
        return 0
    # Ensure b is the shorter sequence for space optimisation
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[len(b)]


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r"\w+", text.lower())


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference.

    ROUGE-L measures the longest common subsequence between two texts,
    capturing sentence-level structure similarity.

    Returns
    -------
    float
        F1 score in [0, 1] where 1 is a perfect match.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    return 2.0 * (precision * recall) / (precision + recall)


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword overlap score (supplementary to ROUGE-L)
# ═══════════════════════════════════════════════════════════════════════════════


def keyword_overlap(prediction: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the prediction text.

    Parameters
    ----------
    prediction:
        The agent-generated draft text.
    keywords:
        List of keywords that should appear in a good draft.

    Returns
    -------
    float
        Fraction of keywords present (0.0–1.0).
    """
    if not keywords:
        return 1.0
    pred_lower = prediction.lower()
    found = sum(1 for kw in keywords if kw.lower() in pred_lower)
    return found / len(keywords)


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval relevance metrics
# ═══════════════════════════════════════════════════════════════════════════════


def _recall_at_k(expected_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    """Recall@K: fraction of expected document IDs present in the top-K results."""
    expected = set(expected_ids)
    retrieved_top = set(retrieved_ids[:k]) if k > 0 else set()
    if not expected:
        return 0.0
    hits = len(expected & retrieved_top)
    return hits / len(expected)


def _precision_at_k(expected_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    """Precision@K: fraction of the top-K retrieved docs that are expected."""
    expected = set(expected_ids)
    top_k = retrieved_ids[:k] if k > 0 else []
    if not top_k:
        return 0.0
    hits = len(set(top_k) & expected)
    return hits / k


def _ndcg_at_k(expected_ids: list[str], retrieved_ids: list[str], k: int, idf: dict[str, float] | None = None) -> float:
    """Normalized Discounted Cumulative Gain@K over retrieved docs.

    Ranking is position in *retrieved_ids*; an item at position i (0-indexed)
    contributes its "relevance" (boolean membership of expected ids) discounted
    by the formula log2(1 + rank_in_seen_window). The first retrieved item
    always contributes log2(1 + 0) = 0 if it is relevant.

    Simplest correct nDCG: each retrieved doc gets +1 if it is in the expected
    set, discounted by 1/(rank+1). The ideal DCG is sum over expected docs in
    retrieved list of 1/(pos+1), normalized by log2(k+1).
    """
    k = max(k, 1)
    ideal_dcg = sum(1.0 / (i + 1) for i in range(min(len(expected_ids), k)))
    n = min(len(retrieved_ids), k)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:n]):
        if doc_id in set(expected_ids):
            dcg += 1.0 / (i + 1)
    if dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def compute_retrieval_metrics(
    expected_ids: list[str],
    retrieved_ids: list[str],
) -> tuple[float, float, float]:
    """Return ``(recall_at_1, precision_at_5, ndcg_at_10)``.

    Parameters
    ----------
    expected_ids:
        Ground-truth document IDs that *should* have been returned by retrieval.
    retrieved_ids:
        Actual document IDs returned by the retriever in order (highest score
        first).

    Returns
    -------
    tuple[float, float, float]
    """
    if not expected_ids and not retrieved_ids:
        return 0.0, 0.0, 0.0

    recall_at_1 = _recall_at_k(expected_ids, retrieved_ids, 1)
    precision_at_5 = _precision_at_k(expected_ids, retrieved_ids, 5)
    ndcg_at_10 = _ndcg_at_k(expected_ids, retrieved_ids, 10)

    return recall_at_1, precision_at_5, ndcg_at_10


def compute_retrieval_pass_rate(recall_at_1: float, precision_at_5: float) -> bool:
    """Return True when both recall@1 >= 0 AND precision@5 >= 0."""
    return bool(recall_at_1 >= 0.0 and precision_at_5 >= 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric data class
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MetricResult:
    """Result of comparing a single ticket's actual output vs expected."""

    ticket_id: str
    category_correct: bool = False
    urgency_correct: bool = False
    draft_rouge_l: float = 0.0
    draft_keyword_overlap: float = 0.0
    escalation_correct: bool = False
    confidence_above_threshold: bool = False
    confidence_correctly_predicts_escalation: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    expected_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "category_correct": self.category_correct,
            "urgency_correct": self.urgency_correct,
            "draft_rouge_l": round(self.draft_rouge_l, 4),
            "draft_keyword_overlap": round(self.draft_keyword_overlap, 4),
            "escalation_correct": self.escalation_correct,
            "confidence_above_threshold": self.confidence_above_threshold,
            "confidence_correctly_predicts_escalation": self.confidence_correctly_predicts_escalation,
            "details": self.details,
            "expected_doc_ids": list(self.expected_doc_ids),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison function
# ═══════════════════════════════════════════════════════════════════════════════


def compare_results(
    actual: dict[str, Any],
    expected: dict[str, Any],
    confidence_threshold: float = 0.7,
) -> MetricResult:
    """Compare actual agent output against expected ground truth.

    Parameters
    ----------
    actual:
        Agent output dict with keys: category, urgency, confidence,
        draft_text (or draft dict), should_escalate.
    expected:
        Expected values from the eval fixture with keys: expected_category,
        expected_urgency, should_escalate, human_approved_draft,
        expected_draft_keywords.
    confidence_threshold:
        The threshold used by the agent to decide escalation.

    Returns
    -------
    MetricResult
        Comparison results for this ticket.
    """
    ticket_id = actual.get("ticket_id", "unknown")

    # ── Category accuracy ────────────────────────────────────────────────────
    actual_category = (actual.get("category") or "").lower()
    expected_category = (expected.get("expected_category") or "").lower()
    category_correct = actual_category == expected_category

    # ── Urgency accuracy ─────────────────────────────────────────────────────
    actual_urgency = (actual.get("urgency") or "").lower()
    expected_urgency = (expected.get("expected_urgency") or "").lower()
    urgency_correct = actual_urgency == expected_urgency

    # ── Draft similarity ─────────────────────────────────────────────────────
    actual_draft = actual.get("draft_text") or actual.get("draft") or ""
    if isinstance(actual_draft, dict):
        actual_draft = actual_draft.get("draft_text", "")
    expected_draft = expected.get("human_approved_draft", "")
    expected_keywords = expected.get("expected_draft_keywords", [])

    draft_rouge = rouge_l_f1(actual_draft, expected_draft)
    draft_kw_overlap = keyword_overlap(actual_draft, expected_keywords)

    # ── Escalation correctness ───────────────────────────────────────────────
    expected_should_escalate = expected.get("should_escalate", False)
    actual_should_escalate = actual.get("should_escalate", False)
    escalation_correct = expected_should_escalate == actual_should_escalate

    # ── Confidence threshold accuracy ────────────────────────────────────────
    confidence = actual.get("confidence", 0.0)
    above_threshold = confidence >= confidence_threshold

    # The confidence correctly predicts escalation when:
    #   - above threshold AND no escalation expected → correct (auto-resolve)
    #   - below threshold AND escalation expected → correct (should escalate)
    #   - above threshold AND escalation expected → incorrect (should've escalated)
    #   - below threshold AND no escalation expected → incorrect (unnecessary escalation)
    if above_threshold and not expected_should_escalate:
        confidence_correctly_predicts = True
    elif not above_threshold and expected_should_escalate:
        confidence_correctly_predicts = True
    elif above_threshold and expected_should_escalate:
        confidence_correctly_predicts = False
    else:
        confidence_correctly_predicts = False

    return MetricResult(
        ticket_id=ticket_id,
        category_correct=category_correct,
        urgency_correct=urgency_correct,
        draft_rouge_l=draft_rouge,
        draft_keyword_overlap=draft_kw_overlap,
        escalation_correct=escalation_correct,
        confidence_above_threshold=above_threshold,
        confidence_correctly_predicts_escalation=confidence_correctly_predicts,
        details={
            "actual_category": actual_category,
            "expected_category": expected_category,
            "actual_urgency": actual_urgency,
            "expected_urgency": expected_urgency,
            "actual_should_escalate": actual_should_escalate,
            "expected_should_escalate": expected_should_escalate,
            "confidence": confidence,
            "draft_length": len(actual_draft),
        },
        expected_doc_ids=list(expected.get("expected_doc_ids") or []),
        retrieved_doc_ids=list(actual.get("retrieved_doc_ids") or []),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate metrics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AggregateMetrics:
    """Aggregated evaluation metrics across all test tickets."""

    total: int = 0
    category_accuracy: float = 0.0
    urgency_accuracy: float = 0.0
    avg_draft_rouge_l: float = 0.0
    avg_draft_keyword_overlap: float = 0.0
    escalation_accuracy: float = 0.0
    confidence_threshold_accuracy: float = 0.0
    pass_rate: float = 0.0
    retrieval_recall_at_1: float = 0.0
    retrieval_precision_at_5: float = 0.0
    retrieval_ndcg_at_10: float = 0.0
    retrieval_pass_rate: float = 0.0
    results: list[MetricResult] = field(default_factory=list)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "category_accuracy": round(self.category_accuracy, 4),
            "urgency_accuracy": round(self.urgency_accuracy, 4),
            "avg_draft_rouge_l": round(self.avg_draft_rouge_l, 4),
            "avg_draft_keyword_overlap": round(self.avg_draft_keyword_overlap, 4),
            "escalation_accuracy": round(self.escalation_accuracy, 4),
            "confidence_threshold_accuracy": round(self.confidence_threshold_accuracy, 4),
            "pass_rate": round(self.pass_rate, 4),
            "retrieval_recall_at_1": round(self.retrieval_recall_at_1, 4),
            "retrieval_precision_at_5": round(self.retrieval_precision_at_5, 4),
            "retrieval_ndcg_at_10": round(self.retrieval_ndcg_at_10, 4),
            "retrieval_pass_rate": round(self.retrieval_pass_rate, 4),
            "confusion_matrix": self.confusion_matrix,
        }


def aggregate_metrics(results: list[MetricResult]) -> AggregateMetrics:
    """Compute aggregate metrics from individual ticket results.

    Parameters
    ----------
    results:
        List of MetricResult from compare_results() calls.

    Returns
    -------
    AggregateMetrics
        Summary metrics with averages and accuracy rates.
    """
    if not results:
        return AggregateMetrics()

    total = len(results)

    category_correct = sum(1 for r in results if r.category_correct)
    urgency_correct = sum(1 for r in results if r.urgency_correct)
    escalation_correct = sum(1 for r in results if r.escalation_correct)
    confidence_correct = sum(1 for r in results if r.confidence_correctly_predicts_escalation)

    avg_rouge = sum(r.draft_rouge_l for r in results) / total
    avg_kw = sum(r.draft_keyword_overlap for r in results) / total

    # A ticket "passes" if category, urgency, and escalation are all correct
    passes = sum(
        1 for r in results
        if r.category_correct and r.urgency_correct and r.escalation_correct
    )

    # Retrieval aggregate metrics
    rec1 = sum(1 for r in results if _recall_at_k(r.expected_doc_ids, r.retrieved_doc_ids, 1) > 0)
    prec5 = sum(
        1 for r in results
        if _precision_at_k(r.expected_doc_ids, r.retrieved_doc_ids, 5) > 0
    )
    ndcg10 = sum(1 for r in results if _ndcg_at_k(r.expected_doc_ids, r.retrieved_doc_ids, 10) > 0)
    rc_rate = sum(1 for r in results if compute_retrieval_pass_rate(
        _recall_at_k(r.expected_doc_ids, r.retrieved_doc_ids, 1),
        _precision_at_k(r.expected_doc_ids, r.retrieved_doc_ids, 5),
    ))

    tp = sum(1 for r in results if r.details.get("expected_should_escalate") and r.details.get("actual_should_escalate"))
    tn = sum(1 for r in results if not r.details.get("expected_should_escalate") and not r.details.get("actual_should_escalate"))
    fp = sum(1 for r in results if not r.details.get("expected_should_escalate") and r.details.get("actual_should_escalate"))
    fn = sum(1 for r in results if r.details.get("expected_should_escalate") and not r.details.get("actual_should_escalate"))

    return AggregateMetrics(
        total=total,
        category_accuracy=category_correct / total,
        urgency_accuracy=urgency_correct / total,
        avg_draft_rouge_l=avg_rouge,
        avg_draft_keyword_overlap=avg_kw,
        escalation_accuracy=escalation_correct / total,
        confidence_threshold_accuracy=confidence_correct / total,
        pass_rate=passes / total,
        retrieval_recall_at_1=rec1 / total,
        retrieval_precision_at_5=prec5 / total,
        retrieval_ndcg_at_10=ndcg10 / total,
        retrieval_pass_rate=rc_rate / total,
        results=results,
        confusion_matrix={
            "escalation": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        },
    )
