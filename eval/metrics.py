"""Evaluation metrics for the triage agent.

Pure-Python metric computation over the per-ticket results produced by
:mod:`eval.harness`.  Everything is implemented with the standard library
only (including ROUGE-L), so evaluation runs identically in a bare CI
environment without pulling extra dependencies.

Metrics computed by :func:`compute_metrics`:

- ``category_accuracy``          — exact match of predicted vs expected category.
- ``urgency_accuracy``           — exact match of predicted vs expected urgency.
- ``category_confusion_matrix``  — 6×6 (rows = actual, cols = predicted).
- ``urgency_confusion_matrix``   — 4×4.
- ``mean_draft_rouge_l``         — mean ROUGE-L F1 of draft vs human draft.
- ``escalation_precision/recall/f1``.
- ``false_escalation_rate``      — fraction of auto-resolve tickets that were escalated.
- ``calibration_error``          — fraction of tickets where confidence < 0.7 did
  NOT predict a wrong answer (see :ref:`confidence_calibrated`).
- ``duration_p50_ms / duration_p95_ms``.
- ``by_source``                  — per-source accuracy breakdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

# The canonical label ordering used everywhere (fixtures use these EXACT strings).
CATEGORIES: list[str] = [
    "BUG",
    "FEATURE_REQUEST",
    "ACCOUNT_ISSUE",
    "BILLING",
    "USAGE_HELP",
    "OTHER",
]

URGENCIES: list[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L (pure Python, no external dependency)
# ═══════════════════════════════════════════════════════════════════════════════


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the Longest Common Subsequence of *a* and *b*.

    Standard DP with a rolling array → O(m·n) time, O(min(m, n)) space.
    """
    if not a or not b:
        return 0
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
    """Lowercase and split text into word tokens (``\\w+``)."""
    return re.findall(r"\w+", text.lower())


def rouge_l_f1(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score between *prediction* and *reference* (0.0–1.0).

    ROUGE-L captures the longest common subsequence of words, so it rewards
    the agent when the draft mirrors the human draft's sentence structure.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


# ═══════════════════════════════════════════════════════════════════════════════
# Small stats helpers
# ═══════════════════════════════════════════════════════════════════════════════


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    ``p`` is a percentage in [0, 100].  Empty input returns 0.0.
    """
    if not sorted_values:
        return 0.0
    if p >= 100:
        return sorted_values[-1]
    if p <= 0:
        return sorted_values[0]
    rank = max(1, round(p / 100.0 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Per-ticket comparison
# ═══════════════════════════════════════════════════════════════════════════════


def compare_single(
    actual: dict[str, Any],
    expected: dict[str, Any],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Compare one ticket's agent output against its ground-truth label.

    Parameters
    ----------
    actual:
        The dict produced by ``EvalHarness.run_single`` — has ``category``,
        ``urgency``, ``confidence``, ``draft``, ``should_escalate``,
        ``latency_ms``.
    expected:
        The fixture label dict for the same ticket — has ``expected_category``,
        ``expected_urgency``, ``should_escalate``, ``human_approved_draft``.
    confidence_threshold:
        Confidence above which the agent's own classification counts as
        "confident".

    Returns
    -------
    dict
        A comparison row with the five spec'd signals plus context:

        - ``category_match``        — bool, exact category match.
        - ``urgency_match``         — bool, exact urgency match.
        - ``draft_similarity``      — ROUGE-L F1 vs ``human_approved_draft``.
        - ``escalation_correct``    — bool, matches expected ``should_escalate``.
        - ``confidence_calibrated`` — bool; a ticket is *calibrated* when a
          correct answer is given confidently (≥ threshold) or a wrong answer
          is given unconfidently (< threshold) — i.e. confidence tracks
          correctness.
    """
    normalized = lambda value: str(value or "").strip().upper()  # noqa: E731

    actual_category = normalized(actual.get("category"))
    expected_category = normalized(expected.get("expected_category"))
    actual_urgency = normalized(actual.get("urgency"))
    expected_urgency = normalized(expected.get("expected_urgency"))

    category_match = actual_category == expected_category
    urgency_match = actual_urgency == expected_urgency

    actual_draft = actual.get("draft") or actual.get("draft_text") or ""
    expected_draft = expected.get("human_approved_draft", "")
    draft_similarity = rouge_l_f1(str(actual_draft), str(expected_draft))

    should_escalate = bool(expected.get("should_escalate"))
    actual_escalate = bool(actual.get("should_escalate"))
    escalation_correct = should_escalate == actual_escalate

    confidence = float(actual.get("confidence") or 0.0)
    correct = category_match and urgency_match
    # Confidence is "calibrated" when it points in the right direction:
    #   confident(≥t) + correct  ⇒ calibrated
    #   unconfident(<t) + wrong  ⇒ calibrated (signals "not sure" on a miss)
    confidence_calibrated = (confidence >= confidence_threshold and correct) or (
        confidence < confidence_threshold and not correct
    )

    return {
        "id": expected.get("id", actual.get("id", actual.get("ticket_id"))),
        "source": expected.get("source", actual.get("source", "unknown")),
        "category_match": category_match,
        "urgency_match": urgency_match,
        "draft_similarity": round(draft_similarity, 4),
        "escalation_correct": escalation_correct,
        "confidence_calibrated": confidence_calibrated,
        "confidence": round(confidence, 4),
        "latency_ms": int(actual.get("latency_ms") or 0),
        "retrieval_count": int(actual.get("retrieval_count") or 0),
        # --- context for reports / debugging ---
        "actual_category": actual_category,
        "expected_category": expected_category,
        "actual_urgency": actual_urgency,
        "expected_urgency": expected_urgency,
        "actual_should_escalate": actual_escalate,
        "expected_should_escalate": should_escalate,
        "draft": str(actual_draft),
        "human_draft": str(expected_draft),
        "error": actual.get("error"),
        "notes": expected.get("notes", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregated metrics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics across all evaluated tickets."""

    total: int = 0
    n_correct_category: int = 0
    n_correct_urgency: int = 0
    category_accuracy: float = 0.0
    urgency_accuracy: float = 0.0
    mean_draft_rouge_l: float = 0.0
    escalation_precision: float = 0.0
    escalation_recall: float = 0.0
    escalation_f1: float = 0.0
    escalation_accuracy: float = 0.0
    false_escalation_rate: float = 0.0
    calibration_error: float = 0.0
    duration_p50_ms: float = 0.0
    duration_p95_ms: float = 0.0
    category_confusion_matrix: list[list[int]] = field(default_factory=list)
    category_labels: list[str] = field(default_factory=lambda: list(CATEGORIES))
    urgency_confusion_matrix: list[list[int]] = field(default_factory=list)
    urgency_labels: list[str] = field(default_factory=lambda: list(URGENCIES))
    escalation_confusion: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict of every metric (rounded)."""
        return {
            "total": self.total,
            "category_accuracy": round(self.category_accuracy, 4),
            "urgency_accuracy": round(self.urgency_accuracy, 4),
            "mean_draft_rouge_l": round(self.mean_draft_rouge_l, 4),
            "escalation_precision": round(self.escalation_precision, 4),
            "escalation_recall": round(self.escalation_recall, 4),
            "escalation_f1": round(self.escalation_f1, 4),
            "escalation_accuracy": round(self.escalation_accuracy, 4),
            "false_escalation_rate": round(self.false_escalation_rate, 4),
            "calibration_error": round(self.calibration_error, 4),
            "duration_p50_ms": round(self.duration_p50_ms, 1),
            "duration_p95_ms": round(self.duration_p95_ms, 1),
            "category_labels": self.category_labels,
            "category_confusion_matrix": self.category_confusion_matrix,
            "urgency_labels": self.urgency_labels,
            "urgency_confusion_matrix": self.urgency_confusion_matrix,
            "escalation_confusion": self.escalation_confusion,
            "by_source": self.by_source,
        }


def _confusion_matrix(
    results: list[dict[str, Any]],
    labels: list[str],
    actual_key: str,
    predicted_key: str,
) -> list[list[int]]:
    """Build a *labels*×*labels* confusion matrix: rows actual, cols predicted."""
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for r in results:
        actual = str(r.get(actual_key) or "").upper()
        predicted = str(r.get(predicted_key) or "").upper()
        ai = idx.get(actual)
        pi = idx.get(predicted)
        if ai is not None and pi is not None:
            matrix[ai][pi] += 1
    return matrix


def compute_metrics(results: list[dict[str, Any]]) -> EvalMetrics:
    """Aggregate single-ticket comparison rows into :class:`EvalMetrics`.

    Parameters
    ----------
    results:
        List of comparison dicts from :func:`compare_single`.
    """
    if not results:
        return EvalMetrics()

    total = len(results)

    cat_correct = sum(1 for r in results if r["category_match"])
    urg_correct = sum(1 for r in results if r["urgency_match"])

    tp = sum(1 for r in results if r["expected_should_escalate"] and r["actual_should_escalate"])
    fp = sum(
        1 for r in results if not r["expected_should_escalate"] and r["actual_should_escalate"]
    )
    fn = sum(
        1 for r in results if r["expected_should_escalate"] and not r["actual_should_escalate"]
    )
    tn = sum(
        1 for r in results if not r["expected_should_escalate"] and not r["actual_should_escalate"]
    )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # False-escalation rate: fraction of auto-resolvable tickets that got escalated.
    auto_resolvable = fp + tn
    false_escalation_rate = _safe_div(fp, auto_resolvable)

    calibration_error = _safe_div(sum(1 for r in results if not r["confidence_calibrated"]), total)

    latencies = sorted(int(r["latency_ms"]) for r in results)

    cat_cm = _confusion_matrix(results, CATEGORIES, "actual_category", "expected_category")
    urg_cm = _confusion_matrix(results, URGENCIES, "actual_urgency", "expected_urgency")

    # Per-source breakdown.
    by_source: dict[str, dict[str, float]] = {}
    sources = sorted({r.get("source", "unknown") for r in results})
    for source in sources:
        rows = [r for r in results if r.get("source") == source]
        n = len(rows)
        by_source[source] = {
            "total": n,
            "category_accuracy": round(
                _safe_div(sum(1 for r in rows if r["category_match"]), n), 4
            ),
            "urgency_accuracy": round(_safe_div(sum(1 for r in rows if r["urgency_match"]), n), 4),
            "escalation_accuracy": round(
                _safe_div(sum(1 for r in rows if r["escalation_correct"]), n), 4
            ),
            "mean_draft_rouge_l": round(_safe_div(sum(r["draft_similarity"] for r in rows), n), 4),
        }

    return EvalMetrics(
        total=total,
        n_correct_category=cat_correct,
        n_correct_urgency=urg_correct,
        category_accuracy=_safe_div(cat_correct, total),
        urgency_accuracy=_safe_div(urg_correct, total),
        mean_draft_rouge_l=_safe_div(sum(r["draft_similarity"] for r in results), total),
        escalation_precision=precision,
        escalation_recall=recall,
        escalation_f1=f1,
        escalation_accuracy=_safe_div(tp + tn, total),
        false_escalation_rate=false_escalation_rate,
        calibration_error=calibration_error,
        duration_p50_ms=percentile(latencies, 50),
        duration_p95_ms=percentile(latencies, 95),
        category_confusion_matrix=cat_cm,
        urgency_confusion_matrix=urg_cm,
        escalation_confusion={"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        by_source=by_source,
    )
