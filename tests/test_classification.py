"""Comprehensive tests for the TicketClassifier classification service.

Covers: prompt template, few-shot examples, token counting, heuristic fallback,
error handling, confidence calibration, factory function, and edge cases.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.schemas import TicketCategory, TicketClassification, UrgencyLevel
from src.services.classification import (
    CATEGORY_KEYWORDS,
    MODEL_COST_RATES,
    SYSTEM_PROMPT,
    URGENCY_KEYWORDS,
    TicketClassifier,
    TokenUsage,
    get_classifier,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def classifier() -> TicketClassifier:
    """Default TicketClassifier for tests (no LLM needed for unit tests)."""
    return TicketClassifier(
        model_name="gpt-4o-mini",
        timeout_seconds=10.0,
        max_retries=3,
        confidence_threshold=0.5,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Template Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptTemplate:
    """Verify the SYSTEM_PROMPT contains required elements."""

    def test_all_six_categories_present(self) -> None:
        for cat in TicketCategory:
            assert cat.value.upper() in SYSTEM_PROMPT, (
                f"Category {cat.value} missing from prompt"
            )

    def test_all_four_urgency_levels_present(self) -> None:
        for urg in UrgencyLevel:
            assert urg.value.upper() in SYSTEM_PROMPT, (
                f"Urgency {urg.value} missing from prompt"
            )

    def test_eight_few_shot_examples(self) -> None:
        # Each example starts with "### Example N"
        example_count = SYSTEM_PROMPT.count("### Example")
        assert example_count == 8, f"Expected 8 few-shot examples, found {example_count}"

    def test_output_format_specified(self) -> None:
        assert "category" in SYSTEM_PROMPT
        assert "urgency" in SYSTEM_PROMPT
        assert "confidence" in SYSTEM_PROMPT
        assert "reasoning" in SYSTEM_PROMPT

    def test_json_format_instruction(self) -> None:
        assert "JSON" in SYSTEM_PROMPT
        assert "no markdown fences" in SYSTEM_PROMPT.lower() or "no markdown" in SYSTEM_PROMPT.lower()

    def test_category_descriptions_present(self) -> None:
        """Each category should have a description in the prompt."""
        assert "Software defect" in SYSTEM_PROMPT or "crash" in SYSTEM_PROMPT.lower()
        assert "New functionality" in SYSTEM_PROMPT or "feature request" in SYSTEM_PROMPT.lower()
        assert "Account access" in SYSTEM_PROMPT or "authentication" in SYSTEM_PROMPT.lower()
        assert "Payment" in SYSTEM_PROMPT or "subscription" in SYSTEM_PROMPT.lower()
        assert "how to use" in SYSTEM_PROMPT.lower() or "questions about" in SYSTEM_PROMPT.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword Pattern Coverage Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKeywordPatterns:
    """Verify the keyword heuristic patterns cover all categories and urgency levels."""

    def test_all_categories_have_keywords(self) -> None:
        for cat in TicketCategory:
            assert cat in CATEGORY_KEYWORDS, f"Category {cat.value} has no keywords"
            assert len(CATEGORY_KEYWORDS[cat]) >= 5, (
                f"Category {cat.value} has fewer than 5 keywords"
            )

    def test_all_urgency_levels_have_keywords(self) -> None:
        for urg in UrgencyLevel:
            assert urg in URGENCY_KEYWORDS, f"Urgency {urg.value} has no keywords"
            assert len(URGENCY_KEYWORDS[urg]) >= 3, (
                f"Urgency {urg.value} has fewer than 3 keywords"
            )

    def test_no_overlapping_category_keywords(self) -> None:
        """No keyword should appear in two category lists."""
        seen: dict[str, str] = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                assert kw not in seen, (
                    f"Keyword '{kw}' appears in both {seen[kw]} and {cat.value}"
                )
                seen[kw] = cat.value


# ═══════════════════════════════════════════════════════════════════════════════
# Token Counting Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenCounting:
    """Verify tiktoken-based token counting and cost estimation."""

    def test_count_tokens_returns_positive(self, classifier: TicketClassifier) -> None:
        tokens = classifier.count_tokens("Hello, world!")
        assert tokens > 0

    def test_count_tokens_empty_string(self, classifier: TicketClassifier) -> None:
        tokens = classifier.count_tokens("")
        assert tokens == 0

    def test_count_tokens_long_text(self, classifier: TicketClassifier) -> None:
        text = "This is a test sentence. " * 100
        tokens = classifier.count_tokens(text)
        assert tokens > 100

    def test_estimate_cost_known_model(self, classifier: TicketClassifier) -> None:
        cost = classifier.estimate_cost(1000, 500)
        assert cost > 0

    def test_estimate_cost_unknown_model_uses_fallback(self) -> None:
        classifier = TicketClassifier(model_name="unknown-model-xyz")
        cost = classifier.estimate_cost(1000, 500)
        assert cost > 0  # Falls back to Sonnet 4 pricing

    def test_estimate_cost_proportional_to_tokens(self, classifier: TicketClassifier) -> None:
        cost_1k = classifier.estimate_cost(1000, 0)
        cost_2k = classifier.estimate_cost(2000, 0)
        assert cost_2k == pytest.approx(cost_1k * 2, rel=1e-6)

    def test_model_cost_rates_complete(self) -> None:
        """All expected models should have pricing defined."""
        expected_models = [
            "claude-opus-5", "claude-opus-4-8", "claude-sonnet-5",
            "claude-haiku-4-5", "gpt-4o", "gpt-4o-mini",
        ]
        for model in expected_models:
            assert model in MODEL_COST_RATES, f"Model {model} missing from MODEL_COST_RATES"
            inp, out = MODEL_COST_RATES[model]
            assert inp > 0 and out > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Token Usage Tracking Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenUsage:
    """Verify the TokenUsage dataclass and tracking methods."""

    def test_token_usage_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.estimated_cost_usd == 0.0

    def test_get_token_usage_initial(self, classifier: TicketClassifier) -> None:
        usage = classifier.get_token_usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model == "gpt-4o-mini"

    def test_get_cost_summary_initial(self, classifier: TicketClassifier) -> None:
        summary = classifier.get_cost_summary()
        assert summary["input_tokens"] == 0
        assert summary["output_tokens"] == 0
        assert summary["estimated_cost_usd"] == 0.0
        assert summary["model"] == "gpt-4o-mini"

    def test_reset_usage(self, classifier: TicketClassifier) -> None:
        classifier._input_tokens = 100
        classifier._output_tokens = 50
        classifier._total_cost = 0.01
        classifier.reset_usage()
        usage = classifier.get_token_usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.estimated_cost_usd == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Response Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseResponse:
    """Verify _parse_response handles various LLM output formats."""

    def test_parse_valid_json(self, classifier: TicketClassifier) -> None:
        raw = json.dumps({
            "category": "BUG",
            "urgency": "HIGH",
            "confidence": 0.9,
            "reasoning": "Clear bug report.",
        })
        result = classifier._parse_response(raw)
        assert result.category == TicketCategory.BUG
        assert result.urgency == UrgencyLevel.HIGH
        assert result.confidence == pytest.approx(0.9, abs=0.01)
        assert "Clear bug report" in result.reasoning

    def test_parse_with_markdown_fences(self, classifier: TicketClassifier) -> None:
        raw = '```json\n{"category": "BILLING", "urgency": "MEDIUM", "confidence": 0.8, "reasoning": "Billing issue."}\n```'
        result = classifier._parse_response(raw)
        assert result.category == TicketCategory.BILLING
        assert result.urgency == UrgencyLevel.MEDIUM

    def test_parse_lowercase_values(self, classifier: TicketClassifier) -> None:
        raw = json.dumps({
            "category": "bug",
            "urgency": "high",
            "confidence": 0.85,
            "reasoning": "Bug.",
        })
        result = classifier._parse_response(raw)
        assert result.category == TicketCategory.BUG
        assert result.urgency == UrgencyLevel.HIGH

    def test_parse_unknown_category_defaults_to_other(self, classifier: TicketClassifier) -> None:
        raw = json.dumps({
            "category": "UNKNOWN_CATEGORY",
            "urgency": "MEDIUM",
            "confidence": 0.5,
            "reasoning": "Unclear.",
        })
        result = classifier._parse_response(raw)
        assert result.category == TicketCategory.OTHER

    def test_parse_unknown_urgency_defaults_to_medium(self, classifier: TicketClassifier) -> None:
        raw = json.dumps({
            "category": "BUG",
            "urgency": "UNKNOWN",
            "confidence": 0.5,
            "reasoning": "Bug.",
        })
        result = classifier._parse_response(raw)
        assert result.urgency == UrgencyLevel.MEDIUM

    def test_parse_confidence_clamped_above(self, classifier: TicketClassifier) -> None:
        raw = json.dumps({
            "category": "BUG",
            "urgency": "LOW",
            "confidence": 1.5,
            "reasoning": "Bug.",
        })
        result = classifier._parse_response(raw)
        assert result.confidence == 1.0

    def test_parse_confidence_clamped_below(self, classifier: TicketClassifier) -> None:
        raw = json.dumps({
            "category": "BUG",
            "urgency": "LOW",
            "confidence": -0.3,
            "reasoning": "Bug.",
        })
        result = classifier._parse_response(raw)
        assert result.confidence == 0.0

    def test_parse_invalid_json_raises(self, classifier: TicketClassifier) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            classifier._parse_response("this is not json")

    def test_parse_missing_confidence_defaults(self, classifier: TicketClassifier) -> None:
        raw = json.dumps({
            "category": "BUG",
            "urgency": "LOW",
            "reasoning": "Bug.",
        })
        result = classifier._parse_response(raw)
        assert result.confidence == 0.5  # default

    def test_parse_truncates_long_reasoning(self, classifier: TicketClassifier) -> None:
        raw = json.dumps({
            "category": "BUG",
            "urgency": "LOW",
            "confidence": 0.8,
            "reasoning": "x" * 3000,
        })
        result = classifier._parse_response(raw)
        assert len(result.reasoning) <= 2000


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic Fallback Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHeuristicFallback:
    """Verify keyword-based heuristic fallback classification."""

    def test_bug_crash_detected(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "The app crashes every time I open it. Error 500 on the server."
        )
        assert result.category == TicketCategory.BUG
        assert result.confidence <= 0.5

    def test_feature_request_detected(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "Please add dark mode support to the web application."
        )
        assert result.category == TicketCategory.FEATURE_REQUEST

    def test_account_issue_detected(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "I can't log in to my account. Password reset isn't working."
        )
        assert result.category == TicketCategory.ACCOUNT_ISSUE

    def test_billing_detected(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "I was charged twice for my subscription. Please refund."
        )
        assert result.category == TicketCategory.BILLING

    def test_usage_help_detected(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "How do I export my data as CSV from the dashboard?"
        )
        assert result.category == TicketCategory.USAGE_HELP

    def test_critical_urgency_detected(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "System is down. Complete outage. All users affected."
        )
        assert result.urgency == UrgencyLevel.CRITICAL

    def test_high_urgency_detected(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "This is urgent and blocking our team. No workaround available."
        )
        assert result.urgency == UrgencyLevel.HIGH

    def test_low_urgency_detected(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "Nice to have: would be great if you could add a dark mode."
        )
        assert result.urgency == UrgencyLevel.LOW

    def test_no_keywords_defaults_to_other_medium(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback("xyzzy foobar baz qux")
        assert result.category == TicketCategory.OTHER
        assert result.urgency == UrgencyLevel.MEDIUM

    def test_heuristic_confidence_capped_at_05(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback(
            "crash error broken bug fails exception 500 outage down critical"
        )
        assert result.confidence <= 0.5

    def test_heuristic_has_reasoning(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback("Login is broken")
        assert "Heuristic fallback" in result.reasoning
        assert "keyword matching" in result.reasoning.lower() or "keyword" in result.reasoning.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence Calibration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfidenceCalibration:
    """Verify confidence calibration dampens hedged predictions."""

    def test_high_confidence_no_hedging_unchanged(self, classifier: TicketClassifier) -> None:
        classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.9,
            reasoning="Clear bug report with no ambiguity.",
        )
        classifier._calibrate_confidence(classification)
        assert classification.confidence == pytest.approx(0.9, abs=0.01)

    def test_high_confidence_with_hedging_reduced(self, classifier: TicketClassifier) -> None:
        classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.HIGH,
            confidence=0.9,
            reasoning="This might be a bug, possibly related to the login module.",
        )
        original = classification.confidence
        classifier._calibrate_confidence(classification)
        assert classification.confidence < original

    def test_low_confidence_not_reduced(self, classifier: TicketClassifier) -> None:
        classification = TicketClassification(
            category=TicketCategory.OTHER,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.4,
            reasoning="Possibly a billing issue, hard to tell.",
        )
        classifier._calibrate_confidence(classification)
        # Low confidence should NOT be reduced further
        assert classification.confidence == pytest.approx(0.4, abs=0.01)

    def test_confidence_capped_at_095(self, classifier: TicketClassifier) -> None:
        classification = TicketClassification(
            category=TicketCategory.BUG,
            urgency=UrgencyLevel.CRITICAL,
            confidence=0.99,
            reasoning="Definite critical bug.",
        )
        classifier._calibrate_confidence(classification)
        assert classification.confidence == 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# Async Classification Tests (with mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════


class TestClassifyAsync:
    """Verify the async classify method with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_classify_success(self, classifier: TicketClassifier) -> None:
        llm_response = json.dumps({
            "category": "BUG",
            "urgency": "HIGH",
            "confidence": 0.92,
            "reasoning": "Clear bug report with error message.",
        })

        with patch(
            "src.services.llm.call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = llm_response
            result = await classifier.classify("The app crashes with error 500.")

            assert isinstance(result, TicketClassification)
            assert result.category == TicketCategory.BUG
            assert result.urgency == UrgencyLevel.HIGH
            assert result.confidence == pytest.approx(0.92, abs=0.01)
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_empty_content_raises(self, classifier: TicketClassifier) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            await classifier.classify("")

    @pytest.mark.asyncio
    async def test_classify_whitespace_only_raises(self, classifier: TicketClassifier) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            await classifier.classify("   \n\t  ")

    @pytest.mark.asyncio
    async def test_classify_llm_failure_uses_heuristic(self, classifier: TicketClassifier) -> None:
        with patch(
            "src.services.llm.call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = RuntimeError("LLM unavailable")
            result = await classifier.classify(
                "I can't log in to my account. Password reset broken."
            )

            # Should fall back to heuristic
            assert isinstance(result, TicketClassification)
            assert result.category == TicketCategory.ACCOUNT_ISSUE

    @pytest.mark.asyncio
    async def test_classify_rate_limit_retries(self, classifier: TicketClassifier) -> None:
        classifier.max_retries = 2

        with patch(
            "src.services.llm.call_llm", new_callable=AsyncMock
        ) as mock_call:
            # First call rate-limited, second succeeds
            mock_call.side_effect = [
                Exception("rate limit 429"),
                json.dumps({
                    "category": "BILLING",
                    "urgency": "MEDIUM",
                    "confidence": 0.8,
                    "reasoning": "Billing question.",
                }),
            ]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await classifier.classify("I need a refund for my subscription.")

            assert result.category == TicketCategory.BILLING
            assert mock_call.call_count == 2

    @pytest.mark.asyncio
    async def test_classify_updates_token_usage(self, classifier: TicketClassifier) -> None:
        llm_response = json.dumps({
            "category": "BUG",
            "urgency": "LOW",
            "confidence": 0.7,
            "reasoning": "Minor bug.",
        })

        with patch(
            "src.services.llm.call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = llm_response
            await classifier.classify("Minor display bug.")

            usage = classifier.get_token_usage()
            assert usage.input_tokens > 0
            assert usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_classify_strips_markdown_fences(self, classifier: TicketClassifier) -> None:
        llm_response = '```json\n{"category": "USAGE_HELP", "urgency": "LOW", "confidence": 0.85, "reasoning": "Help question."}\n```'

        with patch(
            "src.services.llm.call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = llm_response
            result = await classifier.classify("How do I use the dashboard?")

            assert result.category == TicketCategory.USAGE_HELP


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetClassifier:
    """Verify the get_classifier factory function."""

    def test_returns_ticket_classifier(self) -> None:
        with patch("src.services.classification.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                llm_provider="anthropic",
                anthropic_model="claude-sonnet-4-20250514",
                confidence_threshold=0.7,
            )
            classifier = get_classifier()
            assert isinstance(classifier, TicketClassifier)
            assert classifier.model_name == "claude-sonnet-4-20250514"

    def test_openai_provider(self) -> None:
        with patch("src.services.classification.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                llm_provider="openai",
                openai_model="gpt-4o",
                confidence_threshold=0.6,
            )
            classifier = get_classifier()
            assert classifier.model_name == "gpt-4o"
            assert classifier.confidence_threshold == 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Verify handling of edge cases and boundary conditions."""

    def test_classifier_init_defaults(self) -> None:
        c = TicketClassifier()
        assert c.model_name == "claude-sonnet-4-20250514"
        assert c.timeout_seconds == 10.0
        assert c.max_retries == 3
        assert c.confidence_threshold == 0.5

    def test_heuristic_empty_content(self, classifier: TicketClassifier) -> None:
        result = classifier._heuristic_fallback("")
        assert result.category == TicketCategory.OTHER
        assert result.urgency == UrgencyLevel.MEDIUM

    def test_heuristic_long_content(self, classifier: TicketClassifier) -> None:
        long_content = "The login page crashes. " * 1000
        result = classifier._heuristic_fallback(long_content)
        assert result.category == TicketCategory.BUG

    def test_parse_json_with_extra_whitespace(self, classifier: TicketClassifier) -> None:
        raw = '  {"category": "BUG", "urgency": "LOW", "confidence": 0.6, "reasoning": "Bug."}  '
        result = classifier._parse_response(raw)
        assert result.category == TicketCategory.BUG

    def test_usage_summary_after_multiple_classifications(
        self, classifier: TicketClassifier
    ) -> None:
        # Simulate accumulated usage
        classifier._input_tokens = 500
        classifier._output_tokens = 200
        classifier._total_cost = 0.003

        summary = classifier.get_cost_summary()
        assert summary["total_tokens"] == 700
        assert summary["estimated_cost_usd"] == 0.003

    @pytest.mark.asyncio
    async def test_classify_max_retries_exhausted(self, classifier: TicketClassifier) -> None:
        classifier.max_retries = 1

        with patch(
            "src.services.llm.call_llm", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = RuntimeError("Persistent failure")

            # Should fall back to heuristic after retries exhausted
            result = await classifier.classify("Login is broken and crashing.")
            assert isinstance(result, TicketClassification)
            assert result.category == TicketCategory.BUG
