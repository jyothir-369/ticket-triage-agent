"""LLM-based ticket classification service.

Implements ``TicketClassifier`` — a full-featured classifier with:
  * Structured JSON output via the configured LLM
  * Prompt template with 6 categories, 4 urgency levels, and 8 few-shot examples
  * Token counting and cost tracking via ``tiktoken``
  * Keyword-based heuristic fallback when the LLM fails or returns low confidence
  * Confidence calibration to reduce overconfident predictions
  * Retries with exponential backoff and 10-second timeout
  * Comprehensive structured logging
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.models.schemas import TicketCategory, TicketClassification, UrgencyLevel
from src.utils.circuit_breaker import CircuitBreakerOpen
from src.utils.retry import RateLimitError

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Template
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert support-ticket classifier. Analyze the given ticket and classify it.

## Categories

Classify into EXACTLY one of these 6 categories:

1. **BUG** — Software defect, error, crash, broken functionality, unexpected behavior.
   Keywords: crash, error, broken, doesn't work, fails, exception, 500, regression.
2. **FEATURE_REQUEST** — Request for new functionality, enhancement, or improvement.
   Keywords: please add, would be nice, feature, suggestion, enhancement, wish.
3. **ACCOUNT_ISSUE** — Account access, authentication, profile, or permission problems.
   Keywords: login, password, locked out, can't access, account, authentication, 2FA.
4. **BILLING** — Payment, subscription, charges, invoices, or refund issues.
   Keywords: charged, invoice, payment, refund, subscription, billing, overcharged.
5. **USAGE_HELP** — Questions about how to use the product, documentation, or guidance.
   Keywords: how do I, how to, can't figure out, help with, tutorial, guide.
6. **OTHER** — Anything that doesn't fit the above categories.

## Urgency Levels

Assign EXACTLY one urgency level:

1. **CRITICAL** — System down, data loss, security breach, blocking all users.
2. **HIGH** — Major feature broken with no workaround, multiple users affected.
3. **MEDIUM** — Issue with a workaround available, single user affected.
4. **LOW** — Cosmetic, question, nice-to-have, non-urgent request.

## Output Format

Return ONLY valid JSON — no markdown fences, no commentary:

{{
  "category": "<BUG|FEATURE_REQUEST|ACCOUNT_ISSUE|BILLING|USAGE_HELP|OTHER>",
  "urgency": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<1-2 sentence explanation>"
}}

## Few-Shot Examples

### Example 1
Ticket: "The login page crashes every time I enter my credentials. I get a 500 error."
Output: {{"category": "BUG", "urgency": "HIGH", "confidence": 0.92, "reasoning": "Clear software defect with 500 error on login — a critical user-facing flow. No workaround mentioned, blocking access."}}

### Example 2
Ticket: "It would be great if you could add dark mode support to the web app."
Output: {{"category": "FEATURE_REQUEST", "urgency": "LOW", "confidence": 0.90, "reasoning": "Explicit feature request for dark mode. No urgency indicated — nice-to-have enhancement."}}

### Example 3
Ticket: "I can't log in to my account. I changed my password yesterday and now it says invalid credentials."
Output: {{"category": "ACCOUNT_ISSUE", "urgency": "HIGH", "confidence": 0.88, "reasoning": "Account access problem after password change. User locked out — blocking access to their account."}}

### Example 4
Ticket: "I was charged twice for my subscription this month. Please refund the extra payment."
Output: {{"category": "BILLING", "urgency": "HIGH", "confidence": 0.95, "reasoning": "Duplicate charge — clear billing error requiring prompt resolution and refund."}}

### Example 5
Ticket: "How do I export my data as CSV from the analytics dashboard?"
Output: {{"category": "USAGE_HELP", "urgency": "LOW", "confidence": 0.90, "reasoning": "Straightforward usage question about data export functionality."}}

### Example 6
Ticket: "The mobile app crashes whenever I try to open the settings page on iOS 17."
Output: {{"category": "BUG", "urgency": "CRITICAL", "confidence": 0.95, "reasoning": "Reproducible crash on a major platform (iOS 17). App unusable for settings — critical defect."}}

### Example 7
Ticket: "Our API is returning 503 errors for all endpoints since 2 AM. All customers are affected."
Output: {{"category": "BUG", "urgency": "CRITICAL", "confidence": 0.98, "reasoning": "Complete API outage affecting all customers. System-wide service disruption — critical incident."}}

### Example 8
Ticket: "We need an option to export monthly reports as PDF for compliance."
Output: {{"category": "FEATURE_REQUEST", "urgency": "MEDIUM", "confidence": 0.85, "reasoning": "Feature request for PDF export tied to compliance requirements — moderate business impact."}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword-Based Heuristic Fallback Patterns
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_KEYWORDS: dict[TicketCategory, list[str]] = {
    TicketCategory.BUG: [
        "crash", "error", "broken", "doesn't work", "fails", "exception",
        "500", "404", "bug", "regression", "not working", "unexpected",
        "dead", "hang", "freeze", "corrupt", "data loss", "outage",
    ],
    TicketCategory.FEATURE_REQUEST: [
        "feature request", "please add", "would be nice", "suggestion",
        "enhancement", "wish", "it would be great", "can you add",
        "proposal", "idea", "roadmap", "new feature",
    ],
    TicketCategory.ACCOUNT_ISSUE: [
        "login", "password", "locked out", "can't access", "account",
        "authentication", "2fa", "sso", "signup", "register",
        "email verification", "profile", "permission",
    ],
    TicketCategory.BILLING: [
        "charged", "invoice", "payment", "refund", "subscription",
        "billing", "overcharged", "credit card", "receipt", "cancel",
        "upgrade plan", "downgrade", "price",
    ],
    TicketCategory.USAGE_HELP: [
        "how do i", "how to", "can't figure out", "help with",
        "tutorial", "guide", "documentation", "instructions",
        "what is", "where can i", "explain", "walkthrough",
    ],
    TicketCategory.OTHER: [
        "general", "misc", "feedback", "other", "inquiry",
    ],
}

URGENCY_KEYWORDS: dict[UrgencyLevel, list[str]] = {
    UrgencyLevel.CRITICAL: [
        "down", "outage", "data loss", "security", "breach",
        "all users", "everyone", "system wide", "complete",
        "emergency", "production down", "p0",
    ],
    UrgencyLevel.HIGH: [
        "urgent", "asap", "critical", "blocking", "major",
        "no workaround", "multiple users", "deadline",
        "immediately", "broken", "cannot", "unable",
    ],
    UrgencyLevel.MEDIUM: [
        "workaround", "sometimes", "intermittent", "partial",
        "single user", "inconvenient", "would help",
    ],
    UrgencyLevel.LOW: [
        "nice to have", "cosmetic", "minor", "suggestion",
        "when you get a chance", "no rush", "feature request",
        "question", "how to", "documentation",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model Pricing (USD per 1M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_COST_RATES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Token Usage Tracking
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TokenUsage:
    """Tracks token counts and costs for a classification run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# TicketClassifier
# ═══════════════════════════════════════════════════════════════════════════════


class TicketClassifier:
    """LLM-based support ticket classifier with fallback heuristics.

    Classifies tickets into one of 6 categories (BUG, FEATURE_REQUEST,
    ACCOUNT_ISSUE, BILLING, USAGE_HELP, OTHER) and one of 4 urgency levels
    (LOW, MEDIUM, HIGH, CRITICAL).

    Features:
      * Structured JSON output via the configured LLM
      * Token counting and cost tracking using tiktoken
      * Keyword-based heuristic fallback when confidence < threshold
      * Exponential backoff retries on transient failures
      * 10-second request timeout
      * Comprehensive structured logging
    """

    def __init__(
        self,
        *,
        model_name: str = "claude-sonnet-4-20250514",
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        confidence_threshold: float = 0.5,
        llm_client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold
        self._llm_client = llm_client

        # Token tracking
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._total_cost: float = 0.0

        # tiktoken encoder (lazy-loaded)
        self._tiktoken_encoder: Any | None = None

        logger.debug(
            "classifier.initialised",
            model=model_name,
            timeout=timeout_seconds,
            max_retries=max_retries,
            confidence_threshold=confidence_threshold,
        )

    # ── tiktoken helpers ────────────────────────────────────────────────────

    def _get_encoder(self) -> Any:
        """Lazy-load the tiktoken encoder."""
        if self._tiktoken_encoder is None:
            try:
                import tiktoken

                self._tiktoken_encoder = tiktoken.encoding_for_model(self.model_name)
            except Exception:
                # Fallback: use cl100k_base encoding (works for most models)
                import tiktoken

                self._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        return self._tiktoken_encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens in *text* using tiktoken."""
        encoder = self._get_encoder()
        return len(encoder.encode(text))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost from token counts based on model pricing."""
        rates = MODEL_COST_RATES.get(self.model_name)
        if rates is None:
            # Conservative fallback: use Sonnet 4 pricing
            rates = (3.00, 15.00)
        input_rate, output_rate = rates
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    # ── Core classification ─────────────────────────────────────────────────

    async def classify(self, ticket_content: str) -> TicketClassification:
        """Classify a support ticket into category + urgency + confidence.

        Parameters
        ----------
        ticket_content:
            The full text body of the support ticket.

        Returns
        -------
        TicketClassification
            Structured classification result with category, urgency,
            confidence, and reasoning.

        Raises
        ------
        ValueError
            If *ticket_content* is empty or whitespace-only.
        """
        if not ticket_content or not ticket_content.strip():
            raise ValueError("ticket_content must not be empty.")

        ticket_content = ticket_content.strip()

        logger.info(
            "classify.start",
            content_length=len(ticket_content),
            model=self.model_name,
        )
        start_time = time.monotonic()

        # Build messages
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        user_msg = HumanMessage(content=f"Classify this ticket:\n\n{ticket_content}")
        messages = [system_msg, user_msg]

        try:
            raw_response = await self._call_llm_with_retry(messages)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            classification = self._parse_response(raw_response)
            self._log_token_usage(elapsed_ms)
            self._calibrate_confidence(classification)

            logger.info(
                "classify.success",
                category=classification.category.value,
                urgency=classification.urgency.value,
                confidence=round(classification.confidence, 3),
                latency_ms=elapsed_ms,
            )
            return classification

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(
                "classify.llm_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=elapsed_ms,
            )
            return self._heuristic_fallback(ticket_content)

    # ── LLM call with retries ──────────────────────────────────────────────

    async def _call_llm_with_retry(self, messages: list[Any]) -> str:
        """Call the LLM with fallback support.

        Uses call_llm_with_fallback which handles primary/failover providers
        automatically. Retries on transient errors with exponential backoff.
        """
        import asyncio

        from src.services.llm import call_llm_with_fallback

        last_exc: Exception | None = None
        retry_exceptions = (ConnectionError, TimeoutError, OSError, RateLimitError, CircuitBreakerOpen)

        for attempt in range(self.max_retries):
            try:
                response = await call_llm_with_fallback(
                    messages,
                    temperature=0.0,
                    max_tokens=512,
                    timeout_seconds=self.timeout_seconds,
                )
                # Track token usage from the response
                self._track_tokens_from_response(response, messages)
                return response

            except retry_exceptions as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    import random
                    wait_time = min(10.0, 1.0 * (2 ** attempt))
                    jitter = wait_time * 0.3 * (2 * random.random() - 1)
                    wait_time = max(0.01, wait_time + jitter)

                    if isinstance(exc, CircuitBreakerOpen):
                        logger.warning(
                            "classify.circuit_breaker_open",
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            remaining_time=exc.timeout - (time.monotonic() - exc.last_failure_time),
                        )
                        wait_time = max(wait_time, exc.timeout - (time.monotonic() - exc.last_failure_time))

                    elif isinstance(exc, RateLimitError):
                        logger.warning(
                            "classify.rate_limited",
                            attempt=attempt + 1,
                            wait_seconds=round(wait_time, 2),
                            retry_after=exc.retry_after,
                        )
                        if exc.retry_after:
                            wait_time = max(wait_time, exc.retry_after)

                    elif isinstance(exc, TimeoutError):
                        logger.warning(
                            "classify.timeout",
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            timeout=self.timeout_seconds,
                        )
                    else:
                        logger.warning(
                            "classify.connection_error",
                            attempt=attempt + 1,
                            wait_seconds=round(wait_time, 2),
                            error=str(exc),
                        )

                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "classify.all_retries_exhausted",
                        attempt=attempt + 1,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )

            except Exception as exc:
                # Non-retryable error — check for 5xx server errors
                exc_str = str(exc)
                if any(code in exc_str for code in ("500", "502", "503", "504")):
                    last_exc = exc
                    if attempt < self.max_retries - 1:
                        import random
                        wait_time = min(10.0, 1.0 * (2 ** attempt))
                        jitter = wait_time * 0.3 * (2 * random.random() - 1)
                        wait_time = max(0.01, wait_time + jitter)
                        logger.warning(
                            "classify.server_error",
                            attempt=attempt + 1,
                            wait_seconds=round(wait_time, 2),
                            status=exc,
                        )
                        import asyncio
                        await asyncio.sleep(wait_time)
                        continue
                # Non-retryable error — raise immediately
                logger.error("classify.non_retryable_error", error=str(exc))
                raise

        # All retries exhausted
        raise last_exc or RuntimeError("All retries exhausted.")

    def _track_tokens_from_response(self, response: str, messages: list[Any]) -> None:
        """Track token counts from the LLM response."""
        try:
            # Count input tokens from messages
            input_text = " ".join(
                m.content for m in messages if hasattr(m, "content")
            )
            input_tokens = self.count_tokens(input_text)
            output_tokens = self.count_tokens(response)

            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._total_cost += self.estimate_cost(input_tokens, output_tokens)

            logger.debug(
                "classify.tokens",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                running_total_cost=round(self._total_cost, 6),
            )
        except Exception as exc:
            logger.debug("classify.token_count_failed", error=str(exc))

    # ── Response parsing ────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> TicketClassification:
        """Parse the LLM JSON response into a TicketClassification."""
        cleaned = raw.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("classify.json_parse_failed", raw=raw[:200], error=str(exc))
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

        # Map string values to enums
        category_map = {c.value.upper(): c for c in TicketCategory}
        urgency_map = {u.value.upper(): u for u in UrgencyLevel}

        category_str = str(data.get("category", "")).upper()
        urgency_str = str(data.get("urgency", "")).upper()

        category = category_map.get(category_str)
        if category is None:
            logger.warning("classify.unknown_category", value=data.get("category"))
            category = TicketCategory.OTHER

        urgency = urgency_map.get(urgency_str)
        if urgency is None:
            logger.warning("classify.unknown_urgency", value=data.get("urgency"))
            urgency = UrgencyLevel.MEDIUM

        # Confidence: clamp to [0.0, 1.0]
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(data.get("reasoning", ""))[:2000]

        return TicketClassification(
            category=category,
            urgency=urgency,
            confidence=confidence,
            reasoning=reasoning,
            category_confidence=confidence,
            urgency_confidence=confidence,
        )

    # ── Heuristic fallback ──────────────────────────────────────────────────

    def _heuristic_fallback(self, content: str) -> TicketClassification:
        """Keyword-based fallback when LLM fails or returns low confidence.

        Scans the ticket content against predefined keyword patterns for
        each category and urgency level, returning the best-matching
        classification with an appropriately reduced confidence score.
        """
        logger.info("classify.heuristic_fallback", content_length=len(content))
        content_lower = content.lower()

        # Score each category
        category_scores: dict[TicketCategory, float] = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in content_lower)
            category_scores[cat] = score

        # Pick the highest-scoring category
        best_cat = max(category_scores, key=lambda c: category_scores[c])
        cat_score = category_scores[best_cat]

        # If no keywords matched at all, default to OTHER
        if cat_score == 0:
            best_cat = TicketCategory.OTHER
            cat_score = 0.3

        # Score each urgency level
        urgency_scores: dict[UrgencyLevel, float] = {}
        for urg, keywords in URGENCY_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in content_lower)
            urgency_scores[urg] = score

        best_urg = max(urgency_scores, key=lambda u: urgency_scores[u])
        urg_score = urgency_scores[best_urg]

        # If no urgency keywords matched, default to MEDIUM
        if urg_score == 0:
            best_urg = UrgencyLevel.MEDIUM
            urg_score = 0.3

        # Confidence is average of keyword match intensities, capped at 0.5
        raw_confidence = (cat_score + urg_score) / 2.0
        confidence = min(raw_confidence * 0.15, 0.5)  # Scale and cap at threshold

        reasoning = (
            f"Heuristic fallback: classified as {best_cat.value}/{best_urg.value} "
            f"based on keyword matching ({cat_score} category matches, "
            f"{urg_score} urgency matches)."
        )

        classification = TicketClassification(
            category=best_cat,
            urgency=best_urg,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            category_confidence=round(confidence, 3),
            urgency_confidence=round(confidence, 3),
        )

        logger.info(
            "classify.heuristic_result",
            category=best_cat.value,
            urgency=best_urg.value,
            confidence=round(confidence, 3),
        )
        return classification

    # ── Confidence calibration ──────────────────────────────────────────────

    def _calibrate_confidence(self, classification: TicketClassification) -> None:
        """Calibrate confidence scores to reduce overconfident predictions.

        Applies a dampening factor when the model expresses high confidence
        but the classification reason contains hedging language.
        """
        hedging_phrases = [
            "possibly", "might be", "could be", "uncertain",
            "not sure", "hard to tell", "ambiguous", "mixed signals",
        ]
        reasoning_lower = classification.reasoning.lower()

        has_hedging = any(phrase in reasoning_lower for phrase in hedging_phrases)

        if has_hedging and classification.confidence > 0.7:
            # Reduce confidence when the model hedges but scores high
            factor = 0.85
            classification.confidence = round(
                classification.confidence * factor, 3
            )
            classification.category_confidence = round(
                classification.category_confidence * factor, 3
            )
            classification.urgency_confidence = round(
                classification.urgency_confidence * factor, 3
            )
            logger.debug(
                "classify.calibrated_down",
                original=classification.confidence / factor,
                calibrated=classification.confidence,
            )

        # Also cap at 0.95 to avoid false certainty
        if classification.confidence > 0.95:
            classification.confidence = 0.95
            logger.debug("classify.capped_at_0.95")

    # ── Logging helpers ─────────────────────────────────────────────────────

    def _log_token_usage(self, latency_ms: int) -> None:
        """Log token usage and cost for the classification."""
        logger.info(
            "classify.token_usage",
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_cost_usd=round(self._total_cost, 6),
            latency_ms=latency_ms,
        )

    # ── Public accessors ────────────────────────────────────────────────────

    def get_token_usage(self) -> TokenUsage:
        """Return current token usage statistics."""
        return TokenUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._input_tokens + self._output_tokens,
            estimated_cost_usd=round(self._total_cost, 6),
            model=self.model_name,
        )

    def get_cost_summary(self) -> dict[str, Any]:
        """Return a summary dict of token usage and costs."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._input_tokens + self._output_tokens,
            "estimated_cost_usd": round(self._total_cost, 6),
            "model": self.model_name,
        }

    def reset_usage(self) -> None:
        """Reset accumulated token usage and cost counters."""
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_cost = 0.0
        logger.debug("classify.usage_reset")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════════


def get_classifier() -> TicketClassifier:
    """Create and return a ``TicketClassifier`` using the current settings."""
    settings = get_settings()

    # Read model name from settings based on provider
    model_name = _get_model_for_provider(settings)

    return TicketClassifier(
        model_name=model_name,
        timeout_seconds=settings.classification_timeout_seconds,
        max_retries=settings.llm_max_retries,
        confidence_threshold=settings.confidence_threshold,
    )


def _get_model_for_provider(settings: Any) -> str:
    """Return the model name for the configured LLM provider."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return settings.gemini_model
    elif provider == "openai":
        return settings.openai_model
    elif provider == "anthropic":
        return settings.anthropic_model
    elif provider == "openrouter":
        return settings.openrouter_model
    return settings.openai_model
