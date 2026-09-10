"""LLM-based response drafting service.

Implements ``DraftGenerator`` — a full-featured drafting service with:
  * Structured prompt template with system instructions for drafting responses
  * Citation validation against retrieved documents
  * References section generation with full document metadata
  * Draft quality scoring based on length, coverage, and citation usage
  * Safety checks: hallucination detection, PII detection, jailbreak detection
  * Retry with exponential backoff and configurable timeout
  * Comprehensive structured logging
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.models.schemas import (
    Citation,
    DraftResponse,
    RetrievedDocument,
    Ticket,
    TicketCategory,
    TicketClassification,
    UrgencyLevel,
)
from src.utils.circuit_breaker import CircuitBreakerOpen
from src.utils.retry import RateLimitError

logger = structlog.get_logger(__name__)

settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Template
# ═══════════════════════════════════════════════════════════════════════════════

DRAFT_SYSTEM_PROMPT = """\
You are a professional support-response drafter. Given a support ticket, its \
classification, and retrieved context documents, write a helpful, concise, \
and professional response for the customer.

## Response Guidelines

1. **Be helpful and concise**: Address the customer's specific issue directly. \
Avoid unnecessary preamble.
2. **Maintain a professional tone**: Be empathetic and acknowledge the \
customer's frustration if present.
3. **Provide clear steps**: When applicable, provide numbered steps or \
actionable guidance.
4. **Cite sources**: When referencing information from the provided documents, \
use the format [1], [2], etc. Each citation must correspond to a document \
number from the "Related Documents" section.
5. **Ask clarifying questions**: If the issue is ambiguous, ask specific \
questions to better understand the problem.
6. **Match tone to issue type**:
   - Bugs/Issues: Empathetic, solution-focused
   - Feature requests: Encouraging, forward-looking
   - Account issues: Reassuring, security-conscious
   - Billing: Professional, detail-oriented
   - Usage help: Patient, instructional
7. **Keep under 300 words**: Be thorough but concise.

## Output Format

Return ONLY valid JSON — no markdown fences, no commentary:

{{
  "draft_text": "<the full draft response text>",
  "citations": [
    {{
      "claim": "<the specific claim being cited>",
      "source_doc_index": <1-based index from the related documents list>,
      "relevance_score": <0.0 to 1.0>
    }}
  ],
  "reasoning": "<brief explanation of the response strategy>"
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Safety Check Patterns
# ═══════════════════════════════════════════════════════════════════════════════

# PII detection patterns
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("phone_us", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

# Jailbreak detection patterns
JAILBREAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?|prompts?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be|I\s+am)\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+", re.IGNORECASE),
    re.compile(r"override\s+(?:your\s+)?(?:instructions?|rules?|system\s+prompt)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:if|though)\s+(?:you\s+)?(?:have\s+no|don'?t\s+have)\s+", re.IGNORECASE),
    re.compile(r"bypass\s+(?:your\s+)?(?:safety|content|moderation)\s+", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]

# Hallucination indicators - claims that suggest factual assertions outside sources
HALLUCINATION_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"(?:our\s+research\s+(?:shows?|indicates?|suggests?))", re.IGNORECASE),
    re.compile(r"(?:studies?\s+(?:show|indicate|suggest)\s+that)", re.IGNORECASE),
    re.compile(r"(?:according\s+to\s+(?:our|the)\s+(?:data|statistics?|reports?))", re.IGNORECASE),
    re.compile(r"(?:we\s+(?:have\s+)?(?:found|discovered|observed)\s+that)", re.IGNORECASE),
    re.compile(r"(?:industry\s+(?:standards?|best\s+practices?)\s+(?:require|dictate|mandate))", re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Safety Check Results
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SafetyCheckResult:
    """Result of safety checks on a draft response."""

    has_pii: bool = False
    pii_types_found: list[str] = field(default_factory=list)
    has_jailbreak: bool = False
    jailbreak_patterns_matched: list[str] = field(default_factory=list)
    has_hallucination_risk: bool = False
    hallucination_indicators: list[str] = field(default_factory=list)
    is_safe: bool = True

    def __post_init__(self) -> None:
        """Determine overall safety based on individual checks."""
        self.is_safe = not (self.has_pii or self.has_jailbreak or self.has_hallucination_risk)


# ═══════════════════════════════════════════════════════════════════════════════
# Draft Quality Score
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DraftQualityScore:
    """Quality assessment of a draft response."""

    length_score: float = 0.0  # 0-1 based on word count
    citation_score: float = 0.0  # 0-1 based on citation usage
    coverage_score: float = 0.0  # 0-1 based on ticket issue coverage
    overall_score: float = 0.0  # weighted average

    def compute_overall(
        self,
        length_weight: float = 0.3,
        citation_weight: float = 0.3,
        coverage_weight: float = 0.4,
    ) -> float:
        """Compute weighted overall score."""
        self.overall_score = round(
            self.length_score * length_weight
            + self.citation_score * citation_weight
            + self.coverage_score * coverage_weight,
            3,
        )
        return self.overall_score


# ═══════════════════════════════════════════════════════════════════════════════
# Token Usage Tracking
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TokenUsage:
    """Tracks token counts and costs for a drafting run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""


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
# DraftGenerator
# ═══════════════════════════════════════════════════════════════════════════════


class DraftGenerator:
    """LLM-based support response generator with citation validation and safety checks.

    Generates professional draft responses to support tickets using:
    - Retrieved context documents for grounding
    - Citation validation against source documents
    - Quality scoring based on length, coverage, and citation usage
    - Safety checks for PII, jailbreak attempts, and hallucination risks

    Features:
      * Structured JSON output via the configured LLM
      * Token counting and cost tracking
      * Citation validation with automatic rewrite on invalid citations
      * Quality scoring with configurable weights
      * Safety checks for PII, jailbreak, and hallucination indicators
      * Exponential backoff retries on transient failures
      * Configurable request timeout
      * Comprehensive structured logging
    """

    def __init__(
        self,
        *,
        model_name: str = "claude-sonnet-4-20250514",
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        max_rewrite_attempts: int = 2,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_rewrite_attempts = max_rewrite_attempts

        # Token tracking
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._total_cost: float = 0.0

        # tiktoken encoder (lazy-loaded)
        self._tiktoken_encoder: Any | None = None

        logger.debug(
            "draft_generator.initialised",
            model=model_name,
            timeout=timeout_seconds,
            max_retries=max_retries,
            max_rewrite_attempts=max_rewrite_attempts,
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

    # ── Core drafting ───────────────────────────────────────────────────────

    async def generate(
        self,
        ticket: Ticket,
        classification: TicketClassification,
        retrieved_docs: list[RetrievedDocument],
    ) -> DraftResponse:
        """Generate a draft response for a support ticket.

        Parameters
        ----------
        ticket:
            The incoming support ticket.
        classification:
            The classification result for this ticket.
        retrieved_docs:
            Documents retrieved from the vector store for context.

        Returns
        -------
        DraftResponse
            Structured draft response with citations, confidence, and metadata.

        Raises
        ------
        ValueError
            If ticket content is empty.
        RuntimeError
            If all retry attempts are exhausted.
        """
        if not ticket.content or not ticket.content.strip():
            raise ValueError("ticket.content must not be empty.")

        logger.info(
            "draft.generate.start",
            ticket_id=ticket.id,
            category=classification.category.value,
            urgency=classification.urgency.value,
            num_docs=len(retrieved_docs),
            model=self.model_name,
        )
        start_time = time.monotonic()

        # Build user message with ticket, classification, and documents
        user_msg = self._build_user_message(ticket, classification, retrieved_docs)
        messages = [SystemMessage(content=DRAFT_SYSTEM_PROMPT), HumanMessage(content=user_msg)]

        try:
            raw_response = await self._call_llm_with_retry(messages)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            # Parse initial draft
            draft = self._parse_response(raw_response, retrieved_docs)

            # Validate citations and rewrite if needed
            draft, rewrite_count = await self._validate_and_rewrite(
                draft, ticket, classification, retrieved_docs
            )

            # Run safety checks
            safety_result = self._run_safety_checks(draft.draft_text)
            if not safety_result.is_safe:
                logger.warning(
                    "draft.safety_issues_detected",
                    has_pii=safety_result.has_pii,
                    pii_types=safety_result.pii_types_found,
                    has_jailbreak=safety_result.has_jailbreak,
                    has_hallucination_risk=safety_result.has_hallucination_risk,
                )

            # Compute quality score
            quality = self._compute_quality_score(
                draft, ticket, retrieved_docs
            )
            draft.confidence = quality.overall_score

            # Add references section
            draft.draft_text = self._append_references(
                draft.draft_text, retrieved_docs
            )

            # Log token usage
            self._log_token_usage(elapsed_ms)

            logger.info(
                "draft.generate.success",
                ticket_id=ticket.id,
                confidence=round(draft.confidence, 3),
                citations_validated=draft.citations_validated,
                num_citations=len(draft.citations),
                rewrite_count=rewrite_count,
                safety_safe=safety_result.is_safe,
                latency_ms=elapsed_ms,
            )
            return draft

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(
                "draft.generate.failed",
                ticket_id=ticket.id,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=elapsed_ms,
            )
            raise

    # ── Message building ────────────────────────────────────────────────────

    def _build_user_message(
        self,
        ticket: Ticket,
        classification: TicketClassification,
        retrieved_docs: list[RetrievedDocument],
    ) -> str:
        """Build the user message with ticket, classification, and documents."""
        # Build document context
        doc_parts: list[str] = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata_str = ""
            if doc.metadata:
                relevant_meta = {
                    k: v
                    for k, v in doc.metadata.items()
                    if k in ("ticket_id", "subject", "category", "urgency", "source")
                }
                if relevant_meta:
                    metadata_str = f"\nMetadata: {json.dumps(relevant_meta, default=str)}"

            doc_parts.append(
                f"[{i}] (similarity: {doc.similarity_score:.2f}, source: {doc.source})\n"
                f"{doc.content[:800]}{metadata_str}"
            )
        docs_block = "\n\n".join(doc_parts) if doc_parts else "No related documents found."

        # Build classification context
        classification_block = (
            f"Category: {classification.category.value}\n"
            f"Urgency: {classification.urgency.value}\n"
            f"Confidence: {classification.confidence:.2f}"
        )

        return (
            f"## Support Ticket\n"
            f"ID: {ticket.id}\n"
            f"Source: {ticket.source}\n"
            f"Content:\n{ticket.content}\n\n"
            f"## Classification\n{classification_block}\n\n"
            f"## Related Documents\n{docs_block}"
        )

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
                    temperature=0.3,
                    max_tokens=1024,
                    timeout_seconds=self.timeout_seconds,
                )
                # Track token usage
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
                            "draft.circuit_breaker_open",
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            remaining_time=exc.timeout - (time.monotonic() - exc.last_failure_time),
                        )
                        wait_time = max(wait_time, exc.timeout - (time.monotonic() - exc.last_failure_time))

                    elif isinstance(exc, RateLimitError):
                        logger.warning(
                            "draft.rate_limited",
                            attempt=attempt + 1,
                            wait_seconds=round(wait_time, 2),
                            retry_after=exc.retry_after,
                        )
                        if exc.retry_after:
                            wait_time = max(wait_time, exc.retry_after)

                    elif isinstance(exc, TimeoutError):
                        logger.warning(
                            "draft.timeout",
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            timeout=self.timeout_seconds,
                        )
                    else:
                        logger.warning(
                            "draft.connection_error",
                            attempt=attempt + 1,
                            wait_seconds=round(wait_time, 2),
                            error=str(exc),
                        )

                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "draft.all_retries_exhausted",
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
                            "draft.server_error",
                            attempt=attempt + 1,
                            wait_seconds=round(wait_time, 2),
                            status=exc,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                # Non-retryable error — raise immediately
                logger.error("draft.non_retryable_error", error=str(exc))
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
                "draft.tokens",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                running_total_cost=round(self._total_cost, 6),
            )
        except Exception as exc:
            logger.debug("draft.token_count_failed", error=str(exc))

    # ── Response parsing ────────────────────────────────────────────────────

    def _parse_response(
        self, raw: str, retrieved_docs: list[RetrievedDocument]
    ) -> DraftResponse:
        """Parse the LLM JSON response into a DraftResponse."""
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
            logger.warning("draft.json_parse_failed", raw=raw[:200], error=str(exc))
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

        # Extract draft text
        draft_text = str(data.get("draft_text", "")).strip()
        if not draft_text:
            raise ValueError("LLM returned empty draft_text")

        # Extract citations
        raw_citations = data.get("citations", [])
        citations: list[Citation] = []
        for c in raw_citations:
            if not isinstance(c, dict):
                continue
            # Map source_doc_index (1-based) to source_doc_id
            doc_index = c.get("source_doc_index", 1)
            if isinstance(doc_index, int) and 1 <= doc_index <= len(retrieved_docs):
                source_doc = retrieved_docs[doc_index - 1]
                citation = Citation(
                    claim=str(c.get("claim", ""))[:500],
                    source_doc_id=source_doc.id,
                    relevance_score=float(c.get("relevance_score", 1.0)),
                )
                citations.append(citation)

        # Extract reasoning
        reasoning = str(data.get("reasoning", ""))[:2000]

        return DraftResponse(
            draft_text=draft_text,
            citations=citations,
            confidence=0.5,  # Will be updated by quality scoring
            reasoning=reasoning,
            citations_validated=False,
        )

    # ── Citation validation ─────────────────────────────────────────────────

    async def _validate_and_rewrite(
        self,
        draft: DraftResponse,
        ticket: Ticket,
        classification: TicketClassification,
        retrieved_docs: list[RetrievedDocument],
    ) -> tuple[DraftResponse, int]:
        """Validate citations and rewrite draft if invalid citations found.

        Returns the (possibly rewritten) draft and the number of rewrites.
        """
        rewrite_count = 0

        for attempt in range(self.max_rewrite_attempts + 1):
            is_valid, invalid_citations = self._validate_citations(
                draft, retrieved_docs
            )

            if is_valid:
                draft.citations_validated = True
                logger.info(
                    "draft.citations_validated",
                    num_citations=len(draft.citations),
                    attempt=attempt,
                )
                return draft, rewrite_count

            # Log invalid citations
            logger.warning(
                "draft.invalid_citations",
                invalid_citations=invalid_citations,
                attempt=attempt + 1,
            )

            if attempt >= self.max_rewrite_attempts:
                # Max rewrites reached — keep what we have but mark as not fully validated
                logger.warning(
                    "draft.max_rewrite_reached",
                    rewrite_count=rewrite_count,
                )
                return draft, rewrite_count

            # Rewrite draft to fix citations
            draft = await self._rewrite_draft_with_valid_citations(
                draft, invalid_citations, ticket, classification, retrieved_docs
            )
            rewrite_count += 1

        return draft, rewrite_count

    def _validate_citations(
        self,
        draft: DraftResponse,
        retrieved_docs: list[RetrievedDocument],
    ) -> tuple[bool, list[str]]:
        """Validate that all citations refer to actual documents.

        Returns (is_valid, list_of_invalid_citation_descriptions).
        """
        valid_doc_ids = {doc.id for doc in retrieved_docs}
        invalid: list[str] = []

        for citation in draft.citations:
            if citation.source_doc_id not in valid_doc_ids:
                invalid.append(
                    f"Citation '{citation.claim[:50]}...' references "
                    f"doc_id '{citation.source_doc_id}' which is not in retrieved docs"
                )

        # Also check inline citations [N] in the draft text
        inline_citations = re.findall(r"\[(\d+)\]", draft.draft_text)
        for ref_num in inline_citations:
            idx = int(ref_num)
            if idx < 1 or idx > len(retrieved_docs):
                invalid.append(
                    f"Inline citation [{ref_num}] references a document index "
                    f"outside the range of retrieved documents (1-{len(retrieved_docs)})"
                )

        return len(invalid) == 0, invalid

    async def _rewrite_draft_with_valid_citations(
        self,
        original_draft: DraftResponse,
        invalid_citations: list[str],
        ticket: Ticket,
        classification: TicketClassification,
        retrieved_docs: list[RetrievedDocument],
    ) -> DraftResponse:
        """Rewrite the draft to fix invalid citations."""

        rewrite_prompt = f"""\
The previous draft had invalid citations that need to be fixed:

Invalid citations found:
{chr(10).join(f"- {c}" for c in invalid_citations)}

Please rewrite the draft, ensuring:
1. All citations [N] refer to valid document numbers from the related documents list
2. All citation claims in the JSON citations array reference valid document IDs
3. Keep the same helpful tone and content, just fix the citation references
4. Only cite documents that actually support the claims being made

Original draft:
{original_draft.draft_text}

Original citations:
{json.dumps([{"claim": c.claim, "source_doc_id": c.source_doc_id} for c in original_draft.citations], indent=2)}
"""

        try:
            from src.services.llm import call_llm_with_fallback

            # Re-use the same message structure
            user_msg = self._build_user_message(
                ticket, classification, retrieved_docs
            )
            # Append the rewrite instruction
            user_msg += f"\n\n## Rewrite Instructions\n{rewrite_prompt}"

            messages = [
                SystemMessage(content=DRAFT_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]

            raw_response = await call_llm_with_fallback(messages, temperature=0.3, max_tokens=1024)
            self._track_tokens_from_response(raw_response, messages)

            return self._parse_response(raw_response, retrieved_docs)

        except Exception as exc:
            logger.warning("draft.rewrite_failed", error=str(exc))
            # Return original draft if rewrite fails
            return original_draft

    # ── Quality scoring ─────────────────────────────────────────────────────

    def _compute_quality_score(
        self,
        draft: DraftResponse,
        ticket: Ticket,
        retrieved_docs: list[RetrievedDocument],
    ) -> DraftQualityScore:
        """Compute quality score based on length, citation usage, and coverage."""
        quality = DraftQualityScore()

        # Length score: optimal range 100-300 words
        word_count = len(draft.draft_text.split())
        if word_count < 50:
            quality.length_score = 0.2
        elif word_count < 100:
            quality.length_score = 0.5
        elif word_count <= 300:
            quality.length_score = 1.0
        elif word_count <= 500:
            quality.length_score = 0.7
        else:
            quality.length_score = 0.4

        # Citation score: based on usage of available documents
        if retrieved_docs:
            # Check how many documents are actually cited
            cited_doc_ids = {c.source_doc_id for c in draft.citations}
            all_doc_ids = {doc.id for doc in retrieved_docs}
            citation_ratio = len(cited_doc_ids & all_doc_ids) / len(all_doc_ids)
            quality.citation_score = min(citation_ratio * 2, 1.0)  # Scale up, cap at 1.0
        else:
            # No docs available — citation score is neutral
            quality.citation_score = 0.5

        # Coverage score: check if draft addresses ticket keywords
        ticket_words = set(ticket.content.lower().split())
        draft_words = set(draft.draft_text.lower().split())
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "and", "but", "or",
            "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
            "every", "all", "any", "few", "more", "most", "other", "some", "such",
            "than", "too", "very", "just", "about", "also", "here", "there", "when",
            "where", "how", "what", "which", "who", "whom", "this", "that", "these",
            "those", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves",
        }
        meaningful_ticket = ticket_words - stop_words
        meaningful_draft = draft_words - stop_words

        if meaningful_ticket:
            overlap = len(meaningful_ticket & meaningful_draft)
            quality.coverage_score = min(overlap / len(meaningful_ticket), 1.0)
        else:
            quality.coverage_score = 0.5

        # Compute overall score
        quality.compute_overall()

        logger.debug(
            "draft.quality_score",
            length_score=round(quality.length_score, 3),
            citation_score=round(quality.citation_score, 3),
            coverage_score=round(quality.coverage_score, 3),
            overall_score=round(quality.overall_score, 3),
        )

        return quality

    # ── References section ──────────────────────────────────────────────────

    def _append_references(
        self, draft_text: str, retrieved_docs: list[RetrievedDocument]
    ) -> str:
        """Append a References section with full document metadata."""
        if not retrieved_docs:
            return draft_text

        references: list[str] = ["\n\n---\n\n**References:**\n"]
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.metadata
            source_info = f"Source: {doc.source}"
            if metadata.get("ticket_id"):
                source_info += f" | Ticket #{metadata['ticket_id']}"
            if metadata.get("subject"):
                source_info += f" | {metadata['subject']}"
            if metadata.get("category"):
                source_info += f" | Category: {metadata['category']}"

            references.append(
                f"[{i}] {source_info} (similarity: {doc.similarity_score:.2f})"
            )

        return draft_text + "\n" + "\n".join(references)

    # ── Safety checks ───────────────────────────────────────────────────────

    def _run_safety_checks(self, draft_text: str) -> SafetyCheckResult:
        """Run safety checks on the draft text."""
        result = SafetyCheckResult()

        # PII detection
        for pii_type, pattern in PII_PATTERNS:
            matches = pattern.findall(draft_text)
            if matches:
                result.has_pii = True
                result.pii_types_found.append(pii_type)

        # Jailbreak detection
        for pattern in JAILBREAK_PATTERNS:
            if pattern.search(draft_text):
                result.has_jailbreak = True
                result.jailbreak_patterns_matched.append(pattern.pattern)

        # Hallucination detection
        for pattern in HALLUCINATION_INDICATORS:
            if pattern.search(draft_text):
                result.has_hallucination_risk = True
                result.hallucination_indicators.append(pattern.pattern)

        return result

    # ── Logging helpers ─────────────────────────────────────────────────────

    def _log_token_usage(self, latency_ms: int) -> None:
        """Log token usage and cost for the drafting run."""
        logger.info(
            "draft.token_usage",
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
        logger.debug("draft.usage_reset")


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy async function (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════


async def draft_response(
    subject: str,
    body: str,
    retrieved_docs: list[Any],
) -> str:
    """Draft a customer-facing response using the ticket and retrieved context.

    Legacy wrapper that maintains backward compatibility with existing code.
    For new code, prefer using ``DraftGenerator.generate()`` directly.
    """
    from src.services.retrieval import RetrievedDoc

    # Build ticket and classification from subject/body
    ticket = Ticket(
        id="legacy",
        content=f"Subject: {subject}\n\nBody:\n{body}",
        source="legacy",
    )

    # Default classification (will be overridden by DraftGenerator)
    classification = TicketClassification(
        category=TicketCategory.OTHER,
        urgency=UrgencyLevel.MEDIUM,
        confidence=0.5,
        reasoning="Legacy call — no classification provided.",
    )

    # Convert RetrievedDoc to RetrievedDocument if needed
    docs: list[RetrievedDocument] = []
    for doc in retrieved_docs:
        if isinstance(doc, RetrievedDoc):
            docs.append(
                RetrievedDocument(
                    id=str(doc.ticket_id),
                    content=f"{doc.subject}\n{doc.body}",
                    metadata=doc.metadata or {},
                    similarity_score=doc.score,
                    source="past_ticket",
                )
            )
        elif isinstance(doc, RetrievedDocument):
            docs.append(doc)

    gen = get_draft_generator()
    result = await gen.generate(ticket, classification, docs)
    return result.draft_text


# ═══════════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════════


def get_draft_generator() -> DraftGenerator:
    """Create and return a ``DraftGenerator`` using the current settings."""
    settings = get_settings()

    # Read model name from settings based on provider
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        model_name = settings.gemini_model
    elif provider == "openai":
        model_name = settings.openai_model
    elif provider == "anthropic":
        model_name = settings.anthropic_model
    elif provider == "openrouter":
        model_name = settings.openrouter_model
    else:
        model_name = settings.openai_model

    return DraftGenerator(
        model_name=model_name,
        timeout_seconds=settings.drafting_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_rewrite_attempts=2,
    )
