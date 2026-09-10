"""Thin LLM provider adapter — swaps OpenAI / Anthropic without touching agent logic.

Includes circuit breaker integration to prevent cascading failures when the
LLM provider is experiencing issues.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import get_settings
from utils.circuit_breaker import CircuitBreakerOpen, get_circuit_breaker
from utils.retry import RateLimitError

logger = structlog.get_logger(__name__)

settings = get_settings()

# Circuit breaker for LLM API calls: 5 failures in 60s → open for 30s
_llm_breaker = get_circuit_breaker(
    name="llm_service",
    failure_threshold=5,
    timeout=30.0,
)


def get_llm() -> BaseChatModel:
    """Return the configured LLM as a LangChain chat model."""
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
            temperature=0.1,
            max_tokens=1024,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key or None,
            temperature=0.1,
            max_tokens=1024,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider!r}. Use 'openai' or 'anthropic'.")


async def call_llm(
    messages: list[BaseMessage],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout_seconds: float = 10.0,
) -> str:
    """Send a list of messages to the configured LLM and return the text response.

    Parameters
    ----------
    messages:
        List of LangChain messages to send.
    temperature:
        Sampling temperature override.
    max_tokens:
        Maximum tokens in the response.
    timeout_seconds:
        Per-request timeout in seconds (default 10s).

    Returns
    -------
    str
        The LLM response text.

    Raises
    ------
    CircuitBreakerOpen
        When the LLM service circuit breaker is open.
    RateLimitError
        When the API returns a 429 rate limit error.
    TimeoutError
        When the request exceeds the timeout.
    """
    if not _llm_breaker.is_closed:
        logger.warning(
            "llm.circuit_breaker_rejected",
            state=_llm_breaker.state.value,
            failure_count=_llm_breaker._failure_count,
        )
        raise CircuitBreakerOpen(
            "llm_service",
            _llm_breaker._last_failure_time,
            _llm_breaker.timeout,
        )

    llm = get_llm()
    # Re-apply overrides when the caller wants non-default values
    if temperature != 0.1 or max_tokens != 1024:
        llm = llm.with_config(temperature=temperature, max_tokens=max_tokens)

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=timeout_seconds,
        )
        _llm_breaker.record_success()
        return response.content if isinstance(response.content, str) else str(response.content)

    except asyncio.TimeoutError:
        _llm_breaker.record_failure()
        logger.warning(
            "llm.timeout",
            timeout_seconds=timeout_seconds,
            provider=settings.llm_provider,
        )
        raise TimeoutError(f"LLM call timed out after {timeout_seconds}s")

    except Exception as exc:
        _llm_breaker.record_failure()
        # Detect rate limiting (HTTP 429)
        exc_str = str(exc).lower()
        if "429" in exc_str or "rate" in exc_str or "rate_limit" in exc_str:
            logger.warning("llm.rate_limited", provider=settings.llm_provider)
            raise RateLimitError(f"LLM rate limit exceeded: {exc}") from exc
        raise


def get_llm_breaker_stats() -> dict[str, Any]:
    """Return circuit breaker statistics for the LLM service."""
    return _llm_breaker.stats
