"""Thin LLM provider adapter — supports Gemini, OpenAI, Anthropic, and OpenRouter
with automatic fallback and per-provider circuit breakers.

Includes circuit breaker integration to prevent cascading failures when any
LLM provider is experiencing issues.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import get_settings
from src.utils.circuit_breaker import CircuitBreakerOpen, get_circuit_breaker
from src.utils.retry import RateLimitError

logger = structlog.get_logger(__name__)

settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Provider Circuit Breakers
# Each provider has its own breaker so Gemini going down doesn't block OpenRouter
# ═══════════════════════════════════════════════════════════════════════════════

_gemini_breaker = get_circuit_breaker(
    name="llm_gemini",
    failure_threshold=5,
    timeout=30.0,
)

_openai_breaker = get_circuit_breaker(
    name="llm_openai",
    failure_threshold=5,
    timeout=30.0,
)

_anthropic_breaker = get_circuit_breaker(
    name="llm_anthropic",
    failure_threshold=5,
    timeout=30.0,
)

_openrouter_breaker = get_circuit_breaker(
    name="llm_openrouter",
    failure_threshold=5,
    timeout=30.0,
)

_BREAKERS: dict[str, Any] = {
    "gemini": _gemini_breaker,
    "openai": _openai_breaker,
    "anthropic": _anthropic_breaker,
    "openrouter": _openrouter_breaker,
}


def _get_breaker(provider: str) -> Any:
    """Return the circuit breaker for a specific provider."""
    return _BREAKERS.get(provider, _openai_breaker)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Provider Factory
# ═══════════════════════════════════════════════════════════════════════════════


def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return the configured LLM as a LangChain chat model.

    Parameters
    ----------
    provider:
        Which provider to use. If None, uses settings.llm_provider.

    Returns
    -------
    BaseChatModel
        The LangChain chat model for the specified provider.

    Raises
    ------
    ValueError
        If the provider is unsupported or not configured.
    """
    provider = (provider or settings.llm_provider).lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key or None,
            temperature=0.1,
            max_tokens=1024,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            temperature=0.1,
            max_tokens=1024,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key or None,
            base_url=settings.anthropic_base_url,
            temperature=0.1,
            max_tokens=1024,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key or None,
            base_url=settings.openrouter_base_url,
            temperature=0.1,
            max_tokens=1024,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r}. "
        "Use 'openai', 'anthropic', 'gemini', or 'openrouter'."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Single-Provider LLM Call
# ═══════════════════════════════════════════════════════════════════════════════


async def call_llm(
    messages: list[BaseMessage],
    *,
    provider: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout_seconds: float = 10.0,
) -> str:
    """Send a list of messages to a specific LLM provider and return the text response.

    Parameters
    ----------
    messages:
        List of LangChain messages to send.
    provider:
        Provider to use. If None, uses settings.llm_provider.
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
        When the provider's circuit breaker is open.
    RateLimitError
        When the API returns a 429 rate limit error.
    TimeoutError
        When the request exceeds the timeout.
    """
    provider = (provider or settings.llm_provider).lower()
    breaker = _get_breaker(provider)

    if not breaker.is_closed:
        logger.warning(
            "llm.circuit_breaker_rejected",
            provider=provider,
            state=breaker.state.value,
            failure_count=breaker._failure_count,
        )
        raise CircuitBreakerOpen(
            provider,
            breaker._last_failure_time,
            breaker.timeout,
        )

    llm = get_llm(provider)
    # Re-apply overrides when the caller wants non-default values
    if temperature != 0.1 or max_tokens != 1024:
        llm = llm.with_config(temperature=temperature, max_tokens=max_tokens)

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=timeout_seconds,
        )
        breaker.record_success()
        return response.content if isinstance(response.content, str) else str(response.content)

    except TimeoutError:
        breaker.record_failure()
        logger.warning(
            "llm.timeout",
            provider=provider,
            timeout_seconds=timeout_seconds,
        )
        raise TimeoutError(f"LLM call timed out after {timeout_seconds}s")

    except Exception as exc:
        breaker.record_failure()
        # Detect rate limiting (HTTP 429)
        exc_str = str(exc).lower()
        if "429" in exc_str or "rate" in exc_str or "rate_limit" in exc_str:
            logger.warning("llm.rate_limited", provider=provider)
            raise RateLimitError(f"LLM rate limit exceeded: {exc}") from exc
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Call with Automatic Fallback
# ═══════════════════════════════════════════════════════════════════════════════


async def call_llm_with_fallback(
    messages: list[BaseMessage],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout_seconds: float | None = None,
) -> str:
    """Call the primary LLM provider with automatic fallback to secondary.

    Tries the primary provider first. On any exception (timeout, rate limit,
    auth failure, network error), catches it, logs it with full context, and
    tries the fallback provider if configured.

    Parameters
    ----------
    messages:
        List of LangChain messages to send.
    temperature:
        Sampling temperature override.
    max_tokens:
        Maximum tokens in the response.
    timeout_seconds:
        Per-request timeout in seconds. If None, uses settings.llm_timeout_seconds.

    Returns
    -------
    str
        The LLM response text.

    Raises
    ------
    RuntimeError
        If both primary and fallback providers fail.
    """
    timeout = timeout_seconds or settings.llm_timeout_seconds
    primary = settings.llm_provider.lower()
    fallback = settings.llm_fallback_provider.lower() if settings.llm_fallback_provider else None

    # Try primary provider
    try:
        start = time.monotonic()
        result = await call_llm(
            messages,
            provider=primary,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
        )
        elapsed = time.monotonic() - start
        logger.debug(
            "llm.primary_success",
            provider=primary,
            elapsed_seconds=round(elapsed, 3),
        )
        return result

    except Exception as primary_exc:
        elapsed = time.monotonic() - start
        logger.warning(
            "llm.primary_failed",
            provider=primary,
            error_type=type(primary_exc).__name__,
            error=str(primary_exc)[:200],
            elapsed_seconds=round(elapsed, 3),
        )

        # No fallback configured — raise
        if not fallback:
            raise

        # Try fallback provider
        logger.warning(
            "llm.fallback_attempt",
            from_provider=primary,
            to_provider=fallback,
        )
        try:
            start = time.monotonic()
            result = await call_llm(
                messages,
                provider=fallback,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout,
            )
            elapsed = time.monotonic() - start
            logger.warning(
                "llm.fallback_success",
                from_provider=primary,
                to_provider=fallback,
                elapsed_seconds=round(elapsed, 3),
            )
            return result

        except Exception as fallback_exc:
            elapsed = time.monotonic() - start
            logger.error(
                "llm.fallback_failed",
                primary_provider=primary,
                fallback_provider=fallback,
                primary_error=f"{type(primary_exc).__name__}: {str(primary_exc)[:200]}",
                fallback_error=f"{type(fallback_exc).__name__}: {str(fallback_exc)[:200]}",
                elapsed_seconds=round(elapsed, 3),
            )
            raise RuntimeError(
                f"Both LLM providers failed. "
                f"Primary ({primary}): {type(primary_exc).__name__}: {str(primary_exc)[:200]}. "
                f"Fallback ({fallback}): {type(fallback_exc).__name__}: {str(fallback_exc)[:200]}."
            ) from fallback_exc


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker Stats
# ═══════════════════════════════════════════════════════════════════════════════


def get_llm_breaker_stats() -> dict[str, Any]:
    """Return circuit breaker statistics for all LLM providers."""
    return {
        name: breaker.stats
        for name, breaker in _BREAKERS.items()
    }
