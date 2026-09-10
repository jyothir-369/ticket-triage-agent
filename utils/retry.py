"""Retry decorators using tenacity with exponential backoff and jitter.

Provides:
- retry_on_exception: Decorator for async/sync functions with configurable retries
- retry_on_result: Decorator that retries based on return value
- Circuit breaker integration for cascading failure prevention
- Structured logging of retry attempts

Usage::

    from utils.retry import retry_on_exception, retry_on_result

    @retry_on_exception(max_attempts=3, base_delay=1.0)
    async def call_llm(prompt: str) -> str:
        return await llm.ainvoke(prompt)

    @retry_on_result(max_attempts=5, predicate=lambda r: r is None)
    async def fetch_with_retry(url: str) -> dict:
        return await httpx.get(url).json()
"""

from __future__ import annotations

import asyncio
import functools
import random
from typing import Any, Callable, TypeVar

import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    wait_combine,
    before_sleep_log,
    RetryError,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════════════════


class RateLimitError(Exception):
    """Raised when an API rate limit is hit (HTTP 429)."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(message)


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Wait Strategies
# ═══════════════════════════════════════════════════════════════════════════════


def wait_exponential_with_jitter(
    multiplier: float = 2.0,
    min_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter_range: float = 0.3,
) -> Callable[[RetryCallState], float]:
    """Create a wait strategy with exponential backoff and random jitter.

    Parameters
    ----------
    multiplier:
        Multiplier for the exponential backoff (default 2.0).
    min_delay:
        Minimum delay between retries in seconds (default 1.0).
    max_delay:
        Maximum delay between retries in seconds (default 10.0).
    jitter_range:
        Range of random jitter as a fraction of the delay (0.0 to 1.0).

    Returns
    -------
    Callable
        A wait function compatible with tenacity.
    """

    def wait_func(retry_state: RetryCallState) -> float:
        attempt = retry_state.attempt_number or 1
        delay = min(max_delay, min_delay * (multiplier ** (attempt - 1)))
        jitter = delay * jitter_range * (2 * random.random() - 1)
        return max(0.01, delay + jitter)

    return wait_func


# ═══════════════════════════════════════════════════════════════════════════════
# Logging Callbacks
# ═══════════════════════════════════════════════════════════════════════════════


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log each retry attempt with structured data."""
    attempt = retry_state.attempt_number or 1
    exception = retry_state.outcome.exception() if retry_state.outcome else None

    logger.warning(
        "retry.attempt",
        attempt=attempt,
        max_attempts=retry_state.retry_object.stop.max_attempt_number
        if hasattr(retry_state.retry_object.stop, "max_attempt_number")
        else None,
        exception_type=type(exception).__name__ if exception else None,
        exception_message=str(exception) if exception else None,
    )


def _log_retry_giveup(retry_state: RetryCallState) -> None:
    """Log when all retry attempts are exhausted."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None

    logger.error(
        "retry.giveup",
        max_attempts=retry_state.retry_object.stop.max_attempt_number
        if hasattr(retry_state.retry_object.stop, "max_attempt_number")
        else None,
        exception_type=type(exception).__name__ if exception else None,
        exception_message=str(exception) if exception else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Retry Decorators
# ═══════════════════════════════════════════════════════════════════════════════


def retry_on_exception(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter_range: float = 0.3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    circuit_breaker_name: str | None = None,
) -> Callable:
    """Decorator that retries a function on exception with exponential backoff.

    Parameters
    ----------
    max_attempts:
        Maximum number of attempts (including the first one).
    base_delay:
        Base delay in seconds for exponential backoff.
    max_delay:
        Maximum delay between retries in seconds.
    jitter_range:
        Range of random jitter as a fraction (0.0 to 1.0).
    exceptions:
        Tuple of exception types to retry on.
    circuit_breaker_name:
        If provided, integrate with the named circuit breaker.

    Returns
    -------
    Callable
        The decorated function with retry logic.

    Example
    -------
    >>> @retry_on_exception(max_attempts=3, base_delay=1.0)
    ... async def call_api():
    ...     return await httpx.get("https://api.example.com")
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get circuit breaker if configured
            cb = None
            if circuit_breaker_name:
                from utils.circuit_breaker import get_circuit_breaker
                cb = get_circuit_breaker(circuit_breaker_name)

            # Define retry condition
            retry_condition = retry_if_exception(exceptions)

            # Build the retry decorator
            retry_decorator = retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_with_jitter(
                    multiplier=base_delay,
                    min_delay=base_delay,
                    max_delay=max_delay,
                    jitter_range=jitter_range,
                ),
                retry=retry_condition,
                before_sleep=_log_retry_attempt,
                after=_log_retry_giveup,
                reraise=True,
            )

            @retry_decorator
            async def _retry_func():
                if cb:
                    return await cb.call(func, *args, **kwargs)
                return await func(*args, **kwargs)

            return await _retry_func()

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get circuit breaker if configured
            cb = None
            if circuit_breaker_name:
                from utils.circuit_breaker import get_circuit_breaker
                cb = get_circuit_breaker(circuit_breaker_name)

            # Define retry condition
            retry_condition = retry_if_exception(exceptions)

            # Build the retry decorator
            retry_decorator = retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_with_jitter(
                    multiplier=base_delay,
                    min_delay=base_delay,
                    max_delay=max_delay,
                    jitter_range=jitter_range,
                ),
                retry=retry_condition,
                before_sleep=_log_retry_attempt,
                after=_log_retry_giveup,
                reraise=True,
            )

            @retry_decorator
            def _retry_func():
                if cb:
                    return cb.call(func, *args, **kwargs)
                return func(*args, **kwargs)

            return _retry_func()

        # Choose wrapper based on whether the original function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def retry_on_result(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter_range: float = 0.3,
    predicate: Callable[[Any], bool] = lambda r: r is None,
) -> Callable:
    """Decorator that retries based on the return value.

    Parameters
    ----------
    max_attempts:
        Maximum number of attempts (including the first one).
    base_delay:
        Base delay in seconds for exponential backoff.
    max_delay:
        Maximum delay between retries in seconds.
    jitter_range:
        Range of random jitter as a fraction (0.0 to 1.0).
    predicate:
        Function that returns True if the result should trigger a retry.

    Returns
    -------
    Callable
        The decorated function with retry logic.

    Example
    -------
    >>> @retry_on_result(max_attempts=3, predicate=lambda r: r is None)
    ... async def fetch_data():
    ...     return await api.get_data()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            retry_decorator = retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_with_jitter(
                    multiplier=base_delay,
                    min_delay=base_delay,
                    max_delay=max_delay,
                    jitter_range=jitter_range,
                ),
                retry=retry_if_result(predicate),
                before_sleep=_log_retry_attempt,
                after=_log_retry_giveup,
                reraise=True,
            )

            @retry_decorator
            async def _retry_func():
                return await func(*args, **kwargs)

            return await _retry_func()

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            retry_decorator = retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_with_jitter(
                    multiplier=base_delay,
                    min_delay=base_delay,
                    max_delay=max_delay,
                    jitter_range=jitter_range,
                ),
                retry=retry_if_result(predicate),
                before_sleep=_log_retry_attempt,
                after=_log_retry_giveup,
                reraise=True,
            )

            @retry_decorator
            def _retry_func():
                return func(*args, **kwargs)

            return _retry_func()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Common Retry Configurations
# ═══════════════════════════════════════════════════════════════════════════════


def retry_llm_call(
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> Callable:
    """Convenience decorator for retrying LLM API calls.

    Uses circuit breaker integration and handles common LLM errors:
    ConnectionError, TimeoutError, RateLimitError.
    """
    return retry_on_exception(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=10.0,
        jitter_range=0.3,
        exceptions=(ConnectionError, TimeoutError, OSError, RateLimitError),
        circuit_breaker_name="llm_service",
    )


def retry_database_call(
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> Callable:
    """Convenience decorator for retrying database calls."""
    return retry_on_exception(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=10.0,
        jitter_range=0.2,
        exceptions=(ConnectionError, TimeoutError, OSError),
        circuit_breaker_name="database",
    )


def retry_http_call(
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> Callable:
    """Convenience decorator for retrying HTTP calls."""
    return retry_on_exception(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=10.0,
        jitter_range=0.3,
        exceptions=(ConnectionError, TimeoutError, OSError, RateLimitError),
        circuit_breaker_name="http_client",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Manual Retry Helper
# ═══════════════════════════════════════════════════════════════════════════════


async def execute_with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Execute a function with retry logic.

    Parameters
    ----------
    func:
        The function to execute (can be sync or async).
    *args:
        Positional arguments to pass to the function.
    max_attempts:
        Maximum number of attempts.
    base_delay:
        Base delay for exponential backoff.
    max_delay:
        Maximum delay between retries.
    exceptions:
        Tuple of exception types to retry on.
    **kwargs:
        Keyword arguments to pass to the function.

    Returns
    -------
    Any
        The return value of the function.

    Raises
    ------
    Exception
        The last exception if all retries fail.
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except exceptions as exc:
            last_exception = exc

            if attempt < max_attempts:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                jitter = delay * 0.3 * (2 * random.random() - 1)
                delay = max(0.01, delay + jitter)

                logger.warning(
                    "retry.execute.attempt_failed",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_seconds=round(delay, 2),
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )

                await asyncio.sleep(delay)

    logger.error(
        "retry.execute.all_attempts_failed",
        max_attempts=max_attempts,
        exception_type=type(last_exception).__name__ if last_exception else None,
        exception_message=str(last_exception) if last_exception else None,
    )

    raise last_exception  # type: ignore[misc]
