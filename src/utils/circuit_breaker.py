"""Circuit Breaker pattern for resilient service calls.

Implements the classic circuit breaker pattern with three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests are blocked
- HALF_OPEN: After timeout, one test request is allowed through

Usage::

    from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

    breaker = CircuitBreaker(
        name="llm_service",
        failure_threshold=5,
        timeout=30.0,
    )

    try:
        result = await breaker.call(my_async_function, arg1, arg2)
    except CircuitBreakerOpen:
        # Service is unavailable, use fallback
        result = fallback_value
"""

from __future__ import annotations

import asyncio
import enum
import time
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitBreakerOpen(Exception):
    """Raised when a circuit breaker is in OPEN state and rejects a call."""

    def __init__(self, name: str, last_failure_time: float, timeout: float) -> None:
        self.name = name
        self.last_failure_time = last_failure_time
        self.timeout = timeout
        remaining = max(0, timeout - (time.monotonic() - last_failure_time))
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Try again in {remaining:.1f} seconds."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# States
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    """Normal operation — requests pass through."""

    OPEN = "open"
    """Failure threshold exceeded — requests are blocked."""

    HALF_OPEN = "half_open"
    """Testing recovery — one request allowed through."""


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitBreaker:
    """Circuit breaker for protecting service calls.

    Parameters
    ----------
    name:
        Identifier for this circuit breaker (used in logging).
    failure_threshold:
        Number of consecutive failures before opening the circuit.
    timeout:
        Seconds to wait before transitioning from OPEN to HALF_OPEN.
    half_open_max_calls:
        Number of successful calls in HALF_OPEN state before closing.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejected = 0

        logger.debug(
            "circuit_breaker.initialised",
            name=name,
            failure_threshold=failure_threshold,
            timeout=timeout,
        )

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit breaker.

        Automatically transitions from OPEN to HALF_OPEN when timeout expires.
        """
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(
                    "circuit_breaker.half_open",
                    name=self.name,
                    elapsed_seconds=round(elapsed, 2),
                )
        return self._state

    @property
    def is_closed(self) -> bool:
        """Return True when the circuit is closed (allowing requests)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Return True when the circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Return True when the circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def stats(self) -> dict[str, Any]:
        """Return current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "total_rejected": self._total_rejected,
            "last_failure_time": self._last_failure_time,
        }

    def record_success(self) -> None:
        """Record a successful call."""
        self._total_calls += 1
        self._total_successes += 1

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls += 1

            if self._half_open_calls >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(
                    "circuit_breaker.closed",
                    name=self.name,
                    reason="half_open_recovery",
                )
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._total_calls += 1
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Failure during half-open → back to open
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker.opened",
                name=self.name,
                reason="half_open_failure",
                failure_count=self._failure_count,
            )
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit_breaker.opened",
                    name=self.name,
                    reason="threshold_exceeded",
                    failure_count=self._failure_count,
                    threshold=self.failure_threshold,
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info("circuit_breaker.reset", name=self.name)

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function through the circuit breaker.

        Parameters
        ----------
        func:
            The function to call (can be sync or async).
        *args:
            Positional arguments to pass to the function.
        **kwargs:
            Keyword arguments to pass to the function.

        Returns
        -------
        Any
            The return value of the function.

        Raises
        ------
        CircuitBreakerOpen
            When the circuit is open and rejects the call.
        Exception
            Any exception raised by the wrapped function.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            self._total_rejected += 1
            raise CircuitBreakerOpen(
                self.name,
                self._last_failure_time,
                self.timeout,
            )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self.record_success()
            return result

        except Exception:
            self.record_failure()
            raise

    def __enter__(self) -> "CircuitBreaker":
        """Enter context manager — check if circuit is open."""
        current_state = self.state
        if current_state == CircuitState.OPEN:
            self._total_rejected += 1
            raise CircuitBreakerOpen(
                self.name,
                self._last_failure_time,
                self.timeout,
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager — record success or failure."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker Registry
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 30.0,
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                timeout=timeout,
            )
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name, or None if not found."""
        return self._breakers.get(name)

    def get_all(self) -> dict[str, CircuitBreaker]:
        """Return all registered circuit breakers."""
        return dict(self._breakers)

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            breaker.reset()

    def stats(self) -> dict[str, dict[str, Any]]:
        """Return statistics for all circuit breakers."""
        return {name: breaker.stats for name, breaker in self._breakers.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level Registry
# ═══════════════════════════════════════════════════════════════════════════════

_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 30.0,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name.

    Parameters
    ----------
    name:
        Unique identifier for this circuit breaker.
    failure_threshold:
        Number of failures before opening the circuit.
    timeout:
        Seconds to wait before testing recovery.

    Returns
    -------
    CircuitBreaker
        The circuit breaker instance.
    """
    return _registry.get_or_create(
        name,
        failure_threshold=failure_threshold,
        timeout=timeout,
    )


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Return all registered circuit breakers."""
    return _registry.get_all()


def get_circuit_breaker_stats() -> dict[str, dict[str, Any]]:
    """Return statistics for all circuit breakers."""
    return _registry.stats()
