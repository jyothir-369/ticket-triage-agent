"""Structured logging configuration with structlog.

Provides:
- setup_logging() to configure structlog processors
- Correlation ID (ticket_id) support for distributed tracing
- JSON format for production, pretty console for development
- Timezone-aware timestamps
- Log levels: DEBUG (full traces), INFO (key events), WARNING (retries), ERROR (failures)
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from typing import Any

import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import (
    StackInfoRenderer,
    add_log_level,
    format_exc_info,
)
from structlog.stdlib import (
    BoundLogger,
    LoggerFactory,
    add_logger_name,
)

from src.config import get_settings

# ═══════════════════════════════════════════════════════════════════════════════
# Custom Processors
# ═══════════════════════════════════════════════════════════════════════════════


def _add_timestamp_utc(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add ISO 8601 timestamp with timezone to log entries."""
    event_dict["timestamp"] = datetime.now(UTC).isoformat()
    return event_dict


def _add_correlation_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Ensure correlation_id is present in all log entries.

    Uses ticket_id as the correlation_id if not already set.
    """
    if "correlation_id" not in event_dict and "ticket_id" in event_dict:
        event_dict["correlation_id"] = event_dict["ticket_id"]
    return event_dict


def _add_service_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add service name and version context to all log entries."""
    event_dict["service"] = "support-triage-agent"
    event_dict["service_version"] = "0.1.0"
    return event_dict


def _filter_noise(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Filter out noisy log entries that don't provide value."""
    # Remove None values
    return {k: v for k, v in event_dict.items() if v is not None}


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Renderer for Production
# ═══════════════════════════════════════════════════════════════════════════════


def _json_renderer(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> str:
    """Render log entries as JSON for production environments."""
    import json

    return json.dumps(event_dict, default=str, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty Console Renderer for Development
# ═══════════════════════════════════════════════════════════════════════════════


def _pretty_renderer(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> str:
    """Render log entries in a human-readable format for development."""
    # Color mapping for log levels
    colors = {
        "debug": "\033[36m",    # Cyan
        "info": "\033[32m",     # Green
        "warning": "\033[33m",  # Yellow
        "error": "\033[31m",    # Red
        "critical": "\033[35m", # Magenta
    }
    reset = "\033[0m"

    level = event_dict.get("log_level", "").lower()
    color = colors.get(level, "")

    # Build the log line
    timestamp = event_dict.get("timestamp", "")
    log_level = event_dict.get("log_level", "").upper().ljust(8)
    logger_name = event_dict.get("logger", "")
    event = event_dict.get("event", "")

    # Correlation ID
    correlation_id = event_dict.get("correlation_id", "")

    # Key-value pairs (excluding standard fields)
    standard_fields = {
        "timestamp", "log_level", "logger", "event", "correlation_id",
        "service", "service_version",
    }
    kv_pairs = [
        f"{k}={v}"
        for k, v in event_dict.items()
        if k not in standard_fields and v is not None
    ]

    # Format the line
    parts = [
        f"{color}{timestamp}{reset}",
        f"{color}{log_level}{reset}",
        f"[{logger_name}]",
    ]

    if correlation_id:
        parts.append(f"\033[90m({correlation_id})\033[0m")

    parts.append(event)

    if kv_pairs:
        parts.append("\033[90m" + " ".join(kv_pairs) + "\033[0m")

    # Add exception info if present
    exc_info = event_dict.get("exc_info")
    if exc_info:
        import traceback
        exc_text = "".join(traceback.format_exception(*exc_info))
        parts.append(f"\n{color}{exc_text}{reset}")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Setup Functions
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging(
    log_level: str | None = None,
    json_format: bool | None = None,
) -> None:
    """Configure structlog with processors and renderers.

    Parameters
    ----------
    log_level:
        Override the log level (DEBUG, INFO, WARNING, ERROR).
        Defaults to settings.log_level.
    json_format:
        Force JSON (True) or pretty console (False) output.
        Defaults to False for development.
    """
    settings = get_settings()

    if log_level is None:
        log_level = settings.log_level

    if json_format is None:
        json_format = False  # Default to pretty for development

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add a handler that outputs to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(handler)

    # Configure structlog processors
    shared_processors = [
        merge_contextvars,
        add_log_level,
        add_logger_name,
        _add_timestamp_utc,
        _add_correlation_id,
        _add_service_context,
        StackInfoRenderer(),
        format_exc_info,
        _filter_noise,
    ]

    if json_format:
        renderer = _json_renderer
    else:
        renderer = _pretty_renderer

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> BoundLogger:
    """Get a structlog logger with the given name.

    Parameters
    ----------
    name:
        Logger name (typically __name__). If None, uses the root logger.

    Returns
    -------
    BoundLogger
        A configured structlog logger.
    """
    return structlog.get_logger(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Context Manager for Correlation ID
# ═══════════════════════════════════════════════════════════════════════════════


class correlation_id_context:
    """Context manager to bind a correlation_id to all log entries within scope.

    Usage::

        with correlation_id_context("ticket-123"):
            logger.info("Processing ticket")  # Will have correlation_id=ticket-123
    """

    def __init__(self, correlation_id: str) -> None:
        self.correlation_id = correlation_id
        self._token = None

    def __enter__(self) -> correlation_id_context:
        from structlog.contextvars import bind_contextvars
        self._token = bind_contextvars(correlation_id=self.correlation_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        from structlog.contextvars import reset_contextvars
        if self._token is not None:
            reset_contextvars(self._token)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions for Common Log Events
# ═══════════════════════════════════════════════════════════════════════════════


def log_ticket_processed(
    logger: BoundLogger,
    ticket_id: str,
    category: str,
    urgency: str,
    confidence: float,
    should_escalate: bool,
    duration_ms: int,
) -> None:
    """Log a successfully processed ticket."""
    logger.info(
        "ticket.processed",
        ticket_id=ticket_id,
        category=category,
        urgency=urgency,
        confidence=round(confidence, 4),
        should_escalate=should_escalate,
        duration_ms=duration_ms,
    )


def log_ticket_escalated(
    logger: BoundLogger,
    ticket_id: str,
    reason: str,
    confidence: float,
) -> None:
    """Log that a ticket was escalated."""
    logger.warning(
        "ticket.escalated",
        ticket_id=ticket_id,
        reason=reason,
        confidence=round(confidence, 4),
    )


def log_retry_attempt(
    logger: BoundLogger,
    operation: str,
    attempt: int,
    max_attempts: int,
    error: str,
) -> None:
    """Log a retry attempt."""
    logger.warning(
        "retry.attempt",
        operation=operation,
        attempt=attempt,
        max_attempts=max_attempts,
        error=error,
    )


def log_error(
    logger: BoundLogger,
    operation: str,
    error: str,
    error_type: str,
    ticket_id: str | None = None,
) -> None:
    """Log an error."""
    kwargs: dict[str, Any] = {
        "operation": operation,
        "error": error,
        "error_type": error_type,
    }
    if ticket_id:
        kwargs["ticket_id"] = ticket_id

    logger.error("error.occurred", **kwargs)


def log_classification(
    logger: BoundLogger,
    ticket_id: str,
    category: str,
    urgency: str,
    confidence: float,
    duration_ms: int,
) -> None:
    """Log a classification result."""
    logger.info(
        "classification.complete",
        ticket_id=ticket_id,
        category=category,
        urgency=urgency,
        confidence=round(confidence, 4),
        duration_ms=duration_ms,
    )


def log_draft_generated(
    logger: BoundLogger,
    ticket_id: str,
    draft_length: int,
    citation_count: int,
    confidence: float,
    duration_ms: int,
) -> None:
    """Log a draft generation result."""
    logger.info(
        "draft.generated",
        ticket_id=ticket_id,
        draft_length=draft_length,
        citation_count=citation_count,
        confidence=round(confidence, 4),
        duration_ms=duration_ms,
    )
