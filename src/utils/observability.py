"""Observability — OpenTelemetry tracing and custom metrics for the triage agent.

Sets up:
- Tracer with service name "support-triage-agent"
- Span attributes: ticket_id, step, status, latency, model_name, token_count
- Custom spans for each node and tool execution
- Console exporter (always) + OTLP exporter (when endpoint configured)

Custom metrics:
- ticket_processed_total (counter)
- triage_duration_seconds (histogram)
- escalation_rate_gauge
- confidence_score_gauge
- tool_call_count_counter
- llm_token_usage_counter
- retry_count_counter
- error_count_counter
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from src.config import get_settings

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SERVICE_NAME = "support-triage-agent"
SERVICE_VERSION = "0.1.0"

# ═══════════════════════════════════════════════════════════════════════════════
# Resource — identifies this service in traces
# ═══════════════════════════════════════════════════════════════════════════════


def _get_resource() -> Resource:
    """Build the OTel Resource with the current environment from settings."""
    settings = get_settings()
    return Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": SERVICE_VERSION,
            "deployment.environment": settings.environment,
        }
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tracer Provider — console + optional OTLP
# ═══════════════════════════════════════════════════════════════════════════════

_trace_provider: TracerProvider | None = None


def _build_tracer_provider() -> TracerProvider:
    """Create and configure the TracerProvider with console and OTLP exporters."""
    settings = get_settings()
    provider = TracerProvider(resource=_get_resource())

    # Always export to console for debugging
    provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )

    # Add OTLP exporter if endpoint is configured
    if settings.otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.otlp_endpoint,
                insecure=True,
            )
            provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        except Exception:
            pass  # OTLP is optional; console export still works

    return provider


def init_telemetry() -> None:
    """Initialize the global TracerProvider and MeterProvider.

    Call once at application startup (e.g. in FastAPI lifespan).
    """
    global _trace_provider

    if _trace_provider is not None:
        return  # Already initialized

    _trace_provider = _build_tracer_provider()
    trace.set_tracer_provider(_trace_provider)

    # Initialize metrics provider
    _init_metrics_provider()


def shutdown_telemetry() -> None:
    """Flush and shut down all telemetry exporters.

    Call at application shutdown.
    """
    global _trace_provider

    if _trace_provider is not None:
        _trace_provider.shutdown()
        _trace_provider = None


def get_tracer() -> trace.Tracer:
    """Return the application tracer."""
    return trace.get_tracer(SERVICE_NAME, SERVICE_VERSION)


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Metrics
# ═══════════════════════════════════════════════════════════════════════════════

_meter: metrics.Meter | None = None

# Metric instruments (created lazily after meter init)
_ticket_processed_counter: Counter | None = None
_triage_duration_histogram: Histogram | None = None
_escalation_rate_gauge = None  # ObservableGauge
_confidence_score_gauge = None  # ObservableGauge
_tool_call_counter: Counter | None = None
_llm_token_counter: Counter | None = None
_retry_counter: Counter | None = None
_error_counter: Counter | None = None

# In-memory accumulators for observable gauges
_escalation_count = 0
_total_tickets = 0
_confidence_sum = 0.0
_confidence_count = 0


def _init_metrics_provider() -> None:
    """Initialize the MeterProvider and all metric instruments."""
    global _meter
    global _ticket_processed_counter, _triage_duration_histogram
    global _tool_call_counter, _llm_token_counter, _retry_counter, _error_counter

    if _meter is not None:
        return

    reader = PeriodicExportingMetricReader(
        export_interval_millis=30_000,  # Export every 30 seconds
    )
    provider = MeterProvider(resource=_get_resource(), metric_readers=[reader])
    metrics.set_meter_provider(provider)

    _meter = metrics.get_meter(SERVICE_NAME, SERVICE_VERSION)

    # ── Counter: tickets processed ────────────────────────────────────────
    _ticket_processed_counter = _meter.create_counter(
        name="ticket_processed_total",
        description="Total number of tickets processed",
        unit="1",
    )

    # ── Histogram: triage duration ────────────────────────────────────────
    _triage_duration_histogram = _meter.create_histogram(
        name="triage_duration_seconds",
        description="Duration of triage pipeline execution in seconds",
        unit="s",
        # Buckets: 1s, 5s, 10s, 30s, 60s
    )

    # ── Counter: tool calls ───────────────────────────────────────────────
    _tool_call_counter = _meter.create_counter(
        name="tool_call_count_total",
        description="Total number of tool/LLM calls made",
        unit="1",
    )

    # ── Counter: LLM token usage ──────────────────────────────────────────
    _llm_token_counter = _meter.create_counter(
        name="llm_token_usage_total",
        description="Total LLM tokens consumed",
        unit="1",
    )

    # ── Counter: retries ──────────────────────────────────────────────────
    _retry_counter = _meter.create_counter(
        name="retry_count_total",
        description="Total number of retry attempts",
        unit="1",
    )

    # ── Counter: errors ───────────────────────────────────────────────────
    _error_counter = _meter.create_counter(
        name="error_count_total",
        description="Total number of errors encountered",
        unit="1",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Recording Functions
# ═══════════════════════════════════════════════════════════════════════════════


def record_ticket_processed(category: str, status: str) -> None:
    """Record that a ticket was processed."""
    global _ticket_processed_counter

    if _ticket_processed_counter is None:
        return

    _ticket_processed_counter.add(
        1,
        {
            "category": category,
            "status": status,
        },
    )


def record_triage_duration(duration_seconds: float) -> None:
    """Record the duration of a triage pipeline run."""
    global _triage_duration_histogram

    if _triage_duration_histogram is None:
        return

    _triage_duration_histogram.record(duration_seconds)


def record_escalation_rate(escalated: bool) -> None:
    """Update the escalation rate gauge."""
    global _escalation_count, _total_tickets

    _total_tickets += 1
    if escalated:
        _escalation_count += 1


def record_confidence_score(score: float) -> None:
    """Update the confidence score gauge."""
    global _confidence_sum, _confidence_count

    _confidence_sum += score
    _confidence_count += 1


def record_tool_call(tool_name: str, success: bool) -> None:
    """Record a tool/LLM call."""
    global _tool_call_counter

    if _tool_call_counter is None:
        return

    _tool_call_counter.add(
        1,
        {
            "tool": tool_name,
            "success": str(success).lower(),
        },
    )


def record_llm_tokens(model: str, provider: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record LLM token usage."""
    global _llm_token_counter

    if _llm_token_counter is None:
        return

    _llm_token_counter.add(
        prompt_tokens + completion_tokens,
        {
            "model": model,
            "provider": provider,
            "token_type": "total",
        },
    )

    # Also record separately for prompt vs completion
    _llm_token_counter.add(
        prompt_tokens,
        {
            "model": model,
            "provider": provider,
            "token_type": "prompt",
        },
    )
    _llm_token_counter.add(
        completion_tokens,
        {
            "model": model,
            "provider": provider,
            "token_type": "completion",
        },
    )


def record_retry(attempt: int, operation: str) -> None:
    """Record a retry attempt."""
    global _retry_counter

    if _retry_counter is None:
        return

    _retry_counter.add(
        1,
        {
            "operation": operation,
            "attempt": str(attempt),
        },
    )


def record_error(error_type: str, operation: str) -> None:
    """Record an error."""
    global _error_counter

    if _error_counter is None:
        return

    _error_counter.add(
        1,
        {
            "error_type": error_type,
            "operation": operation,
        },
    )


def get_escalation_rate() -> float:
    """Return the current escalation rate (0.0 to 1.0)."""
    if _total_tickets == 0:
        return 0.0
    return _escalation_count / _total_tickets


def get_average_confidence() -> float:
    """Return the current average confidence score (0.0 to 1.0)."""
    if _confidence_count == 0:
        return 0.0
    return _confidence_sum / _confidence_count


# ═══════════════════════════════════════════════════════════════════════════════
# Span Helpers
# ═══════════════════════════════════════════════════════════════════════════════


@contextmanager
def create_span(
    name: str,
    *,
    ticket_id: str | None = None,
    step: str | None = None,
    status: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Context manager that creates an OpenTelemetry span with common attributes.

    Parameters
    ----------
    name:
        Span name (e.g. "classify", "retrieve", "draft").
    ticket_id:
        The ticket being processed (added as span attribute).
    step:
        Pipeline step name (added as span attribute).
    status:
        Initial status string (added as span attribute).
    attributes:
        Additional key-value attributes to set on the span.

    Yields
    ------
    trace.Span
        The active span (can be used to add more attributes).
    """
    tracer = get_tracer()

    attrs: dict[str, Any] = {}
    if ticket_id:
        attrs["ticket_id"] = ticket_id
    if step:
        attrs["step"] = step
    if status:
        attrs["status"] = status
    if attributes:
        attrs.update(attributes)

    with tracer.start_as_current_span(name, attributes=attrs) as span:
        yield span


@contextmanager
def create_node_span(
    node_name: str,
    ticket_id: str | None = None,
) -> Generator[trace.Span, None, None]:
    """Create a span for a graph node execution.

    Automatically records start time and sets the node name.
    """
    with create_span(
        f"node.{node_name}",
        ticket_id=ticket_id,
        step=node_name,
        status="running",
    ) as span:
        start = time.monotonic()
        try:
            yield span
            elapsed = time.monotonic() - start
            span.set_attribute("status", "completed")
            span.set_attribute("duration_ms", int(elapsed * 1000))
        except Exception as exc:
            elapsed = time.monotonic() - start
            span.set_attribute("status", "failed")
            span.set_attribute("duration_ms", int(elapsed * 1000))
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("error.message", str(exc))
            span.record_exception(exc)
            raise


@contextmanager
def create_tool_span(
    tool_name: str,
    ticket_id: str | None = None,
    **attributes: Any,
) -> Generator[trace.Span, None, None]:
    """Create a span for a tool execution."""
    with create_span(
        f"tool.{tool_name}",
        ticket_id=ticket_id,
        step=f"tool_{tool_name}",
        status="running",
        attributes=attributes,
    ) as span:
        start = time.monotonic()
        try:
            yield span
            elapsed = time.monotonic() - start
            span.set_attribute("status", "completed")
            span.set_attribute("duration_ms", int(elapsed * 1000))
            record_tool_call(tool_name, success=True)
        except Exception as exc:
            elapsed = time.monotonic() - start
            span.set_attribute("status", "failed")
            span.set_attribute("duration_ms", int(elapsed * 1000))
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("error.message", str(exc))
            span.record_exception(exc)
            record_tool_call(tool_name, success=False)
            raise


@contextmanager
def create_llm_span(
    model: str,
    provider: str,
    ticket_id: str | None = None,
) -> Generator[trace.Span, None, None]:
    """Create a span for an LLM call."""
    with create_span(
        f"llm.{provider}.{model}",
        ticket_id=ticket_id,
        step="llm_call",
        status="running",
        attributes={
            "model_name": model,
            "llm.provider": provider,
        },
    ) as span:
        start = time.monotonic()
        try:
            yield span
            elapsed = time.monotonic() - start
            span.set_attribute("status", "completed")
            span.set_attribute("duration_ms", int(elapsed * 1000))
        except Exception as exc:
            elapsed = time.monotonic() - start
            span.set_attribute("status", "failed")
            span.set_attribute("duration_ms", int(elapsed * 1000))
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("error.message", str(exc))
            span.record_exception(exc)
            raise
