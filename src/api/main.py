"""FastAPI entry point — exposes triage, trace, and metrics endpoints.

Includes graceful shutdown handling:
- SIGTERM/SIGINT → stop accepting new tasks, wait for in-flight tasks (30s max)
- Close database connections, Qdrant client, Redis client
- Log shutdown progress
"""

from __future__ import annotations

import asyncio
import signal
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.graph import AgentExecutor, shutdown as graceful_shutdown, is_shutting_down
from src.config import get_settings
from src.models.database import Base, get_db_session, get_engine
from src.models.schemas import (
    DashboardMetrics,
    Ticket,
    TicketCategory,
    TicketStatus,
    UrgencyLevel,
)
from src.models.ticket import (
    TicketModel,
    TicketCategory as DBTicketCategory,
    TicketStatus as DBTicketStatus,
    TicketUrgency as DBTicketUrgency,
)
from src.models.trace import TraceModel

logger = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan — create tables on startup / graceful shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    # NOTE: In production, run `alembic upgrade head` before starting the app.
    # The create_all below is a dev convenience fallback for SQLite / local dev.
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("api.startup", db=settings.database_url.split("@")[-1])

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def _handle_signal(sig_name: str) -> None:
        logger.info("api.signal_received", signal=sig_name)
        # Schedule graceful shutdown
        loop.create_task(_shutdown_handler(sig_name))

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s.name))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for all signals
            pass

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    await _shutdown_handler("lifespan_exit")


async def _shutdown_handler(signal_name: str = "unknown") -> None:
    """Handle graceful shutdown: stop tasks, close connections."""
    if is_shutting_down():
        return  # Already shutting down

    logger.info("api.shutdown.initiated", signal=signal_name)

    # Graceful shutdown of agent tasks (waits up to 30s)
    await graceful_shutdown(timeout_seconds=30.0)

    # Close database engine
    try:
        engine = get_engine()
        await engine.dispose()
        logger.info("api.shutdown.database_disposed")
    except Exception as exc:
        logger.warning("api.shutdown.database_close_failed", error=str(exc))

    logger.info("api.shutdown.complete", signal=signal_name)


app = FastAPI(
    title="Support-Ticket Triage Agent",
    description=(
        "Automated triage pipeline for support tickets — "
        "classifies, retrieves context, drafts responses, and "
        "routes low-confidence cases to human reviewers."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "api.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# OpenTelemetry instrumentation (optional)
# ---------------------------------------------------------------------------

if _HAS_OTEL:
    FastAPIInstrumentor.instrument_app(app)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        "api.http_error",
        status=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        "api.unhandled_error",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Include route modules
# ---------------------------------------------------------------------------

from src.api.tickets import router as tickets_router  # noqa: E402
from src.api.dashboard import router as dashboard_router  # noqa: E402

app.include_router(tickets_router)
app.include_router(dashboard_router)


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


class HealthCheckResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str
    timestamp: str
    version: str
    checks: dict[str, "HealthCheckResult"]


class HealthCheckResult(BaseModel):
    """Individual health check result."""

    status: str
    latency_ms: float
    message: str | None = None


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check endpoint",
    description="Verifies all dependencies (database, Qdrant, Redis, LLM) are reachable.",
)
async def health_check():
    """Check health of all service dependencies."""
    import asyncio
    from datetime import datetime, timezone

    from src.config import get_settings
    from src.utils.circuit_breaker import get_circuit_breaker_stats

    settings = get_settings()
    checks: dict[str, HealthCheckResult] = {}
    overall_status = "healthy"

    async def check_database() -> HealthCheckResult:
        """Check database connectivity."""
        start = time.monotonic()
        try:
            from src.models.database import get_engine

            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(select(1))
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status="healthy",
                latency_ms=round(latency, 2),
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status="unhealthy",
                latency_ms=round(latency, 2),
                message=str(exc),
            )

    async def check_qdrant() -> HealthCheckResult:
        """Check Qdrant connectivity."""
        start = time.monotonic()
        try:
            from qdrant_client import AsyncQdrantClient

            client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
            await client.get_collections()
            await client.close()
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status="healthy",
                latency_ms=round(latency, 2),
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status="unhealthy",
                latency_ms=round(latency, 2),
                message=str(exc),
            )

    async def check_redis() -> HealthCheckResult:
        """Check Redis connectivity."""
        start = time.monotonic()
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(settings.redis_url)
            await client.ping()
            await client.close()
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status="healthy",
                latency_ms=round(latency, 2),
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status="degraded",
                latency_ms=round(latency, 2),
                message=str(exc),
            )

    async def check_llm() -> HealthCheckResult:
        """Check LLM provider connectivity."""
        start = time.monotonic()
        try:
            from src.services.llm import get_llm

            llm = get_llm()
            # Just verify we can create the client (not make a real call)
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status="healthy",
                latency_ms=round(latency, 2),
                message=f"Provider: {settings.llm_provider}",
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return HealthCheckResult(
                status="unhealthy",
                latency_ms=round(latency, 2),
                message=str(exc),
            )

    # Run all checks concurrently
    db_check, qdrant_check, redis_check, llm_check = await asyncio.gather(
        check_database(),
        check_qdrant(),
        check_redis(),
        check_llm(),
    )

    checks["database"] = db_check
    checks["qdrant"] = qdrant_check
    checks["redis"] = redis_check
    checks["llm"] = llm_check

    # Determine overall status
    for check in checks.values():
        if check.status == "unhealthy":
            overall_status = "unhealthy"
            break
        elif check.status == "degraded" and overall_status != "unhealthy":
            overall_status = "degraded"

    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="0.1.0",
        checks=checks,
    )


@app.get(
    "/health/ready",
    summary="Readiness probe",
    description="Returns 200 if the service is ready to accept traffic.",
)
async def readiness_check():
    """Kubernetes readiness probe — verify critical dependencies."""
    from src.models.database import get_engine

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(select(1))
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Not ready: {exc}")


@app.get(
    "/health/live",
    summary="Liveness probe",
    description="Returns 200 if the service is alive.",
)
async def liveness_check():
    """Kubernetes liveness probe — always returns 200 if the process is running."""
    return {"status": "alive"}


@app.get(
    "/metrics/circuit-breakers",
    summary="Circuit breaker statistics",
    description="Returns statistics for all circuit breakers.",
)
async def circuit_breaker_stats():
    """Return statistics for all circuit breakers."""
    from src.utils.circuit_breaker import get_circuit_breaker_stats

    return get_circuit_breaker_stats()


@app.get(
    "/metrics/resources",
    summary="Resource usage and limits",
    description="Returns concurrency, timeout, and resource limit metrics.",
)
async def resource_metrics():
    """Return resource usage and limit metrics."""
    from src.agent.graph import (
        _triage_semaphore,
        _inflight_tasks,
        is_shutting_down,
    )

    return {
        "concurrency": {
            "max_concurrent": settings.max_concurrent_triages,
            "available_slots": _triage_semaphore._value,
            "inflight_tasks": len(_inflight_tasks),
        },
        "timeouts": {
            "triage_timeout_seconds": settings.triage_timeout_seconds,
            "tool_timeout_seconds": settings.classification_timeout_seconds,
        },
        "shutdown": {
            "is_shutting_down": is_shutting_down(),
        },
    }
