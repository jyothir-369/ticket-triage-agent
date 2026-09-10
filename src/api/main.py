"""FastAPI entry point — exposes triage, trace, and metrics endpoints."""

from __future__ import annotations

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

from src.agent.graph import AgentExecutor
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
# Lifespan — create tables on startup / dispose on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("api.startup", db=settings.database_url.split("@")[-1])
    yield
    await engine.dispose()
    logger.info("api.shutdown")


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
