"""SQLAlchemy async engine, session factory, and retry-aware connection management."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Base declarative — imported by all ORM models
# ═══════════════════════════════════════════════════════════════════════════════

from sqlalchemy.orm import DeclarativeBase  # noqa: E402 — must be before Base


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in the project."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Engine — lazy factory so tests can inject their own engine
# ═══════════════════════════════════════════════════════════════════════════════

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return (and lazily create) the default async engine.

    Supports both local Postgres and serverless Neon with appropriate
    connection pool settings.
    """
    global _engine
    if _engine is None:
        from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

        from sqlalchemy.pool import NullPool

        from src.config import get_settings
        settings = get_settings()
        url = settings.database_url

        # Handle SQLite (tests) — skip SSL and pool args
        if url.startswith("sqlite+aiosqlite://"):
            _engine = create_async_engine(
                url,
                echo=False,
                pool_pre_ping=True,
            )
            return _engine

        # For PostgreSQL: asyncpg doesn't understand libpq query params like
        # sslmode, channel_binding, etc. Strip them from the URL and handle
        # SSL via connect_args instead.
        connect_args: dict = {}
        if url.startswith("postgresql"):
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)

            # Remove libpq-specific params that asyncpg doesn't understand
            asyncpg_incompatible = {"sslmode", "channel_binding", "sslrootcert", "sslcert", "sslkey"}
            cleaned_params = {
                k: v for k, v in query_params.items()
                if k.lower() not in asyncpg_incompatible
            }

            # Rebuild URL without incompatible params
            clean_query = urlencode(cleaned_params, doseq=True) if cleaned_params else ""
            url = urlunparse(parsed._replace(query=clean_query))

            # Set SSL via connect_args if sslmode was in the URL
            sslmode = query_params.get("sslmode", [None])[0]
            if sslmode and sslmode != "disable":
                connect_args["ssl"] = True

        # Serverless Postgres (Neon): use NullPool to avoid holding connections
        if settings.database_use_nullpool:
            _engine = create_async_engine(
                url,
                echo=False,
                poolclass=NullPool,
                connect_args=connect_args,
                pool_pre_ping=True,
            )
        else:
            _engine = create_async_engine(
                url,
                echo=False,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_pre_ping=True,
                pool_recycle=1800,
                pool_timeout=settings.database_connect_timeout,
                connect_args=connect_args,
            )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return (and lazily create) the default session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


# Expose engine for backward compatibility (triggers lazy creation)
@property  # type: ignore[misc]
def engine() -> AsyncEngine:  # noqa: D103
    return get_engine()


# ═══════════════════════════════════════════════════════════════════════════════
# Retry decorator for transient database errors
# ═══════════════════════════════════════════════════════════════════════════════


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    reraise=True,
)
async def _check_connectivity() -> None:
    """Lightweight ping to verify the connection pool is alive."""
    async with get_engine().connect() as conn:
        await conn.execute(text("SELECT 1"))


# ═══════════════════════════════════════════════════════════════════════════════
# Session context manager — commit / rollback handled automatically
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a session and commits on clean exit.

    Usage::

        async with get_session() as session:
            session.add(obj)
        # auto-committed here; rolled back on exception
    """
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI dependency — generator-style for ``Depends()``
# ═══════════════════════════════════════════════════════════════════════════════


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (FastAPI dependency)."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# Retry-aware session helper — wraps any coroutine in retry + session
# ═══════════════════════════════════════════════════════════════════════════════


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    reraise=True,
)
async def execute_with_retry(coro_factory: Any) -> Any:
    """Run an async database operation with automatic retry on transient failures.

    Parameters
    ----------
    coro_factory:
        A *callable* that returns a fresh coroutine each time it is called.
        This is necessary because coroutines cannot be re-awaited.
        Example: ``lambda: _do_query(session)``
    """
    return await coro_factory()
