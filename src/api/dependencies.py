"""Shared FastAPI dependencies — injected via ``Depends()`` in route handlers."""

from __future__ import annotations

from src.config import Settings, get_settings
from src.models.database import get_db_session
from src.repository import TicketRepository


async def get_settings_dep() -> Settings:
    """Return the cached application settings."""
    return get_settings()


async def get_repository() -> TicketRepository:
    """Return a fresh TicketRepository instance."""
    return TicketRepository()


__all__ = [
    "get_db_session",
    "get_settings_dep",
    "get_repository",
]
