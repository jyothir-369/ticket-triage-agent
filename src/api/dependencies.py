"""Shared FastAPI dependencies — injected via ``Depends()`` in route handlers."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from src.config import Settings, get_settings
from src.models.database import get_db_session
from src.repository import TicketRepository


async def get_settings_dep() -> Settings:
    """Return the cached application settings."""
    return get_settings()


async def get_repository() -> TicketRepository:
    """Return a fresh TicketRepository instance."""
    return TicketRepository()


async def verify_api_key(request: Request) -> None:
    """Verify an API key (or Bearer token) against the configured credential list.

    Accepts either ``X-API-Key`` header or ``Authorization: Bearer <token>``.
    The list of valid keys is read from the ``API_KEYS`` env var (comma-separated).

    Raises HTTPException 401 when authentication fails.
    """
    settings = get_settings()
    api_key = request.headers.get("X-API-Key") or ""
    if not api_key:
        bearer = request.headers.get("Authorization", "")
        if bearer.lower().startswith("bearer "):
            api_key = bearer[7:]

    valid_keys = get_settings().api_keys or []
    if "*" not in valid_keys and api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )


__all__ = [
    "get_db_session",
    "get_repository",
    "get_settings_dep",
    "verify_api_key",
]
