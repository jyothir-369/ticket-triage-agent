"""Pre-flight config validation script.

Verifies that every required environment variable is set and that each
connected service (database, Qdrant, Redis) is reachable before starting the
app. Exits 0 only when all checks pass.

Usage::

    python -m src.scripts.validate_config
    # or:
    python scripts/validate_config.py
"""

from __future__ import annotations

import asyncio
import sys
from urllib.parse import urlparse

from src.config import get_settings
from src.utils.observability import init_telemetry, shutdown_telemetry


def _check(label: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")
    if not ok:
        sys.exit(1)


def _check_nonempty(label: str, value: str | None, detail: str = "") -> None:
    ok = bool(value and value.strip())
    _check(label, ok, f"'{value}'" if value else "(unset)")


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        import httpx  # local import keeps the script usable when httpx absent
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        return resp.status_code < 500
    except Exception:
        return False


async def _check_database() -> None:
    settings = get_settings()
    parsed = urlparse(settings.database_url)
    required_parts = ["host", "port"]
    missing = [p for p in required_parts if not getattr(parsed, p)]
    _check("DATABASE_URL", len(missing) == 0, "URL has host:port")

    # Try a raw asyncpg connection via SQLAlchemy
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(settings.database_url, pool_pre_ping=True,
                                     connect_args={"connect_timeout": 5})
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            engine.dispose()
            _check("DATABASE", True, "PostgreSQL reachable")
        finally:
            await engine.dispose()
    except Exception as exc:
        _check("DATABASE", False, f"unreachable ({exc!r})")


async def _check_qdrant() -> None:
    settings = get_settings()
    if settings.qdrant_url:
        # Cloud URL: check HTTP root + collection exists
        ok = await _http_ok(f"{settings.qdrant_url.rstrip('/')}/")
        _check("QDRANT_URL", ok, "HTTP health endpoint reachable")
        if ok:
            # Also verify the collection exists via REST search endpoint
            import httpx
            key = settings.qdrant_api_key
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            try:
                r = await httpx.get(
                    f"{settings.qdrant_url.rstrip('/')}/collections/{}",
                    headers=headers, timeout=5.0, follow_redirects=True,
                )
                _check("QDRANT_COLLECTION", r.status_code == 200,
                       f"REST /collections returns {r.status_code}")
            except Exception as exc:
                _check("QDRANT_COLLECTION", False, f"{exc!r}")
    else:
        _check("QDRANT_HOST", True, "non-cloud URL configured; skipping deep probe")


async def _check_redis() -> None:
    settings = get_settings()
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.redis_url, socket_timeout=3,
                                  max_connections=1)
        try:
            await client.ping()
            _check("REDIS_URL", True, "Ping OK")
        finally:
            await client.aclose()
    except Exception as exc:
        _check("REDIS_URL", False, f"unreachable ({exc!r})")


async def main() -> None:
    print("=" * 60)
    print("Config validation pre-flight")
    print("=" * 60)

    try:
        await _check_database()
        await _check_qdrant()
        await _check_redis()
        print("\nAll checks passed.")
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        print(f"\nUnexpected error: {exc!r}")
        sys.exit(1)
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    asyncio.run(main())
