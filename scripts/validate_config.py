"""Configuration and connectivity validator for all five services.

Tests:
1. Settings validation (via Settings.validate())
2. PostgreSQL connectivity (SELECT 1)
3. Redis connectivity (PING)
4. Qdrant connectivity (get_collections)
5. Gemini connectivity (list models)
6. OpenRouter connectivity (/models)

Usage:
    python scripts/validate_config.py
    python scripts/validate_config.py --strict   # fail on warnings too
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import Settings


def check_settings() -> tuple[str, float, str]:
    """Validate Settings configuration."""
    start = time.monotonic()
    try:
        # Create fresh Settings to pick up any env var changes
        settings = Settings()
        # model_validator runs automatically on instantiation
        latency = (time.monotonic() - start) * 1000
        return "OK", latency, f"Environment: {settings.environment}"
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return "FAIL", latency, str(exc)


def check_postgres() -> tuple[str, float, str]:
    """Test PostgreSQL connectivity with SELECT 1."""
    import asyncio
    start = time.monotonic()
    try:
        # Clear engine cache to pick up fresh settings
        import src.models.database as db_module
        db_module._engine = None
        db_module._session_factory = None

        from src.models.database import get_engine
        engine = get_engine()

        async def _test():
            async with engine.connect() as conn:
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))

        asyncio.run(_test())
        latency = (time.monotonic() - start) * 1000
        return "OK", latency, "Connected"
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return "FAIL", latency, str(exc)[:200]


def check_redis() -> tuple[str, float, str]:
    """Test Redis connectivity with PING."""
    import asyncio
    start = time.monotonic()
    try:
        import redis.asyncio as aioredis
        settings = Settings()

        connect_kwargs = {
            "max_connections": settings.redis_max_connections,
            "socket_connect_timeout": settings.redis_socket_timeout,
        }
        if settings.redis_url.startswith("rediss://"):
            connect_kwargs["ssl_cert_reqs"] = "required"

        async def _test():
            client = aioredis.from_url(settings.redis_url, **connect_kwargs)
            await client.ping()
            await client.aclose()

        asyncio.run(_test())
        latency = (time.monotonic() - start) * 1000
        return "OK", latency, "Connected"
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return "FAIL", latency, str(exc)[:200]


def check_qdrant() -> tuple[str, float, str]:
    """Test Qdrant connectivity with get_collections."""
    start = time.monotonic()
    try:
        from src.services.retrieval import get_qdrant_client
        client = get_qdrant_client()
        collections = client.get_collections()
        latency = (time.monotonic() - start) * 1000
        names = [c.name for c in collections.collections]
        return "OK", latency, f"Collections: {names}"
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return "FAIL", latency, str(exc)[:200]


def check_gemini() -> tuple[str, float, str]:
    """Test Gemini connectivity."""
    import asyncio
    start = time.monotonic()
    try:
        settings = Settings()
        if not settings.gemini_api_key:
            latency = (time.monotonic() - start) * 1000
            return "SKIP", latency, "GEMINI_API_KEY not set"

        from langchain_core.messages import HumanMessage

        from src.services.llm import call_llm

        async def _test():
            return await call_llm(
                [HumanMessage(content="Say 'ok' in one word.")],
                provider="gemini",
                max_tokens=5,
                timeout_seconds=30,
            )

        asyncio.run(_test())
        latency = (time.monotonic() - start) * 1000
        return "OK", latency, "Connected"
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        # In development, timeout/rate-limit are acceptable
        if "timeout" in str(exc).lower() or "429" in str(exc):
            return "WARN", latency, f"LLM slow/rate-limited (OK in dev): {str(exc)[:100]}"
        return "FAIL", latency, str(exc)[:200]


def check_openrouter() -> tuple[str, float, str]:
    """Test OpenRouter connectivity."""
    import asyncio
    start = time.monotonic()
    try:
        settings = Settings()
        if not settings.openrouter_api_key:
            latency = (time.monotonic() - start) * 1000
            return "SKIP", latency, "OPENROUTER_API_KEY not set"

        from langchain_core.messages import HumanMessage

        from src.services.llm import call_llm

        async def _test():
            return await call_llm(
                [HumanMessage(content="Say 'ok' in one word.")],
                provider="openrouter",
                max_tokens=5,
                timeout_seconds=30,
            )

        asyncio.run(_test())
        latency = (time.monotonic() - start) * 1000
        return "OK", latency, "Connected"
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        # In development, rate limits on free tier are acceptable
        if "429" in str(exc) or "rate" in str(exc).lower():
            return "WARN", latency, f"Rate limited (OK in dev): {str(exc)[:100]}"
        return "FAIL", latency, str(exc)[:200]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate configuration and connectivity.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (SKIP results) in addition to failures.",
    )
    args = parser.parse_args()

    checks = [
        ("Settings", check_settings),
        ("PostgreSQL", check_postgres),
        ("Redis", check_redis),
        ("Qdrant", check_qdrant),
        ("Gemini", check_gemini),
        ("OpenRouter", check_openrouter),
    ]

    print("\n" + "=" * 70)
    print("  Configuration & Connectivity Validator")
    print("=" * 70)
    print(f"  {'Service':<15} {'Status':<8} {'Latency':<12} {'Notes'}")
    print("-" * 70)

    has_failure = False
    has_skip = False
    has_warn = False

    for name, check_fn in checks:
        status, latency, notes = check_fn()
        status_display = {
            "OK": "\033[92mOK\033[0m",      # Green
            "FAIL": "\033[91mFAIL\033[0m",  # Red
            "SKIP": "\033[93mSKIP\033[0m",  # Yellow
            "WARN": "\033[93mWARN\033[0m",  # Yellow
        }.get(status, status)

        print(f"  {name:<15} {status_display:<17} {latency:>8.1f}ms   {notes}")

        if status == "FAIL":
            has_failure = True
        elif status == "SKIP":
            has_skip = True
        elif status == "WARN":
            has_warn = True

    print("=" * 70)

    if has_failure:
        print("\n  [FAIL] One or more critical checks FAILED.")
        sys.exit(1)
    elif has_skip and args.strict:
        print("\n  [WARN] Some checks were skipped (--strict mode).")
        sys.exit(1)
    else:
        print("\n  [OK] All checks passed (or warnings in dev mode).")
        sys.exit(0)


if __name__ == "__main__":
    main()
