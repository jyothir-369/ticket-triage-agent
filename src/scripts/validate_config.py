#!/usr/bin/env python
"""Configuration validation script.

Run at container startup (or in CI) to verify all required settings are present
and valid for the current environment.

Exit codes:
  0 - All checks passed
  1 - One or more validation errors
  2 - Runtime import/config error
"""

from __future__ import annotations

import sys
from typing import NoReturn

from src.config import get_settings


def _exit_error(msg: str) -> NoReturn:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def _exit_warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def validate_production_requirements(s: "Settings") -> list[str]:
    """Return list of validation error messages for production env."""
    errors = []

    # At least one LLM provider configured
    has_llm = any([
        s.gemini_api_key,
        s.openai_api_key,
        s.anthropic_api_key,
        s.openrouter_api_key,
    ])
    if not has_llm:
        errors.append(
            "In production, at least one LLM API key must be set "
            "(GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY)"
        )

    # Database must not be localhost
    if "localhost" in s.database_url or "127.0.0.1" in s.database_url:
        errors.append(
            "In production, DATABASE_URL must not point to localhost"
        )

    # Require API keys to be configured (auth enabled)
    if not s.api_keys:
        errors.append(
            "In production, API_KEYS must be configured (at least one API key)"
        )

    # Require reply delivery mechanism
    if s.reply_delivery_provider == "none":
        errors.append(
            "In production, REPLY_DELIVERY_PROVIDER must be 'email' or 'webhook' "
            "(cannot be 'none' for FR4 compliance)"
        )

    return errors


def validate_all_environments(s: "Settings") -> list[str]:
    """Return list of validation error messages that apply to all envs."""
    errors = []

    # Qdrant collection vector size must match embedding provider
    if s.embedding_provider == "openai" and s.embedding_vector_size != 1536:
        errors.append(
            f"EMBEDDING_VECTOR_SIZE ({s.embedding_vector_size}) must be 1536 "
            "for OpenAI text-embedding-ada-002"
        )

    # SSL mode for database in non-development
    if s.environment != "development" and s.database_ssl_mode == "disable":
        errors.append(
            f"DATABASE_SSL_MODE cannot be 'disable' in {s.environment} environment"
        )

    # Valid reply delivery configuration
    if s.reply_delivery_provider == "email":
        if not s.smtp_host or not s.smtp_from_address:
            errors.append(
                "When REPLY_DELIVERY_PROVIDER=email, SMTP_HOST and SMTP_FROM_ADDRESS must be set"
            )
    elif s.reply_delivery_provider == "webhook":
        if not s.reply_webhook_url:
            errors.append(
                "When REPLY_DELIVERY_PROVIDER=webhook, REPLY_WEBHOOK_URL must be set"
            )

    # Redis TLS in production
    if s.environment == "production" and s.redis_url.startswith("redis://"):
        errors.append(
            "In production, REDIS_URL must use rediss:// (TLS) not redis://"
        )

    return errors


def main() -> int:
    print("=" * 60)
    print("Support-Ticket Triage Agent — Configuration Validation")
    print("=" * 60)
    print(f"Environment: {getattr(get_settings(), 'environment', 'unknown')}")

    try:
        settings = get_settings()
    except Exception as e:
        print(f"[ERROR] Failed to load settings: {e}", file=sys.stderr)
        return 2

    all_errors: list[str] = []

    # Common validations
    all_errors.extend(validate_all_environments(settings))

    # Environment-specific validations
    if settings.environment == "production":
        all_errors.extend(validate_production_requirements(settings))

    # Report results
    if all_errors:
        print("\n[FAIL] Configuration validation failed:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("\n[OK] All configuration checks passed.")
    print(f"  LLM Provider: {settings.llm_provider} (fallback: {settings.llm_fallback_provider or 'none'})")
    print(f"  Embedding: {settings.embedding_provider or 'auto'} ({settings.embedding_vector_size} dims)")
    print(f"  Reply Delivery: {settings.reply_delivery_provider}")
    print(f"  API Auth: {'enabled' if settings.api_keys else 'disabled'}")
    print(f"  CORS: restricted (env={settings.environment})")
    return 0


if __name__ == "__main__":
    sys.exit(main())