"""Application settings — pydantic-settings with full field descriptions and validation."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file.

    Every field maps 1:1 to an env var (case-insensitive). Defaults are
    sensible for local development against the docker-compose stack.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ────────────────────────────────────────────────────────────────

    database_url: str = Field(
        default="postgresql+asyncpg://triage_user:triage_pass@localhost:5432/triage_db",
        description="SQLAlchemy async connection string for PostgreSQL.",
    )

    # ── Qdrant (Vector DB) ─────────────────────────────────────────────────────

    qdrant_host: str = Field(
        default="localhost",
        description="Hostname where Qdrant is running.",
    )
    qdrant_port: int = Field(
        default=6333,
        ge=1,
        le=65535,
        description="gRPC/HTTP port for Qdrant.",
    )
    qdrant_collection: str = Field(
        default="tickets",
        description="Qdrant collection name used for ticket vector storage.",
    )

    # ── LLM Provider ────────────────────────────────────────────────────────────

    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Which LLM backend to use. Swappable without touching agent logic.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (required when llm_provider='openai').",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model identifier.",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key (required when llm_provider='anthropic').",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model identifier.",
    )

    # ── Redis (background jobs / caching) ──────────────────────────────────────

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for Inngest and caching.",
    )

    # ── Agent Configuration ────────────────────────────────────────────────────

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum classification confidence to auto-triage. "
            "Tickets below this threshold are escalated to a human."
        ),
    )
    max_loop_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of agent loop iterations before forced escalation.",
    )
    triage_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="p95 latency budget for a single triage run (seconds).",
    )

    # ── Observability (OpenTelemetry) ──────────────────────────────────────────

    otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP gRPC endpoint for OpenTelemetry trace export.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Python log level for structlog.",
    )

    # ── Evaluation ──────────────────────────────────────────────────────────────

    eval_tickets_path: str = Field(
        default="./data/eval_tickets.json",
        description="Path to the labeled eval ticket fixture (JSON).",
    )

    # ── Validators ──────────────────────────────────────────────────────────────

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, v: str) -> str:
        if not v.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError(
                "database_url must use the asyncpg or aiosqlite driver "
                f"(got {v.split('://')[0]}://)"
            )
        return v

    @field_validator("qdrant_host")
    @classmethod
    def _validate_qdrant_host(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("qdrant_host must not be empty.")
        return v.strip()

    @field_validator("redis_url")
    @classmethod
    def _validate_redis_url(cls, v: str) -> str:
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("redis_url must start with redis:// or rediss://")
        return v

    @field_validator("otlp_endpoint")
    @classmethod
    def _validate_otlp_endpoint(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("otlp_endpoint must be a full URL (http://… or https://…)")
        return v

    @field_validator("eval_tickets_path")
    @classmethod
    def _validate_eval_path(cls, v: str) -> str:
        if not v.endswith(".json"):
            raise ValueError("eval_tickets_path must point to a .json file")
        return v

    # ── Derived helpers ─────────────────────────────────────────────────────────

    def require_openai_key(self) -> str:
        """Return the OpenAI key or raise if missing when OpenAI is selected."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER='openai'")
        return self.openai_api_key

    def require_anthropic_key(self) -> str:
        """Return the Anthropic key or raise if missing when Anthropic is selected."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER='anthropic'")
        return self.anthropic_api_key


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    The cache is per-process, so calling ``get_settings()`` hundreds of
    times costs only one parse + validation pass.
    """
    return Settings()
