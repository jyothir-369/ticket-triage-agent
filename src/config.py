"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object — values come from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Database ────────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://triage_user:triage_pass@localhost:5432/triage_db"

    # ── Qdrant ──────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "tickets"

    # ── LLM Provider ────────────────────────────────────────────────────────────
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # ── Redis ───────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Agent Config ────────────────────────────────────────────────────────────
    confidence_threshold: float = 0.7
    max_loop_retries: int = 3
    triage_timeout_seconds: int = 30

    # ── Observability ───────────────────────────────────────────────────────────
    otlp_endpoint: str = "http://localhost:4317"
    log_level: str = "INFO"

    # ── Evaluation ──────────────────────────────────────────────────────────────
    eval_tickets_path: str = "./data/eval_tickets.json"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
