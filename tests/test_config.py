"""Tests for the configuration module — validators and defaults."""

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings


class TestSettingsDefaults:
    def test_default_llm_provider(self):
        s = Settings()
        assert s.llm_provider == "openai"

    def test_default_confidence_threshold(self):
        s = Settings()
        assert s.confidence_threshold == 0.7

    def test_default_max_loop_retries(self):
        s = Settings()
        assert s.max_loop_retries == 3

    def test_default_triage_timeout(self):
        s = Settings()
        assert s.triage_timeout_seconds == 30

    def test_default_qdrant(self):
        s = Settings()
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333

    def test_default_eval_path(self):
        s = Settings()
        assert s.eval_tickets_path.endswith(".json")


class TestSettingsValidators:
    def test_database_url_rejects_sync_driver(self):
        with pytest.raises(ValidationError, match="asyncpg or aiosqlite"):
            Settings(database_url="postgresql://localhost/db")

    def test_database_url_accepts_asyncpg(self):
        s = Settings(database_url="postgresql+asyncpg://user:pass@localhost/db")
        assert "asyncpg" in s.database_url

    def test_database_url_accepts_aiosqlite(self):
        s = Settings(database_url="sqlite+aiosqlite:///test.db")
        assert "aiosqlite" in s.database_url

    def test_qdrant_host_empty_rejected(self):
        with pytest.raises(ValidationError, match="qdrant_host must not be empty"):
            Settings(qdrant_host="   ")

    def test_redis_url_must_start_with_redis(self):
        with pytest.raises(ValidationError, match="redis://"):
            Settings(redis_url="http://localhost:6379")

    def test_otlp_endpoint_must_be_url(self):
        with pytest.raises(ValidationError, match="full URL"):
            Settings(otlp_endpoint="localhost:4317")

    def test_eval_path_must_be_json(self):
        with pytest.raises(ValidationError, match="\.json"):
            Settings(eval_tickets_path="./data/tickets.csv")

    def test_confidence_threshold_in_range(self):
        s = Settings(confidence_threshold=0.5)
        assert s.confidence_threshold == 0.5

    def test_confidence_threshold_too_high(self):
        with pytest.raises(ValidationError):
            Settings(confidence_threshold=1.5)

    def test_max_loop_retries_in_range(self):
        s = Settings(max_loop_retries=5)
        assert s.max_loop_retries == 5

    def test_max_loop_retries_too_high(self):
        with pytest.raises(ValidationError):
            Settings(max_loop_retries=20)


class TestSettingsHelpers:
    def test_require_openai_key_missing(self):
        s = Settings(llm_provider="openai", openai_api_key="")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            s.require_openai_key()

    def test_require_openai_key_present(self):
        s = Settings(llm_provider="openai", openai_api_key="sk-test")
        assert s.require_openai_key() == "sk-test"

    def test_require_anthropic_key_missing(self):
        s = Settings(llm_provider="anthropic", anthropic_api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            s.require_anthropic_key()

    def test_require_anthropic_key_present(self):
        s = Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")
        assert s.require_anthropic_key() == "sk-ant-test"


class TestGetSettingsSingleton:
    def test_returns_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
