"""Tests for the configuration module."""

from src.config import Settings, get_settings


def test_settings_defaults():
    s = Settings()
    assert s.llm_provider in ("openai", "anthropic")
    assert s.confidence_threshold == 0.7
    assert s.max_loop_retries == 3
    assert s.triage_timeout_seconds == 30
    assert s.qdrant_host == "localhost"
    assert s.qdrant_port == 6333


def test_get_settings_returns_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
