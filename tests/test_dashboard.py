"""Tests for the Streamlit dashboard — pure logic and API client functions.

Tests the helper functions and authentication logic without launching Streamlit.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


def test_api_get_constructs_correct_url():
    """api_get should construct the correct URL from API_BASE and endpoint."""
    from dashboard.app import api_get

    with patch("dashboard.app.httpx") as mock_httpx:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_resp

        result = api_get("/health")
        mock_httpx.get.assert_called_once()
        call_args = mock_httpx.get.call_args
        assert "/health" in call_args[0][0]


def test_api_post_constructs_correct_url():
    """api_post should construct the correct URL and pass json_data."""
    from dashboard.app import api_post

    with patch("dashboard.app.httpx") as mock_httpx:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ticket_id": "123"}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_resp

        result = api_post("/tickets/", json_data={"content": "test"})
        mock_httpx.post.assert_called_once()
        call_args = mock_httpx.post.call_args
        assert "/tickets/" in call_args[0][0]
        assert call_args[1]["json"] == {"content": "test"}


def test_api_get_returns_none_on_error():
    """api_get should return None and not raise on HTTP errors."""
    from dashboard.app import api_get

    with patch("dashboard.app.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("Connection refused")
        result = api_get("/health")
        assert result is None


def test_api_post_returns_none_on_error():
    """api_post should return None and not raise on HTTP errors."""
    from dashboard.app import api_post

    with patch("dashboard.app.httpx") as mock_httpx:
        mock_httpx.post.side_effect = Exception("Connection refused")
        result = api_post("/tickets/", json_data={"content": "test"})
        assert result is None


def test_login_default_credentials():
    """Default credentials should be admin/admin."""
    from dashboard.app import DEFAULT_USERNAME, DEFAULT_PASSWORD

    assert DEFAULT_USERNAME == "admin"
    assert DEFAULT_PASSWORD == "admin"


def test_login_env_override():
    """Credentials should be overridable via environment variables."""
    with patch.dict(os.environ, {"DASHBOARD_USERNAME": "custom_user", "DASHBOARD_PASSWORD": "custom_pass"}):
        # Re-import to pick up env changes
        import importlib
        import dashboard.app
        importlib.reload(dashboard.app)

        assert dashboard.app.DEFAULT_USERNAME == "custom_user"
        assert dashboard.app.DEFAULT_PASSWORD == "custom_pass"


def test_logout_clears_session_state():
    """logout() should clear authenticated state from session."""
    import streamlit as st
    from dashboard.app import logout

    # Mock streamlit session state
    with patch.object(st, "session_state", {"authenticated": True, "username": "admin"}):
        with patch.object(st, "rerun"):
            logout()
            assert st.session_state.get("authenticated") is False
