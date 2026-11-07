"""API client for the triage backend + shared formatting helpers.

Every fetch goes through :func:`api_get` / :func:`api_post`, which normalise
failures into ``st.session_state["api_error"]`` (consumed by the banner in
``app.py``). Pages never see raw exceptions — they get ``None`` and decide how
to render an error / empty / partial state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import httpx
import streamlit as st

API_BASE = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_USERNAME = os.environ.get("DASHBOARD_USERNAME", "admin")
DEFAULT_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "admin")
# Dashboard→backend auth key; falls back to DASHBOARD_API_KEY env var.
try:
    from src.config import get_settings
    _DASHBOARD_API_KEY = get_settings().dashboard_api_key
except Exception:  # noqa: BLE001 — never break the dashboard on config errors
    _DASHBOARD_API_KEY = ""

REQUEST_TIMEOUT = 15.0
POST_TIMEOUT = 90.0


def record_api_error(error: dict | None) -> None:
    """Store the most recent API error (or clear it on success)."""
    st.session_state["api_error"] = error


def _describe(e: Exception) -> dict:
    if isinstance(e, httpx.ConnectError):
        return {"type": "connection", "detail": f"Could not reach the API at {API_BASE}."}
    if isinstance(e, httpx.TimeoutException):
        return {"type": "timeout", "detail": "The API request timed out."}
    if isinstance(e, httpx.HTTPStatusError):
        body = e.response.text[:300]
        return {"type": "http", "status": e.response.status_code, "detail": body}
    return {"type": "unknown", "detail": f"{type(e).__name__}: {str(e)[:200]}"}


def api_get(endpoint: str, params: dict | None = None, timeout: float = REQUEST_TIMEOUT):
    """GET + parse JSON. Returns parsed JSON or ``None`` on failure."""
    headers = {}
    if _DASHBOARD_API_KEY:
        headers["X-API-Key"] = _DASHBOARD_API_KEY
    try:
        resp = httpx.get(f"{API_BASE}{endpoint}", params=params, timeout=timeout, headers=headers)
        resp.raise_for_status()
        record_api_error(None)
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — normalised below
        record_api_error(_describe(exc))
        return None


def api_post(endpoint: str, json_data: dict | None = None, timeout: float = POST_TIMEOUT):
    """POST (or PATCH/DELETE when ``method`` is overridden) + parse JSON."""
    headers = {}
    if _DASHBOARD_API_KEY:
        headers["X-API-Key"] = _DASHBOARD_API_KEY
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", json=json_data, timeout=timeout, headers=headers)
        resp.raise_for_status()
        record_api_error(None)
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        record_api_error(_describe(exc))
        return None


def api_patch(endpoint: str, json_data: dict | None = None, timeout: float = POST_TIMEOUT):
    """PATCH a resource and parse JSON (used for draft updates)."""
    headers = {}
    if _DASHBOARD_API_KEY:
        headers["X-API-Key"] = _DASHBOARD_API_KEY
    try:
        resp = httpx.patch(f"{API_BASE}{endpoint}", json=json_data, timeout=timeout, headers=headers)
        resp.raise_for_status()
        record_api_error(None)
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        record_api_error(_describe(exc))
        return None


def check_backend_health() -> dict | None:
    """Thin wrapper around ``/health`` — returns the payload or ``None``.

    Uses the same timeout as other API calls for consistency. Before v1.3.2,
    the health-check used a hard-coded 8-second timeout, which meant slow but
    reachable dependencies would trigger a "connection"-style offline error.
    The request-timeout and error-type mismatch caused the dashboard to show
    "API offline" even when the FastAPI process was up and serving health.
    The backend health endpoint aggregates per‑dependency status, so an HTTP
    200 with any unhealthy/lost dependency is a "degraded" situation, not a
    genuine offline event. Only failures to even reach FastAPI (timeout,
    connection refused, DNS error) should render as "API offline".

    Health probes intentionally skip ``X-API-Key`` — they are side-channel
    Kubernetes readiness / liveness checks that must never fail just because
    auth is engaged. Ticket endpoints below all require it.
    """
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        record_api_error(None)
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        record_api_error(_describe(exc))
        return None


# ---------------------------------------------------------------------------
# Concurrent raw fetch (no session-state writes in worker threads)
# ---------------------------------------------------------------------------


def _raw_fetch(path: str, params: dict | None, timeout: float) -> tuple[str, dict | None]:
    """One HTTP GET; returns (path, json) or (path, None) on failure."""
    headers = {} if not _DASHBOARD_API_KEY else {"X-API-Key": _DASHBOARD_API_KEY}
    try:
        resp = httpx.get(f"{API_BASE}{path}", params=params, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return path, resp.json()
    except Exception:  # noqa: BLE001
        return path, None


def fetch_many(
    endpoints: list[tuple[str, str, dict | None]],
    timeout: float = 20.0,
) -> dict:
    """Fetch several endpoints concurrently.

    Each entry is ``(label, path, params)``. Returns ``{label: json|None}``.
    Safe to call from worker threads — it never writes to ``st.session_state``;
    the caller owns error handling. Labels must be unique.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _run(ep: tuple[str, str, dict | None]) -> tuple[str, dict | None]:
        _, path, params = ep
        return ep[0], _raw_fetch(path, params, timeout)[1]

    results: dict = {}
    with ThreadPoolExecutor(max_workers=min(len(endpoints), 8)) as pool:
        for label, payload in pool.map(_run, endpoints):
            results[label] = payload
    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_timestamp(ts: str | None, full: bool = False) -> str:
    """Format an ISO timestamp to a relative string, or absolute when ``full``."""
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if full:
            return dt.strftime("%b %d, %Y %H:%M")
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt
        if delta.days > 0:
            return f"{delta.days}d ago"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes}m ago"
        return "Just now"
    except Exception:
        return str(ts)[:16]


def format_duration(ms: int | None) -> str:
    """Format milliseconds into a readable duration."""
    if ms is None:
        return "—"
    if ms < 1_000:
        return f"{ms}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60000:.1f}m"


def safe_json(value: str | None) -> dict | list | None:
    """Parse a JSON string defensively; returns ``None`` when unparseable."""
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def preview(text: str | None, limit: int = 200) -> str:
    """Short preview with ellipsis."""
    if not text:
        return ""
    return text[:limit] + ("…" if len(text) > limit else "")


def is_valid_uuid_ish(ticket_id: str) -> bool:
    """Loose sanity check for a ticket id (UUID or hex string)."""
    return bool(ticket_id) and len(ticket_id) >= 6