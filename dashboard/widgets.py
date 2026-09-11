"""Reusable UI components: badges, meters, states, truncation, clipboard.

These render HTML inside ``st.markdown(..., unsafe_allow_html=True)`` and
expose the handful of interactive pieces Streamlit needs (buttons, toggles,
downloads) behind small, consistent wrappers so every page follows the same
error / empty / partial / loading story.
"""

from __future__ import annotations

import html
import json

import streamlit as st

from dashboard.api import api_patch, api_post
from dashboard.theme import CATEGORY_COLORS, STATUS_COLORS, URGENCY_COLORS


# ---------------------------------------------------------------------------
# Badges & meters
# ---------------------------------------------------------------------------

STATUS_ICONS = {
    "resolved": "✅",
    "approved": "✅",
    "escalated": "⚠️",
    "awaiting_review": "👁️",
    "processing": "⏳",
    "pending": "⏸️",
    "open": "📥",
    "classified": "🏷️",
    "failed": "❌",
    "rejected": "🚫",
}

URGENCY_ICONS = {
    "critical": "🔴",
    "urgent": "🟠",
    "high": "🟡",
    "medium": "🟢",
    "low": "⚪",
}


def status_badge(status: str) -> str:
    """HTML badge for a lifecycle status."""
    css = f"badge-{status}" if status else "badge-pending"
    return (
        f'<span class="badge {css}">{STATUS_ICONS.get(status or "", "•")} '
        f"{(status or '—').upper()}</span>"
    )


def category_badge(category: str | None) -> str:
    """HTML badge for a ticket category."""
    if not category:
        return '<span class="badge badge-pending">—</span>'
    color = CATEGORY_COLORS.get(category, "#6366f1")
    label = category.replace("_", " ").upper()
    return (
        f'<span class="badge" style="background:{color}22;color:{color};'
        f'box-shadow:inset 0 0 0 1px {color}55;">{label}</span>'
    )


def urgency_badge(urgency: str | None) -> str:
    """HTML badge for an urgency level."""
    if not urgency:
        return '<span class="badge badge-pending">—</span>'
    css = f"badge-{urgency}" if urgency in URGENCY_ICONS else "badge-pending"
    return f'<span class="badge {css}">{URGENCY_ICONS.get(urgency, "•")} {urgency.upper()}</span>'


def confidence_meter(value: float | None, large: bool = False) -> str:
    """HTML confidence bar with a % label and colour coding."""
    if value is None:
        return '<span class="placeholder-text">Not yet generated</span>'
    pct = max(0.0, min(1.0, value))
    css = "confidence-low" if pct < 0.5 else "confidence-medium" if pct <= 0.7 else "confidence-high"
    size = " confidence-large" if large else ""
    bar_w = int(round(pct * 100))
    return f"""
    <div style="display:flex;align-items:center;gap:10px;">
        <div class="confidence-bar-bg{size}" style="flex:1;max-width:260px;">
            <div class="confidence-bar-fill {css}" style="width:{bar_w}%;"></div>
        </div>
        <span style="font-size:1rem;font-weight:700;color:var(--text-900);min-width:44px;">{bar_w}%</span>
    </div>
    """


def confidence_label(value: float | None) -> str:
    """Human label + colour group for a confidence score."""
    if value is None:
        return "N/A"
    if value < 0.5:
        return "⚠️ Low"
    if value <= 0.7:
        return "🟡 Medium"
    return "🟢 High"


# ---------------------------------------------------------------------------
# Section / card helpers
# ---------------------------------------------------------------------------


def section_header(icon: str, title: str, sub: str | None = None) -> None:
    parts = [f'<div class="section-header"><span class="icon">{icon}</span><h2>{title}</h2>']
    if sub:
        parts.append(f'<span class="small muted" style="margin-left:6px;">{sub}</span>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def card_open() -> None:
    st.markdown('<div class="s-card">', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading / empty / error / partial states
# ---------------------------------------------------------------------------


def loading(message: str = "Loading…"):
    """Context-manager spinner with a descriptive message."""
    return st.spinner(message)


def empty_state(
    icon: str,
    heading: str,
    message: str,
    primary_label: str | None = None,
    primary_key: str | None = None,
) -> bool:
    """Render an empty state. Returns True when the primary action is clicked."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <h3>{heading}</h3>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if primary_label:
        return st.button(primary_label, key=primary_key, type="primary", use_container_width=False)
    return False


def error_state(icon: str, heading: str, message: str, retry_key: str | None = None) -> bool:
    """Render an error state; returns True when Retry is clicked."""
    st.markdown(
        f"""
        <div class="empty-state" style="border-color:var(--danger);">
            <div class="empty-icon">{icon}</div>
            <h3>{heading}</h3>
            <p style="font-family:var(--font-mono);font-size:0.82rem;color:var(--text-400);">{message}</p>
            <p class="small" style="font-size:0.78rem;color:var(--text-400);">
                Error type: {icon} · No stack trace was shown intentionally.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if retry_key:
        return st.button("↻ Retry", key=retry_key, type="primary", use_container_width=False)
    return False


def network_banner() -> bool:
    """Top-of-page banner when the API is unreachable. Returns True on retry click."""
    err = st.session_state.get("api_error")
    if not err:
        return False
    label = {
        "connection": "Cannot reach the API server",
        "timeout": "API request timed out",
        "http": f"API returned HTTP {err.get('status')}",
        "unknown": "Unexpected API error",
    }.get(err.get("type"), "API error")
    st.markdown(
        f"""
        <div class="network-banner">
            <span>📡 <strong>{label}</strong> — {err.get('detail', '')[:160]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.button("↻ Retry connection", key="retry_conn", type="primary")


def partial_banner(title: str, detail: str) -> None:
    """Flag partial data: what succeeded vs. what failed."""
    st.markdown(
        f'<div class="partial-banner">⚠️ <strong>{title}</strong> — {detail}</div>',
        unsafe_allow_html=True,
    )


def placeholder(text: str = "Not yet generated") -> None:
    """Inline placeholder for missing/optional fields."""
    st.markdown(f'<span class="placeholder-text">{text}</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Long content truncation with "show more"
# ---------------------------------------------------------------------------


def _truncated_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def show_more_text(text: str | None, limit: int, key: str) -> None:
    """Render text with an expand/collapse toggle past ``limit`` characters."""
    if not text:
        placeholder()
        return
    open_state = st.session_state.get(f"showmore_{key}", False)
    shown = text if open_state else _truncated_text(text, limit)
    st.markdown(
        f"""<div style="white-space:pre-wrap;font-size:0.9rem;color:var(--text-700);
        line-height:1.55;background:var(--surface-50);border:1px solid var(--surface-200);
        border-radius:var(--radius-md);padding:14px 16px;">{shown}</div>""",
        unsafe_allow_html=True,
    )
    if len(text) > limit:
        if st.button(("Show less" if open_state else "Show more"), key=f"sm_btn_{key}", type="secondary"):
            st.session_state[f"showmore_{key}"] = not open_state
            st.rerun()


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------


def copy_button(label: str, text: str, key: str, icon: str = "📋") -> bool:
    """Button that copies ``text`` to the clipboard via injected JS.

    Uses a hidden textarea + ``execCommand('copy')`` (works in sandboxed
    iframes), then falls back to the clipboard API.
    """
    if not text:
        st.button(label, key=key, disabled=True)
        return False
    payload = html.escape(json.dumps(text, ensure_ascii=False))
    if st.button(f"{icon} {label}", key=key):
        st.components.v1.html(
            f"""<div><textarea id="c">{payload}</textarea><script>
            var t=document.getElementById('c');
            t.focus(); t.select(); t.setSelectionRange(0, t.value.length);
            var ok=false;
            try {{ ok=document.execCommand('copy'); }} catch(e) {{ ok=false; }}
            if(!ok && navigator.clipboard && navigator.clipboard.writeText) {{
              navigator.clipboard.writeText(t.value);
            }}
            </script></div>""",
            height=0,
            width=0,
        )
        st.toast("Copied to clipboard", icon="✅")
        return True
    return False


def download_text_button(label: str, text: str, filename: str, key: str, icon: str = "⬇️") -> None:
    """Download button for trace / metadata as a text file."""
    if not text:
        st.button(label, key=key, disabled=True)
        return
    st.download_button(
        label=f"{icon} {label}",
        data=text,
        file_name=filename,
        mime="text/plain",
        key=key,
    )


# ---------------------------------------------------------------------------
# Action helpers (shared between queue & detail)
# ---------------------------------------------------------------------------

USERNAME_KEY = "username"


def approve_ticket(ticket_id: str) -> dict | None:
    """Approve a draft; returns the API payload or None."""
    with st.spinner("Approving draft…"):
        return api_post(
            f"/tickets/{ticket_id}/approve",
            json_data={"reviewer": st.session_state.get(USERNAME_KEY, "user")},
            timeout=30,
        )


def escalate_ticket(ticket_id: str, reason: str) -> dict | None:
    """Manually escalate a ticket with a reason."""
    with st.spinner("Escalating ticket…"):
        return api_post(
            f"/tickets/{ticket_id}/escalate",
            json_data={
                "reason": reason,
                "reviewer": st.session_state.get(USERNAME_KEY, "user"),
            },
            timeout=30,
        )


def update_draft(ticket_id: str, draft_text: str) -> dict | None:
    """Persist an edited draft before approve."""
    return api_patch(f"/tickets/{ticket_id}/draft", json_data={"draft_text": draft_text})


def rerun_triage(ticket_id: str) -> dict | None:
    """Re-trigger the triage pipeline for an existing ticket."""
    with st.spinner("Starting triage pipeline…"):
        return api_post(f"/tickets/{ticket_id}/triage", timeout=30)