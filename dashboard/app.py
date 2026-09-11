"""Streamlit dashboard — triage metrics, ticket queue, and trace viewer.

Production-grade SaaS-style dashboard for support team daily use.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Triage Agent",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_USERNAME = os.environ.get("DASHBOARD_USERNAME", "admin")
DEFAULT_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "admin")


# ── Custom CSS ─────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Color Palette (Light Mode) ── */
:root {
    --brand-500: #6366f1;
    --brand-600: #4f46e5;
    --brand-700: #4338ca;
    --brand-50: #eef2ff;
    --brand-100: #e0e7ff;

    --surface-0: #ffffff;
    --surface-50: #f8fafc;
    --surface-100: #f1f5f9;
    --surface-200: #e2e8f0;
    --surface-300: #cbd5e1;

    --text-900: #0f172a;
    --text-700: #334155;
    --text-500: #64748b;
    --text-400: #94a3b8;
    --text-300: #cbd5e1;

    --success: #10b981;
    --success-bg: #d1fae5;
    --warning: #f59e0b;
    --warning-bg: #fef3c7;
    --danger: #ef4444;
    --danger-bg: #fee2e2;
    --info: #3b82f6;
    --info-bg: #dbeafe;

    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);

    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;

    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Dark Mode ── */
@media (prefers-color-scheme: dark) {
    :root {
        --surface-0: #0f172a;
        --surface-50: #1e293b;
        --surface-100: #334155;
        --surface-200: #475569;
        --surface-300: #64748b;

        --text-900: #f1f5f9;
        --text-700: #e2e8f0;
        --text-500: #94a3b8;
        --text-400: #64748b;
        --text-300: #475569;

        --brand-50: #1e1b4b;
        --brand-100: #312e81;
        --success-bg: #064e3b;
        --warning-bg: #78350f;
        --danger-bg: #7f1d1d;
        --info-bg: #1e3a5f;

        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.4);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.5);
    }
}

/* ── Global Resets ── */
.stApp {
    font-family: var(--font-sans) !important;
}

section[data-testid="stSidebar"] {
    background: var(--surface-0) !important;
    border-right: 1px solid var(--surface-200) !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-700);
}

/* ── Sidebar Branding ── */
.sidebar-brand {
    padding: 16px 0 8px 0;
    border-bottom: 1px solid var(--surface-200);
    margin-bottom: 16px;
}

.sidebar-brand h2 {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-900);
    margin: 0;
    letter-spacing: -0.02em;
}

.sidebar-brand .subtitle {
    font-size: 0.8rem;
    color: var(--text-500);
    margin-top: 2px;
}

/* ── KPI Cards ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 20px 24px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--brand-500);
}

.kpi-card .label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-500);
    margin-bottom: 8px;
}

.kpi-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-900);
    line-height: 1.1;
    letter-spacing: -0.02em;
}

.kpi-card .trend {
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 6px;
}

.kpi-card .trend.up { color: var(--success); }
.kpi-card .trend.down { color: var(--danger); }
.kpi-card .trend.neutral { color: var(--text-400); }

.kpi-card.accent {
    border-left: 4px solid var(--brand-500);
}

.kpi-card.success {
    border-left: 4px solid var(--success);
}

.kpi-card.warning {
    border-left: 4px solid var(--warning);
}

.kpi-card.danger {
    border-left: 4px solid var(--danger);
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--surface-200);
}

.section-header h2 {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-900);
    margin: 0;
    letter-spacing: -0.01em;
}

.section-header .icon {
    font-size: 1.25rem;
}

/* ── Cards ── */
.s-card {
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 20px 24px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 16px;
}

.s-card-elevated {
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 20px 24px;
    box-shadow: var(--shadow-md);
    margin-bottom: 16px;
}

/* ── Status Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    white-space: nowrap;
}

.badge-resolved { background: var(--success-bg); color: var(--success); }
.badge-escalated { background: var(--warning-bg); color: var(--warning); }
.badge-processing { background: var(--info-bg); color: var(--info); }
.badge-pending { background: var(--surface-100); color: var(--text-500); }
.badge-failed { background: var(--danger-bg); color: var(--danger); }
.badge-approved { background: var(--success-bg); color: var(--success); }

.badge-urgent { background: var(--danger-bg); color: var(--danger); }
.badge-high { background: #fff7ed; color: #ea580c; }
.badge-medium { background: var(--warning-bg); color: #b45309; }
.badge-low { background: var(--success-bg); color: #059669; }

/* ── Confidence Meter ── */
.confidence-bar-bg {
    width: 100%;
    height: 8px;
    background: var(--surface-200);
    border-radius: 999px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}

.confidence-high { background: var(--success); }
.confidence-medium { background: var(--warning); }
.confidence-low { background: var(--danger); }

/* ── Trace Timeline ── */
.trace-timeline {
    display: flex;
    align-items: flex-start;
    gap: 0;
    padding: 16px 0;
    overflow-x: auto;
}

.trace-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 120px;
    position: relative;
    flex-shrink: 0;
}

.trace-step:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 18px;
    left: calc(50% + 18px);
    width: calc(100% - 36px);
    height: 3px;
    background: var(--surface-300);
    z-index: 0;
}

.trace-step.completed:not(:last-child)::after {
    background: var(--success);
}

.trace-step.failed:not(:last-child)::after {
    background: var(--danger);
}

.trace-dot {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    z-index: 1;
    border: 3px solid var(--surface-0);
    box-shadow: var(--shadow-sm);
}

.trace-dot.completed { background: var(--success); color: white; }
.trace-dot.failed { background: var(--danger); color: white; }
.trace-dot.running { background: var(--info); color: white; animation: pulse 1.5s infinite; }
.trace-dot.pending { background: var(--surface-200); color: var(--text-500); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.trace-label {
    margin-top: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-700);
    text-align: center;
    max-width: 100px;
}

.trace-duration {
    font-size: 0.7rem;
    color: var(--text-400);
    margin-top: 2px;
}

/* ── Login ── */
.login-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
}

.login-card {
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-lg);
    padding: 40px 48px;
    box-shadow: var(--shadow-lg);
    max-width: 420px;
    width: 100%;
}

.login-card h1 {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--text-900);
    text-align: center;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}

.login-card .login-subtitle {
    text-align: center;
    color: var(--text-500);
    font-size: 0.9rem;
    margin-bottom: 28px;
}

.login-card .brand-icon {
    text-align: center;
    font-size: 3rem;
    margin-bottom: 16px;
}

/* ── Ticket Row ── */
.ticket-row {
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer;
}

.ticket-row:hover {
    border-color: var(--brand-500);
    box-shadow: var(--shadow-md);
}

.ticket-row .ticket-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.ticket-row .ticket-id {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--brand-500);
}

.ticket-row .ticket-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

.ticket-row .ticket-preview {
    font-size: 0.9rem;
    color: var(--text-700);
    line-height: 1.5;
    margin-top: 4px;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: var(--text-500);
}

.empty-state .empty-icon {
    font-size: 3rem;
    margin-bottom: 12px;
    opacity: 0.5;
}

.empty-state h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-700);
    margin-bottom: 4px;
}

.empty-state p {
    font-size: 0.9rem;
}

/* ── Table Overrides ── */
.stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* ── Button Overrides ── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"] {
    background: var(--brand-500) !important;
    border-color: var(--brand-500) !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--brand-600) !important;
    border-color: var(--brand-600) !important;
}

/* ── Tab Overrides ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--surface-100);
    border-radius: var(--radius-sm);
    padding: 3px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

.stTabs [aria-selected="true"] {
    background: var(--surface-0) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Expander Overrides ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid var(--surface-200);
    margin: 20px 0;
}

/* ── Filter Bar ── */
.filter-bar {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    padding: 12px 16px;
    background: var(--surface-50);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    margin-bottom: 16px;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .kpi-row {
        grid-template-columns: repeat(2, 1fr);
    }
    .kpi-card .value {
        font-size: 1.5rem;
    }
    .trace-timeline {
        flex-direction: column;
        align-items: stretch;
    }
    .trace-step:not(:last-child)::after {
        display: none;
    }
}

/* ── Remove Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--surface-100);
}
::-webkit-scrollbar-thumb {
    background: var(--surface-300);
    border-radius: 3px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Authentication ──────────────────────────────────────────────────────────────


def check_login() -> bool:
    """Check if the user is authenticated via session state."""
    return st.session_state.get("authenticated", False)


def login_form() -> None:
    """Render the login form and handle authentication."""
    # Center everything vertically with spacer
    st.markdown('<div style="height: 12vh;"></div>', unsafe_allow_html=True)

    # Center column layout — branding + form in the same column
    _, center, _ = st.columns([1, 1.5, 1])
    with center:
        # Branding — sits above the form in the same column
        st.markdown(
            """
            <div style="text-align:center; margin-bottom: 24px;">
                <div style="font-size: 3rem; margin-bottom: 8px;">🎫</div>
                <h1 style="font-size: 1.75rem; font-weight: 800; color: var(--text-900);
                            margin: 0 0 4px 0; letter-spacing: -0.02em;">Triage Agent</h1>
                <p style="color: var(--text-500); font-size: 0.9rem; margin: 0;">
                    Sign in to access the support dashboard
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Form — inside a styled card container
        with st.form("login_form"):
            username = st.text_input(
                "Username", placeholder="Enter your username", label_visibility="visible"
            )
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password", label_visibility="visible"
            )
            submitted = st.form_submit_button(
                "Sign In", use_container_width=True, type="primary"
            )

            if submitted:
                if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")


def logout() -> None:
    """Clear authentication state and rerun."""
    st.session_state["authenticated"] = False
    st.session_state.pop("username", None)
    st.rerun()


# ── Sidebar ────────────────────────────────────────────────────────────────────

if check_login():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>🎫 Triage Agent</h2>
                <div class="subtitle">Support Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"Signed in as **{st.session_state.get('username', 'user')}**")

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📊 Overview", "📋 Tickets", "🔍 Trace Viewer"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if st.button("Sign Out", use_container_width=True):
            logout()
else:
    page = None


# ── Helpers ────────────────────────────────────────────────────────────────────


def api_get(endpoint: str) -> dict | None:
    """Make a GET request to the API with error handling."""
    try:
        resp = httpx.get(f"{API_BASE}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot connect to the API server. Please ensure it's running.")
        return None
    except httpx.TimeoutException:
        st.error("API request timed out. Please try again.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API returned error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_post(endpoint: str, json_data: dict | None = None) -> dict | None:
    """Make a POST request to the API with error handling."""
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", json=json_data, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot connect to the API server. Please ensure it's running.")
        return None
    except httpx.TimeoutException:
        st.error("API request timed out. The operation may have completed — refresh to check.")
        return None
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text[:200]
        st.error(f"API error ({e.response.status_code}): {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def status_badge(status: str) -> str:
    """Return HTML for a status badge."""
    status_map = {
        "resolved": ("✅", "badge-resolved"),
        "escalated": ("⚠️", "badge-escalated"),
        "processing": ("⏳", "badge-processing"),
        "pending": ("⏸", "badge-pending"),
        "failed": ("❌", "badge-failed"),
        "approved": ("✅", "badge-approved"),
        "awaiting_review": ("👁", "badge-escalated"),
        "rejected": ("🚫", "badge-failed"),
    }
    icon, css_class = status_map.get(status, ("•", "badge-pending"))
    return f'<span class="badge {css_class}">{icon} {status.upper()}</span>'


def urgency_badge(urgency: str) -> str:
    """Return HTML for an urgency badge."""
    urgency_map = {
        "critical": ("badge-urgent"),
        "urgent": ("badge-urgent"),
        "high": ("badge-high"),
        "medium": ("badge-medium"),
        "low": ("badge-low"),
    }
    css_class = urgency_map.get(urgency, "badge-pending")
    return f'<span class="badge {css_class}">{urgency.upper() if urgency else "N/A"}</span>'


def confidence_meter(value: float) -> str:
    """Return HTML for a confidence bar."""
    if value >= 0.8:
        css_class = "confidence-high"
    elif value >= 0.5:
        css_class = "confidence-medium"
    else:
        css_class = "confidence-low"
    pct = int(value * 100)
    return f"""
    <div style="display:flex;align-items:center;gap:8px;">
        <div class="confidence-bar-bg" style="flex:1;">
            <div class="confidence-bar-fill {css_class}" style="width:{pct}%;"></div>
        </div>
        <span style="font-size:0.8rem;font-weight:600;color:var(--text-700);min-width:36px;">{pct}%</span>
    </div>
    """


def section_header(icon: str, title: str) -> None:
    """Render a section header."""
    st.markdown(
        f'<div class="section-header"><span class="icon">{icon}</span><h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, heading: str, message: str) -> None:
    """Render an empty state."""
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


def format_timestamp(ts: str | None) -> str:
    """Format an ISO timestamp to a human-readable relative string."""
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
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
        return ts[:16] if len(ts) > 16 else ts


# ── Metrics Page ───────────────────────────────────────────────────────────────


def render_metrics():
    """Render the overview / metrics dashboard."""
    section_header("📊", "Dashboard Overview")

    data = api_get("/dashboard/metrics")
    if not data:
        empty_state(
            "📡",
            "No Data Available",
            "Could not fetch metrics. Make sure the API server is running on "
            f"`{API_BASE}`.",
        )
        return

    # ── KPI Row ────────────────────────────────────────────────────────────────
    total = data.get("total_tickets", 0)
    processed = data.get("processed_tickets", 0)
    escalated = data.get("escalated_tickets", 0)
    avg_conf = data.get("avg_confidence")
    avg_steps = data.get("avg_steps", 0)
    success_rate = data.get("success_rate", 0)
    escalation_rate = data.get("escalation_rate", 0)

    conf_pct = f"{avg_conf:.0%}" if avg_conf else "N/A"

    # Row 1: Core counts
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets", total)
    c2.metric("Processed", processed)
    c3.metric("Escalated", escalated, delta=f"{escalated}" if escalated else None,
              delta_color="inverse" if escalated else "off")
    c4.metric("Avg Confidence", conf_pct)

    # Row 2: Rates
    c5, c6, c7 = st.columns(3)
    c5.metric("Success Rate", f"{success_rate:.0%}")
    c6.metric("Escalation Rate", f"{escalation_rate:.0%}")
    c7.metric("Avg Steps", f"{avg_steps:.1f}")

    st.markdown("---")

    # ── Charts Row ─────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        section_header("📁", "Category Distribution")
        cat_dist = data.get("category_distribution", {})
        if cat_dist:
            chart_colors = [
                "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
                "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe",
            ]
            fig = px.pie(
                values=list(cat_dist.values()),
                names=list(cat_dist.keys()),
                color_discrete_sequence=chart_colors,
                hole=0.45,
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont_size=12,
                textfont_color="white",
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("📁", "No Categories Yet", "Ticket categories will appear here once tickets are triaged.")

    with col_b:
        section_header("⚡", "Urgency Distribution")
        urg_dist = data.get("urgency_distribution", {})
        if urg_dist:
            urg_colors = {
                "critical": "#ef4444",
                "urgent": "#f97316",
                "high": "#f59e0b",
                "medium": "#eab308",
                "low": "#22c55e",
            }
            labels = list(urg_dist.keys())
            values = list(urg_dist.values())
            colors = [urg_colors.get(l, "#94a3b8") for l in labels]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color=colors,
                        marker_line_width=0,
                        text=values,
                        textposition="outside",
                        textfont_size=12,
                    )
                ]
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Count",
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("⚡", "No Urgency Data", "Urgency distribution will appear after triage runs.")

    # ── Recent Activity ────────────────────────────────────────────────────────
    recent = data.get("recent_activity", [])
    if recent:
        section_header("🕐", "Recent Activity")
        for activity in recent[:5]:
            tid = activity.get("ticket_id", "N/A")[:8]
            action = activity.get("action", "N/A")
            ts = format_timestamp(activity.get("timestamp"))
            st.markdown(
                f'<div class="ticket-row" style="cursor:default;">'
                f'<div class="ticket-header">'
                f'<span class="ticket-id">#{tid}...</span>'
                f'{status_badge(action)}'
                f'</div>'
                f'<span style="font-size:0.8rem;color:var(--text-400);">{ts}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Ticket Queue Page ──────────────────────────────────────────────────────────


def render_queue():
    """Render the ticket queue with escalation management and ticket creation."""
    section_header("📋", "Ticket Queue")

    # ── Tab Layout ─────────────────────────────────────────────────────────────
    tab_escalated, tab_create = st.tabs(["⚠️ Escalated Queue", "➕ Create Ticket"])

    # ── Escalated Tickets Tab ──────────────────────────────────────────────────
    with tab_escalated:
        escalated = api_get("/tickets/escalated/list")
        tickets = escalated.get("tickets", []) if escalated else []
        total = escalated.get("total", 0) if escalated else 0

        if tickets:
            # Summary bar
            st.markdown(
                f"""
                <div class="filter-bar">
                    <span style="font-weight:600;color:var(--text-700);">📊 {total} escalated ticket{"s" if total != 1 else ""} requiring attention</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for t in tickets:
                tid_short = t["ticket_id"][:8]
                category = t.get("category") or "N/A"
                urgency = t.get("urgency") or "N/A"
                reason = t.get("escalation_reason") or "No reason provided"
                created = format_timestamp(t.get("created_at"))

                with st.expander(
                    f"#{tid_short}...  ·  {category}  ·  {urgency.upper()}",
                    expanded=False,
                ):
                    st.markdown(
                        f"""
                        <div style="margin-bottom:12px;">
                            {status_badge("escalated")}
                            {urgency_badge(urgency)}
                        </div>
                        <div style="margin-bottom:12px;">
                            <div style="font-size:0.8rem;font-weight:600;color:var(--text-500);text-transform:uppercase;letter-spacing:0.03em;margin-bottom:4px;">Preview</div>
                            <div style="font-size:0.9rem;color:var(--text-700);line-height:1.5;background:var(--surface-50);padding:12px;border-radius:var(--radius-sm);border:1px solid var(--surface-200);">
                                {t.get('content_preview', 'No preview available.')}
                            </div>
                        </div>
                        <div style="margin-bottom:8px;">
                            <div style="font-size:0.8rem;font-weight:600;color:var(--text-500);text-transform:uppercase;letter-spacing:0.03em;margin-bottom:4px;">Escalation Reason</div>
                            <div style="font-size:0.85rem;color:var(--text-700);font-style:italic;">
                                "{reason}"
                            </div>
                        </div>
                        <div style="font-size:0.8rem;color:var(--text-400);">Created: {created}</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("---")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(
                            "✅ Approve & Send",
                            key=f"approve_{t['ticket_id']}",
                            use_container_width=True,
                            type="primary",
                        ):
                            with st.spinner("Approving..."):
                                result = api_post(
                                    f"/tickets/{t['ticket_id']}/approve",
                                    json_data={"reviewer": st.session_state.get("username", "user")},
                                )
                            if result:
                                st.success(f"Approved! Status: {result['status']}")
                                time.sleep(0.5)
                                st.rerun()
                    with col_b:
                        if st.button(
                            "🚫 Reject",
                            key=f"reject_{t['ticket_id']}",
                            use_container_width=True,
                        ):
                            with st.spinner("Rejecting..."):
                                result = api_post(
                                    f"/tickets/{t['ticket_id']}/escalate",
                                    json_data={
                                        "reason": "Rejected via dashboard",
                                        "reviewer": st.session_state.get("username", "user"),
                                    },
                                )
                            if result:
                                st.success(f"Rejected. Status: {result['status']}")
                                time.sleep(0.5)
                                st.rerun()
        else:
            empty_state(
                "🎉",
                "All Clear!",
                "No escalated tickets in the queue. All tickets have been handled.",
            )

    # ── Create Ticket Tab ──────────────────────────────────────────────────────
    with tab_create:
        st.markdown(
            '<div style="margin-bottom:16px;"><p style="color:var(--text-500);font-size:0.9rem;">Create a new support ticket to be triaged by the AI agent.</p></div>',
            unsafe_allow_html=True,
        )

        with st.form("create_ticket", clear_on_submit=True):
            content = st.text_area(
                "Ticket Content",
                placeholder="Subject: Unable to reset password\n\nBody: I've been trying to reset my password for the past hour but the reset link never arrives in my inbox. I've checked spam too. Please help.",
                height=160,
            )

            col1, col2 = st.columns(2)
            with col1:
                source = st.selectbox(
                    "Source",
                    ["api", "email", "github", "intercom"],
                    format_func=lambda x: {
                        "api": "🔌 API",
                        "email": "📧 Email",
                        "github": "🐙 GitHub",
                        "intercom": "💬 Intercom",
                    }.get(x, x),
                )
            with col2:
                source_id = st.text_input(
                    "Source ID",
                    placeholder="e.g. GH-123, INC-4567",
                    help="Optional external reference ID for deduplication.",
                )

            submitted = st.form_submit_button(
                "🚀 Create & Triage",
                use_container_width=True,
                type="primary",
            )

            if submitted:
                if not content.strip():
                    st.error("Please enter ticket content.")
                else:
                    payload = {"content": content.strip(), "source": source}
                    if source_id.strip():
                        payload["source_id"] = source_id.strip()

                    with st.spinner("Creating ticket..."):
                        result = api_post("/tickets/", json_data=payload)

                    if result:
                        st.success(f"Ticket created: `{result['ticket_id'][:12]}...`")

                        if result.get("is_duplicate"):
                            st.warning("⚠️ Duplicate detected — returned existing ticket.")

                        # Offer to trigger triage
                        if result.get("status") in ("pending", "open"):
                            st.markdown("---")
                            st.markdown("**Next step:** Start the AI triage pipeline?")
                            if st.button(
                                "🤖 Start Triage",
                                key=f"triage_{result['ticket_id']}",
                                type="primary",
                            ):
                                with st.spinner("Starting triage pipeline..."):
                                    triage_result = api_post(f"/tickets/{result['ticket_id']}/triage")
                                if triage_result:
                                    st.success(f"✅ {triage_result.get('message', 'Triage started!')}")
                                    st.info(f"Track progress in the 🔍 Trace Viewer with ticket ID: `{result['ticket_id'][:12]}...`")


# ── Trace Viewer Page ──────────────────────────────────────────────────────────


def render_trace():
    """Render the trace viewer with horizontal timeline and step details."""
    section_header("🔍", "Trace Viewer")

    # ── Input Row ──────────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticket_id = st.text_input(
            "Ticket ID",
            value=st.session_state.get("trace_ticket_id", ""),
            placeholder="Enter ticket UUID to view trace",
            label_visibility="collapsed",
        )
    with col_btn:
        load_trace = st.button("🔍 Load Trace", use_container_width=True, type="primary", key="load_trace_btn")

    if load_trace and ticket_id:
        st.session_state["trace_ticket_id"] = ticket_id

    ticket_id = st.session_state.get("trace_ticket_id", ticket_id)

    if not ticket_id:
        empty_state(
            "🔍",
            "Enter a Ticket ID",
            "Paste a ticket UUID above to view its triage pipeline trace.",
        )
        return

    # ── Fetch Status + Trace ───────────────────────────────────────────────────
    status_data = api_get(f"/tickets/{ticket_id}/status")
    trace_data = api_get(f"/tickets/{ticket_id}/trace")

    # ── Ticket Status Card ─────────────────────────────────────────────────────
    if status_data:
        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
        status = status_data.get("status", "N/A")
        category = status_data.get("category") or "N/A"
        urgency = status_data.get("urgency") or "N/A"
        confidence = status_data.get("confidence")

        status_html = f"""
        <div class="s-card-elevated">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                <div>
                    <span style="font-family:var(--font-mono);font-size:0.95rem;font-weight:600;color:var(--brand-500);">#{ticket_id[:12]}...</span>
                    <span style="margin-left:8px;">{status_badge(status)}</span>
                    <span style="margin-left:4px;">{urgency_badge(urgency)}</span>
                </div>
                <div style="display:flex;gap:16px;align-items:center;">
                    <span style="font-size:0.8rem;color:var(--text-500);">Category: <strong>{category}</strong></span>
                    <span style="font-size:0.8rem;color:var(--text-500);">Created: <strong>{format_timestamp(status_data.get('created_at'))}</strong></span>
                </div>
            </div>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)

        # Confidence meter
        if confidence is not None:
            st.markdown(
                f"""
                <div style="padding:4px 0 8px 0;">
                    <span style="font-size:0.8rem;font-weight:600;color:var(--text-500);margin-right:8px;">Confidence:</span>
                    {confidence_meter(confidence)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Trace Timeline ─────────────────────────────────────────────────────────
    if trace_data:
        steps = trace_data.get("steps", [])
        total_duration = trace_data.get("total_duration_ms")
        loop_count = trace_data.get("loop_count")

        if steps:
            section_header("🔗", "Pipeline Timeline")

            # Summary metrics
            if total_duration or loop_count:
                summary_parts = []
                if total_duration:
                    summary_parts.append(f"⏱ **Total Duration:** {total_duration}ms")
                if loop_count:
                    summary_parts.append(f"🔄 **Loop Iterations:** {loop_count}")
                st.markdown("  ·  ".join(summary_parts))

            # Horizontal timeline (HTML)
            timeline_html = '<div class="trace-timeline">'
            for i, step in enumerate(steps):
                status = step.get("status", "unknown")
                if status == "completed":
                    dot_class = "completed"
                    icon = "✓"
                elif status == "failed":
                    dot_class = "failed"
                    icon = "✗"
                elif status in ("running", "in_progress"):
                    dot_class = "running"
                    icon = "⟳"
                else:
                    dot_class = "pending"
                    icon = str(i + 1)

                step_name = step.get("step", f"Step {i + 1}")
                duration = step.get("duration_ms")
                duration_str = f"{duration}ms" if duration else ""

                timeline_html += f"""
                <div class="trace-step {status}">
                    <div class="trace-dot {dot_class}">{icon}</div>
                    <div class="trace-label">{step_name}</div>
                    <div class="trace-duration">{duration_str}</div>
                </div>
                """
            timeline_html += "</div>"
            st.markdown(timeline_html, unsafe_allow_html=True)

            st.markdown("---")

            # ── Step Details ───────────────────────────────────────────────────
            section_header("📄", "Step Details")

            for i, step in enumerate(steps):
                status = step.get("status", "unknown")
                step_name = step.get("step", f"Step {i + 1}")
                duration = step.get("duration_ms")
                error = step.get("error")
                data = step.get("data")

                icon = {"completed": "✅", "failed": "❌", "running": "🔄"}.get(status, "○")

                with st.expander(
                    f"{icon} Step {i + 1}: {step_name} — {status.upper()}"
                    + (f"  ({duration}ms)" if duration else ""),
                    expanded=(status == "failed"),
                ):
                    # Metadata
                    meta_parts = [f"**Status:** {status}"]
                    if duration:
                        meta_parts.append(f"**Duration:** {duration}ms")
                    if step.get("timestamp"):
                        meta_parts.append(f"**Time:** {step['timestamp']}")
                    st.markdown("  ·  ".join(meta_parts))

                    # Error
                    if error:
                        st.error(f"**Error:** {error}")

                    # Step data
                    if data:
                        st.markdown("**Step Data:**")
                        st.json(data)
        else:
            empty_state(
                "📋",
                "No Trace Steps",
                "This ticket has not been triaged yet, or no trace data was recorded.",
            )

        # Final status
        final_status = trace_data.get("final_status")
        if final_status:
            st.markdown(
                f"""
                <div class="s-card" style="text-align:center;padding:16px;">
                    <span style="font-size:0.85rem;color:var(--text-500);">Final Status: </span>
                    {status_badge(final_status)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif status_data:
        empty_state(
            "📋",
            "No Trace Available",
            "This ticket exists but has no trace data. It may not have been triaged yet.",
        )


# ── Render ─────────────────────────────────────────────────────────────────────

if not check_login():
    login_form()
elif page == "📊 Overview":
    render_metrics()
elif page == "📋 Tickets":
    render_queue()
elif page == "🔍 Trace Viewer":
    render_trace()
