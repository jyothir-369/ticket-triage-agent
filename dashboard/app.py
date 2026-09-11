"""Streamlit dashboard — entry point for the triage SaaS console.

Routes to the metrics / queue / trace pages, handles auth, global chrome
(sidebar branding, health dot, dark mode), the shared top bar with
breadcrumbs, and the network-failure banner.

Note: Streamlit executes this file as a bare script (``__main__``), not as a
package module, so we must place the project root on ``sys.path`` before any
``from dashboard.* import …`` so absolute imports resolve regardless of the
launch working directory.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from dashboard.api import API_BASE, DEFAULT_PASSWORD, DEFAULT_USERNAME, check_backend_health
from dashboard.page_metrics import render_metrics_page
from dashboard.page_queue import render_create_ticket, render_queue_page
from dashboard.page_trace import render_trace_page
from dashboard.theme import CUSTOM_CSS, apply_theme_marker
from dashboard.widgets import network_banner

st.set_page_config(
    page_title="Triage Agent — Support Console",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Seed session state
# ---------------------------------------------------------------------------

for _k, _v in {
    "authenticated": False,
    "username": DEFAULT_USERNAME,
    "page": "metrics",
    "selected_ticket_id": None,
    "dark_mode": False,
    "api_error": None,
    "last_refreshed": None,
    "show_create": False,
    "bulk_selected": [],
    "q_page": 1,
    "q_page_size": 20,
    "q_view": "Cards",
    "queue_cache": {},
    "trace_focus_idx": 0,
    "metrics_range": "Last 30 days",
}.items():
    st.session_state.setdefault(_k, _v)

apply_theme_marker()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Health indicator (auto-refresh every 30s)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=20, show_spinner=False)
def cached_health() -> dict | None:
    """Health payload cached for 20s (backend /health probes the LLM, which is slow)."""
    try:
        return check_backend_health()
    except Exception:  # noqa: BLE001 — never break the sidebar chrome
        return None


@st.fragment(run_every=30)
def health_fragment() -> None:
    health = cached_health()

    if health and health.get("status") == "healthy":
        cls, label = "healthy", "API healthy"
    elif health and health.get("status") == "degraded":
        cls, label = "degraded", "API degraded"
    else:
        cls, label = "down", "API offline"
    st.markdown(
        f'''<div class="health-row" title="Backend connectivity, refreshed every 30s;">
            <span class="health-dot {cls}"></span><span>{label}</span></div>''',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def login_form() -> None:
    st.markdown('<div class="login-page">', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        st.markdown(
            """<div class="login-card">
                <div class="brand-icon">🎫</div>
                <h1>Triage Agent</h1>
                <p class="login-subtitle">Automated support-ticket triage — classify, retrieve, draft, resolve.</p>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            c1, c2 = st.columns([1, 1])
            with c1:
                remember = st.checkbox("Remember me", value=False,
                                       help="Keeps you signed in for this console session.")
            with c2:
                st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

        if submitted:
            errors = []
            if not username.strip():
                errors.append("Username is required.")
            if not password:
                errors.append("Password is required.")
            if errors:
                st.markdown(
                    '<div class="field-error">' + "<br>".join(errors) + "</div>",
                    unsafe_allow_html=True,
                )
            elif username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["remember_me"] = remember
                st.toast(f"Welcome back, {username}", icon="👋")
                st.rerun()
            else:
                st.markdown(
                    '<div class="field-error">Invalid credentials. Check your username and password.</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Default credentials for this demo: `admin` / `admin`")


def logout() -> None:
    st.session_state["authenticated"] = False
    st.session_state.pop("username", None)
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _navigate(page: str) -> None:
    st.session_state["page"] = page
    if page != "queue":
        st.session_state["selected_ticket_id"] = None
    st.session_state.pop("show_create", None)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """<div class="sidebar-brand">
                <div class="brand-mark">🎫</div>
                <div class="brand-word">
                    <h3>Triage Agent</h3>
                    <div class="subtitle">SUPPORT CONSOLE</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        username = st.session_state.get("username", "user")
        role = "Support Lead" if username == DEFAULT_USERNAME else "Operator"
        initial = (username[:2] or "U").upper()
        st.markdown(
            f"""<div class="user-chip">
                <div class="user-avatar">{initial}</div>
                <div class="user-info">
                    <div class="name">{username}</div>
                    <div class="role">{role}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        health_fragment()

        # Shared key = the page itself, so breadcrumbs / actions that set
        # session_state["page"] are reflected here and vice versa.
        st.radio(
            "Navigation",
            ["metrics", "queue", "trace"],
            format_func=lambda k: {
                "metrics": "📊 Overview",
                "queue": "📋 Ticket Queue",
                "trace": "🔍 Trace Viewer",
            }.get(k, k),
            key="page",
            label_visibility="collapsed",
        )

        st.markdown("---")

        now_dark = st.session_state.get("dark_mode", False)
        st.toggle("🌙 Dark mode", value=now_dark, key="dark_mode",
                  help="Switch the console between light and dark themes.")

        st.caption(f"API: {API_BASE}")

        if st.button("Sign out", key="sidebar_logout", use_container_width=True):
            logout()


# ---------------------------------------------------------------------------
# Top bar with breadcrumbs
# ---------------------------------------------------------------------------


def top_bar() -> None:
    page = st.session_state.get("page", "metrics")
    selected = st.session_state.get("selected_ticket_id")

    with st.container(border=True):
        b1, b2, b3, b4 = st.columns([0.45, 0.65, 3.4, 2.6], vertical_alignment="center")
        with b1:
            home = st.button("🏠 Home", key="crumb_home", use_container_width=True)
        with b2:
            queue_crumb = st.button("📋 Queue", key="crumb_queue", use_container_width=True)

        if selected:
            b3_extra = (
                '<span class="crumb-sep">/</span>'
                f'<span class="crumb-current">Ticket #{selected[:12]}…</span>'
            )
            b4_extra = f'<span class="context">Opening ticket <span class="mono">{selected[:16]}…</span></span>'
        elif page == "metrics":
            b3_extra = '<span class="crumb-sep">/</span><span class="crumb-current">Overview</span>'
            b4_extra = '<span class="context">Live metrics · grouped by the selected range</span>'
        elif page == "trace":
            b3_extra = '<span class="crumb-sep">/</span><span class="crumb-current">Trace Viewer</span>'
            b4_extra = '<span class="context">Pipeline traces</span>'
        else:
            b3_extra = '<span class="crumb-sep">/</span><span class="crumb-current">Queue</span>'
            b4_extra = '<span class="context">Triage queue</span>'

        with b3:
            st.markdown(
                f'<div class="crumbs">Home{b3_extra}</div>',
                unsafe_allow_html=True,
            )
        with b4:
            st.markdown(b4_extra, unsafe_allow_html=True)

    if home:
        _navigate("metrics")
        st.rerun()
    if queue_crumb:
        _navigate("queue")
        st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def main() -> None:
    # Global network banner (hidden on the login screen)
    if st.session_state.get("authenticated") and network_banner():
        st.rerun()

    if not st.session_state.get("authenticated"):
        login_form()
        return

    render_sidebar()
    top_bar()

    page = st.session_state.get("page", "metrics")
    if page == "metrics":
        render_metrics_page()
    elif page == "queue":
        render_queue_page()
        if st.session_state.get("show_create"):
            render_create_ticket()
            st.session_state["show_create"] = False
    elif page == "trace":
        render_trace_page()
    else:
        render_metrics_page()


main()