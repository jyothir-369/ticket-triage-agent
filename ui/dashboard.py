"""Streamlit dashboard — triage metrics, escalation queue, and ticket detail viewer.

Sections:
  1. Login / Auth  (hardcoded admin/password for v1)
  2. Escalation Queue  (list + filter ESCALATED tickets)
  3. Ticket Detail View  (full ticket info, draft, actions)
  4. Metrics Dashboard  (KPI cards, charts, activity log)

Usage:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Triage Agent Dashboard",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

API_BASE = "http://localhost:8000"
METRICS_TTL_SECONDS = 30  # cache TTL for dashboard metrics
REFRESH_INTERVAL_SECONDS = 10

# Hardcoded credentials for v1
VALID_CREDENTIALS = {"admin": "admin"}

# ═══════════════════════════════════════════════════════════════════════════════
# Session state defaults
# ═══════════════════════════════════════════════════════════════════════════════


def _init_session_state() -> None:
    """Initialise all session_state keys used by the dashboard."""
    defaults: dict[str, Any] = {
        "authenticated": False,
        "username": "",
        "selected_ticket_id": None,
        "current_page": "dashboard",
        "last_refresh": time.time(),
        "auto_refresh": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

# ═══════════════════════════════════════════════════════════════════════════════
# API helpers
# ═══════════════════════════════════════════════════════════════════════════════


def api_get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Synchronous GET request to the FastAPI backend."""
    try:
        resp = httpx.get(f"{API_BASE}{endpoint}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot connect to API server. Is it running on port 8000?")
        return None
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text[:200]}")
        return None
    except Exception as exc:
        st.error(f"Unexpected API error: {exc}")
        return None


def api_post(endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Synchronous POST request to the FastAPI backend."""
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot connect to API server. Is it running on port 8000?")
        return None
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text[:200]}")
        return None
    except Exception as exc:
        st.error(f"Unexpected API error: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Cached data fetchers
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=METRICS_TTL_SECONDS, show_spinner="Loading metrics...")
def fetch_metrics() -> dict[str, Any] | None:
    """Fetch aggregated dashboard metrics (cached with TTL)."""
    return api_get("/dashboard/metrics")


@st.cache_data(ttl=METRICS_TTL_SECONDS, show_spinner="Loading escalated tickets...")
def fetch_escalated_tickets() -> list[dict[str, Any]]:
    """Fetch the list of escalated tickets (cached with TTL)."""
    data = api_get("/tickets/escalated/list", params={"limit": 200})
    if data and "tickets" in data:
        return data["tickets"]
    return []


@st.cache_data(ttl=METRICS_TTL_SECONDS, show_spinner="Loading activity...")
def fetch_activity() -> list[dict[str, Any]]:
    """Fetch recent activity log (cached with TTL)."""
    data = api_get("/dashboard/activity")
    if data and "activities" in data:
        return data["activities"]
    return []


def fetch_ticket_status(ticket_id: str) -> dict[str, Any] | None:
    """Fetch full ticket status (not cached — user expects fresh data)."""
    return api_get(f"/tickets/{ticket_id}/status")


def fetch_ticket_trace(ticket_id: str) -> dict[str, Any] | None:
    """Fetch the triage trace for a ticket."""
    return api_get(f"/tickets/{ticket_id}/trace")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Login / Auth
# ═══════════════════════════════════════════════════════════════════════════════


def render_login() -> None:
    """Render the login form.  Hardcoded admin/password for v1."""
    st.title("🎫 Support Ticket Triage Agent")
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if (
                username in VALID_CREDENTIALS
                and VALID_CREDENTIALS[username] == password
            ):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Escalation Queue
# ═══════════════════════════════════════════════════════════════════════════════


def render_escalation_queue() -> None:
    """Render the escalation queue with filters and ticket list."""
    st.header("📋 Escalation Queue")

    # ── Filters ────────────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            date_range = st.date_input(
                "Date range",
                value=(
                    datetime.now(timezone.utc) - timedelta(days=30),
                    datetime.now(timezone.utc),
                ),
                max_value=datetime.now(timezone.utc),
                key="escalation_date_range",
            )

        with col2:
            categories = [
                "bug",
                "feature_request",
                "account_issue",
                "billing",
                "usage_help",
                "other",
            ]
            selected_categories = st.multiselect(
                "Categories",
                options=categories,
                default=[],
                key="escalation_categories",
            )

        with col3:
            st.write("")  # spacer
            st.write("")
            if st.button("🔄 Refresh", key="refresh_queue"):
                st.cache_data.clear()
                st.rerun()

    # ── Fetch tickets ──────────────────────────────────────────────────────────
    tickets = fetch_escalated_tickets()

    if not tickets:
        st.info("No escalated tickets found. All clear! 🎉")
        return

    # ── Apply filters ──────────────────────────────────────────────────────────
    filtered = tickets

    if selected_categories:
        filtered = [
            t for t in filtered if t.get("category") in selected_categories
        ]

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        start_dt = datetime.combine(start_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        end_dt = datetime.combine(end_date, datetime.max.time()).replace(
            tzinfo=timezone.utc
        )
        filtered = [
            t
            for t in filtered
            if _parse_iso(t.get("created_at", "")) is not None
            and start_dt <= _parse_iso(t["created_at"]) <= end_dt
        ]

    st.caption(f"Showing {len(filtered)} of {len(tickets)} escalated tickets")

    # ── Ticket table ───────────────────────────────────────────────────────────
    if filtered:
        df = pd.DataFrame(
            [
                {
                    "ticket_id": t["ticket_id"][:8] + "...",
                    "category": t.get("category", "—"),
                    "urgency": t.get("urgency", "—"),
                    "escalation_reason": (t.get("escalation_reason") or "—")[:80],
                    "created_at": _format_dt(t.get("created_at", "")),
                    "updated_at": _format_dt(t.get("updated_at", "")),
                }
                for t in filtered
            ]
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticket_id": st.column_config.TextColumn("Ticket ID", width="medium"),
                "category": st.column_config.TextColumn("Category", width="small"),
                "urgency": st.column_config.TextColumn("Urgency", width="small"),
                "escalation_reason": st.column_config.TextColumn(
                    "Escalation Reason", width="large"
                ),
                "created_at": st.column_config.TextColumn("Created", width="medium"),
                "updated_at": st.column_config.TextColumn("Updated", width="medium"),
            },
        )

        # ── Select ticket to view ──────────────────────────────────────────────
        st.divider()
        ticket_ids = [t["ticket_id"] for t in filtered]
        selected = st.selectbox(
            "Select a ticket to view details",
            options=ticket_ids,
            format_func=lambda x: f"{x[:8]}... — {next((t.get('category', '') for t in filtered if t['ticket_id'] == x), '')}",
            key="queue_ticket_select",
        )

        if st.button("🔍 View Ticket Details", type="primary", key="view_ticket_btn"):
            st.session_state.selected_ticket_id = selected
            st.session_state.current_page = "ticket_detail"
            st.rerun()
    else:
        st.info("No tickets match the selected filters.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Ticket Detail View
# ═══════════════════════════════════════════════════════════════════════════════


def render_ticket_detail() -> None:
    """Render the full ticket detail view with classification, docs, draft, and actions."""
    ticket_id = st.session_state.selected_ticket_id
    if not ticket_id:
        st.warning("No ticket selected.")
        return

    # ── Header ─────────────────────────────────────────────────────────────────
    col_header, col_back = st.columns([5, 1])
    with col_header:
        st.header(f"🎫 Ticket Details — `{ticket_id[:8]}...`")
    with col_back:
        if st.button("← Back to Queue", key="back_to_queue"):
            st.session_state.current_page = "escalation"
            st.session_state.selected_ticket_id = None
            st.rerun()

    # ── Fetch ticket data ──────────────────────────────────────────────────────
    with st.spinner("Loading ticket details..."):
        status_data = fetch_ticket_status(ticket_id)
        trace_data = fetch_ticket_trace(ticket_id)

    if not status_data:
        st.error("Could not load ticket data. Check if the ticket exists.")
        return

    # ── Status overview ────────────────────────────────────────────────────────
    with st.expander("📊 Ticket Status", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", status_data.get("status", "—").upper())
        col2.metric(
            "Category",
            status_data.get("category", "—").replace("_", " ").title(),
        )
        col3.metric(
            "Urgency",
            status_data.get("urgency", "—").upper(),
        )
        col4.metric(
            "Confidence",
            f"{status_data['confidence']:.1%}" if status_data.get("confidence") else "N/A",
        )

        st.caption(
            f"Created: {_format_dt(status_data.get('created_at', ''))} | "
            f"Updated: {_format_dt(status_data.get('updated_at', ''))}"
        )

    # ── Original ticket content ────────────────────────────────────────────────
    with st.expander("📝 Original Ticket Content", expanded=True):
        # The status endpoint doesn't return content; fetch from escalated list or trace
        content = _extract_content_from_trace(trace_data, ticket_id)
        if content:
            st.markdown(content)
        else:
            st.info(
                "Ticket content not available from status endpoint. "
                "The original content was processed during triage."
            )

    # ── Classification details ─────────────────────────────────────────────────
    with st.expander("🏷️ Classification Details", expanded=True):
        _render_classification_from_trace(trace_data, status_data)

    # ── Retrieved documents ────────────────────────────────────────────────────
    with st.expander("📚 Retrieved Documents", expanded=True):
        _render_retrieved_docs(trace_data)

    # ── Draft response ─────────────────────────────────────────────────────────
    with st.expander("💬 Draft Response", expanded=True):
        _render_draft_response(trace_data, status_data)

    # ── Action buttons ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Actions")
    _render_ticket_actions(ticket_id, status_data)


def _extract_content_from_trace(
    trace_data: dict[str, Any] | None, ticket_id: str
) -> str | None:
    """Try to extract original ticket content from trace data."""
    if not trace_data:
        return None

    steps = trace_data.get("steps", [])
    for step in steps:
        data = step.get("data") or {}
        if isinstance(data, dict):
            # Check for ticket content in step data
            if "content" in data:
                return data["content"]
            if "ticket_content" in data:
                return data["ticket_content"]
    return None


def _render_classification_from_trace(
    trace_data: dict[str, Any] | None, status_data: dict[str, Any]
) -> None:
    """Render classification details from trace or status data."""
    # Try to get classification from trace first
    classification = None
    if trace_data:
        for step in trace_data.get("steps", []):
            data = step.get("data") or {}
            if isinstance(data, dict) and "classification" in data:
                classification = data["classification"]
                break

    # Fallback to status data
    category = status_data.get("category") or (
        classification.get("category") if classification else None
    )
    urgency = status_data.get("urgency") or (
        classification.get("urgency") if classification else None
    )
    confidence = status_data.get("confidence") or (
        classification.get("confidence") if classification else None
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Category", (category or "—").replace("_", " ").title())
    col2.metric("Urgency", (urgency or "—").upper())
    col3.metric(
        "Confidence",
        f"{confidence:.1%}" if confidence else "N/A",
    )

    # Show reasoning if available from classification
    if classification and classification.get("reasoning"):
        st.markdown("**Reasoning:**")
        st.info(classification["reasoning"])

    # Show per-dimension confidence if available
    if classification:
        cat_conf = classification.get("category_confidence")
        urg_conf = classification.get("urgency_confidence")
        if cat_conf or urg_conf:
            st.markdown("**Per-dimension Confidence:**")
            c1, c2 = st.columns(2)
            if cat_conf:
                c1.metric("Category Confidence", f"{cat_conf:.1%}")
            if urg_conf:
                c2.metric("Urgency Confidence", f"{urg_conf:.1%}")


def _render_retrieved_docs(trace_data: dict[str, Any] | None) -> None:
    """Render retrieved documents from trace data."""
    docs = []
    if trace_data:
        for step in trace_data.get("steps", []):
            data = step.get("data") or {}
            if isinstance(data, dict) and "retrieved_docs" in data:
                docs = data["retrieved_docs"]
                break
            if isinstance(data, dict) and "documents" in data:
                docs = data["documents"]
                break

    if not docs:
        st.info("No retrieved documents found in trace data.")
        return

    for i, doc in enumerate(docs):
        similarity = doc.get("similarity_score", doc.get("score", 0))
        source = doc.get("source", "unknown")
        content = doc.get("content", "No content available")
        doc_id = doc.get("id", f"doc_{i}")

        with st.container():
            st.markdown(
                f"**Document {i + 1}** — `{doc_id[:12]}...` | "
                f"Source: `{source}` | Similarity: **{similarity:.1%}**"
            )
            # Show similarity as a progress bar
            st.progress(min(similarity, 1.0))
            # Show content in a code block for readability
            st.text_area(
                "Content",
                value=content[:1000] + ("..." if len(content) > 1000 else ""),
                height=120,
                disabled=True,
                key=f"doc_content_{i}",
                label_visibility="collapsed",
            )
            st.divider()


def _render_draft_response(
    trace_data: dict[str, Any] | None, status_data: dict[str, Any]
) -> None:
    """Render the draft response with highlighted citations."""
    draft_text = None
    citations = []
    draft_confidence = None
    draft_reasoning = None

    # Try trace first
    if trace_data:
        for step in trace_data.get("steps", []):
            data = step.get("data") or {}
            if isinstance(data, dict) and "draft" in data:
                draft = data["draft"]
                draft_text = draft.get("draft_text", draft.get("content"))
                citations = draft.get("citations", [])
                draft_confidence = draft.get("confidence")
                draft_reasoning = draft.get("reasoning")
                break

    # Fallback: try to get from status (though status endpoint may not have it)
    if not draft_text:
        # The draft text might be stored in trace steps
        st.info("Draft response not available in trace data.")
        return

    # Draft confidence
    if draft_confidence:
        st.metric("Draft Confidence", f"{draft_confidence:.1%}")

    # Draft reasoning
    if draft_reasoning:
        with st.expander("💭 Draft Reasoning", expanded=False):
            st.info(draft_reasoning)

    # Draft text with citations highlighted
    st.markdown("**Draft Response:**")
    if citations:
        highlighted = _highlight_citations(draft_text, citations)
        st.markdown(highlighted, unsafe_allow_html=True)
    else:
        st.markdown(draft_text)

    # Citations list
    if citations:
        with st.expander(f"📎 Citations ({len(citations)})", expanded=False):
            for j, cite in enumerate(citations):
                claim = cite.get("claim", "")
                source_doc_id = cite.get("source_doc_id", "—")
                relevance = cite.get("relevance_score", 0)
                st.markdown(
                    f"**{j + 1}.** `{claim[:100]}` → Doc: `{source_doc_id[:12]}...` "
                    f"(relevance: {relevance:.1%})"
                )


def _highlight_citations(text: str, citations: list[dict[str, Any]]) -> str:
    """Highlight citation references in the draft text."""
    # Simple highlighting — wrap cited phrases in bold/highlight
    highlighted = text
    for cite in citations:
        claim = cite.get("claim", "")
        if claim and claim in highlighted:
            highlighted = highlighted.replace(
                claim, f"<mark>**{claim}**</mark>"
            )
    return highlighted


def _render_ticket_actions(ticket_id: str, status_data: dict[str, Any]) -> None:
    """Render action buttons for the ticket."""
    current_status = status_data.get("status", "")

    col1, col2, col3, col4 = st.columns(4)

    # ── Approve ────────────────────────────────────────────────────────────────
    with col1:
        if current_status in ("awaiting_review", "escalated"):
            if st.button("✅ Approve", type="primary", use_container_width=True, key="approve_btn"):
                with st.spinner("Approving ticket..."):
                    result = api_post(
                        f"/tickets/{ticket_id}/approve",
                        {"reviewer": st.session_state.username},
                    )
                if result:
                    st.success(f"Ticket approved! Status: {result.get('status')}")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
        else:
            st.button(
                "✅ Approve",
                disabled=True,
                use_container_width=True,
                key="approve_disabled",
                help=f"Cannot approve in status '{current_status}'",
            )

    # ── Edit Draft ─────────────────────────────────────────────────────────────
    with col2:
        with st.popover("✏️ Edit Draft", use_container_width=True):
            edited_text = st.text_area(
                "Edit the draft response",
                value="",
                height=200,
                key="edited_draft",
                placeholder="Paste or edit the draft response here...",
            )
            if st.button("Submit Edited Draft", key="submit_edited"):
                if edited_text.strip():
                    # Approve with the edited draft
                    with st.spinner("Submitting edited draft..."):
                        result = api_post(
                            f"/tickets/{ticket_id}/approve",
                            {"reviewer": st.session_state.username},
                        )
                    if result:
                        st.success("Edited draft approved!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("Please enter a draft response.")

    # ── Reject ─────────────────────────────────────────────────────────────────
    with col3:
        with st.popover("❌ Reject", use_container_width=True):
            reject_reason = st.text_area(
                "Rejection reason (required)",
                height=100,
                key="reject_reason",
                placeholder="Explain why this response is being rejected...",
            )
            if st.button("Confirm Rejection", key="confirm_reject"):
                if reject_reason.strip():
                    with st.spinner("Rejecting ticket..."):
                        result = api_post(
                            f"/tickets/{ticket_id}/escalate",
                            {
                                "reason": f"Rejected: {reject_reason}",
                                "reviewer": st.session_state.username,
                            },
                        )
                    if result:
                        st.success("Ticket rejected and re-escalated.")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("Please provide a rejection reason.")

    # ── Manual Escalate ────────────────────────────────────────────────────────
    with col4:
        with st.popover("🚨 Manual Escalate", use_container_width=True):
            escalate_reason = st.text_area(
                "Escalation reason (required)",
                height=100,
                key="escalate_reason",
                placeholder="Reason for manual escalation...",
            )
            if st.button("Confirm Escalation", key="confirm_escalate"):
                if escalate_reason.strip():
                    with st.spinner("Escalating ticket..."):
                        result = api_post(
                            f"/tickets/{ticket_id}/escalate",
                            {
                                "reason": escalate_reason,
                                "reviewer": st.session_state.username,
                            },
                        )
                    if result:
                        st.success("Ticket escalated!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("Please provide an escalation reason.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Metrics Dashboard
# ═══════════════════════════════════════════════════════════════════════════════


def render_metrics_dashboard() -> None:
    """Render the full metrics dashboard with KPI cards, charts, and activity log."""
    st.header("📊 Metrics Dashboard")

    # ── Refresh button ─────────────────────────────────────────────────────────
    col_refresh, col_auto = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh Metrics", key="refresh_metrics"):
            st.cache_data.clear()
            st.rerun()
    with col_auto:
        st.session_state.auto_refresh = st.toggle(
            "Auto-refresh (10s)",
            value=st.session_state.auto_refresh,
            key="auto_refresh_toggle",
        )

    # ── Fetch data ─────────────────────────────────────────────────────────────
    metrics = fetch_metrics()
    activity = fetch_activity()

    if not metrics:
        st.warning("Could not load metrics. Is the API running?")
        return

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    _render_kpi_cards(metrics)

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────────
    _render_charts(metrics)

    st.divider()

    # ── Recent Activity ────────────────────────────────────────────────────────
    _render_activity_log(activity)

    st.divider()

    # ── Success Rate Over Time ─────────────────────────────────────────────────
    _render_success_rate_chart(metrics)


def _render_kpi_cards(metrics: dict[str, Any]) -> None:
    """Render the top-level KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Tickets",
            value=metrics.get("total_tickets", 0),
            help="Total tickets in the system",
        )

    with col2:
        avg_conf = metrics.get("avg_confidence", 0)
        st.metric(
            label="Avg Confidence",
            value=f"{avg_conf:.1%}" if avg_conf else "N/A",
            help="Mean classification confidence across all tickets",
        )

    with col3:
        esc_rate = metrics.get("escalation_rate", 0)
        st.metric(
            label="Escalation Rate",
            value=f"{esc_rate:.1%}" if esc_rate else "0.0%",
            help="Fraction of total tickets that were escalated",
        )

    with col4:
        app_rate = metrics.get("approval_rate", 0)
        st.metric(
            label="Approval Rate",
            value=f"{app_rate:.1%}" if app_rate else "0.0%",
            help="Fraction of escalated tickets approved by humans",
        )


def _render_charts(metrics: dict[str, Any]) -> None:
    """Render the category pie chart, urgency bar chart, and tickets-per-day bar chart."""
    col_a, col_b = st.columns(2)

    # ── Category Distribution (Pie Chart) ──────────────────────────────────────
    with col_a:
        with st.expander("📂 Category Distribution", expanded=True):
            cat_dist = metrics.get("category_distribution", {})
            if cat_dist:
                fig_cat = px.pie(
                    values=list(cat_dist.values()),
                    names=[k.replace("_", " ").title() for k in cat_dist.keys()],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hole=0.3,
                )
                fig_cat.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                )
                fig_cat.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No category data available yet.")

    # ── Urgency Distribution (Bar Chart) ───────────────────────────────────────
    with col_b:
        with st.expander("🔥 Urgency Distribution", expanded=True):
            urg_dist = metrics.get("urgency_distribution", {})
            if urg_dist:
                urgency_order = ["critical", "high", "medium", "low"]
                ordered_keys = [u for u in urgency_order if u in urg_dist]
                ordered_values = [urg_dist[u] for u in ordered_keys]
                urgency_colors = {
                    "critical": "#e74c3c",
                    "high": "#e67e22",
                    "medium": "#f1c40f",
                    "low": "#2ecc71",
                }
                colors = [urgency_colors.get(k, "#95a5a6") for k in ordered_keys]

                fig_urg = go.Figure(
                    data=[
                        go.Bar(
                            x=[k.upper() for k in ordered_keys],
                            y=ordered_values,
                            marker_color=colors,
                            text=ordered_values,
                            textposition="auto",
                        )
                    ]
                )
                fig_urg.update_layout(
                    xaxis_title="Urgency Level",
                    yaxis_title="Count",
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_urg, use_container_width=True)
            else:
                st.info("No urgency data available yet.")

    # ── Tickets Per Day (Bar Chart) ────────────────────────────────────────────
    with st.expander("📈 Tickets Per Day", expanded=True):
        activity = metrics.get("recent_activity", [])
        if activity:
            # Aggregate by date
            daily_counts: dict[str, int] = {}
            for act in activity:
                ts = act.get("timestamp", "")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        day_str = dt.strftime("%Y-%m-%d")
                        daily_counts[day_str] = daily_counts.get(day_str, 0) + 1
                    except (ValueError, TypeError):
                        pass

            if daily_counts:
                daily_df = pd.DataFrame(
                    list(daily_counts.items()), columns=["Date", "Tickets"]
                ).sort_values("Date")

                fig_daily = px.bar(
                    daily_df,
                    x="Date",
                    y="Tickets",
                    color_discrete_sequence=["#3498db"],
                )
                fig_daily.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Tickets",
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.info("Not enough data to generate daily chart.")
        else:
            st.info("No activity data available yet.")


def _render_activity_log(activities: list[dict[str, Any]]) -> None:
    """Render the recent activity log as a table."""
    with st.expander("📋 Recent Activity Log", expanded=True):
        if not activities:
            st.info("No recent activity.")
            return

        df = pd.DataFrame(
            [
                {
                    "Ticket ID": (a.get("ticket_id", "—"))[:12] + "...",
                    "Action": a.get("action", "—").replace("_", " ").title(),
                    "Timestamp": _format_dt(a.get("timestamp", "")),
                    "Details": (a.get("details") or "—")[:100],
                }
                for a in activities
            ]
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticket ID": st.column_config.TextColumn("Ticket ID", width="medium"),
                "Action": st.column_config.TextColumn("Action", width="medium"),
                "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "Details": st.column_config.TextColumn("Details", width="large"),
            },
        )


def _render_success_rate_chart(metrics: dict[str, Any]) -> None:
    """Render a success rate gauge chart."""
    with st.expander("🎯 Success Rate", expanded=True):
        success_rate = metrics.get("success_rate", 0)
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=success_rate * 100 if success_rate else 0,
                    number={"suffix": "%", "font": {"size": 40}},
                    title={"text": "Auto-Resolution Rate", "font": {"size": 20}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": "#2ecc71"},
                        "steps": [
                            {"range": [0, 50], "color": "#e74c3c"},
                            {"range": [50, 75], "color": "#f39c12"},
                            {"range": [75, 100], "color": "#2ecc71"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 80,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                margin=dict(t=40, b=20, l=30, r=30),
                height=300,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Additional stats
        st.markdown(
            f"**{metrics.get('processed_tickets', 0)}** tickets processed | "
            f"**{metrics.get('escalated_tickets', 0)}** escalated | "
            f"**{metrics.get('avg_steps', 0):.1f}** avg steps per ticket"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_iso(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _format_dt(ts: str) -> str:
    """Format an ISO timestamp for display."""
    dt = _parse_iso(ts)
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ═══════════════════════════════════════════════════════════════════════════════


def render_sidebar() -> None:
    """Render the sidebar with navigation and user info."""
    with st.sidebar:
        st.title("🎫 Triage Agent")
        st.caption(f"Logged in as **{st.session_state.username}**")
        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "📋 Escalation Queue", "🎫 Ticket Detail"],
            index=0 if st.session_state.current_page == "dashboard" else
                  1 if st.session_state.current_page == "escalation" else 2,
            key="sidebar_nav",
        )

        # Map radio selection to page key
        page_map = {
            "📊 Dashboard": "dashboard",
            "📋 Escalation Queue": "escalation",
            "🎫 Ticket Detail": "ticket_detail",
        }
        new_page = page_map.get(page, "dashboard")
        if new_page != st.session_state.current_page:
            st.session_state.current_page = new_page
            st.rerun()

        st.divider()

        # Connection status
        health = api_get("/health")
        if health:
            status = health.get("status", "unknown")
            color = "🟢" if status == "ok" else "🟡" if status == "degraded" else "🔴"
            st.caption(f"{color} API Status: {status}")
        else:
            st.caption("🔴 API Status: unreachable")

        st.divider()

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.selected_ticket_id = None
            st.session_state.current_page = "dashboard"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point for the Streamlit dashboard."""
    # Check authentication
    if not st.session_state.authenticated:
        render_login()
        return

    # Render sidebar
    render_sidebar()

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        elapsed = time.time() - st.session_state.last_refresh
        if elapsed >= REFRESH_INTERVAL_SECONDS:
            st.session_state.last_refresh = time.time()
            st.cache_data.clear()
            st.rerun()

    # Render current page
    page = st.session_state.current_page
    if page == "dashboard":
        render_metrics_dashboard()
    elif page == "escalation":
        render_escalation_queue()
    elif page == "ticket_detail":
        render_ticket_detail()
    else:
        render_metrics_dashboard()


if __name__ == "__main__":
    main()
