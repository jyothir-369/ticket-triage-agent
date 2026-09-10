"""Streamlit dashboard — triage metrics, ticket queue, and trace viewer."""

import json
from pathlib import Path

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Triage Agent Dashboard",
    page_icon="🎫",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🎫 Triage Agent")
page = st.sidebar.radio("Navigate", ["📊 Metrics", "📋 Ticket Queue", "🔍 Trace Viewer"])

# ── Helpers ────────────────────────────────────────────────────────────────────

def api_get(endpoint: str) -> dict | None:
    try:
        resp = httpx.get(f"{API_BASE}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, json_data: dict | None = None) -> dict | None:
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", json=json_data, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Metrics Page ───────────────────────────────────────────────────────────────

def render_metrics():
    st.title("📊 Triage Metrics")

    data = api_get("/dashboard/metrics")
    if not data:
        st.warning("Could not fetch metrics. Is the API running?")
        return

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets", data.get("total_tickets", 0))
    col2.metric("Processed", data.get("processed_tickets", 0))
    col3.metric("Escalated", data.get("escalated_tickets", 0))
    col4.metric(
        "Avg Confidence",
        f"{data['avg_confidence']:.1%}" if data.get("avg_confidence") else "N/A",
    )

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Steps", f"{data.get('avg_steps', 0):.1f}")
    col6.metric(
        "Success Rate",
        f"{data.get('success_rate', 0):.1%}",
    )
    col7.metric(
        "Escalation Rate",
        f"{data.get('escalation_rate', 0):.1%}",
    )

    st.divider()

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Category Distribution")
        cat_dist = data.get("category_distribution", {})
        if cat_dist:
            fig = px.pie(
                values=list(cat_dist.values()),
                names=list(cat_dist.keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data yet.")

    with col_b:
        st.subheader("Urgency Distribution")
        urg_dist = data.get("urgency_distribution", {})
        if urg_dist:
            labels = list(urg_dist.keys())
            values = list(urg_dist.values())
            colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"][: len(labels)]
            fig = go.Figure(
                data=[go.Bar(x=labels, y=values, marker_color=colors)]
            )
            fig.update_layout(xaxis_title="Urgency", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No urgency data yet.")

    # Recent activity
    recent = data.get("recent_activity", [])
    if recent:
        st.subheader("Recent Activity")
        for activity in recent:
            st.markdown(
                f"- **{activity.get('ticket_id', 'N/A')}** — "
                f"{activity.get('action', 'N/A')} "
                f"({activity.get('timestamp', '')})"
            )


# ── Ticket Queue Page ──────────────────────────────────────────────────────────

def render_queue():
    st.title("📋 Ticket Queue")

    # Fetch escalated tickets from the API
    escalated = api_get("/tickets/escalated/list")
    if escalated and escalated.get("tickets"):
        st.subheader(f"Escalated Tickets ({escalated.get('total', 0)})")
        for t in escalated["tickets"]:
            with st.expander(
                f"#{t['ticket_id'][:8]}... — {t.get('category', 'N/A')} / {t.get('urgency', 'N/A')}",
                expanded=False,
            ):
                st.write(f"**Preview:** {t.get('content_preview', '')}")
                st.write(f"**Category:** {t.get('category', 'N/A')}")
                st.write(f"**Urgency:** {t.get('urgency', 'N/A')}")
                st.write(f"**Escalation Reason:** {t.get('escalation_reason', 'N/A')}")
                st.write(f"**Created:** {t.get('created_at', 'N/A')}")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Approve", key=f"approve_{t['ticket_id']}"):
                        result = api_post(
                            f"/tickets/{t['ticket_id']}/approve",
                            json_data={"reviewer": "streamlit-user"},
                        )
                        if result:
                            st.success(f"Status: {result['status']}")
                with col_b:
                    if st.button("🚫 Reject", key=f"reject_{t['ticket_id']}"):
                        result = api_post(
                            f"/tickets/{t['ticket_id']}/escalate",
                            json_data={
                                "reason": "Rejected via dashboard",
                                "reviewer": "streamlit-user",
                            },
                        )
                        if result:
                            st.success(f"Status: {result['status']}")
    else:
        st.info("No escalated tickets. All clear! 🎉")

    # Create new ticket
    st.divider()
    st.subheader("Create New Ticket")
    with st.form("create_ticket"):
        content = st.text_area("Ticket Content", placeholder="Subject: ...\n\nBody:\n...")
        source = st.selectbox("Source", ["api", "email", "github", "intercom"])
        source_id = st.text_input("Source ID (optional)", placeholder="e.g. GH-123")
        submitted = st.form_submit_button("Create Ticket")
        if submitted and content:
            payload = {"content": content, "source": source}
            if source_id:
                payload["source_id"] = source_id
            result = api_post("/tickets/", json_data=payload)
            if result:
                st.success(f"Ticket created: {result['ticket_id']}")
                if result.get("is_duplicate"):
                    st.warning("Duplicate detected — returned existing ticket.")


# ── Trace Viewer Page ──────────────────────────────────────────────────────────

def render_trace():
    st.title("🔍 Trace Viewer")

    ticket_id = st.text_input("Ticket ID", value="", placeholder="Enter ticket UUID")

    if st.button("Load Trace") and ticket_id:
        data = api_get(f"/tickets/{ticket_id}/trace")
        if data:
            st.subheader(f"Trace for Ticket {data['ticket_id']}")
            st.metric("Steps", len(data.get("steps", [])))

            st.subheader("Decision Steps")
            for i, step in enumerate(data.get("steps", []), 1):
                with st.container():
                    status_icon = "✅" if step.get("status") == "completed" else "❌" if step.get("status") == "failed" else "🔄"
                    st.markdown(
                        f"**Step {i}:** {status_icon} `{step.get('step', 'N/A')}` — "
                        f"Status: `{step.get('status', 'N/A')}`"
                    )
                    if step.get("duration_ms"):
                        st.caption(f"⏱ {step['duration_ms']}ms")
                    if step.get("data"):
                        with st.expander("View Data"):
                            st.json(step["data"])
                    if step.get("error"):
                        st.error(f"Error: {step['error']}")
        else:
            st.warning("No trace found for this ticket.")

    # Also show status
    if ticket_id:
        status_data = api_get(f"/tickets/{ticket_id}/status")
        if status_data:
            st.divider()
            st.subheader("Ticket Status")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Status", status_data.get("status", "N/A"))
            col2.metric("Category", status_data.get("category", "N/A"))
            col3.metric("Urgency", status_data.get("urgency", "N/A"))
            col4.metric(
                "Confidence",
                f"{status_data['confidence']:.1%}" if status_data.get("confidence") else "N/A",
            )


# ── Render ─────────────────────────────────────────────────────────────────────

if page == "📊 Metrics":
    render_metrics()
elif page == "📋 Ticket Queue":
    render_queue()
elif page == "🔍 Trace Viewer":
    render_trace()
