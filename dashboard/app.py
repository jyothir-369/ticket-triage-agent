"""Streamlit dashboard — triage metrics, ticket queue, and trace viewer."""

import json
from pathlib import Path

import httpx
import pandas as pd
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


def api_post(endpoint: str) -> dict | None:
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Metrics Page ───────────────────────────────────────────────────────────────

def render_metrics():
    st.title("📊 Triage Metrics")

    data = api_get("/dashboard/triage-metrics")
    if not data:
        st.warning("Could not fetch metrics. Is the API running?")
        return

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets", data["total_tickets"])
    col2.metric("Triaged", data["triaged"])
    col3.metric("Escalated", data["escalated"])
    col4.metric(
        "Avg Confidence",
        f"{data['avg_confidence']:.1%}" if data["avg_confidence"] else "N/A",
    )

    col5, col6 = st.columns(2)
    col5.metric(
        "Avg Latency",
        f"{data['avg_latency_ms']:.0f}ms" if data["avg_latency_ms"] else "N/A",
    )
    escalation_rate = (
        data["escalated"] / data["total_tickets"] if data["total_tickets"] else 0
    )
    col6.metric("Escalation Rate", f"{escalation_rate:.1%}")

    st.divider()

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Category Breakdown")
        if data["category_breakdown"]:
            fig = px.pie(
                values=list(data["category_breakdown"].values()),
                names=list(data["category_breakdown"].keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Urgency Breakdown")
        if data["urgency_breakdown"]:
            urgency_order = ["P0", "P1", "P2", "P3"]
            labels = [u for u in urgency_order if u in data["urgency_breakdown"]]
            values = [data["urgency_breakdown"][u] for u in labels]
            colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
            fig = go.Figure(
                data=[go.Bar(x=labels, y=values, marker_color=colors[: len(labels)])]
            )
            fig.update_layout(xaxis_title="Urgency", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)


# ── Ticket Queue Page ──────────────────────────────────────────────────────────

def render_queue():
    st.title("📋 Ticket Queue")

    # Load sample tickets for display
    eval_path = Path("./data/eval_tickets.json")
    if eval_path.exists():
        tickets = json.loads(eval_path.read_text())
    else:
        tickets = []
        st.info("No tickets found. Run `make db-seed` to populate sample data.")
        return

    for t in tickets:
        with st.expander(f"#{t['id']} — {t['subject']}", expanded=False):
            st.write(f"**Body:** {t['body']}")
            st.write(f"**Expected Category:** {t['expected_category']}")
            st.write(f"**Expected Urgency:** {t['expected_urgency']}")

            if st.button(f"⚡ Triage Ticket #{t['id']}", key=f"triage_{t['id']}"):
                with st.spinner("Running triage..."):
                    result = api_post(f"/tickets/{t['id']}/triage")
                if result:
                    st.success(f"Decision: **{result['decision'].upper()}**")
                    st.json(result)


# ── Trace Viewer Page ──────────────────────────────────────────────────────────

def render_trace():
    st.title("🔍 Trace Viewer")

    ticket_id = st.number_input("Ticket ID", min_value=1, value=1, step=1)

    if st.button("Load Trace"):
        data = api_get(f"/tickets/{ticket_id}/trace")
        if data:
            st.subheader(f"Trace for Ticket #{data['ticket_id']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Status", data["status"])
            col2.metric("Steps", data["total_steps"])
            col3.metric("Latency", f"{data['latency_ms']}ms" if data["latency_ms"] else "N/A")

            st.subheader("Decision Steps")
            for step in data["steps"]:
                with st.container():
                    st.markdown(
                        f"**Step {step['order']}:** `{step['step_type']}` — {step['output_summary'] or '…'}"
                    )
                    if step.get("latency_ms"):
                        st.caption(f"⏱ {step['latency_ms']}ms")
        else:
            st.warning("No trace found for this ticket.")


# ── Render ─────────────────────────────────────────────────────────────────────

if page == "📊 Metrics":
    render_metrics()
elif page == "📋 Ticket Queue":
    render_queue()
elif page == "🔍 Trace Viewer":
    render_trace()
