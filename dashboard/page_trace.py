"""Standalone Trace Viewer page — enter a ticket id to see its full pipeline
timeline with download / copy affordances (Sub-Phase 2.3).
"""

from __future__ import annotations

import streamlit as st

from dashboard.api import api_get, format_timestamp
from dashboard.trace import render_trace_actions, render_trace_timeline, trace_final_status_card
from dashboard.widgets import empty_state, section_header, status_badge, urgency_badge


def render_trace_page() -> None:
    section_header("🔍", "Trace Viewer", "Inspect the full triage pipeline for any ticket")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticket_id = st.text_input(
            "Ticket ID",
            value=st.session_state.get("trace_ticket_id", ""),
            placeholder="Enter a ticket UUID to view its trace",
            key="trace_page_input",
        )
    with col_btn:
        submit = st.button("🔍 Load Trace", key="trace_load", type="primary", use_container_width=True)

    if submit and ticket_id.strip():
        st.session_state["trace_ticket_id"] = ticket_id.strip()
        st.rerun()

    ticket_id = st.session_state.get("trace_ticket_id", "").strip()
    if not ticket_id:
        empty_state(
            "🔍",
            "Enter a Ticket ID",
            "Paste a ticket UUID above to view its triage pipeline trace."
            "\n\nTip: open any ticket from the Queue, then copy its ID.",
        )
        return

    with st.spinner("Loading trace…"):
        status_data = api_get(f"/tickets/{ticket_id}/status")
        trace = api_get(f"/tickets/{ticket_id}/trace")

    if trace is None:
        is_404 = (st.session_state.get("api_error") or {}).get("status") == 404
        empty_state(
            "🕳️", "Ticket Not Found" if is_404 else "Trace Unavailable",
            "No ticket with that ID exists." if is_404 else "The API couldn't return a trace right now.",
            primary_label="↻ Retry", primary_key="trace_retry",
        )
        return

    if status_data:
        st.markdown(
            f"""<div class="s-card-elevated" style="margin-bottom:16px;">
                <div class="space-between">
                    <div class="flex-row">
                        <span class="mono" style="font-weight:700;color:var(--brand-500);">#{ticket_id[:16]}</span>
                        {status_badge(status_data.get('status'))}
                        {urgency_badge(status_data.get('urgency'))}
                    </div>
                    <span class="small muted">Category: <strong>{status_data.get('category') or '—'}</strong> · Created {format_timestamp(status_data.get('created_at'))}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
    render_trace_timeline(
        trace.get("steps") or [],
        trace.get("total_duration_ms"),
        trace.get("loop_count"),
    )
    render_trace_actions(trace, ticket_id)
    trace_final_status_card(trace)