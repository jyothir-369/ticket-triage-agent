"""Ticket detail view (Sub-Phase 2.2): a dedicated page with header row,
six tabs, a prominent confidence meter, and a sticky bottom actions bar.

Opened from the queue — the open ticket id lives in
``st.session_state["selected_ticket_id"]``.
"""

from __future__ import annotations

import json

import streamlit as st

from dashboard.api import api_get, fetch_many, format_timestamp, safe_json
from dashboard.theme import is_dark
from dashboard.trace import (
    render_trace_actions,
    render_trace_timeline,
    trace_final_status_card,
)
from dashboard.widgets import (
    approve_ticket,
    category_badge,
    confidence_label,
    confidence_meter,
    copy_button,
    empty_state,
    escalate_ticket,
    partial_banner,
    placeholder,
    rerun_triage,
    section_header,
    show_more_text,
    status_badge,
    update_draft,
    urgency_badge,
)


def _invalidate_detail() -> None:
    """Drop the detail cache whenever a mutation (approve/reject/edit) happens."""
    detail_bundle.clear()
    st.session_state["queue_cache"] = {}


@st.cache_data(ttl=20, show_spinner=False)
def detail_bundle(ticket_id: str) -> dict:
    """Full ticket + trace, fetched concurrently and cached for 20s."""
    return fetch_many(
        [
            ("detail", f"/tickets/{ticket_id}", None),
            ("trace", f"/tickets/{ticket_id}/trace", None),
        ],
        timeout=25.0,
    )


def _back_to_queue() -> None:
    st.session_state["selected_ticket_id"] = None
    st.session_state.pop("detail_draft_edit", None)
    st.rerun()


# ---------------------------------------------------------------------------
# Trace JSON parsers
# ---------------------------------------------------------------------------


def _trace_payload(detail: dict) -> dict:
    return safe_json(detail.get("trace")) or {}


def _classification_from_trace(detail: dict) -> dict | None:
    """Pull the classification record from trace JSON or the classify step."""
    tp = _trace_payload(detail)
    if tp.get("classification"):
        return tp["classification"]
    return None


def _retrieved_docs_from_trace(detail: dict) -> list[dict]:
    return _trace_payload(detail).get("retrieved_docs") or []


def _citation_origin_url(doc: dict) -> str | None:
    md = doc.get("metadata") or {}
    for k in ("url", "link", "source_url"):
        if md.get(k):
            return str(md[k])
    return None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def render_ticket_detail(ticket_id: str) -> None:
    top_col, back_col = st.columns([5.4, 0.6])
    with top_col:
        copy_button("Copy ticket ID", ticket_id, f"copytid_{ticket_id[:8]}", icon="🔗")
    with back_col:
        if st.button("← Back to Queue", key="detail_back", use_container_width=True):
            _back_to_queue()

    with st.spinner("Loading ticket…"):
        bundle = detail_bundle(ticket_id)
    detail = bundle.get("detail")
    trace = bundle.get("trace")

    if detail is None:
        err = st.session_state.get("api_error") or {}
        clicked = empty_state(
            "🕳️", "Ticket Not Found",
            "This ticket doesn't exist or has been removed.",
            primary_label="← Back to Queue", primary_key="detail_missing_back",
        )
        if clicked:
            _back_to_queue()
        return

    # ── Header row ──────────────────────────────────────────────────────────
    st.markdown(
        f"""<div class="s-card-elevated">
            <div class="space-between">
                <div class="flex-row">
                    <span class="mono" style="font-size:1rem;font-weight:700;color:var(--brand-500);">#{detail['ticket_id'][:20]}</span>
                    {status_badge(detail.get('status'))}
                    {category_badge(detail.get('category'))}
                    {urgency_badge(detail.get('urgency'))}
                </div>
                <span class="small muted">Created {format_timestamp(detail.get('created_at'), full=True)}</span>
            </div>
            <div style="margin-top:14px;">
                <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-500);margin-bottom:6px;">Classification confidence</div>
                {confidence_meter(detail.get('confidence'), large=True)}
                <div class="small muted" style="margin-top:4px;">{confidence_label(detail.get('confidence'))} · meets threshold at 70%</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_overview, tab_class, tab_retr, tab_draft, tab_trace, tab_actions = st.tabs(
        ["📄 Overview", "🏷️ Classification", "📚 Retrieval", "✍️ Draft", "🔗 Trace", "⚙️ Actions"]
    )

    with tab_overview:
        _render_overview(detail)
    with tab_class:
        _render_classification(detail)
    with tab_retr:
        _render_retrieval(detail)
    with tab_draft:
        _render_draft(detail, ticket_id)
    with tab_trace:
        _render_trace(detail, trace, ticket_id)
    with tab_actions:
        _render_actions(detail, ticket_id)

    _render_sticky_actions(detail, ticket_id)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _render_overview(detail: dict) -> None:
    section_header("📄", "Original Ticket")
    show_more_text(detail.get("content"), 500, f"ov_content_{detail['ticket_id'][:8]}")
    st.markdown("---")
    section_header("ℹ️", "Metadata")
    md = safe_json(detail.get("metadata")) or {}
    meta_items = [
        ("Source", detail.get("source") or "api"),
        ("Source ID", detail.get("source_id")),
        ("Customer email", md.get("customer_email") or (detail.get("email"))),
        ("Status", detail.get("status")),
        ("Created", format_timestamp(detail.get("created_at"), full=True)),
        ("Updated", format_timestamp(detail.get("updated_at"), full=True)),
        ("Processed", format_timestamp(detail.get("processed_at"), full=True)),
        ("Reviewed", format_timestamp(detail.get("reviewed_at"), full=True)),
        ("Review decision", detail.get("review_decision")),
        ("Escalation reason", detail.get("escalation_reason")),
    ]
    for label, val in meta_items:
        if val in (None, "", "N/A"):
            val_html = '<span class="placeholder-text">Not yet generated</span>'
        else:
            val_html = f"<strong>{str(val)}</strong>"
        st.markdown(
            f"""<div style="display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid var(--surface-100);">
            <span class="small muted" style="min-width:130px;">{label}</span><span style="text-align:right;">{val_html}</span></div>""",
            unsafe_allow_html=True,
        )


def _render_classification(detail: dict) -> None:
    section_header("🏷️", "Classification")
    cls = _classification_from_trace(detail)

    if not cls:
        placeholder("No classification has been generated yet.")
        if detail.get("category") or detail.get("confidence") is not None:
            partial_banner("Partial classification", "Stored at the lifecycle level, but no reasoning trail was captured.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class="s-card">
                <div class="label" style="font-size:0.72rem;font-weight:700;text-transform:uppercase;color:var(--text-500);">Category</div>
                <div style="font-size:1.2rem;font-weight:700;color:var(--text-900);margin-top:4px;">{category_badge(cls.get('category'))}</div>
                <div class="muted" style="font-size:0.8rem;margin-top:4px;">Category confidence</div>
                {confidence_meter(cls.get('category_confidence'))}
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="s-card">
                <div class="label" style="font-size:0.72rem;font-weight:700;text-transform:uppercase;color:var(--text-500);">Urgency</div>
                <div style="font-size:1.2rem;font-weight:700;color:var(--text-900);margin-top:4px;">{urgency_badge(cls.get('urgency'))}</div>
                <div class="muted" style="font-size:0.8rem;margin-top:4px;">Urgency confidence</div>
                {confidence_meter(cls.get('urgency_confidence'))}
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    section_header("🧠", "Model Reasoning")
    show_more_text(cls.get("reasoning") or "No reasoning recorded.", 600, f"cls_{detail['ticket_id'][:8]}")

    if cls.get("degraded"):
        partial_banner(
            "Degraded classification",
            "The LLM was unavailable, so a keyword heuristic fallback was used. "
            + str(cls.get("degradation_reason") or "Low confidence."),
        )


def _render_retrieval(detail: dict) -> None:
    section_header("📚", "Retrieved Documents")
    docs = _retrieved_docs_from_trace(detail)

    if not docs:
        placeholder("No documents were retrieved for this ticket.")
        st.markdown(
            '<p class="small muted">This can happen when the vector store is empty, unavailable, or returned no relevant matches.</p>',
            unsafe_allow_html=True,
        )
        return

    for i, d in enumerate(docs, start=1):
        url = _citation_origin_url(d)
        title = d.get("source") or d.get("id") or f"Doc {i}"
        card_html = f"""
        <div class="s-card" style="margin-bottom:12px;">
            <div class="space-between" style="margin-bottom:6px;">
                <span style="font-weight:700;color:var(--text-900);">[{i}] {title}</span>
                <span class="muted small">{d.get('id', '')[:24]}</span>
            </div>
        """
        if url:
            card_html += f'<div class="small" style="margin-bottom:6px;"><a href="{url}" target="_blank" rel="noopener">{url[:80]}</a></div>'
        st.markdown(card_html, unsafe_allow_html=True)
        show_more_text(d.get("content"), 300, f"doc_{detail['ticket_id'][:8]}_{i}")
        sim = d.get("similarity_score")
        if sim is not None:
            st.markdown(
                f"""<div style="margin-top:8px;font-size:0.75rem;color:var(--text-500);">Similarity</div>
                {confidence_meter(float(sim))}""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_draft(detail: dict, ticket_id: str) -> None:
    section_header("✍️", "Draft Response")
    draft_text = detail.get("draft_text")

    if not draft_text:
        placeholder("Not yet generated — run triage to produce a draft.")
        st.markdown("---")
        # Still expose the edit slot so a human can write one from scratch
        _render_draft_editor(ticket_id)
        return

    editing = st.session_state.get(f"draft_edit_{ticket_id[:8]}", False)
    if editing:
        _render_draft_editor(ticket_id, initial=draft_text)
        return

    st.markdown(draft_text)
    show_more = len(draft_text) > 1000
    if show_more:
        st.caption("Draft is long — it is shown in full above. Copy or edit it below.")

    b1, b2, b3 = st.columns(3)
    with b1:
        copy_button("Copy draft", draft_text, f"copydraft_{ticket_id[:8]}", icon="📋")
    with b2:
        if st.button("✏️ Edit draft", key=f"edit_on_{ticket_id[:8]}", use_container_width=True):
            st.session_state[f"draft_edit_{ticket_id[:8]}"] = True
            st.rerun()
    with b3:
        st.caption(f"{len(draft_text)} characters")

    # Citations
    cites = safe_json(detail.get("draft_citations"))
    if cites:
        st.markdown("---")
        section_header("🔖", "Citations")
        for i, c in enumerate(cites, start=1):
            rel = c.get("relevance_score")
            rel_s = f"{float(rel):.0%}" if isinstance(rel, (int, float)) else (rel or "—")
            doc_id = str(c.get("source_doc_id", "?"))[:24]
            claim = str(c.get("claim", ""))
            if len(claim) > 300:
                claim = claim[:300] + "…"
            st.markdown(
                f"""<div class="s-card" style="padding:12px 16px;">
                    <div class="small muted">[{i}] Doc {doc_id} · relevance {rel_s}</div>
                    <div style="font-size:0.9rem;color:var(--text-700);margin-top:4px;">{claim}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_draft_editor(ticket_id: str, initial: str = "") -> None:
    new_text = st.text_area(
        "Edit draft response",
        value=initial,
        key=f"draft_editor_{ticket_id[:8]}",
        height=240,
        help="Edits are saved back to the ticket before any approve action.",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save draft", key=f"draft_save_{ticket_id[:8]}", type="primary", use_container_width=True):
            if not new_text.strip():
                st.error("Draft cannot be empty.")
            elif update_draft(ticket_id, new_text.strip()):
                st.toast("Draft saved", icon="💾")
                st.session_state[f"draft_edit_{ticket_id[:8]}"] = False
                st.rerun()
            else:
                st.error("Could not save draft.")
    with c2:
        if st.button("Cancel", key=f"draft_cancel_{ticket_id[:8]}", use_container_width=True):
            st.session_state[f"draft_edit_{ticket_id[:8]}"] = False
            st.rerun()


def _render_trace(detail: dict, trace: dict | None, ticket_id: str) -> None:
    section_header("🔗", "Pipeline Trace")
    if not trace:
        placeholder("No trace recorded for this ticket yet.")
        return
    steps = trace.get("steps") or []
    render_trace_timeline(steps, trace.get("total_duration_ms"), trace.get("loop_count"))
    st.markdown("---")
    render_trace_actions(trace, ticket_id)
    trace_final_status_card(trace)


def _render_actions(detail: dict, ticket_id: str) -> None:
    section_header("⚙️", "Ticket Actions")
    status = detail.get("status")

    st.markdown(
        f'<div class="s-card"><span class="small muted">Current status: </span>{status_badge(status)}'
        f'<div class="small muted" style="margin-top:6px;">Review decision: <strong>{detail.get("review_decision") or "—"}</strong></div></div>',
        unsafe_allow_html=True,
    )

    can_approve = status in ("awaiting_review", "escalated")

    with st.popover("✅ Approve draft", key=f"act_app_{ticket_id[:8]}", disabled=not can_approve,
                    help="Send the approved reply to the customer." if can_approve else "Only tickets awaiting review can be approved."):
        st.markdown("**Approve and send?**")
        st.markdown("This marks the ticket resolved and mocks sending the reply to the customer.")
        if st.button("Confirm approve", key=f"act_app_go_{ticket_id[:8]}", type="primary", use_container_width=True):
            res = approve_ticket(ticket_id)
            if res:
                st.toast("Draft approved", icon="✅")
                st.rerun()
            else:
                st.error("Approval failed.")

    with st.popover("✏️ Edit & approve", key=f"act_ea_{ticket_id[:8]}", disabled=not can_approve):
        new_text = st.text_area("Edited draft", value=detail.get("draft_text") or "", height=160,
                                key=f"ea_text_{ticket_id[:8]}")
        if st.button("Save & approve", key=f"act_ea_go_{ticket_id[:8]}", type="primary", use_container_width=True):
            if not new_text.strip():
                st.error("Draft cannot be empty.")
            elif update_draft(ticket_id, new_text.strip()):
                st.toast("Draft updated", icon="💾")
                if approve_ticket(ticket_id):
                    st.toast("Draft approved", icon="✅")
                    st.rerun()
                else:
                    st.error("Draft saved but approval failed.")
            else:
                st.error("Could not save draft.")

    with st.popover("🚫 Reject", key=f"act_rej_{ticket_id[:8]}"):
        reason = st.text_area("Reason for rejection", key=f"rej_text_{ticket_id[:8]}",
                              placeholder="Why is this draft being rejected?")
        if st.button("Confirm reject", key=f"act_rej_go_{ticket_id[:8]}", use_container_width=True):
            if not reason.strip():
                st.error("A reason is required to reject.")
            elif escalate_ticket(ticket_id, f"Rejected by reviewer: {reason.strip()}"):
                st.toast("Ticket rejected", icon="🚫")
                st.rerun()
            else:
                st.error("Reject failed.")

    st.markdown("---")

    with st.popover("🤖 Re-run triage", key=f"act_rerun_{ticket_id[:8]}"):
        st.markdown("Re-run the full triage pipeline (classify → retrieve → draft → escalation gate).")
        if st.button("Start re-triage", key=f"act_rerun_go_{ticket_id[:8]}", type="primary", use_container_width=True):
            res = rerun_triage(ticket_id)
            if res:
                st.toast("Triage restarted", icon="🤖")
                st.success(res.get("message", "Triage started."))
                st.rerun()
            else:
                st.error("Could not start triage.")

    with st.popover("🔺 Escalate manually", key=f"act_esc_{ticket_id[:8]}"):
        reason = st.text_area("Escalation reason", key=f"esc_text_{ticket_id[:8]}",
                              placeholder="Why is human review needed?")
        if st.button("Confirm escalation", key=f"act_esc_go_{ticket_id[:8]}", type="primary", use_container_width=True):
            if not reason.strip():
                st.error("A reason is required to escalate.")
            elif escalate_ticket(ticket_id, reason.strip()):
                st.toast("Ticket escalated", icon="🔺")
                st.rerun()
            else:
                st.error("Escalation failed.")


def _render_sticky_actions(detail: dict, ticket_id: str) -> None:
    """Fixed action bar that stays visible while scrolling."""
    status = detail.get("status")
    can_approve = status in ("awaiting_review", "escalated")

    st.markdown('<div class="actions-bar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Approve", key=f"sticky_app_{ticket_id[:8]}", type="primary",
                     use_container_width=True, disabled=not can_approve):
            if approve_ticket(ticket_id):
                st.toast("Draft approved", icon="✅")
                st.session_state["queue_cache"] = {}
                st.rerun()
    with c2:
        if st.button("🚫 Reject", key=f"sticky_rej_{ticket_id[:8]}", use_container_width=True,
                     disabled=(status == "escalated")):
            st.session_state[f"sticky_rejecting_{ticket_id[:8]}"] = True
            st.rerun()
        if st.session_state.get(f"sticky_rejecting_{ticket_id[:8]}", False) and status != "escalated":
            reason = st.text_input("Reject reason", key=f"sticky_rej_text_{ticket_id[:8]}",
                                   placeholder="Required")
            if st.button("Confirm", key=f"sticky_rej_go_{ticket_id[:8]}", type="primary"):
                if not reason.strip():
                    st.error("Reason required.")
                elif escalate_ticket(ticket_id, f"Rejected by reviewer: {reason.strip()}"):
                    st.toast("Ticket rejected", icon="🚫")
                    st.session_state.pop(f"sticky_rejecting_{ticket_id[:8]}", None)
                    st.session_state["queue_cache"] = {}
                    st.rerun()
    with c3:
        if st.button("🔺 Escalate", key=f"sticky_esc_{ticket_id[:8]}", use_container_width=True,
                     disabled=(status == "escalated")):
            st.session_state[f"sticky_escalating_{ticket_id[:8]}"] = True
            st.rerun()
        if st.session_state.get(f"sticky_escalating_{ticket_id[:8]}", False) and status != "escalated":
            reason = st.text_input("Escalation reason", key=f"sticky_esc_text_{ticket_id[:8]}",
                                   placeholder="Required")
            if st.button("Confirm", key=f"sticky_esc_go_{ticket_id[:8]}", type="primary"):
                if not reason.strip():
                    st.error("Reason required.")
                elif escalate_ticket(ticket_id, reason.strip()):
                    st.toast("Ticket escalated", icon="🔺")
                    st.session_state.pop(f"sticky_escalating_{ticket_id[:8]}", None)
                    st.session_state["queue_cache"] = {}
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)