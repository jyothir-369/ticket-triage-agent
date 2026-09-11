"""Ticket queue (Sub-Phase 2.1): search, filters, sort, pagination, bulk
actions, refresh, empty states — plus the in-place "create ticket" flow.

Filter state lives in ``st.session_state`` via stable widget keys (``q_*``),
so it survives reruns and navigation. Rows are rendered as rich cards, and a
Table view provides Streamlit's native keyboard-selection (arrows + Enter).
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.api import api_get, api_post, format_timestamp, preview
from dashboard.theme import is_dark
from dashboard.widgets import (
    approve_ticket,
    category_badge,
    confidence_meter,
    empty_state,
    escalate_ticket,
    section_header,
    status_badge,
    urgency_badge,
)

QUEUE_PAGE_SIZE = 20
STATUS_OPTIONS = ["pending", "processing", "resolved", "escalated", "failed",
                  "awaiting_review", "approved", "rejected", "open", "classified"]
CATEGORY_OPTIONS = ["bug", "feature_request", "account_issue", "billing", "usage_help", "other"]
URGENCY_OPTIONS = ["low", "medium", "high", "critical"]
SORT_FIELDS = {
    "created_at": "Created",
    "updated_at": "Updated",
    "confidence": "Confidence",
    "urgency": "Urgency",
    "status": "Status",
}


# ---------------------------------------------------------------------------
# Fetch helper with a short-lived cache (Refresh clears it)
# ---------------------------------------------------------------------------


def _fetch_page(params: dict) -> dict | None:
    sig = str(sorted(params.items()))
    cache_key = "queue_cache"
    now = time.time()
    cached = st.session_state.get(cache_key, {})
    if cached.get("sig") == sig and now - cached.get("ts", 0) < 8:
        return cached["data"]
    data = api_get("/tickets/list", params=params)
    if data is not None:
        st.session_state[cache_key] = {"sig": sig, "ts": time.time(), "data": data}
    return data


# ---------------------------------------------------------------------------
# Queue page
# ---------------------------------------------------------------------------


def _reset_page() -> None:
    st.session_state["q_page"] = 1


def render_queue_page() -> None:
    """Main queue entry point — delegates to detail when a ticket is open."""
    if st.session_state.get("selected_ticket_id"):
        from dashboard.page_detail import render_ticket_detail

        return render_ticket_detail(st.session_state["selected_ticket_id"])

    section_header("📋", "Ticket Queue", "Search, filter, triage, and act across every ticket")

    # ── Top bar: refresh + last refreshed + create ─────────────────────────
    top_col, mid_col, create_col = st.columns([2.2, 2.4, 1.4])
    # Refresh
    with top_col:
        if st.button("↻ Refresh", key="q_refresh", use_container_width=True):
            st.session_state["queue_cache"] = {}
            st.session_state["last_refreshed"] = datetime.now()
            st.toast("Queue refreshed", icon="🔄")
            st.rerun()
    with mid_col:
        last_ts = st.session_state.get("last_refreshed")
        st.caption(
            f"Last refreshed: {last_ts.strftime('%H:%M:%S') if last_ts else '—'}"
        )

    with create_col:
        if st.button("➕ New Ticket", key="q_new_ticket", use_container_width=True, type="primary"):
            st.session_state["show_create"] = True

    # ── Filters ─────────────────────────────────────────────────────────────
    with st.expander("🔎 Filters", expanded=True):
        search = st.text_input(
            "Search tickets",
            key="q_search",
            placeholder="Search by ticket ID, subject, or content substring…",
            help="Matches against ticket ID and full content.",
        )
        f1, f2, f3 = st.columns(3)
        with f1:
            cats = st.multiselect("Category", CATEGORY_OPTIONS, key="q_cats",
                                  placeholder="All categories",
                                  format_func=lambda c: c.replace("_", " ").title())
        with f2:
            urgs = st.multiselect("Urgency", URGENCY_OPTIONS, key="q_urgs",
                                  placeholder="All urgency levels",
                                  format_func=lambda c: c.upper())
        with f3:
            stats = st.multiselect("Status", STATUS_OPTIONS, key="q_stats",
                                   placeholder="All statuses",
                                   format_func=lambda c: c.replace("_", " ").title())
        f4, f5, f6 = st.columns(3)
        with f4:
            date_from = st.date_input("Created after", value=None, key="q_date_from")
        with f5:
            date_to = st.date_input("Created before", value=None, key="q_date_to")
        with f6:
            sort_by = st.selectbox("Sort by", list(SORT_FIELDS.keys()), index=0, key="q_sort_by",
                                   format_func=lambda k: SORT_FIELDS[k])
            sort_dir = st.selectbox("Direction", ["desc", "asc"], index=0, key="q_sort_dir",
                                    format_func=lambda d: "Newest first" if d == "desc" else "Oldest first")

    # ── Build query params ─────────────────────────────────────────────────
    params: dict = {}
    if search.strip():
        params["search"] = search.strip()
    if cats:
        params["category"] = ",".join(cats)
    if urgs:
        params["urgency"] = ",".join(urgs)
    if stats:
        params["status"] = ",".join(stats)
    if date_from:
        params["created_after"] = datetime.combine(date_from, datetime.min.time()).isoformat()
    if date_to:
        params["created_before"] = datetime.combine(date_to, datetime.max.time()).isoformat()
    params["sort_by"] = sort_by
    params["sort_dir"] = sort_dir
    params["page_size"] = st.session_state.get("q_page_size", QUEUE_PAGE_SIZE)
    page = max(1, int(st.session_state.get("q_page", 1)))
    params["page"] = page

    with st.spinner("Loading tickets…"):
        data = _fetch_page(params)

    if data is None:
        empty_state(
            "📡", "Could not load the queue",
            "The API didn't respond. Check the backend and retry.",
            primary_label="↻ Retry", primary_key="q_retry",
        )
        return

    tickets = data.get("tickets", [])
    total = data.get("total", 0)
    total_pages = max(1, data.get("total_pages", 1))
    page = min(page, total_pages)  # clamp after filtering shrinks the set

    # ── View toggle + page size ────────────────────────────────────────────
    vt1, vt2 = st.columns([4, 1])
    with vt1:
        view = st.radio("View", ["Cards", "Table"], key="q_view", horizontal=True,
                        label_visibility="collapsed",
                        help="Table view supports native keyboard navigation — arrow keys move, Enter selects.")
    with vt2:
        ps_options = [10, 20, 50]
        cur_ps = int(st.session_state.get("q_page_size", QUEUE_PAGE_SIZE))
        st.selectbox(
            "Page size",
            ps_options,
            index=ps_options.index(cur_ps) if cur_ps in ps_options else 1,
            key="q_page_size",
            format_func=lambda n: f"{n} / page",
        )

    # ── Summary + showing X–Y of Z ─────────────────────────────────────────
    if total == 0:
        if any([search, cats, urgs, stats, date_from, date_to]):
            clicked = empty_state(
                "🔍", "No tickets match your filters",
                "Try broadening the search terms or removing a few filters.",
                primary_label="🧹 Clear filters", primary_key="q_clear",
            )
            if clicked:
                for k in ["q_search", "q_cats", "q_urgs", "q_stats", "q_date_from", "q_date_to"]:
                    st.session_state.pop(k, None)
                st.session_state["q_page"] = 1
                st.rerun()
        else:
            empty_state(
                "🎉", "Queue is clear", "No tickets yet. Create your first ticket to get started.",
                primary_label="➕ Create your first ticket", primary_key="q_first",
            )
            if st.session_state.get("q_first"):
                st.session_state["show_create"] = True
        return

    start = (page - 1) * len(tickets) + 1
    end = start + len(tickets) - 1
    st.markdown(
        f'<div class="filter-bar"><span style="font-weight:600;color:var(--text-700);">'
        f"Showing <span class='mono'>{start}–{end}</span> of <strong>{total}</strong> tickets"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # ── Bulk selection state ───────────────────────────────────────────────
    st.session_state.setdefault("bulk_selected", [])
    page_ids = [t["ticket_id"] for t in tickets]

    if view == "Table":
        _render_table(tickets, stats, page_ids)
    else:
        _render_cards(tickets, page_ids)

    _render_bulk_bar(page_ids)

    _render_pagination(page, total_pages, total)


# ---------------------------------------------------------------------------
# Card view
# ---------------------------------------------------------------------------


def _select_row(ticket_id: str) -> None:
    st.session_state["selected_ticket_id"] = ticket_id
    st.session_state["trace_focus_idx"] = 0


def _render_cards(tickets: list, page_ids: list) -> None:
    for t in tickets:
        tid = t["ticket_id"]
        cb_col, body = st.columns([0.35, 4.35])
        with cb_col:
            st.checkbox(
                "Select " + tid[:8],
                key=f"sel_{tid}",
                label_visibility="collapsed",
                help=f"Select ticket {tid[:12]}… for bulk actions.",
            )
        with body:
            with st.container(border=True):
                st.markdown(
                    f"""<div class="ticket-row" style="border:none;box-shadow:none;margin:0;padding:0;">
                        <div class="ticket-top">
                            <span class="ticket-id">#{tid[:12]}…</span>
                            <span class="ticket-meta">
                                {status_badge(t.get("status"))}
                                {category_badge(t.get("category"))}
                                {urgency_badge(t.get("urgency"))}
                            </span>
                        </div>
                        <div class="ticket-preview">{preview(t.get("content_preview") or t.get("content"), 260)}</div>
                        <div class="ticket-sub">
                            <span>🕐 Created {format_timestamp(t.get("created_at"))}</span>
                            <span>✏️ Updated {format_timestamp(t.get("updated_at"))}</span>
                            <span>Source: {t.get("source", "api")}</span>
                            <span>Confidence: {f"{t['confidence']:.0%}" if t.get("confidence") is not None else '—'}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("Open ticket", key=f"open_{tid}", use_container_width=True):
                        _select_row(tid)
                        st.rerun()
                with c2:
                    st.caption(tid)


def _render_table(tickets: list, stats: list, page_ids: list) -> None:
    """Dataframe view — native keyboard selection (arrows + Enter)."""
    rows = []
    for t in tickets:
        rows.append({
            "ID": t["ticket_id"][:12],
            "Status": (t.get("status") or "").upper(),
            "Category": (t.get("category") or "—").replace("_", " ").title(),
            "Urgency": (t.get("urgency") or "—").upper(),
            "Confidence": (f"{t['confidence']:.0%}" if t.get("confidence") is not None else "—"),
            "Created": format_timestamp(t.get("created_at")),
            "Source": t.get("source", "api"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        key="q_table",
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        height=min(len(rows) * 38 + 40, 560),
    )
    sel = st.session_state.get("q_table", {}).get("selection", {}).get("rows", [])
    if sel and st.button("Open selected ticket ▸", key="q_open_table", type="primary"):
        idx = sel[0]
        _select_row(tickets[idx]["ticket_id"])
        st.rerun()


def _render_bulk_bar(page_ids: list) -> None:
    """Bulk-select actions bar shown when at least one ticket is selected."""
    selected = [k[4:] for k in st.session_state if k.startswith("sel_") and st.session_state[k]]
    st.session_state["bulk_selected"] = selected

    if not selected:
        return

    st.markdown(
        f'<div class="actions-bar" style="position:static;margin-bottom:16px;">'
        f'<span style="font-weight:700;color:var(--text-900);">✅ {len(selected)} selected</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        esc_pop = st.popover("⚠️ Escalate selected", key=f"bulk_esc_pop_{st.session_state.get('bulk_pop', 0)}",
                             help="Send selected tickets to human review with a shared reason.")
        with esc_pop:
            reason = st.text_area("Escalation reason", key="bulk_esc_reason",
                                  placeholder="Why are these tickets being escalated together?")
            if st.button("Confirm escalation", key="bulk_esc_go", type="primary", use_container_width=True):
                if not reason.strip():
                    st.error("Reason is required to escalate.")
                else:
                    ok, fail = 0, 0
                    for tid in selected:
                        if escalate_ticket(tid, reason.strip()):
                            ok += 1
                        else:
                            fail += 1
                    if ok:
                        st.toast(f"Escalated {ok} ticket(s)", icon="⚠️")
                    if fail:
                        st.error(f"Failed to escalate {fail} ticket(s).")
                    _clear_selection(page_ids)
                    st.rerun()
    with c2:
        if st.button("✅ Approve selected", key="bulk_app", use_container_width=True):
            ok, fail = 0, 0
            for tid in selected:
                if approve_ticket(tid):
                    ok += 1
                else:
                    fail += 1
            if ok:
                st.toast(f"Approved {ok} ticket(s)", icon="✅")
            if fail:
                st.error(f"Could not approve {fail} ticket(s) — check their status.")
            _clear_selection(page_ids)
            st.rerun()
    with c3:
        if st.button("🧹 Clear selection", key="bulk_clear", use_container_width=True):
            _clear_selection(page_ids)
            st.rerun()


def _clear_selection(page_ids: list) -> None:
    for pid in page_ids:
        st.session_state.pop(f"sel_{pid}", None)
    st.session_state["bulk_selected"] = []


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def _render_pagination(page: int, total_pages: int, total: int) -> None:
    if total_pages <= 1:
        return
    p1, p2, p3 = st.columns([1, 2, 1])
    with p1:
        if st.button("← Previous", key="q_prev", disabled=(page <= 1), use_container_width=True):
            st.session_state["q_page"] = max(1, page - 1)
            st.rerun()
    with p2:
        st.markdown(
            f'<div style="text-align:center;font-weight:600;color:var(--text-700);padding-top:6px;">'
            f"Page <span class='mono'>{page}</span> of {total_pages}</div>",
            unsafe_allow_html=True,
        )
        st.number_input(
            "Jump to page", min_value=1, max_value=total_pages,
            value=page, key="q_page_jump", label_visibility="collapsed",
            help=f"Jump to a page (1–{total_pages}).",
        )
        if int(st.session_state.get("q_page_jump", page)) != page:
            st.session_state["q_page"] = int(st.session_state.get("q_page_jump"))
            st.rerun()
    with p3:
        if st.button("Next →", key="q_next", disabled=(page >= total_pages), use_container_width=True):
            st.session_state["q_page"] = min(total_pages, page + 1)
            st.rerun()


# ---------------------------------------------------------------------------
# Create ticket flow
# ---------------------------------------------------------------------------


def render_create_ticket() -> None:
    """Inline create-ticket form (mirrors the API contract)."""
    section_header("➕", "Create Ticket")
    st.markdown(
        '<p style="color:var(--text-500);font-size:0.9rem;">Create a new support ticket to be triaged by the AI agent.</p>',
        unsafe_allow_html=True,
    )

    with st.form("create_ticket_form", clear_on_submit=True):
        content = st.text_area(
            "Ticket content *",
            placeholder="Subject: Unable to reset password\n\nBody: I've been trying to reset my password for the past hour but the reset link never arrives…",
            height=150,
        )
        c1, c2 = st.columns(2)
        with c1:
            source = st.selectbox(
                "Source", ["api", "email", "github", "intercom"],
                format_func=lambda x: {"api": "🔌 API", "email": "📧 Email",
                                       "github": "🐙 GitHub", "intercom": "💬 Intercom"}.get(x, x),
            )
        with c2:
            source_id = st.text_input("Source ID", placeholder="e.g. GH-123, INC-4567")
        submitted = st.form_submit_button("🚀 Create & Triage", type="primary", use_container_width=True)

    if submitted:
        if not content.strip():
            st.error("Ticket content is required.")
        else:
            payload = {"content": content.strip(), "source": source}
            if source_id.strip():
                payload["source_id"] = source_id.strip()
            with st.spinner("Creating ticket…"):
                result = api_post("/tickets/", json_data=payload)
            if result:
                st.toast("Ticket created", icon="🎫")
                st.success(f"Created `{result['ticket_id'][:12]}…`")
                if result.get("is_duplicate"):
                    st.warning("Duplicate detected — returned the existing ticket.")
                st.session_state["show_create"] = False
                st.session_state["selected_ticket_id"] = result["ticket_id"]
                st.session_state["queue_cache"] = {}
                st.rerun()