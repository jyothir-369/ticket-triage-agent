"""Horizontal pipeline-trace viewer — shared by the ticket detail Trace tab
and the standalone Trace Viewer page.

Layout: a horizontal row of step cards joined by arrows. Clicking a card
toggles an inline panel revealing the full formatted step data below it.
"""

from __future__ import annotations

import html
import json

import streamlit as st

from dashboard.api import format_duration
from dashboard.widgets import copy_button, download_text_button, empty_state


def _step_summary(step: dict) -> str:
    """One-line human summary of a step's key output."""
    data = step.get("data") or {}
    status = str(step.get("status", "")).lower()
    name = str(step.get("step", "")).lower()

    if status == "failed":
        return f"✗ {html.escape(str(step.get('error') or 'failed'))[:90]}"
    if status == "skipped":
        return "skipped"

    if "classif" in name:
        cat = data.get("category", "—")
        urg = data.get("urgency", "—")
        conf = data.get("confidence")
        conf_s = f"{float(conf):.0%}" if conf is not None else "—"
        return f"{cat} · {urg} · {conf_s}".upper()
    if "retrieve" in name:
        n = data.get("retrieval_count", 0)
        return f"{n} doc(s)" + (" · degraded" if data.get("degraded") else "")
    if "draft" in name:
        return f"{data.get('draft_length', 0)} chars · {data.get('citation_count', 0)} citations"
    if "escalat" in name:
        esc = data.get("should_escalate")
        if esc is None:
            return "→ routed"
        return "→ human review" if esc else "→ auto-resolved"
    if "finaliz" in name:
        return str(data.get("final_status") or step.get("status") or "").upper()
    if "manual" in name or name == "escalate":
        return html.escape(str(data.get("reason") or "manual escalation"))[:90]
    if name == "route":
        return str(data.get("path") or data.get("target") or "").upper()

    # Generic: show first two scalar-ish values
    parts = []
    for k, v in list(data.items()):
        if isinstance(v, (str, int, float, bool)):
            parts.append(f"{k}={v}")
        if len(parts) >= 2:
            break
    return html.escape(", ".join(parts))[:90] if parts else ""


def _status_meta(status: str) -> tuple[str, str]:
    """Maps a step status to (css class, icon)."""
    s = str(status or "").lower()
    if s == "completed":
        return "completed", "✓"
    if s in ("failed", "error", "escalated"):
        return "failed", "✗"
    if s == "skipped":
        return "skipped", "–"
    if s in ("running", "in_progress"):
        return "running", "⏳"
    return "pending", str(len("")) or "•" if False else "·"


def _step_icon(status: str) -> str:
    _, icon = _status_meta(status)
    return icon


def _pretty(step: dict) -> str:
    return json.dumps(step, indent=2, default=str)


def render_trace_timeline(steps: list[dict], total_duration: int | None, loop_count: int | None) -> None:
    """Render the horizontal timeline with inline expand panels."""
    if not steps:
        empty_state("📋", "No Trace Steps", "This ticket has not been triaged yet, or no trace data was recorded.")
        return

    # Scrubber for jumping between steps
    focus_idx = st.session_state.get("trace_focus_idx", 0)
    if len(steps) > 4:
        focus_idx = st.slider(
            "Focus step",
            min_value=0,
            max_value=len(steps) - 1,
            value=focus_idx,
            key="trace_slider",
            help="Use the slider to jump to a specific pipeline step.",
        )
        st.session_state["trace_focus_idx"] = focus_idx

    # ── Badges row ──
    meta_html = []
    if total_duration:
        meta_html.append(f'<span class="badge badge-processing">⏱ total {format_duration(total_duration)}</span>')
    if loop_count:
        meta_html.append(f'<span class="badge badge-warning">🔄 loop x{loop_count}</span>')
    meta_html.append(f'<span class="badge badge-pending">{len(steps)} steps</span>')
    st.markdown(
        f'<div class="flex-row" style="margin-bottom:10px;">{" ".join(meta_html)}</div>',
        unsafe_allow_html=True,
    )

    # ── Build HTML cards with inline panels ──
    cards = []
    for i, step in enumerate(steps):
        css, icon = _status_meta(step.get("status"))
        name = html.escape(str(step.get("step") or f"Step {i + 1}"))
        dur = format_duration(step.get("duration_ms"))
        summary = _step_summary(step)
        status_txt = html.escape(str(step.get("status") or "pending").upper())
        pretty = _pretty(step)

        card = f"""
        <div style="display:flex;align-items:stretch;">
            <div class="trace-step-card" id="tstep_{i}"
                 onclick="toggleTracePanel({i})"
                 tabindex="0" role="button" aria-label="Expand step {html.escape(name)}"
                 onkeydown="if(event.key==='Enter'||event.key===' '){{toggleTracePanel({i});event.preventDefault();}}"
                 data-focused="{1 if i == focus_idx else 0}">
                <div class="trace-step-head">
                    <span class="trace-status-icon {css}">{html.escape(_step_icon(step.get('status')))}</span>
                    <span class="trace-step-name">{name}</span>
                </div>
                <span class="trace-step-status">{status_txt}</span>
                <div class="trace-step-dur">{dur}</div>
                <div class="trace-step-summary">{summary}</div>
            </div>
            <div class="trace-connector">{'→' if i < len(steps) - 1 else ''}</div>
        </div>
        <div id="tpanel_{i}" class="s-card" style="display:none;margin-top:6px;">
            <div class="space-between" style="margin-bottom:6px;">
                <span style="font-weight:700;color:var(--text-900);">{name}</span>
                <span class="small muted">{status_txt} · {dur}</span>
            </div>
            <pre style="font-family:var(--font-mono);font-size:0.8rem;color:var(--text-700);
                 background:var(--surface-50);padding:12px;border-radius:var(--radius-sm);
                 overflow-x:auto;max-height:420px;">{html.escape(pretty)}</pre>
        </div>
        """
        cards.append(card)

    st.markdown('<div class="trace-timeline">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    # JS toggle + focus styling
    st.markdown(
        """
        <script>
        function toggleTracePanel(idx){
          var p=document.getElementById('tpanel_'+idx);
          if(p){p.style.display = (p.style.display==='none') ? 'block' : 'none';}
        }
        var all=document.querySelectorAll('[data-focused="1"]');
        if(all.length){all[0].scrollIntoView({block:'nearest',inline:'center'});}
        </script>
        """,
        unsafe_allow_html=True,
    )


def render_trace_actions(trace: dict, ticket_id: str) -> None:
    """Download-as-JSON + copy-as-text buttons for a trace."""
    tickets = (
        json.dumps(trace.get("steps", []), indent=2, default=str)
        if trace.get("steps")
        else ""
    )
    full_json = json.dumps(trace, indent=2, default=str)
    text_blob = "\n".join(
        f"{i + 1}. [{s.get('status','?')}] {s.get('step')} "
        f"({format_duration(s.get('duration_ms'))})"
        for i, s in enumerate(trace.get("steps", []))
    )

    c1, c2 = st.columns(2)
    with c1:
        download_text_button("Download trace as JSON", full_json, f"trace_{ticket_id}.json", f"dl_{ticket_id}", icon="⬇️")
    with c2:
        copy_button("Copy trace as text", text_blob or full_json, f"copytxt_{ticket_id}", icon="📋")


def trace_final_status_card(trace: dict) -> None:
    """Final status pill at the bottom of the trace view."""
    final = trace.get("final_status") or trace.get("status")
    if not final:
        return
    from dashboard.widgets import status_badge

    st.markdown(
        f'<div class="s-card" style="text-align:center;padding:14px;">'
        f'<span class="small muted">Final status: </span>{status_badge(final)}</div>',
        unsafe_allow_html=True,
    )