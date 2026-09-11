"""Metrics dashboard (Sub-Phase 2.4): KPI cards with sparklines + trends,
stacked time-series, distribution charts, confidence histogram, escalation
reasons, and model/cost cards. A date-range selector re-renders everything.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.api import fetch_many, record_api_error
from dashboard.theme import CATEGORY_COLORS, URGENCY_COLORS, chart_colors
from dashboard.widgets import empty_state, section_header

RANGES = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "All time": 4000,
}


@st.cache_data(ttl=45, show_spinner=False)
def metrics_bundle(days: int) -> dict:
    """Fetch every metrics endpoint concurrently and cache the bundle.

    Returns ``{label: payload|None}`` keyed by stable labels. Cached for 45s
    so navigating between pages never re-hammers the (slow) backend.
    """
    return fetch_many(
        [
            ("ts_cur", "/dashboard/timeseries", {"days": days}),
            ("ts_prev", "/dashboard/timeseries", {"days": min(days * 2, 4000)}),
            ("metrics_cur", "/dashboard/metrics", {"days": days}),
            ("metrics_prev", "/dashboard/metrics", {"days": min(days * 2, 4000)}),
            ("hist", "/dashboard/confidence-histogram", {"days": days}),
            ("reasons", "/dashboard/escalation-reasons", {"days": days}),
        ],
        timeout=25.0,
    )


def _kpi_sparkline(values: list, color: str) -> go.Figure:
    """Tiny inline line chart for a KPI card."""
    xs = list(range(len(values)))
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=values,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color + "26",
            hovertemplate="%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=44,
        margin=dict(t=2, b=2, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _trend_html(values: list) -> str:
    """Arrow + percentage change comparing the two halves of ``values``."""
    if len(values) < 2:
        return '<span class="trend flat">— no trend data</span>'
    half = len(values) // 2
    prev = values[:half]
    cur = values[-half:]
    if not prev:
        return '<span class="trend flat">—</span>'
    p = sum(prev) / len(prev)
    c = sum(cur) / len(cur)
    if p == 0 and c == 0:
        return '<span class="trend flat">→ stable</span>'
    if p == 0:
        return '<span class="trend up">▲ new</span>'
    delta = (c - p) / p * 100
    if abs(delta) < 0.5:
        return '<span class="trend flat">→ stable</span>'
    cls = "up" if delta > 0 else "down"
    arrow = "▲" if delta > 0 else "▼"
    return f'<span class="trend {cls}">{arrow} {abs(delta):.1f}%</span>'


def _render_kpi(label: str, raw_value: float, fmt: callable, spark: list, accent: str, spark_color: str) -> None:
    """Render one KPI card column: value, trend, sparkline (last 7 periods)."""
    st.markdown(
        f"""
        <div class="kpi-card {accent}">
            <div class="label">{label}</div>
            <div class="value">{fmt(raw_value)}</div>
            <div>{_trend_html(spark)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    spark_series = spark[-7:] if len(spark) > 7 else spark
    st.plotly_chart(_kpi_sparkline(spark_series, spark_color), use_container_width=True, config={"displayModeBar": False})


def render_metrics_page() -> None:
    section_header("📊", "Dashboard Overview", "Live triage metrics across the support pipeline")

    # Date range selector (persists in session_state)
    default_range = st.session_state.get("metrics_range", "Last 30 days")
    selected = st.pills(
        "Time range",
        list(RANGES.keys()),
        default=default_range,
        key="metrics_range_widget",
        label_visibility="collapsed",
        help="Scope every metric and chart to this window.",
    )
    if selected:
        st.session_state["metrics_range"] = selected

    days = RANGES.get(st.session_state.get("metrics_range", "Last 30 days"), 30)
    c = chart_colors()

    with st.spinner("Loading metrics…"):
        bundle = metrics_bundle(days)

    ts = bundle.get("ts_cur")
    ts_double = bundle.get("ts_prev") or ts
    metrics = bundle.get("metrics_cur")
    hist = bundle.get("hist")
    reasons = bundle.get("reasons")
    prev_metrics = bundle.get("metrics_prev")

    if metrics is None and ts is None:
        record_api_error({"type": "connection", "detail": "All metrics endpoints failed to respond.", "endpoint": "/dashboard/*"})
        empty_state(
            "📡", "Dashboard unavailable",
            "Could not fetch metrics from the API. Check the service and retry.",
        )
        return
    metrics = metrics or {}

    # ── Time-series for KPI sparklines & stacked chart ──
    points = (ts or {}).get("points", [])
    daily_total = [p["total"] for p in points]
    daily_processed = [p["resolved"] + p["escalated"] for p in points]
    daily_escalated = [p["escalated"] for p in points]
    daily_confidence = [p["avg_confidence"] if p["avg_confidence"] is not None else 0 for p in points]
    daily_success = [
        round(p["resolved"] / p["total"], 3) if p["total"] else 0 for p in points
    ]
    prev_points = (ts_double or {}).get("points", [])
    prev_total = [p["total"] for p in prev_points]
    prev_processed = [p["resolved"] + p["escalated"] for p in prev_points]
    prev_escalated = [p["escalated"] for p in prev_points]
    prev_confidence = [p["avg_confidence"] if p["avg_confidence"] is not None else 0 for p in prev_points]
    prev_success = [round(p["resolved"] / p["total"], 3) if p["total"] else 0 for p in prev_points]

    total = metrics.get("total_tickets", 0)
    fmt_int = lambda v: f"{int(v):,}"  # noqa: E731
    fmt_pct = lambda v: f"{v:.0%}"  # noqa: E731

    # ── KPI row ──
    st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
    k5 = st.columns(5)
    with k5[0]:
        _render_kpi("Total Tickets", total, fmt_int, daily_total + prev_total, "accent", c["brand"])
    with k5[1]:
        _render_kpi("Processed", metrics.get("processed_tickets", 0), fmt_int, daily_processed + prev_processed, "ok", c["green"])
    with k5[2]:
        _render_kpi("Escalated", metrics.get("escalated_tickets", 0), fmt_int, daily_escalated + prev_escalated, "warn", c["amber"])
    with k5[3]:
        avg_conf = metrics.get("avg_confidence") or 0.0
        _render_kpi("Avg Confidence", avg_conf, fmt_pct, daily_confidence + prev_confidence, "accent", c["blue"])
    with k5[4]:
        _render_kpi("Success Rate", metrics.get("success_rate", 0), fmt_pct, daily_success + prev_success, "ok", c["green"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Time-series stacked area ──
    if points:
        section_header("📈", "Tickets Processed by Day", f"Last {len(points)} days (resolved · escalated · pending)")
        display = points[-120:] if len(points) > 120 else points
        labels = [p["date"] for p in display]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels, y=[p["resolved"] for p in display], mode="lines",
            name="Resolved", stackgroup="one", line=dict(width=0, color=c["green"]),
            hovertemplate="%{x}<br>Resolved: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=labels, y=[p["escalated"] for p in display], mode="lines",
            name="Escalated", stackgroup="one", line=dict(width=0, color=c["amber"]),
            hovertemplate="%{x}<br>Escalated: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=labels, y=[p["pending"] for p in display], mode="lines",
            name="Pending", stackgroup="one", line=dict(width=0, color=c["text"]),
            hovertemplate="%{x}<br>Pending: %{y}<extra></extra>",
        ))
        fig.update_layout(
            height=320, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["text"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(showgrid=False, color=c["axis"], tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor=c["grid"], color=c["axis"], zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Two-column distribution row ──
    col_a, col_b = st.columns(2)

    with col_a:
        # Category distribution — horizontal bar, sorted, with %
        cat_dist = metrics.get("category_distribution", {})
        if cat_dist:
            section_header("📁", "Category Distribution")
            total_cats = sum(cat_dist.values())
            items = sorted(cat_dist.items(), key=lambda kv: kv[1], reverse=True)
            labels = [k.replace("_", " ").title() for k, _ in items]
            values = [v for _, v in items]
            colors = [CATEGORY_COLORS.get(k, "#6366f1") for k, _ in items]
            pcts = [f"{v / total_cats:.0%}" for v in values]
            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=colors, text=pcts, textposition="outside",
                hovertemplate="%{y}: %{x} tickets (%{text})<extra></extra>",
            ))
            fig.update_layout(
                height=300, margin=dict(t=6, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=c["text"]),
                xaxis=dict(showgrid=True, gridcolor=c["grid"], color=c["axis"]),
                yaxis=dict(showgrid=False, color=c["axis"]),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("📁", "No Categories Yet", "Categories appear once tickets are triaged.")

    with col_b:
        # Urgency distribution — bar with previous-period overlay
        urg_dist = metrics.get("urgency_distribution", {})
        if urg_dist:
            section_header("⚡", "Urgency Distribution", "solid = current · outline = previous period")
            # previous period = 2N window minus N window
            prev_urg = (prev_metrics or {}).get("urgency_distribution", {})
            all_keys = []
            for k in list(urg_dist.keys()) + list(prev_urg.keys()):
                if k not in all_keys:
                    all_keys.append(k)
            order = ["critical", "urgent", "high", "medium", "low"]
            all_keys.sort(key=lambda k: order.index(k) if k in order else 99)
            labels = [k.title() for k in all_keys]
            cur_vals = [urg_dist.get(k, 0) for k in all_keys]
            # prev window value = value in 2N window that is *not* part of current N window
            prev_vals = [
                max(0, (prev_urg.get(k, 0) - urg_dist.get(k, 0))) for k in all_keys
            ]
            colors = [URGENCY_COLORS.get(k, "#94a3b8") for k in all_keys]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels, y=cur_vals, name="This period",
                marker_color=colors, text=cur_vals, textposition="outside",
                hovertemplate="%{x}: %{y} tickets<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=labels, y=prev_vals, name="Previous period",
                marker=dict(color="rgba(0,0,0,0)", line=dict(color=c["axis"], width=1.5, dash="dot")),
                text=prev_vals, textposition="outside",
                hovertemplate="%{x}: %{y} tickets<extra></extra>",
            ))
            fig.update_layout(
                barmode="group",
                height=300, margin=dict(t=6, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=c["text"]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                xaxis=dict(showgrid=False, color=c["axis"]),
                yaxis=dict(showgrid=True, gridcolor=c["grid"], color=c["axis"], zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("⚡", "No Urgency Data", "Urgency distribution appears after triage runs.")

    # ── Confidence histogram ──
    hist_buckets = (hist or {}).get("buckets", [])
    if hist_buckets:
        section_header("🎯", "Confidence Histogram", "Tickets bucketed by classification confidence")
        labels = [b["range_label"] for b in hist_buckets]
        counts = [b["count"] for b in hist_buckets]
        colors = []
        for b in hist_buckets:
            mid = (b["min_val"] + b["max_val"]) / 2
            colors.append(c["red"] if mid < 0.5 else c["amber"] if mid <= 0.7 else c["green"])
        fig = go.Figure(go.Bar(
            x=labels, y=counts, marker_color=colors,
            text=counts, textposition="outside",
            hovertemplate="Confidence %{x}: %{y} tickets<extra></extra>",
        ))
        fig.update_layout(
            height=280, margin=dict(t=6, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["text"]),
            xaxis=dict(showgrid=False, color=c["axis"], categoryorder="array", categoryarray=labels),
            yaxis=dict(showgrid=True, gridcolor=c["grid"], color=c["axis"], zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Escalation reasons + model/cost cards ──
    reason_items = (reasons or {}).get("reasons", [])
    col_c, col_d = st.columns([3, 2])

    with col_c:
        if reason_items:
            section_header("🚨", "Top Escalation Reasons")
            labels = [r["reason"][:48] + ("…" if len(r["reason"]) > 48 else "") for r in reason_items]
            values = [r["count"] for r in reason_items]
            fig = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.35,
                marker=dict(colors=[c["brand"], c["amber"], c["red"], c["blue"], c["green"], c["violet"]]),
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ))
            fig.update_layout(
                height=320, margin=dict(t=6, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=c["text"]),
                legend=dict(orientation="v", font=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("🚨", "No Escalations Yet", "Escalation reasons will appear once tickets are escalated.")

    with col_d:
        section_header("⚙️", "Runtime")

        st.markdown(
            f"""<div class="s-card">
                <div class="label" style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-500);">LLM Provider</div>
                <div style="font-size:1.05rem;font-weight:700;color:var(--text-900);margin-top:4px;">🤖 gemini</div>
                <div class="small muted" style="margin-top:2px;">primary · openrouter fallback</div>
                <div class="small muted" style="margin-top:8px;">Per-ticket provider attribution is not stored by the backend yet.</div>
            </div>""",
            unsafe_allow_html=True,
        )

        cost = metrics.get("cost_per_ticket")
        if cost is not None and cost > 0:
            cost_value = f"${cost:.4f}"
            cost_note = "estimated LLM spend per triaged ticket"
        else:
            cost_value = "Not tracked"
            cost_note = "the backend does not record LLM cost per ticket yet"
        st.markdown(
            f"""<div class="s-card">
                <div class="label" style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-500);">Avg Cost / Ticket</div>
                <div style="font-size:1.3rem;font-weight:800;color:var(--text-900);margin-top:4px;">
                  {cost_value}
                </div>
                <div class="small muted" style="margin-top:2px;">{cost_note}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Recent activity
        recent = metrics.get("recent_activity", [])
        if recent:
            st.markdown(
                '<div class="label" style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-500);margin-bottom:6px;">Recent Activity</div>',
                unsafe_allow_html=True,
            )
            for a in recent[:5]:
                tid = str(a.get("ticket_id", "?"))[:8]
                action = a.get("action", "?")
                ts = format_timestamp(a.get("timestamp"))
                st.markdown(
                    f'<div class="ticket-row" style="cursor:default;padding:10px 14px;">'
                    f'<div class="ticket-top"><span class="ticket-id">#{tid}…</span><span>{status_badge_short(action)}</span></div>'
                    f'<div class="ticket-sub">{ts}</div></div>',
                    unsafe_allow_html=True,
                )


def status_badge_short(status: str) -> str:
    """Local status badge (avoids circular import with widgets)."""
    from dashboard.theme import STATUS_COLORS

    color = STATUS_COLORS.get(status, "#94a3b8")
    from dashboard.widgets import STATUS_ICONS

    icon = STATUS_ICONS.get(status, "•")
    return (
        f'<span class="badge" style="background:{color}22;color:{color};">'
        f"{icon} {status.upper()}</span>"
    )