"""Theme tokens, global CSS, and dark-mode helpers for the dashboard.

Dark mode is driven by the ``data-theme`` attribute on the document root
(the ``<html>`` element), which Streamlit already sets based on its own
theme. We mirror that attribute from JS on first paint and swap it when the
sidebar toggle flips, then let CSS variables do the rest.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Color / chart helpers
# ---------------------------------------------------------------------------


def is_dark() -> bool:
    """Whether the manual dark-mode toggle is active."""
    return bool(st.session_state.get("dark_mode", False))


def chart_colors() -> dict:
    """Plotly-facing color tokens that track the active theme.

    Card background stays transparent so the CSS surface colour shows through;
    text / grid colours flip with the theme.
    """
    dark = is_dark()
    return {
        "text": "#94a3b8" if dark else "#475569",
        "grid": "rgba(100,116,139,0.15)" if dark else "rgba(100,116,139,0.12)",
        "surface": "rgba(0,0,0,0)",
        "axis": "#64748b",
        "brand": "#6366f1",
        "brand_soft": "#818cf8",
        "green": "#10b981",
        "amber": "#f59e0b",
        "red": "#ef4444",
        "blue": "#3b82f6",
        "violet": "#8b5cf6",
        "cyan": "#06b6d4",
    }


CATEGORY_COLORS = {
    "bug": "#ef4444",
    "feature_request": "#8b5cf6",
    "account_issue": "#3b82f6",
    "billing": "#f59e0b",
    "usage_help": "#10b981",
    "other": "#94a3b8",
}

URGENCY_COLORS = {
    "critical": "#ef4444",
    "urgent": "#f97316",
    "high": "#f59e0b",
    "medium": "#eab308",
    "low": "#22c55e",
}

STATUS_COLORS = {
    "resolved": "#10b981",
    "approved": "#10b981",
    "escalated": "#f59e0b",
    "processing": "#3b82f6",
    "awaiting_review": "#8b5cf6",
    "pending": "#94a3b8",
    "open": "#94a3b8",
    "failed": "#ef4444",
    "rejected": "#ef4444",
    "classified": "#06b6d4",
}


THEME_JS = """
<script>
/* Keep Streamlit's root data-theme in sync with the manually toggled theme. */
var storageKey = 'triage_dark_mode';
(function applyTheme() {
  try {
    var saved = sessionStorage.getItem(storageKey);
    var desired = saved === '1' ? 'dark' : saved === '0' ? 'light' : null;
    if (desired) document.documentElement.setAttribute('data-theme', desired);
  } catch (e) {}
})();
</script>
"""


def apply_theme_marker() -> None:
    """Mirror the current ``dark_mode`` flag onto the document root."""
    dark = is_dark()
    theme = "dark" if dark else "light"
    st.markdown(
        f"""
        <script>
        (function(){{
          document.documentElement.setAttribute('data-theme', '{theme}');
          try {{ sessionStorage.setItem('{THEME_JS_STORAGE_KEY}', '{1 if dark else 0}'); }} catch(e) {{}}
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )


THEME_JS_STORAGE_KEY = "triage_dark_mode"


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>
/* ═══ Color Palette — Light (default) ═══ */
:root {{
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
    --success-text: #047857;
    --warning: #f59e0b;
    --warning-bg: #fef3c7;
    --warning-text: #b45309;
    --danger: #ef4444;
    --danger-bg: #fee2e2;
    --danger-text: #b91c1c;
    --info: #3b82f6;
    --info-bg: #dbeafe;
    --info-text: #1d4ed8;

    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.06);
    --shadow-lg: 0 12px 24px -6px rgb(0 0 0 / 0.14), 0 4px 8px -4px rgb(0 0 0 / 0.08);

    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;

    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    --font-mono: 'JetBrains Mono', 'SFMono-Regular', 'Fira Code', ui-monospace, monospace;
}}

/* ═══ Dark Palette — system-triggered (unless user pinned light) ═══ */
@media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
        --surface-0: #0b1220;
        --surface-50: #111a2b;
        --surface-100: #1e293b;
        --surface-200: #2d3a4f;
        --surface-300: #475569;

        --text-900: #f1f5f9;
        --text-700: #dbe3ee;
        --text-500: #94a3b8;
        --text-400: #64748b;
        --text-300: #475569;

        --brand-50: #1e1b4b;
        --brand-100: #312e81;
        --success-bg: #052e22;
        --success-text: #6ee7b7;
        --warning-bg: #33240a;
        --warning-text: #fcd34d;
        --danger-bg: #3a0d12;
        --danger-text: #fca5a5;
        --info-bg: #0c2540;
        --info-text: #93c5fd;

        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.4);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.5);
        --shadow-lg: 0 12px 24px -6px rgb(0 0 0 / 0.6);
    }}
}}

/* ═══ Dark Palette — manually toggled dark ═══ */
html[data-theme="dark"] {{
    --surface-0: #0b1220;
    --surface-50: #111a2b;
    --surface-100: #1e293b;
    --surface-200: #2d3a4f;
    --surface-300: #475569;

    --text-900: #f1f5f9;
    --text-700: #dbe3ee;
    --text-500: #94a3b8;
    --text-400: #64748b;
    --text-300: #475569;

    --brand-50: #1e1b4b;
    --brand-100: #312e81;
    --success-bg: #052e22;
    --success-text: #6ee7b7;
    --warning-bg: #33240a;
    --warning-text: #fcd34d;
    --danger-bg: #3a0d12;
    --danger-text: #fca5a5;
    --info-bg: #0c2540;
    --info-text: #93c5fd;

    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.4);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.5);
    --shadow-lg: 0 12px 24px -6px rgb(0 0 0 / 0.6);
}}

/* ═══ Base ═══ */
.stApp {{
    font-family: var(--font-sans) !important;
    color: var(--text-700);
    background: var(--surface-50);
}}

html, body, [class*="css"] {{
    font-family: var(--font-sans);
}}

section[data-testid="stSidebar"] {{
    background: var(--surface-0) !important;
    border-right: 1px solid var(--surface-200) !important;
}}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: var(--text-700);
}}

/* ═══ Typography ═══ */
h1, h2, h3, h4 {{
    font-family: var(--font-sans);
    color: var(--text-900);
    letter-spacing: -0.02em;
}}

/* ═══ Shared top bar / breadcrumbs ═══ */
.top-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    padding: 10px 16px;
    margin: -0.6rem 0 20px 0;
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
}}

.crumbs {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    color: var(--text-500);
    flex-wrap: wrap;
}}

.crumbs a {{
    color: var(--brand-500);
    text-decoration: none;
    font-weight: 600;
    cursor: pointer;
}}

.crumbs a:hover {{ text-decoration: underline; }}
.crumb-sep {{ color: var(--text-300); font-size: 0.8rem; }}
.crumb-current {{ color: var(--text-700); font-weight: 700; }}

.top-bar .context {{
    font-size: 0.8rem;
    color: var(--text-500);
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ═══ Sidebar branding ═══ */
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 0 12px 0;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--surface-200);
}}

.brand-mark {{
    width: 40px; height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--brand-500), var(--brand-700));
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem;
    box-shadow: var(--shadow-md);
    flex-shrink: 0;
}}

.brand-word h3 {{
    font-size: 1.05rem; font-weight: 800; color: var(--text-900); margin: 0;
    line-height: 1.15;
}}

.brand-word .subtitle {{
    font-size: 0.72rem; color: var(--text-500); letter-spacing: 0.03em;
}}

.user-chip {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px;
    background: var(--surface-50);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    margin: 4px 0 12px 0;
}}

.user-avatar {{
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, var(--success), var(--info));
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
}}

.user-info .name {{ font-size: 0.85rem; font-weight: 700; color: var(--text-900); }}
.user-info .role {{ font-size: 0.72rem; color: var(--text-500); }}

/* ═══ Health indicator ═══ */
.health-row {{
    display: flex; align-items: center; gap: 8px;
    font-size: 0.78rem; color: var(--text-500);
    padding: 8px 12px;
    background: var(--surface-50);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    margin-bottom: 12px;
}}

.health-dot {{
    width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
}}
.health-dot.healthy {{ background: var(--success); box-shadow: 0 0 0 3px var(--success-bg); }}
.health-dot.degraded {{ background: var(--warning); box-shadow: 0 0 0 3px var(--warning-bg); }}
.health-dot.down {{ background: var(--danger); box-shadow: 0 0 0 3px var(--danger-bg); }}

/* ═══ KPI cards + sparklines ═══ */
.kpi-card {{
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 16px 18px 12px 18px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
    height: 100%;
}}

.kpi-card:hover {{ box-shadow: var(--shadow-md); border-color: var(--brand-500); }}

.kpi-card .label {{
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--text-500); margin-bottom: 6px;
}}

.kpi-card .value {{
    font-size: 1.7rem; font-weight: 800; color: var(--text-900);
    line-height: 1.1; letter-spacing: -0.02em;
}}

.kpi-card .trend {{
    font-size: 0.78rem; font-weight: 600; margin-top: 6px;
    display: flex; align-items: center; gap: 4px;
}}

.trend.up {{ color: var(--success-text); }}
.trend.down {{ color: var(--danger-text); }}
.trend.flat {{ color: var(--text-400); }}

.kpi-card.accent {{ border-left: 4px solid var(--brand-500); }}
.kpi-card.ok {{ border-left: 4px solid var(--success); }}
.kpi-card.warn {{ border-left: 4px solid var(--warning); }}
.kpi-card.bad {{ border-left: 4px solid var(--danger); }}

/* ═══ Section headers ═══ */
.section-header {{
    display: flex; align-items: center; gap: 10px;
    margin: 8px 0 14px 0; padding-bottom: 10px;
    border-bottom: 1px solid var(--surface-200);
}}
.section-header .icon {{ font-size: 1.15rem; }}
.section-header h2 {{ font-size: 1.1rem; font-weight: 700; color: var(--text-900); margin: 0; }}

/* ═══ Cards ═══ */
.s-card {{
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 14px;
}}
.s-card-elevated {{
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    box-shadow: var(--shadow-md);
    margin-bottom: 14px;
}}

/* ═══ Badges ═══ */
.badge {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.03em;
    white-space: nowrap;
}}
.badge-resolved {{ background: var(--success-bg); color: var(--success-text); }}
.badge-approved {{ background: var(--success-bg); color: var(--success-text); }}
.badge-escalated {{ background: var(--warning-bg); color: var(--warning-text); }}
.badge-processing {{ background: var(--info-bg); color: var(--info-text); }}
.badge-awaiting_review {{ background: var(--brand-100); color: var(--brand-700); }}
.badge-pending {{ background: var(--surface-100); color: var(--text-500); }}
.badge-open {{ background: var(--surface-100); color: var(--text-500); }}
.badge-classified {{ background: var(--info-bg); color: var(--info-text); }}
.badge-failed {{ background: var(--danger-bg); color: var(--danger-text); }}
.badge-rejected {{ background: var(--danger-bg); color: var(--danger-text); }}

.badge-urgent, .badge-critical {{ background: var(--danger-bg); color: var(--danger-text); }}
.badge-high {{ background: var(--warning-bg); color: #c2410c; }}
html[data-theme="dark"] .badge-high {{ color: var(--warning-text); }}
.badge-medium {{ background: var(--warning-bg); color: var(--warning-text); }}
.badge-low {{ background: var(--success-bg); color: var(--success-text); }}

/* ═══ Confidence meter ═══ */
.confidence-bar-bg {{
    width: 100%; height: 9px;
    background: var(--surface-200);
    border-radius: 999px; overflow: hidden;
}}
.confidence-bar-fill {{
    height: 100%; border-radius: 999px;
    transition: width 0.5s ease;
}}
.confidence-high {{ background: var(--success); }}
.confidence-medium {{ background: var(--warning); }}
.confidence-low {{ background: var(--danger); }}

.confidence-large {{ height: 14px; }}

/* ═══ Trace timeline ═══ */
.trace-timeline {{
    display: flex; align-items: stretch; gap: 0;
    padding: 8px 0 12px 0;
    overflow-x: auto;
}}
.trace-step-card {{
    display: flex; flex-direction: column;
    min-width: 150px; max-width: 190px;
    border: 1px solid var(--surface-200);
    background: var(--surface-0);
    border-radius: var(--radius-md);
    padding: 12px 12px;
    position: relative; flex-shrink: 0;
    box-shadow: var(--shadow-sm);
    cursor: pointer;
    transition: border-color 0.15s ease, transform 0.15s ease;
}}
.trace-step-card:hover {{ border-color: var(--brand-500); transform: translateY(-1px); }}

.trace-step-head {{
    display: flex; align-items: center; gap: 6px; margin-bottom: 6px;
}}

.trace-status-icon {{
    width: 20px; height: 20px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; color: #fff; flex-shrink: 0;
}}
.trace-status-icon.completed {{ background: var(--success); }}
.trace-status-icon.failed {{ background: var(--danger); }}
.trace-status-icon.skipped {{ background: var(--surface-300); color: var(--text-500); }}
.trace-status-icon.running {{ background: var(--info); animation: tracePulse 1.5s infinite; }}
.trace-status-icon.pending {{ background: var(--surface-300); color: var(--text-500); }}

.trace-step-name {{
    font-size: 0.8rem; font-weight: 700; color: var(--text-900);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.trace-step-status {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.03em; color: var(--text-400);
}}
.trace-step-dur {{
    font-size: 0.72rem; color: var(--text-500);
}}
.trace-step-output {{
    font-size: 0.72rem; color: var(--text-500);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-top: 4px;
}}
.trace-step-summary {{
    margin-top: 8px; font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-700);
}}

.trace-connector {{
    align-self: center; flex-shrink: 0;
    font-size: 1rem; color: var(--text-300);
    padding: 0 2px;
}}
html[data-theme="dark"] .trace-connector {{ color: var(--surface-300); }}

@keyframes tracePulse {{
    0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }}
}}

/* ═══ Ticket rows ═══ */
.ticket-row {{
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
.ticket-row:hover {{ border-color: var(--brand-500); box-shadow: var(--shadow-md); }}

.ticket-row .ticket-top {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 10px; flex-wrap: wrap; margin-bottom: 6px;
}}
.ticket-id {{ font-family: var(--font-mono); font-size: 0.82rem; font-weight: 700; color: var(--brand-500); }}
.ticket-meta {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }}
.ticket-preview {{ font-size: 0.88rem; color: var(--text-700); line-height: 1.5; }}
.ticket-sub {{ font-size: 0.76rem; color: var(--text-400); margin-top: 6px; display: flex; gap: 12px; flex-wrap: wrap; }}

.ticket-row.kb-focused {{ border-color: var(--brand-500); box-shadow: 0 0 0 3px var(--brand-500); }}

/* ═══ Empty / error / partial states ═══ */
.empty-state {{
    text-align: center; padding: 44px 24px;
    color: var(--text-500);
    border: 1px dashed var(--surface-300);
    border-radius: var(--radius-md);
    background: var(--surface-0);
}}
.empty-state .empty-icon {{ font-size: 2.8rem; margin-bottom: 10px; opacity: 0.6; }}
.empty-state h3 {{ font-size: 1.05rem; font-weight: 700; color: var(--text-700); margin-bottom: 4px; }}
.empty-state p {{ font-size: 0.88rem; margin: 0 0 14px 0; }}

.network-banner {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; flex-wrap: wrap;
    padding: 12px 16px; margin-bottom: 16px;
    border-radius: var(--radius-md);
    background: var(--danger-bg);
    border: 1px solid var(--danger);
    color: var(--danger-text);
    font-size: 0.88rem; font-weight: 600;
}}

.partial-banner {{
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px; margin-bottom: 14px;
    border-radius: var(--radius-md);
    background: var(--warning-bg);
    border: 1px solid var(--warning);
    color: var(--warning-text);
    font-size: 0.85rem;
}}

.placeholder-text {{
    color: var(--text-400); font-style: italic; font-size: 0.88rem;
}}

/* ═══ Sticky actions bar ═══ */
.actions-bar {{
    position: sticky; bottom: 0;
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 12px 16px; margin-top: 20px;
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg);
    z-index: 50;
}}

/* ═══ Filter bar ═══ */
.filter-bar {{
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    padding: 12px 16px;
    background: var(--surface-50);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    margin-bottom: 16px;
}}

/* ═══ Login ═══ */
.login-page {{
    min-height: calc(100vh - 2rem);
    display: flex; align-items: center; justify-content: center;
    background:
        radial-gradient(1200px 500px at 15% -10%, var(--brand-100), transparent 60%),
        radial-gradient(1000px 500px at 110% 10%, var(--info-bg), transparent 55%),
        var(--surface-50);
    border-radius: var(--radius-lg);
    padding: 24px;
}}

.login-card {{
    background: var(--surface-0);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-lg);
    padding: 36px 44px;
    box-shadow: var(--shadow-lg);
    max-width: 440px; width: 100%;
}}
.login-card .brand-icon {{ text-align: center; font-size: 2.6rem; margin-bottom: 10px; }}
.login-card h1 {{ font-size: 1.6rem; font-weight: 800; color: var(--text-900); text-align: center; margin: 0 0 4px 0; }}
.login-card .login-subtitle {{ text-align: center; color: var(--text-500); font-size: 0.9rem; margin-bottom: 24px; }}

.field-error {{
    color: var(--danger-text);
    background: var(--danger-bg);
    border: 1px solid var(--danger);
    border-radius: var(--radius-sm);
    padding: 8px 12px;
    font-size: 0.8rem; font-weight: 600;
    margin-top: 8px;
}}

/* ═══ Buttons ═══ */
.stButton > button {{
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
}}
.stButton > button[kind="primary"] {{
    background: var(--brand-500) !important;
    border-color: var(--brand-500) !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: var(--brand-600) !important;
    border-color: var(--brand-600) !important;
}}

/* ═══ Tabs ═══ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: var(--surface-100);
    border-radius: var(--radius-sm);
    padding: 3px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}}
.stTabs [aria-selected="true"] {{
    background: var(--surface-0) !important;
    box-shadow: var(--shadow-sm) !important;
}}

/* ═══ Table / dataframe ═══ */
.stDataFrame {{ border-radius: var(--radius-md) !important; overflow: hidden; }}

/* ═══ Expander ═══ */
.streamlit-expanderHeader {{ font-weight: 600 !important; font-size: 0.9rem !important; }}

/* ═══ Divider ═══ */
hr {{ border: none; border-top: 1px solid var(--surface-200); margin: 18px 0; }}

/* ═══ Shimmer skeleton ═══ */
.skeleton {{
    background: linear-gradient(90deg, var(--surface-100) 25%, var(--surface-200) 50%, var(--surface-100) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.3s infinite;
    border-radius: var(--radius-md);
}}
.skeleton-card {{ height: 96px; margin-bottom: 12px; }}
.skeleton-line {{ height: 14px; margin-bottom: 8px; }}
@keyframes shimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* ═══ Misc ═══ */
.mono {{ font-family: var(--font-mono); }}
.small {{ font-size: 0.8rem; }}
.muted {{ color: var(--text-500); }}
.flex-row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.space-between {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}

/* Focus visibility for keyboard nav */
:focus-visible {{
    outline: 2px solid var(--brand-500);
    outline-offset: 2px;
}}

/* ═══ Streamlit chrome cleanup ═══ */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ═══ Scrollbar ═══ */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--surface-100); }}
::-webkit-scrollbar-thumb {{ background: var(--surface-300); border-radius: 3px; }}

/* ═══ Responsive ═══ */
@media (max-width: 768px) {{
    .login-card {{ padding: 26px 22px; }}
    .kpi-card .value {{ font-size: 1.35rem; }}
    .trace-step-card {{ min-width: 130px; }}
}}
</style>
"""