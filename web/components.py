"""
web/components.py - Bento UI 卡片、表格、指標與 Plotly 圖表元件
===================================================================================
Provides Ultra-High-Contrast Pro Dark Financial Terminal CSS (TradingView / Bloomberg Style),
horizontal factor exposure bars, macro ribbon CSS, and reusable UI helpers for effortless readability.
"""

from __future__ import annotations

import streamlit as st


def inject_terminal_and_portal_css() -> None:
    """
    Inject Ultra-High-Contrast Pro Dark Financial Terminal CSS.
    Enforces crystal-clear white & gold typography (#FFFFFF, #F1F5F9, #FBBF24) on rich obsidian
    cards (#161F30) with solid contrast borders (#334155), completely eliminating faint or washed-out text.
    """
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Noto+Sans+TC:wght@400;500;700;900&family=JetBrains+Mono:wght@500;600;700;800&display=swap');

@keyframes liveDotPulse {
    0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    50%      { transform: scale(1.2); opacity: 0.85; box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
}

/* 1. Global Viewport & Background */
html, body, [class*="css"], .stApp {
    background-color: #0B0F19 !important;
    color: #FFFFFF !important;
    font-family: 'Inter', 'Noto Sans TC', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 2. Global High-Contrast Typography */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

p, span, div, li {
    color: #F1F5F9;
    font-weight: 500;
}

strong, b {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* 3. Sidebar High-Contrast Styling */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1.5px solid #374151 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #F9FAFB !important;
    font-weight: 600 !important;
}

/* 4. Streamlit Form Widgets & Inputs */
label, [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-size: 0.94rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px !important;
}

/* 5. Top Macro Status Ribbon */
.macro-ribbon {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    background: linear-gradient(90deg, #111827 0%, #161F30 100%);
    border: 1.5px solid #334155;
    border-left: 4px solid #38BDF8;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}

.macro-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05rem;
    font-weight: 800;
    color: #FFFFFF;
}

.live-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #10B981;
    display: inline-block;
    animation: liveDotPulse 2s infinite;
}

.macro-pills {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.macro-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.86rem;
    color: #E2E8F0;
    background: rgba(30, 41, 59, 0.75);
    padding: 5px 12px;
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

/* 6. Metrics Cards (High Luminance) */
div[data-testid="stMetric"] {
    background: #161F30 !important;
    border: 1.5px solid #334155 !important;
    border-top: 3.5px solid #38BDF8 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
}

div[data-testid="stMetricLabel"] p {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    color: #CBD5E1 !important;
}

div[data-testid="stMetricValue"] div {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums !important;
    font-size: 1.50rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
}

div[data-testid="stMetricDelta"] div {
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}

/* 7. Bento Grid Cards & Hero Verdict Banner */
.bento-card {
    position: relative;
    overflow: hidden;
    background: #161F30 !important;
    border: 1.5px solid #334155 !important;
    border-top: 3.5px solid #38BDF8 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.40) !important;
}

.verdict-banner {
    background: linear-gradient(135deg, #161F30 0%, #1E293B 100%) !important;
    border: 1.5px solid #38BDF8 !important;
    border-radius: 14px !important;
    padding: 16px 22px !important;
    margin-bottom: 16px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 14px !important;
}

.bento-title {
    font-size: 0.95rem !important;
    letter-spacing: 0.4px !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    margin-bottom: 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.18) !important;
    padding-bottom: 8px !important;
}

.exec-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 12px !important;
    margin-top: 10px !important;
}

.exec-cell {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
}

.exec-label {
    font-size: 0.82rem !important;
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}

.exec-val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.18rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
}

.mono-num {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums !important;
    font-weight: 700 !important;
    color: #FFFFFF;
}

/* 8. Semantic Badges (Bold & Readable) */
.badge-bull {
    background: rgba(16, 185, 129, 0.22) !important;
    color: #34D399 !important;
    border: 1.5px solid #10B981 !important;
    padding: 5px 14px !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}

.badge-bear {
    background: rgba(244, 63, 94, 0.22) !important;
    color: #FB7185 !important;
    border: 1.5px solid #F43F5E !important;
    padding: 5px 14px !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}

.badge-warn {
    background: rgba(245, 158, 11, 0.22) !important;
    color: #FBBF24 !important;
    border: 1.5px solid #F59E0B !important;
    padding: 5px 14px !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}

/* 9. Tabs High-Contrast Bar */
button[data-baseweb="tab"] {
    font-weight: 700 !important;
    font-size: 1.00rem !important;
    color: #CBD5E1 !important;
    padding: 10px 18px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom-color: #38BDF8 !important;
    font-weight: 800 !important;
}

/* 10. Expanders */
div[data-testid="stExpander"] {
    background: #161F30 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
}

div[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
}

div[data-testid="stExpander"] summary:hover {
    color: #38BDF8 !important;
}

div[data-testid="stExpander"] * {
    color: #F1F5F9 !important;
}

/* 11. Tables & Markdown Tables */
table {
    color: #FFFFFF !important;
    width: 100% !important;
}

th {
    background-color: #1E293B !important;
    color: #38BDF8 !important;
    font-weight: 700 !important;
    font-size: 0.90rem !important;
    border-bottom: 2px solid #334155 !important;
    padding: 8px 12px !important;
}

td {
    color: #F8FAFC !important;
    font-size: 0.88rem !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12) !important;
    padding: 8px 12px !important;
}

.factor-row {
    margin-bottom: 12px;
}

.factor-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 5px;
    color: #F1F5F9;
}

.factor-track {
    width: 100%;
    height: 9px;
    background: #1E293B;
    border-radius: 999px;
    overflow: hidden;
    position: relative;
    border: 1px solid #334155;
}

.factor-fill-pos {
    height: 100%;
    background: linear-gradient(90deg, #0EA5E9, #10B981);
    border-radius: 999px;
}

.factor-fill-neg {
    height: 100%;
    background: linear-gradient(90deg, #F59E0B, #F43F5E);
    border-radius: 999px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_factor_bars_html(
    title: str,
    subtitle: str,
    factor_dict: dict,
    max_scale: float = 3.0,
    unit: str = "σ",
) -> str:
    """
    Render sleek, high-contrast horizontal factor exposure bars inside a Bento card.
    Uses zero leading indentation so CommonMark never wraps inner tags into <pre><code>.
    """
    rows_list: list[str] = []
    for label, val in factor_dict.items():
        val_f = float(val)
        pct = min(100.0, max(6.0, (abs(val_f) / max_scale) * 100.0))
        fill_cls = "factor-fill-pos" if val_f >= 0 else "factor-fill-neg"
        val_color = "#34D399" if val_f > 0 else ("#FB7185" if val_f < 0 else "#CBD5E1")
        dir_tag = "偏多正向" if val_f > 0.05 else ("偏空負向" if val_f < -0.05 else "中性均衡")
        row_str = (
            f'<div class="factor-row">'
            f'<div class="factor-header">'
            f'<span style="color:#F1F5F9; font-weight:600;">{label} <small style="color:#94A3B8; font-size:0.78rem;">({dir_tag})</small></span>'
            f'<span class="mono-num" style="color: {val_color}; font-size:0.95rem; font-weight:800;">{val_f:+.2f} {unit}</span>'
            f'</div>'
            f'<div class="factor-track">'
            f'<div class="{fill_cls}" style="width: {pct:.1f}%;"></div>'
            f'</div>'
            f'</div>'
        )
        rows_list.append(row_str)

    rows_html = "".join(rows_list)
    return (
        f'<div class="bento-card">'
        f'<div class="bento-title">'
        f'<span style="color:#FFFFFF; font-size:0.92rem; font-weight:800;">{title}</span>'
        f'<span style="color: #38BDF8; font-size: 0.85rem; font-weight: 700;">{subtitle}</span>'
        f'</div>'
        f'{rows_html}'
        f'</div>'
    )
