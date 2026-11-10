"""
web/app.py - 清晰直觀的港股機構級量化決策終端 (Clean & User-Friendly Edition)
===================================================================================
Preserves all institutional mathematical engines (t-1 price decoupling, HKEX statutory fees,
Almgren-Chriss impact slippage, 365D TTM dividend yield engine, Ledoit-Wolf/RMT denoising,
and Britten-Jones de-biased backtesting) while presenting a clean, progressive-disclosure UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_CURRENT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _CURRENT_DIR if (_CURRENT_DIR / "config" / "settings.py").exists() else _CURRENT_DIR.parent
_PKG_STR = str(_PACKAGE_ROOT)

sys.path = [p for p in sys.path if Path(p).resolve() != _PACKAGE_ROOT.parent]
if _PKG_STR not in sys.path:
    sys.path.insert(0, _PKG_STR)

for _mod_name in list(sys.modules.keys()):
    if _mod_name.split(".")[0] in ("config", "data", "engine", "web", "hk_quant_system"):
        _mod_obj = sys.modules.get(_mod_name)
        _mod_file = getattr(_mod_obj, "__file__", None)
        if _mod_file and not str(Path(_mod_file).resolve()).startswith(_PKG_STR):
            del sys.modules[_mod_name]

from config.settings import (
    DEFAULT_ALMGREN_GAMMA,
    DEFAULT_BROKER_COMM_RATE,
    DEFAULT_MIN_BROKER_COMM_HKD,
    HK_DEFAULT_LOT_SIZES,
    is_hk_short_eligible,
    load_portfolio_config,
    normalize_hk_ticker,
    save_portfolio_config,
)
from data.futu_gateway import (
    check_futu_opend_connection,
    get_futu_order_book,
)
from data.loader import fetch_hk_macro_regime
from engine.alpha import (
    analyze_stock_trend,
    generate_day_trading_signal,
)
from engine.risk import analyze_portfolio_health
from engine.backtest import evaluate_historical_prediction_accuracy
from web.components import (
    inject_terminal_and_portal_css,
    render_factor_bars_html,
)

st.set_page_config(
    page_title="香港股市量化分析終端 | HKEX AlphaEngine",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_terminal_and_portal_css()


# ==========================================
# Cached Quantitative Data Pipelines
# ==========================================
@st.cache_data(ttl=5, show_spinner=False)
def cached_order_book(ticker: str, host: str, port: int, spot_price: float, lot_size: int):
    return get_futu_order_book(
        ticker,
        num=10,
        host=host,
        port=port,
        spot_price_fallback=spot_price,
        lot_size_fallback=lot_size,
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_macro_regime() -> dict:
    return fetch_hk_macro_regime()


@st.cache_data(ttl=60, show_spinner=False)
def cached_day_trading_signal(
    ticker: str,
    min_adv_hkd: float,
    interval: str,
    broker_comm_rate: float,
    min_broker_comm_hkd: float,
    almgren_gamma: float,
):
    return generate_day_trading_signal(
        ticker,
        min_adv_hkd=min_adv_hkd,
        interval=interval,
        broker_comm_rate=broker_comm_rate,
        min_broker_comm_hkd=min_broker_comm_hkd,
        almgren_gamma=almgren_gamma,
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_stock_trend(ticker: str):
    return analyze_stock_trend(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_portfolio_health(
    holdings_json: str,
    total_cap: float,
    rf: float,
    max_w: float,
    min_adv: float,
    method: str,
    trials: int,
    cov_est: str,
    max_to: float,
    comm_rate: float,
    min_comm: float,
    gamma: float,
):
    holdings = json.loads(holdings_json)
    return analyze_portfolio_health(
        holdings,
        total_capital=total_cap,
        rf_rate=rf,
        max_single_weight=max_w,
        min_adv_hkd=min_adv,
        alloc_method=method,
        num_trials=trials,
        cov_estimator=cov_est,
        max_turnover=max_to,
        broker_comm_rate=comm_rate,
        min_broker_comm_hkd=min_comm,
        almgren_gamma=gamma,
    )


@st.cache_data(ttl=120, show_spinner=False)
def cached_historical_accuracy(
    ticker: str,
    horizon: int,
    history_period: str,
    atr_stop_mult: float,
    signal_z_threshold: float,
    comm_rate: float,
    min_comm: float,
    gamma: float,
):
    return evaluate_historical_prediction_accuracy(
        ticker=ticker,
        forward_horizon=horizon,
        history_period=history_period,
        atr_stop_mult=atr_stop_mult,
        signal_z_threshold=signal_z_threshold,
        broker_comm_rate=comm_rate,
        min_broker_comm_hkd=min_comm,
        almgren_gamma=gamma,
    )


# ==========================================
# Reactive State & Clean Sidebar Navigation
# ==========================================
if "portfolio_config" not in st.session_state:
    st.session_state.portfolio_config = load_portfolio_config()
if "last_saved_hash" not in st.session_state:
    st.session_state.last_saved_hash = json.dumps(
        st.session_state.portfolio_config, sort_keys=True, ensure_ascii=False
    )

cfg = st.session_state.portfolio_config

QUICK_WATCHLIST = {
    "0005.HK": "0005.HK 匯豐控股",
    "0700.HK": "0700.HK 騰訊控股",
    "9988.HK": "9988.HK 阿里巴巴-W",
    "3690.HK": "3690.HK 美團-W",
    "1810.HK": "1810.HK 小米集團-W",
    "0388.HK": "0388.HK 香港交易所",
    "0941.HK": "0941.HK 中國移動",
    "1211.HK": "1211.HK 比亞迪股份",
}

with st.sidebar:
    st.markdown("### 🎯 港股分析控制台")

    preset_t = st.selectbox(
        "熱門港股快速選擇",
        options=list(QUICK_WATCHLIST.keys()),
        format_func=lambda k: QUICK_WATCHLIST[k],
        index=0,
    )
    custom_t = st.text_input("或輸入任意港股代號 (例如 5.hk / 0823.HK)", value=preset_t).strip().upper()
    active_ticker = normalize_hk_ticker(custom_t) if custom_t else preset_t

    interval_choice = st.segmented_control(
        "K 線分析週期 (Tab 1)",
        options=["15m", "5m", "1d"],
        default="15m",
    ) or "15m"

    if st.button("🔄 重新整理最新市場報價", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.toast("⚡ 已清除快取並重新抓取最新市場數據！")
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💼 投資組合快速設定 (Tab 3)")
    total_capital_hkd = st.number_input(
        "目標總資金 (HKD)",
        min_value=10000.0,
        value=float(cfg.get("total_capital_hkd", 300000.0)),
        step=10000.0,
    )

    method_options = {
        "hrp": "HRP 階層風險平價 (推薦穩健)",
        "risk_parity": "Risk Parity 等風險貢獻",
        "min_variance": "Minimum Variance 最小波動",
    }
    curr_method = cfg.get("alloc_method", "hrp")
    alloc_method = st.selectbox(
        "資產配置模型",
        options=list(method_options.keys()),
        format_func=lambda k: method_options[k],
        index=list(method_options.keys()).index(curr_method) if curr_method in method_options else 0,
    )

    # All advanced mathematical parameters tucked neatly inside an expander
    with st.expander("⚙️ 進階量化與港交所費率設定", expanded=False):
        cov_options = {
            "ledoit_wolf": "Ledoit-Wolf 壓縮共變異數",
            "rmt": "RMT 隨機矩陣特徵值去噪",
        }
        curr_cov = cfg.get("cov_estimator", "ledoit_wolf")
        cov_estimator = st.selectbox(
            "矩陣去噪演算法",
            options=list(cov_options.keys()),
            format_func=lambda k: cov_options[k],
            index=list(cov_options.keys()).index(curr_cov) if curr_cov in cov_options else 0,
        )
        rf_rate_pct = st.slider(
            "無風險利率 Rf (% 年化)",
            min_value=0.0,
            max_value=8.0,
            value=float(cfg.get("risk_free_rate", 0.04)) * 100.0,
            step=0.25,
        )
        max_pos_pct = st.slider(
            "單一持股權重上限 (%)",
            min_value=15.0,
            max_value=60.0,
            value=float(cfg.get("max_single_position_pct", 0.35)) * 100.0,
            step=5.0,
        )
        max_turnover_pct = st.slider(
            "單次調倉換手率上限 (%)",
            min_value=10.0,
            max_value=100.0,
            value=float(cfg.get("max_turnover", 0.40)) * 100.0,
            step=5.0,
        )
        broker_comm_bps = st.number_input(
            "經紀佣金率 (% 預設 0.03%)",
            min_value=0.0,
            max_value=0.50,
            value=float(cfg.get("broker_comm_rate", DEFAULT_BROKER_COMM_RATE)) * 100.0,
            step=0.01,
            format="%.3f",
        )
        min_broker_comm = st.number_input(
            "最低經紀佣金 (HKD 預設 $15)",
            min_value=0.0,
            max_value=200.0,
            value=float(cfg.get("min_broker_comm_hkd", DEFAULT_MIN_BROKER_COMM_HKD)),
            step=5.0,
        )
        almgren_gamma = st.slider(
            "Almgren-Chriss 衝擊係數 γ",
            min_value=0.10,
            max_value=1.00,
            value=float(cfg.get("almgren_gamma", DEFAULT_ALMGREN_GAMMA)),
            step=0.05,
        )
        min_adv_input = st.number_input(
            "最低 20 日均成交額門檻 (HKD)",
            min_value=0,
            value=int(cfg.get("min_adv_hkd", 5000000)),
            step=1000000,
        )
        num_trials = st.number_input(
            "回測多重檢定次數 (DSR K)",
            min_value=1,
            max_value=500,
            value=int(cfg.get("num_backtest_trials", 5)),
            step=1,
        )

    with st.expander("🔌 富途 (Futu OpenD) L2 報價連線", expanded=False):
        futu_enabled = st.checkbox(
            "啟用富途 Level 2 即時報價",
            value=bool(cfg.get("enable_futu", False)),
            key="futu_enable_toggle",
        )
        f_host = st.text_input("FutuOpenD 位址", value=str(cfg.get("futu_host", "127.0.0.1")))
        f_port = st.number_input("連接埠 (Port)", min_value=1000, max_value=65535, value=int(cfg.get("futu_port", 11111)))
        cfg["enable_futu"] = futu_enabled
        cfg["futu_host"] = f_host
        cfg["futu_port"] = int(f_port)
        if st.button("測試 FutuOpenD 連線", use_container_width=True):
            if check_futu_opend_connection(f_host, int(f_port), timeout=0.6):
                st.success("🟢 FutuOpenD 已連線！")
            else:
                st.info("⚪ 未偵測到 FutuOpenD，自動使用 Yahoo Finance 數據。")

    # Sync config changes
    cfg["total_capital_hkd"] = float(total_capital_hkd)
    cfg["risk_free_rate"] = float(rf_rate_pct) / 100.0
    cfg["max_single_position_pct"] = float(max_pos_pct) / 100.0
    cfg["max_turnover"] = float(max_turnover_pct) / 100.0
    cfg["min_adv_hkd"] = float(min_adv_input)
    cfg["alloc_method"] = alloc_method
    cfg["cov_estimator"] = cov_estimator
    cfg["broker_comm_rate"] = float(broker_comm_bps) / 100.0
    cfg["min_broker_comm_hkd"] = float(min_broker_comm)
    cfg["almgren_gamma"] = float(almgren_gamma)
    cfg["num_backtest_trials"] = int(num_trials)

    new_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    if new_hash != st.session_state.last_saved_hash:
        if save_portfolio_config(cfg):
            st.session_state.last_saved_hash = new_hash


# ==========================================
# Top Macro Regime Status Ribbon
# ==========================================
macro_info = cached_macro_regime()
hsi_dist = macro_info.get(
    "hsi_dist_pct",
    (macro_info["hsi_close"] - macro_info["hsi_sma200"]) / (macro_info["hsi_sma200"] + 1e-9),
)
regime_badge = (
    '<span class="badge-bull">牛市偏多環境</span>'
    if not macro_info["bear_market"]
    else '<span class="badge-bear">熊市防禦環境</span>'
)

st.markdown(
    f"""
<div class="macro-ribbon">
    <div class="macro-brand">
        <span class="live-dot"></span>
        <span>HKEX 港股量化決策系統</span>
        <span style="color: #38BDF8; font-size: 0.85rem; font-weight: 600;">當前標的：{active_ticker}</span>
    </div>
    <div class="macro-pills">
        <div class="macro-item">
            <span>恆生指數:</span>
            <span class="mono-num">{macro_info['hsi_close']:,.0f}</span>
            <span class="mono-num" style="color: {'#34D399' if hsi_dist >= 0 else '#FB7185'};">({hsi_dist*100:+.1f}% vs 200MA)</span>
        </div>
        <div class="macro-item">
            <span>美債 10Y 殖利率:</span>
            <span class="mono-num">{macro_info['tnx_yield']:.2f}%</span>
        </div>
        <div class="macro-item">
            <span>大盤狀態:</span>
            {regime_badge}
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# Main 4 Clean Workspace Tabs
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    f"⚡ 1. 即時報價與買賣點 ({active_ticker})",
    f"🔍 2. 趨勢勝率與基本面 ({active_ticker})",
    "💼 3. 投資組合與智能調倉",
    f"🎯 4. 歷史勝率回測檢驗 ({active_ticker})",
])


# ==========================================
# TAB 1: Clean Intraday Execution & Action Plan
# ==========================================
with tab1:
    with st.spinner(f"正在計算 {active_ticker} ({interval_choice}) 實盤買賣點與交易成本..."):
        day_res, day_err = cached_day_trading_signal(
            active_ticker,
            float(min_adv_input),
            interval_choice,
            float(cfg["broker_comm_rate"]),
            float(cfg["min_broker_comm_hkd"]),
            float(cfg["almgren_gamma"]),
        )

    if day_err or not day_res:
        st.error(f"無法載入 {active_ticker}：{day_err}")
    else:
        # Row 1: 6 Core Metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("最新實盤現價", f"${day_res['price']:.2f}", f"決策價 ${day_res['decision_price']:.2f}")
        m2.metric("次根開盤預估撮合價", f"${day_res['execution_price']:.2f}", f"含滑價 {day_res['microstructure']['slippage_bps']:.1f} bps")
        m3.metric("AI 上漲趨勢機率", f"{day_res['up_prob']*100:.1f}%", f"合成 Z {day_res['composite_zscore']:+.2f}σ")
        m4.metric("日內 VWAP 均價 (09:30重置)", f"${day_res['vwap']:.2f}", f"{((day_res['decision_price']-day_res['vwap'])/day_res['vwap'])*100:+.2f}%")
        m5.metric("時段標準化量能分位", f"P{day_res['vol_pct_rank']:.0f}", f"{day_res['vol_ratio']:.2f}x 同刻均量")
        m6.metric(f"每手入場市值 ({day_res['lot_size']}股)", f"${day_res['lot_cost_hkd']:,.0f}", f"法定 Tick ${day_res['tick_size']}")

        # Row 2: Clear Action Plan Card + Factor Summary Bars
        b_left, b_right = st.columns([1.15, 1.15])

        with b_left:
            prob_val = day_res["up_prob"]
            sig_badge_cls = "badge-bull" if prob_val >= 0.60 else ("badge-bear" if prob_val <= 0.40 else "badge-warn")
            sfc_badge = (
                '<span class="badge-bull">可沽空名單</span>'
                if day_res["short_eligible"]
                else '<span class="badge-warn">僅限做多</span>'
            )
            mc = day_res["microstructure"]
            st.markdown(
                f"""
            <div class="bento-card">
                <div class="bento-title">
                    <span>🎯 實盤操作建議與止損止盈參考點位</span>
                    {sfc_badge}
                </div>
                <div style="margin: 6px 0 12px 0;">
                    <span class="{sig_badge_cls}">{day_res['action']}</span>
                </div>
                <div class="exec-grid">
                    <div class="exec-cell">
                        <div class="exec-label">建議進場參考價 (次根開盤+滑價)</div>
                        <div class="exec-val" style="color:#38BDF8;">${day_res['execution_price']:.2f}</div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">日內多空分水嶺 (VWAP 均價)</div>
                        <div class="exec-val" style="color:#FBBF24;">${day_res['vwap']:.2f} <span style="font-size:0.82rem; color:#CBD5E1;">(RSI: {day_res['rsi']:.1f})</span></div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">建議止損防守價 (ATR 動態止損)</div>
                        <div class="exec-val" style="color: #FB7185;">${day_res['stop_loss_price']:.2f}</div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">建議目標止盈價 (1:2 風報比)</div>
                        <div class="exec-val" style="color: #34D399;">${day_res['take_profit_price']:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:12px; padding-top:10px; border-top:1px solid #334155; font-size:0.82rem; color:#CBD5E1; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                    <span>預估單筆總成本: <strong class="mono-num" style="color:#FFFFFF;">${mc['total_frictions_hkd']:.1f} ({mc['total_cost_bps']:.1f} bps)</strong></span>
                    <span>其中印花稅: <strong class="mono-num">${mc['stamp_duty_hkd']:.0f}</strong></span>
                    <span>佣金+規費: <strong class="mono-num">${mc['broker_commission_hkd']+mc['sfc_levy_hkd']+mc['afrc_levy_hkd']+mc['hkex_fee_hkd']:.1f}</strong></span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with b_right:
            st.markdown(
                render_factor_bars_html(
                    title="📊 短線五大動能指標強弱一覽",
                    subtitle=f"綜合評分 Z: {day_res['composite_zscore']:+.2f}σ (正值偏多 / 負值偏空)",
                    factor_dict=day_res["factor_zscores"],
                    max_scale=3.0,
                    unit="σ",
                ),
                unsafe_allow_html=True,
            )

        # Row 3: Clean Price, VWAP & Volume Chart
        df_d = day_res["history"]
        fig_3p = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.74, 0.26],
            subplot_titles=(
                f"{day_res['ticker']} K線走勢、日內 VWAP 均價線與 ATR 通道 ({day_res['interval']})",
                "成交量與 20 期均量",
            ),
        )

        fig_3p.add_trace(
            go.Candlestick(
                x=df_d.index,
                open=df_d["Open"],
                high=df_d["High"],
                low=df_d["Low"],
                close=df_d["Close"],
                name="K線",
                increasing_line_color="#10B981",
                decreasing_line_color="#F43F5E",
            ),
            row=1,
            col=1,
        )

        if "VWAP" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["VWAP"],
                    mode="lines",
                    name="VWAP 均價線",
                    line=dict(color="#F59E0B", width=2.2),
                ),
                row=1,
                col=1,
            )

        if "ATR_Upper" in df_d.columns and "ATR_Lower" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["ATR_Upper"],
                    mode="lines",
                    name="+1.5 ATR 通道",
                    line=dict(color="rgba(56, 189, 248, 0.35)", width=1, dash="dot"),
                ),
                row=1,
                col=1,
            )
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["ATR_Lower"],
                    mode="lines",
                    name="-1.5 ATR 通道",
                    line=dict(color="rgba(56, 189, 248, 0.35)", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(56, 189, 248, 0.05)",
                ),
                row=1,
                col=1,
            )

        vol_colors = np.where(df_d["Close"] >= df_d["Open"], "#10B981", "#F43F5E")
        fig_3p.add_trace(
            go.Bar(x=df_d.index, y=df_d["Volume"], name="成交量", marker_color=vol_colors, opacity=0.85),
            row=2,
            col=1,
        )
        if "Vol_MA20" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["Vol_MA20"],
                    mode="lines",
                    name="20期均量",
                    line=dict(color="#0284C7", width=1.4),
                ),
                row=2,
                col=1,
            )

        fig_3p.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#161F30",
            plot_bgcolor="#0B0F19",
            font=dict(color="#FFFFFF", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=15, r=15, t=35, b=15),
        )
        fig_3p.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
        if day_res["interval"] in ("5m", "15m"):
            fig_3p.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour"),
                    dict(bounds=[12, 13], pattern="hour"),
                ]
            )
        fig_3p.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
        st.plotly_chart(fig_3p, use_container_width=True)

        # Progressive Disclosure: Level 2 Order Book tucked neatly in an expander
        with st.expander("🏛️ 展開檢視：港股 Level 2 買賣十檔盤口深度與微觀價差", expanded=False):
            ob_res = cached_order_book(
                ticker=active_ticker,
                host=cfg.get("futu_host", "127.0.0.1"),
                port=int(cfg.get("futu_port", 11111)),
                spot_price=float(day_res["price"]),
                lot_size=int(day_res["lot_size"]),
            )
            obi_color = "#34D399" if ob_res["order_book_imbalance"] >= 0 else "#FB7185"
            obi_text = "買盤佔優" if ob_res["order_book_imbalance"] >= 0 else "賣盤佔優"
            st.markdown(
                f"最優買賣價差: **${ob_res['spread']:.3f}** ({ob_res['spread_ticks']} Tick) ｜ "
                f"微觀中間價: **${ob_res['micro_price']:.3f}** ｜ "
                f"盤口力量失衡 (OBI): <span style='color:{obi_color}; font-weight:700;'>{ob_res['order_book_imbalance']*100:+.1f}% ({obi_text})</span> ｜ "
                f"數據來源: `{ob_res['source']}`",
                unsafe_allow_html=True,
            )
            ob_col1, ob_col2 = st.columns(2)
            max_vol_scale = max(1, max(b["volume"] for b in ob_res["bids"] + ob_res["asks"]))
            with ob_col1:
                df_bids = pd.DataFrame(ob_res["bids"])
                df_bids["檔位"] = [f"買 {i}" for i in df_bids["level"]]
                st.dataframe(
                    df_bids[["檔位", "price", "volume", "orders"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "檔位": st.column_config.TextColumn("買方檔位"),
                        "price": st.column_config.NumberColumn("委買價", format="$%.3f"),
                        "volume": st.column_config.ProgressColumn("委買股數", format="%d", min_value=0, max_value=max_vol_scale),
                        "orders": st.column_config.NumberColumn("委託筆數", format="%d 筆"),
                    },
                )
            with ob_col2:
                df_asks = pd.DataFrame(ob_res["asks"])
                df_asks["檔位"] = [f"賣 {i}" for i in df_asks["level"]]
                st.dataframe(
                    df_asks[["檔位", "price", "volume", "orders"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "檔位": st.column_config.TextColumn("賣方檔位"),
                        "price": st.column_config.NumberColumn("委賣價", format="$%.3f"),
                        "volume": st.column_config.ProgressColumn("委賣股數", format="%d", min_value=0, max_value=max_vol_scale),
                        "orders": st.column_config.NumberColumn("委託筆數", format="%d 筆"),
                    },
                )


# ==========================================
# TAB 2: Clean Multi-Factor Trend & Dividend Analysis
# ==========================================
with tab2:
    with st.spinner(f"正在分析 {active_ticker} 的多因子趨勢勝率與真實 TTM 股息率..."):
        trend_res, trend_err = cached_stock_trend(active_ticker)

    if trend_err or not trend_res:
        st.error(f"無法分析 {active_ticker}：{trend_err}")
    else:
        # Row 1: 6 Core Metrics
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("最新現價 / 次日撮合價", f"${trend_res['price']:.2f}", f"撮合 ${trend_res['execution_price']:.2f}")
        c2.metric("多因子綜合勝率 P(Up)", f"{trend_res['up_prob']*100:.1f}%", f"合成 Z {trend_res['composite_zscore']:+.2f}σ")
        c3.metric("短期戰術上漲機率", f"{trend_res['short_term_prob']*100:.1f}%", f"20D Z {trend_res['z_20d']:+.2f}σ")
        c4.metric("長期戰略上漲機率", f"{trend_res['long_term_prob']*100:.1f}%", f"200MA {trend_res['dist_200ma']*100:+.1f}%")
        c5.metric("滾動股息殖利率", f"{trend_res['dividend_yield']*100:.2f}%", "前復權已計入")
        c6.metric("股東權益報酬 (ROE)", f"{trend_res['roe']*100:.1f}%", "財報質量因子")

        # Row 2: Factor Bars + Plain-Language Summary
        t2_left, t2_right = st.columns([1.15, 1.15])
        with t2_left:
            z_dict_display = {
                row["factor_name"]: float(row["z_score"])
                for row in trend_res["factor_exposure_matrix"]
            }
            st.markdown(
                render_factor_bars_html(
                    title="🏛️ 四大核心因子強弱評分 (動量 / 價值 / 低波 / 資金流)",
                    subtitle=f"{trend_res['trend']}",
                    factor_dict=z_dict_display,
                    max_scale=3.0,
                    unit="σ",
                ),
                unsafe_allow_html=True,
            )
        with t2_right:
            bullets_html = "".join(
                f'<div style="padding: 8px 0; border-bottom: 1px solid rgba(148,163,184,0.18); font-size: 0.88rem; color: #F1F5F9;">{r}</div>'
                for r in trend_res["reasons"]
            )
            st.markdown(
                f"""
            <div class="bento-card">
                <div class="bento-title">
                    <span>📋 多因子綜合診斷結論</span>
                    <span class="mono-num" style="color:#34D399;">綜合勝率: {trend_res['up_prob']*100:.1f}%</span>
                </div>
                {bullets_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Row 3: Daily K-line + 50MA/200MA + Momentum Z-score
        df_h = trend_res["history"]
        fig_t2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
            subplot_titles=(
                f"{trend_res['ticker']} 日K線走勢 + 50MA (藍線) / 200MA 牛熊分界線 (黃線)",
                "20 日波段動量強弱指標 (Z-Score)",
            ),
        )
        fig_t2.add_trace(
            go.Candlestick(
                x=df_h.index,
                open=df_h["Open"],
                high=df_h["High"],
                low=df_h["Low"],
                close=df_h["Close"],
                name="日K線",
                increasing_line_color="#10B981",
                decreasing_line_color="#F43F5E",
            ),
            row=1,
            col=1,
        )

        if "SMA50" in df_h.columns:
            fig_t2.add_trace(
                go.Scatter(
                    x=df_h.index,
                    y=df_h["SMA50"],
                    mode="lines",
                    name="50MA (中線趨勢)",
                    line=dict(color="#0284C7", width=1.8),
                ),
                row=1,
                col=1,
            )
        if "SMA200" in df_h.columns:
            fig_t2.add_trace(
                go.Scatter(
                    x=df_h.index,
                    y=df_h["SMA200"],
                    mode="lines",
                    name="200MA (長線牛熊線)",
                    line=dict(color="#F59E0B", width=2.2),
                ),
                row=1,
                col=1,
            )

        if "Z_20D" in df_h.columns:
            z_colors = np.where(df_h["Z_20D"] >= 0, "#10B981", "#F43F5E")
            fig_t2.add_trace(
                go.Bar(
                    x=df_h.index,
                    y=df_h["Z_20D"],
                    name="20日動量 Z-Score",
                    marker_color=z_colors,
                    opacity=0.85,
                ),
                row=2,
                col=1,
            )
            fig_t2.add_hline(y=1.5, line_dash="dot", line_color="rgba(16,185,129,0.5)", row=2, col=1)
            fig_t2.add_hline(y=-1.5, line_dash="dot", line_color="rgba(244,63,94,0.5)", row=2, col=1)

        fig_t2.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#161F30",
            plot_bgcolor="#0B0F19",
            font=dict(color="#FFFFFF", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=15, r=15, t=35, b=15),
        )
        fig_t2.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
        fig_t2.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
        st.plotly_chart(fig_t2, use_container_width=True)

        # Progressive Disclosure: VIF Matrix & Factor Correlation Heatmap tucked in expander
        with st.expander("📐 展開檢視：因子權重明細表、Rank-IC 與共線性診斷 (VIF 矩陣)", expanded=False):
            exp_col, corr_col = st.columns([1.35, 0.95])
            with exp_col:
                df_exp = pd.DataFrame(trend_res["factor_exposure_matrix"])
                st.dataframe(
                    df_exp[[
                        "factor_name",
                        "z_score",
                        "rank_ic",
                        "ic_weight_pct",
                        "marginal_contribution_z",
                        "vif",
                        "collinearity_status",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "factor_name": st.column_config.TextColumn("核心因子名稱"),
                        "z_score": st.column_config.NumberColumn("標準化 Z-Score", format="%+.2fσ"),
                        "rank_ic": st.column_config.NumberColumn("滾動 Rank IC", format="%+.3f"),
                        "ic_weight_pct": st.column_config.ProgressColumn("IC 動態權重", format="%.1f%%", min_value=0, max_value=100),
                        "marginal_contribution_z": st.column_config.NumberColumn("邊際貢獻度", format="%+.3fσ"),
                        "vif": st.column_config.NumberColumn("共線性 VIF", format="%.2f"),
                        "collinearity_status": st.column_config.TextColumn("獨立性診斷"),
                    },
                )
            with corr_col:
                f_corr = trend_res["factor_corr_matrix"].copy()
                short_names = ["動量", "價值質量", "低波品質", "流動資金"]
                f_corr.index = short_names
                f_corr.columns = short_names
                fig_fcorr = px.imshow(
                    f_corr, text_auto=".2f", color_continuous_scale="Tealrose", zmin=-1, zmax=1
                )
                fig_fcorr.update_layout(
                    height=240,
                    paper_bgcolor="#161F30",
                    font=dict(color="#FFFFFF", size=12),
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_fcorr, use_container_width=True)


# ==========================================
# TAB 3: Clean Portfolio Optimization & Blotter
# ==========================================
with tab3:
    st.markdown("##### 🚀 快速切換精選投資組合範本 (或展開下方表格自訂持股)")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🌐 港股科技巨頭 (0700/9988/3690/1810)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0700.HK", "name": "騰訊控股", "lot_size": 100, "shares": 300, "entry_price": 380.0, "short_eligible": True},
                {"ticker": "9988.HK", "name": "阿里巴巴-W", "lot_size": 100, "shares": 500, "entry_price": 85.0, "short_eligible": True},
                {"ticker": "3690.HK", "name": "美團-W", "lot_size": 100, "shares": 400, "entry_price": 130.0, "short_eligible": True},
                {"ticker": "1810.HK", "name": "小米集團-W", "lot_size": 200, "shares": 2000, "entry_price": 25.0, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col2:
        if st.button("🏛️ 藍籌金融核心 (0005/0388/1299/0939)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0005.HK", "name": "匯豐控股", "lot_size": 400, "shares": 1200, "entry_price": 68.0, "short_eligible": True},
                {"ticker": "0388.HK", "name": "香港交易所", "lot_size": 100, "shares": 300, "entry_price": 280.0, "short_eligible": True},
                {"ticker": "1299.HK", "name": "友邦保險", "lot_size": 200, "shares": 1000, "entry_price": 62.0, "short_eligible": True},
                {"ticker": "0939.HK", "name": "建設銀行", "lot_size": 1000, "shares": 10000, "entry_price": 5.8, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col3:
        if st.button("🚗 新能源與製造 (1211/2015/0175/2382)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "1211.HK", "name": "比亞迪股份", "lot_size": 500, "shares": 500, "entry_price": 260.0, "short_eligible": True},
                {"ticker": "2015.HK", "name": "理想汽車-W", "lot_size": 100, "shares": 600, "entry_price": 100.0, "short_eligible": True},
                {"ticker": "0175.HK", "name": "吉利汽車", "lot_size": 1000, "shares": 4000, "entry_price": 12.0, "short_eligible": True},
                {"ticker": "2382.HK", "name": "舜宇光學", "lot_size": 100, "shares": 500, "entry_price": 55.0, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col4:
        if st.button("🛡️ 高息防禦公用 (0941/0883/0002/0003)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0941.HK", "name": "中國移動", "lot_size": 500, "shares": 1500, "entry_price": 72.0, "short_eligible": True},
                {"ticker": "0883.HK", "name": "中國海洋石油", "lot_size": 1000, "shares": 3000, "entry_price": 18.0, "short_eligible": True},
                {"ticker": "0002.HK", "name": "中電控股", "lot_size": 500, "shares": 1500, "entry_price": 65.0, "short_eligible": True},
                {"ticker": "0003.HK", "name": "中華煤氣", "lot_size": 1000, "shares": 5000, "entry_price": 6.2, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()

    with st.expander("✏️ 展開編輯：自訂持倉股票代號與目前股數", expanded=False):
        qa_c1, qa_c2, qa_c3 = st.columns([1.5, 1.0, 1.0])
        with qa_c1:
            add_ticker_input = st.text_input("➕ 輸入新增港股代號 (如 1810.HK)", key="quick_add_input")
        with qa_c2:
            add_shares_input = st.number_input("目前持有股數", min_value=0, value=500, step=100, key="quick_add_shares")
        with qa_c3:
            st.write("")
            st.write("")
            if st.button("加入持倉組合", type="secondary", use_container_width=True):
                if add_ticker_input.strip():
                    norm_add = normalize_hk_ticker(add_ticker_input.strip())
                    curr_tickers = [h.get("ticker", "") for h in cfg.get("holdings", [])]
                    if norm_add not in curr_tickers:
                        cfg["holdings"].append({
                            "ticker": norm_add,
                            "name": norm_add,
                            "lot_size": HK_DEFAULT_LOT_SIZES.get(norm_add, 100),
                            "shares": int(add_shares_input),
                            "entry_price": 0.0,
                            "short_eligible": is_hk_short_eligible(norm_add),
                        })
                        save_portfolio_config(cfg)
                        st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
                        st.rerun()

        df_edit_in = pd.DataFrame(cfg.get("holdings", []))
        if df_edit_in.empty:
            df_edit_in = pd.DataFrame(columns=["ticker", "name", "lot_size", "shares"])
        else:
            for col in ["ticker", "name", "lot_size", "shares"]:
                if col not in df_edit_in.columns:
                    df_edit_in[col] = ""

        edited_df = st.data_editor(
            df_edit_in[["ticker", "name", "lot_size", "shares"]],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ticker": st.column_config.TextColumn("港股代號", required=True),
                "name": st.column_config.TextColumn("股票名稱", required=True),
                "lot_size": st.column_config.NumberColumn("每手股數", min_value=1, step=50, required=True),
                "shares": st.column_config.NumberColumn("目前持有股數", min_value=0, step=100, required=True),
            },
            key="holdings_editor",
        )

        existing_map = {normalize_hk_ticker(h.get("ticker", "")): h for h in cfg.get("holdings", [])}
        new_holdings = []
        for _, row in edited_df.iterrows():
            t_str = str(row.get("ticker", "")).strip()
            if not t_str:
                continue
            norm_tk = normalize_hk_ticker(t_str)
            base_item = dict(existing_map.get(norm_tk, {}))
            base_item.update({
                "ticker": norm_tk,
                "name": str(row.get("name", norm_tk)).strip(),
                "lot_size": int(row.get("lot_size") or HK_DEFAULT_LOT_SIZES.get(norm_tk, 100)),
                "shares": int(row.get("shares") or 0),
            })
            new_holdings.append(base_item)

        if new_holdings != cfg.get("holdings", []):
            cfg["holdings"] = new_holdings
            if save_portfolio_config(cfg):
                st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)

    holdings_serialized = json.dumps(cfg["holdings"], ensure_ascii=False, sort_keys=True)

    with st.spinner("正在計算最佳化權重與實盤調倉建議..."):
        health_res = cached_portfolio_health(
            holdings_serialized,
            float(total_capital_hkd),
            float(rf_rate_pct) / 100.0,
            float(max_pos_pct) / 100.0,
            float(min_adv_input),
            alloc_method,
            int(num_trials),
            cov_estimator,
            float(max_turnover_pct) / 100.0,
            float(cfg["broker_comm_rate"]),
            float(cfg["min_broker_comm_hkd"]),
            float(cfg["almgren_gamma"]),
        )

    if not health_res:
        st.error("無法計算組合風險，請確認持倉代號是否有效。")
    else:
        bt = health_res["backtest"]

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("現有持倉總市值", f"${health_res['total_value']:,.0f}", f"目標池 ${total_capital_hkd:,.0f}")
        k2.metric("組合風控健康度", f"{health_res['health_score']}/100", health_res["health_status"].split(" ")[1])
        if bt:
            k3.metric("滾動回測年化回報", f"{bt['cagr']*100:+.2f}%", f"總回報 {bt['total_return']*100:+.1f}%")
            k4.metric(f"夏普比率 (Rf={rf_rate_pct:.1f}%)", f"{bt['sharpe']:.2f}", f"波動率 {bt['ann_volatility']*100:.1f}%")
            k5.metric("DSR 抗過擬合信心度", f"{bt['dsr_prob']*100:.1f}%", f"p={bt['dsr_pvalue']:.3f}")
            k6.metric("最大回撤 / 1D CVaR", f"{bt['max_drawdown']*100:.1f}%", f"CVaR {health_res['cvar_95']*100:.2f}%")
        else:
            k3.metric("組合年化波動率", f"{health_res['ann_volatility']*100:.2f}%")
            k4.metric("現有夏普比率", f"{health_res['sharpe_ratio']:.2f}")
            k5.metric("單日尾部 CVaR (95%)", f"{health_res['cvar_95']*100:.2f}%")
            k6.metric("去噪後平均相關性", f"{health_res['avg_correlation']:.2f}")

        # Primary Execution Blotter upfront for clarity
        st.markdown("#### 🛒 智能調倉建議執行單 (自動對齊港股每手股數與法定規費)")
        exec_df = pd.DataFrame(health_res["execution_orders"])
        exec_df["target_weight_pct"] = exec_df["target_weight"] * 100.0
        st.dataframe(
            exec_df[[
                "ticker",
                "name",
                "price",
                "lot_size",
                "target_weight_pct",
                "current_shares",
                "target_shares",
                "action",
                "estimated_turnover",
                "total_friction_hkd",
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("代號"),
                "name": st.column_config.TextColumn("名稱"),
                "price": st.column_config.NumberColumn("現價", format="$%.2f"),
                "lot_size": st.column_config.NumberColumn("每手股數", format="%d"),
                "target_weight_pct": st.column_config.ProgressColumn("目標最佳權重", format="%.1f%%", min_value=0, max_value=100),
                "current_shares": st.column_config.NumberColumn("目前股數", format="%d"),
                "target_shares": st.column_config.NumberColumn("建議目標股數", format="%d"),
                "action": st.column_config.TextColumn("調倉操作指令"),
                "estimated_turnover": st.column_config.NumberColumn("預估交易金額", format="$%,.0f"),
                "total_friction_hkd": st.column_config.NumberColumn("預估總規費+滑價", format="$%,.1f"),
            },
        )

        chart_left, chart_right = st.columns([1.35, 1.0])
        with chart_left:
            if bt:
                fig_bt = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.68, 0.32],
                    subplot_titles=(
                        f"投資組合歷史回測淨值 ({method_options[alloc_method]})",
                        "歷史回撤幅度 (Drawdown %)",
                    ),
                )
                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["strategy_nav"],
                        mode="lines",
                        name=f"{alloc_method.upper()} 策略淨值",
                        line=dict(color="#10B981", width=2.4),
                    ),
                    row=1,
                    col=1,
                )
                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["benchmark_nav"],
                        mode="lines",
                        name="等權重基準",
                        line=dict(color="#94A3B8", width=1.5, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["drawdown_series"] * 100.0,
                        mode="lines",
                        name="回撤 (%)",
                        line=dict(color="#F43F5E", width=1.6),
                        fill="tozeroy",
                        fillcolor="rgba(244, 63, 94, 0.18)",
                    ),
                    row=2,
                    col=1,
                )
                fig_bt.update_layout(
                    height=420,
                    paper_bgcolor="#161F30",
                    plot_bgcolor="#0B0F19",
                    font=dict(color="#FFFFFF", size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=15, r=15, t=35, b=15),
                )
                fig_bt.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
                fig_bt.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
                st.plotly_chart(fig_bt, use_container_width=True)

        with chart_right:
            curr_w = health_res["weights"]
            opt_w = health_res["optimal_weights"]
            tickers_all = list(opt_w.keys())
            df_w = pd.DataFrame({
                "Ticker": tickers_all,
                "目前權重 (%)": [curr_w.get(t, 0.0) * 100.0 for t in tickers_all],
                "建議權重 (%)": [opt_w.get(t, 0.0) * 100.0 for t in tickers_all],
            })
            fig_w = go.Figure()
            fig_w.add_trace(
                go.Bar(
                    y=df_w["Ticker"],
                    x=df_w["目前權重 (%)"],
                    orientation="h",
                    name="目前權重",
                    marker_color="#64748B",
                )
            )
            fig_w.add_trace(
                go.Bar(
                    y=df_w["Ticker"],
                    x=df_w["建議權重 (%)"],
                    orientation="h",
                    name="建議目標權重",
                    marker_color="#0284C7",
                )
            )
            fig_w.update_layout(
                title="⚖️ 目前持倉 vs 建議目標權重對比",
                barmode="group",
                height=420,
                paper_bgcolor="#161F30",
                plot_bgcolor="#0B0F19",
                font=dict(color="#FFFFFF", size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=15, r=15, t=45, b=15),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with st.expander("🔥 展開檢視：去噪後資產相關性矩陣與風控診斷明細", expanded=False):
            d_c1, d_c2 = st.columns([1.0, 1.2])
            with d_c1:
                corr_df = health_res["corr_matrix"]
                fig_heat = px.imshow(
                    corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1
                )
                fig_heat.update_layout(
                    height=300,
                    paper_bgcolor="#161F30",
                    font=dict(color="#FFFFFF", size=12),
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            with d_c2:
                for w in health_res["warnings"]:
                    st.markdown(f"- {w}")


# ==========================================
# TAB 4: Authentic, Un-Clipped Historical Backtest Validation
# ==========================================
with tab4:
    cfg_c1, cfg_c2, cfg_c3 = st.columns([1.2, 1.0, 1.2])
    horizon_presets = {
        5: "短線：未來 5 個交易日 (5D)",
        10: "波段：未來 10 個交易日 (10D)",
        21: "月線：未來 1 個月 (21D)",
        63: "季線：未來 3 個月 (63D)",
        126: "半年：未來 6 個月 (126D)",
        252: "年線：未來 1 年 (252D)",
    }
    with cfg_c1:
        preset_k = st.selectbox(
            "選擇前瞻預測天數 (k)",
            options=list(horizon_presets.keys()),
            format_func=lambda k: horizon_presets[k],
            index=1,
        )
    with cfg_c2:
        hist_len_map = {
            "1y": "近 1 年 (約252日)",
            "3y": "近 3 年 (約756日)",
            "5y": "近 5 年 (約1260日)",
        }
        hist_period_sel = st.selectbox(
            "歷史回測檢驗區間",
            options=list(hist_len_map.keys()),
            format_func=lambda k: hist_len_map[k],
            index=1,
        )
    with cfg_c3:
        with st.expander("⚙️ 進階止損與訊號過濾門檻設定", expanded=False):
            atr_stop_mult_sel = st.slider(
                "ATR 動態止損倍數",
                min_value=1.0,
                max_value=4.0,
                value=2.0,
                step=0.25,
            )
            sig_thresh_sel = st.slider(
                "訊號觸發門檻 (|Z| ≥ σ)",
                min_value=0.0,
                max_value=1.5,
                value=0.25,
                step=0.05,
            )

    with st.spinner(f"正在對 {active_ticker} 執行真實無裁切歷史回測檢驗..."):
        acc_res, acc_err = cached_historical_accuracy(
            ticker=active_ticker,
            horizon=int(preset_k),
            history_period=hist_period_sel,
            atr_stop_mult=float(atr_stop_mult_sel),
            signal_z_threshold=float(sig_thresh_sel),
            comm_rate=float(cfg["broker_comm_rate"]),
            min_comm=float(cfg["min_broker_comm_hkd"]),
            gamma=float(cfg["almgren_gamma"]),
        )

    if acc_err or not acc_res:
        st.error(f"無法完成歷史準確度驗證：{acc_err}")
    else:
        # Clear Plain-Language Verdict Banner at the top of Tab 4
        hr_val = acc_res["hit_rate"] * 100.0
        ic_val = acc_res["mean_ic"]
        pf_val = acc_res["profit_factor"]
        if hr_val >= 52.5 and ic_val >= 0.02 and pf_val >= 1.15:
            v_badge = '<span class="badge-bull">🟢 模型表現強健 (適合順勢波段操作)</span>'
            v_desc = f"在過去 {hist_len_map[hist_period_sel]} 中，{active_ticker} 的多因子訊號展現出顯著的正向預測力（勝率 {hr_val:.1f}%、Rank IC {ic_val:+.3f}、獲利因子 {pf_val:.2f}），高分組 (Q5) 顯著跑贏低分組 (Q1)。"
        elif hr_val >= 49.5 and ic_val >= -0.01:
            v_badge = '<span class="badge-warn">🟡 模型表現中性 (建議搭配嚴格止損或區間操作)</span>'
            v_desc = f"在過去 {hist_len_map[hist_period_sel]} 中，{active_ticker} 走勢偏向區間震盪，因子預測力溫和（勝率 {hr_val:.1f}%、Rank IC {ic_val:+.3f}），建議僅在訊號極強時出手。"
        else:
            v_badge = '<span class="badge-bear">🔴 趨勢動量逆風 (歷史勝率偏低，不宜單純追高)</span>'
            v_desc = f"在過去 {hist_len_map[hist_period_sel]} 中，{active_ticker} 受制於長線震盪或假突破洗盤，順勢動量因子出現負相關（勝率 {hr_val:.1f}%、Rank IC {ic_val:+.3f}、獲利因子 {pf_val:.2f}），反映該股在此週期不適合盲目順勢追漲。"

        st.markdown(
            f"""
            <div class="verdict-banner">
                <div>
                    <div style="font-size:1.02rem; font-weight:800; color:#FFFFFF; margin-bottom:6px;">
                        📊 {active_ticker} 歷史回測總評診斷（前瞻 {preset_k} 個交易日）
                    </div>
                    <div style="font-size:0.88rem; color:#CBD5E1; max-width:820px;">
                        {v_desc}
                    </div>
                </div>
                <div>{v_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Row 1: 6 Core Authentic KPI Cards
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric(
            "歷史方向勝率 (Hit Rate)",
            f"{acc_res['hit_rate']*100:.1f}%",
            f"有效訊號 {acc_res['sample_size']} 次",
        )
        a2.metric(
            "平均 Rank IC (秩相關)",
            f"{acc_res['mean_ic']:+.3f}",
            f"全樣本 IC {acc_res['overall_rank_ic']:+.3f}",
        )
        ic_ir_label = (
            "顯著正向預測" if acc_res["ic_ir"] >= 0.30
            else ("反向/均值回歸" if acc_res["ic_ir"] <= -0.20 else "溫和/震盪")
        )
        a3.metric(
            "IC 資訊比率 (IC-IR)",
            f"{acc_res['ic_ir']:+.2f}",
            ic_ir_label,
        )
        a4.metric(
            "年化超額回報 (vs 恆指)",
            f"{acc_res['annualized_alpha']*100:+.2f}%",
            f"策略年化 {acc_res['strategy_ann_return']*100:+.1f}%",
        )
        a5.metric(
            "獲利因子 (Profit Factor)",
            f"{acc_res['profit_factor']:.2f}",
            f"盈虧比 {acc_res['win_loss_ratio']:.2f}x",
        )
        a6.metric(
            "平均實際持倉週期",
            f"{acc_res['avg_holding_days']:.1f} 日",
            f"單筆期望 {acc_res['expected_return_pct']:+.2f}%",
        )

        # Row 2: Quantile Monotonicity Bar Chart + Cumulative Quantile Curves
        q_left, q_right = st.columns([1.05, 1.25])
        q_df = pd.DataFrame(acc_res["quantile_summary"])

        with q_left:
            bar_colors = ["#F43F5E", "#FB923C", "#94A3B8", "#0284C7", "#10B981"]
            fig_qbar = go.Figure(
                go.Bar(
                    x=q_df["quantile"],
                    y=q_df["mean_forward_return_pct"],
                    marker_color=bar_colors,
                    text=[f"{v:+.2f}%" for v in q_df["mean_forward_return_pct"]],
                    textposition="outside",
                )
            )
            fig_qbar.update_layout(
                title=f"📊 五分位數收益檢驗 (Q1極空 至 Q5極多 | 多空差 {acc_res['q5_q1_spread_pct']:+.2f}%)",
                height=390,
                paper_bgcolor="#161F30",
                plot_bgcolor="#0B0F19",
                font=dict(color="#FFFFFF", size=12),
                yaxis_title=f"前瞻 {acc_res['forward_horizon']} 日平均收益率 (%)",
                margin=dict(l=15, r=15, t=45, b=15),
            )
            fig_qbar.update_xaxes(showgrid=False)
            fig_qbar.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
            st.plotly_chart(fig_qbar, use_container_width=True)

        with q_right:
            fig_qnav = go.Figure()
            q_palette = {
                "Q1 (極空)": "#F43F5E",
                "Q2 (偏空)": "#FB923C",
                "Q3 (中性)": "#94A3B8",
                "Q4 (偏多)": "#0284C7",
                "Q5 (極多)": "#10B981",
            }
            for q_name, nav_s in acc_res["quantile_cum_curves"].items():
                fig_qnav.add_trace(
                    go.Scatter(
                        x=nav_s.index,
                        y=nav_s.values,
                        mode="lines",
                        name=q_name,
                        line=dict(color=q_palette.get(q_name, "#0284C7"), width=2.4 if q_name in ("Q5 (極多)", "Q1 (極空)") else 1.4),
                    )
                )
            fig_qnav.update_layout(
                title=f"📈 分位數累積淨值走勢 (單調性分數 = {acc_res['monotonicity_score']:+.2f})",
                height=390,
                paper_bgcolor="#161F30",
                plot_bgcolor="#0B0F19",
                font=dict(color="#FFFFFF", size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="累積淨值 (基準 = 1.0)",
                margin=dict(l=15, r=15, t=45, b=15),
            )
            fig_qnav.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
            fig_qnav.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
            st.plotly_chart(fig_qnav, use_container_width=True)

        # Progressive Disclosure: Advanced Statistical Diagnostics tucked inside expander
        with st.expander("🔬 展開檢視：進階統計散點圖、MAE/MFE 最大浮虧分佈、混淆矩陣與指標判讀表", expanded=False):
            sc_left, sc_right = st.columns([1.15, 1.15])
            sc_df = acc_res["scatter_df"]
            reg = acc_res["regression"]

            with sc_left:
                fig_sc = go.Figure()
                fig_sc.add_trace(
                    go.Scatter(
                        x=sc_df["pred_score"],
                        y=sc_df["forward_return_pct"],
                        mode="markers",
                        name="歷史樣本點",
                        marker=dict(
                            size=6.5,
                            color=sc_df["pred_prob_pct"],
                            colorscale="Tealgrn",
                            opacity=0.72,
                            showscale=True,
                            colorbar=dict(title="上漲機率%"),
                        ),
                        text=sc_df["date"],
                        hovertemplate="日期: %{text}<br>預測得分: %{x:.1f}<br>實際前瞻漲跌: %{y:+.2f}%<extra></extra>",
                    )
                )
                x_line = np.linspace(float(sc_df["pred_score"].min()), float(sc_df["pred_score"].max()), 50)
                y_line = reg["slope"] * x_line + reg["intercept"]
                fig_sc.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"回歸趨勢線 (R² = {reg['r_squared']:.3f})",
                        line=dict(color="#F59E0B", width=2.5, dash="dash"),
                    )
                )
                fig_sc.update_layout(
                    title=f"🎯 預測得分 vs 未來 {acc_res['forward_horizon']} 日實際漲跌幅 (R² = {reg['r_squared']:.3f})",
                    height=350,
                    paper_bgcolor="#161F30",
                    plot_bgcolor="#0B0F19",
                    font=dict(color="#FFFFFF", size=12),
                    xaxis_title="預測得分 (-100 至 +100)",
                    yaxis_title=f"未來 {acc_res['forward_horizon']} 日實際漲跌 (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=15, r=15, t=45, b=15),
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            with sc_right:
                cm = acc_res["confusion_matrix"]
                st.markdown(
                    f"""
                    <div class="bento-card" style="margin-top:8px;">
                        <div class="bento-title">
                            <span>🧩 多空預測混淆矩陣與停損統計</span>
                            <span class="mono-num" style="color:#FBBF24;">假陽性率 FPR: {cm['fpr']*100:.1f}%</span>
                        </div>
                        <div class="exec-grid">
                            <div class="exec-cell">
                                <div class="exec-label">看多且實際上漲 (TP)</div>
                                <div class="exec-val" style="color:#34D399;">{cm['tp']} 次 <span style="font-size:0.78rem; color:#CBD5E1;">(精確率 {cm['precision']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">看多但實際下跌 (FP)</div>
                                <div class="exec-val" style="color:#FB7185;">{cm['fp']} 次 <span style="font-size:0.78rem; color:#CBD5E1;">(FPR {cm['fpr']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">看空且成功避險 (TN)</div>
                                <div class="exec-val" style="color:#38BDF8;">{cm['tn']} 次 <span style="font-size:0.78rem; color:#CBD5E1;">(觸發止損率 {cm['stop_trigger_rate']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">看空但標的逆勢上漲 (FN)</div>
                                <div class="exec-val" style="color:#FBBF24;">{cm['fn']} 次 <span style="font-size:0.78rem; color:#CBD5E1;">(召回率 {cm['recall']*100:.1f}%)</span></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("""
| 指標名稱 | 弱勢 / 反向 (`< 基準`) | 中性震盪 (`正常範圍`) | 強健有效 (`優秀表現`) | 實戰意義說明 |
| :--- | :--- | :--- | :--- | :--- |
| **歷史方向勝率 (Hit Rate)** | `< 48.0%` (動量逆風) | `48.0% ~ 52.5%` | `> 53.0%` (顯著優勢) | 訊號預測方向與實際漲跌一致的比例。 |
| **平均 Rank IC (秩相關)** | `< 0.000` (負相關/均值回歸) | `0.000 ~ +0.025` | `> +0.030` (顯著正向 Alpha) | 衡量因子分數高低與未來實際報酬高低的排序相關性。 |
| **獲利因子 (Profit Factor)** | `< 1.00` (總虧損 > 總獲利) | `1.00 ~ 1.25` | `> 1.25` (健康正向期望) | 總獲利金額 ÷ 總虧損金額（已計入止損與交易規費）。 |
| **單調性評分 (Monotonicity)**| `< 0.00` (順序倒掛) | `0.00 ~ +0.50` | `+0.60 ~ +1.00` (階梯遞增) | 檢驗 Q1 到 Q5 是否呈現「分數越高、報酬越高」的階梯關係。 |
            """)