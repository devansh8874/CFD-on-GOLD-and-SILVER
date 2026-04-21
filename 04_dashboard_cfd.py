"""
================================================================================
    CFD GOLD & SILVER — INTERACTIVE STREAMLIT DASHBOARD  [v1]
    MBA Dissertation: Design and Performance Evaluation of Rule-Based
    Algorithmic Trading Strategies in Gold and Silver Markets
    S.V. National Institute of Technology, Surat — Dept. of Management Studies
  FEATURES:
    ├─ Live candlestick charts (Plotly) with overlaid indicators
    ├─ Signal panel: rule-based T1/T2/T3 signal + ML consensus
    ├─ Model performance comparison: XGB / RF / DT-Bag / SGB / LSTM / RNN
    ├─ Walk-forward precision chart
    ├─ Feature importance (XGB + RF)
    ├─ Backtest equity curve + trade log
    ├─ Comparative analysis: Rules vs ML-filtered backtest
    ├─ Regime overlay (bull/bear zones on candlestick)
    ├─ Volume analysis panel
    └─ Instrument switcher: GOLD_CFD / SILVER_CFD

  RUN:  streamlit run 05_dashboard_cfd.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, glob, json
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold & Silver Algorithmic Trading — MBA Dissertation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  THEME — dark terminal aesthetic with gold/silver accents
# ─────────────────────────────────────────────────────────────────────────────
GOLD   = "#F5C842"
SILVER = "#C0C0C0"
GREEN  = "#00E676"
RED    = "#FF1744"
ORANGE = "#FF9100"
BLUE   = "#448AFF"
PURPLE = "#E040FB"
BG     = "#0D0F14"
CARD   = "#141820"
BORDER = "#1E2433"
TEXT   = "#E8EAF0"
DIM    = "#8892A4"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{
      background-color: {BG};
      color: {TEXT};
      font-family: 'Space Grotesk', sans-serif;
  }}

  .stApp {{ background-color: {BG}; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background-color: {CARD};
      border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stMultiSelect label {{
      color: {DIM} !important;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }}

  /* Metric cards */
  .metric-card {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 8px;
  }}
  .metric-label {{
      font-size: 0.70rem;
      color: {DIM};
      text-transform: uppercase;
      letter-spacing: 0.10em;
      margin-bottom: 4px;
  }}
  .metric-value {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.5rem;
      font-weight: 700;
      color: {TEXT};
  }}
  .metric-delta {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.80rem;
      margin-top: 2px;
  }}

  /* Signal badge */
  .signal-badge {{
      display: inline-block;
      padding: 6px 20px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.1rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
  }}
  .sig-buy     {{ background: rgba(0,230,118,0.15); color: {GREEN};  border: 1px solid {GREEN};  }}
  .sig-strong  {{ background: rgba(0,230,118,0.25); color: {GREEN};  border: 2px solid {GREEN};  }}
  .sig-watch   {{ background: rgba(255,145,0,0.15); color: {ORANGE}; border: 1px solid {ORANGE}; }}
  .sig-sell    {{ background: rgba(255,23,68,0.15);  color: {RED};   border: 1px solid {RED};    }}
  .sig-hold    {{ background: rgba(68,138,255,0.10); color: {BLUE};  border: 1px solid {BLUE};   }}
  .sig-skip    {{ background: rgba(136,146,164,0.10);color: {DIM};   border: 1px solid {BORDER}; }}

  /* Section header */
  .section-header {{
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: {DIM};
      border-bottom: 1px solid {BORDER};
      padding-bottom: 6px;
      margin: 20px 0 12px 0;
  }}

  /* Tier row */
  .tier-row {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px 0;
      border-bottom: 1px solid {BORDER};
      font-size: 0.88rem;
  }}
  .tier-label {{
      font-family: 'JetBrains Mono', monospace;
      color: {DIM};
      width: 60px;
      flex-shrink: 0;
  }}
  .tier-ok   {{ color: {GREEN}; }}
  .tier-fail {{ color: {RED};   }}
  .tier-warn {{ color: {ORANGE}; }}

  /* Model table */
  .model-row {{
      display: flex;
      align-items: center;
      padding: 7px 0;
      border-bottom: 1px solid {BORDER};
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      gap: 8px;
  }}
  .model-name {{ width: 80px; color: {DIM}; }}
  .model-rec  {{ width: 70px; font-weight: 700; }}
  .model-bar  {{ flex: 1; height: 4px; background: {BORDER}; border-radius: 2px; position: relative; }}
  .model-fill {{ height: 100%; border-radius: 2px; }}

  /* Scrollable trade log */
  .trade-log {{
      max-height: 320px;
      overflow-y: auto;
      border: 1px solid {BORDER};
      border-radius: 6px;
      padding: 0;
  }}
  .trade-row {{
      display: flex;
      gap: 12px;
      padding: 8px 14px;
      border-bottom: 1px solid {BORDER};
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      align-items: center;
  }}
  .trade-row:hover {{ background: {BORDER}; }}
  .outcome-win  {{ color: {GREEN}; }}
  .outcome-loss {{ color: {RED};   }}
  .outcome-exit {{ color: {ORANGE}; }}

  div[data-testid="stHorizontalBlock"] > div {{
      border-right: none;
  }}

  .stTabs [data-baseweb="tab-list"] {{
      background: {CARD};
      border-bottom: 1px solid {BORDER};
      gap: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
      color: {DIM};
      font-size: 0.80rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 10px 20px;
  }}
  .stTabs [aria-selected="true"] {{
      color: {GOLD} !important;
      border-bottom: 2px solid {GOLD} !important;
      background: transparent !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{
      padding: 16px 0 0 0;
  }}

  hr {{ border-color: {BORDER}; margin: 8px 0; }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
INDICATORS_DIR  = "02_data/with_indicators"
SIGNALS_DIR     = "02_data/signals"
ML_DIR          = "02_data/ml_results"
BT_DIR          = "02_data/backtest"
EQUITY_DIR      = "02_data/equity_curves"
BACKTEST_CSV    = "02_data/backtest_metrics.csv"
PERF_CSV        = "02_data/strategy_performance.csv"
SIGNALS_SUMMARY = "02_data/signals_summary.csv"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
    xaxis=dict(gridcolor=BORDER, zeroline=False, showline=False),
    yaxis=dict(gridcolor=BORDER, zeroline=False, showline=False),
    margin=dict(l=50, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font_size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
                font=dict(size=10)),
)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADERS  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_indicators(symbol):
    path = f"{INDICATORS_DIR}/{symbol}.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(ttl=300)
def load_signal(symbol):
    path = f"{SIGNALS_DIR}/{symbol}_signal.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(ttl=300)
def load_ml_summary():
    path = f"{ML_DIR}/ml_summary.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_ml_result(symbol):
    path = f"{ML_DIR}/{symbol}_ml.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_backtest(symbol):
    path = f"{BT_DIR}/{symbol}_bt.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(ttl=300)
def load_comparative():
    path = f"{BT_DIR}/comparative_analysis.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_signals_summary():
    if not os.path.exists(SIGNALS_SUMMARY):
        return pd.DataFrame()
    return pd.read_csv(SIGNALS_SUMMARY)

@st.cache_data(ttl=300)
def load_equity_curve(symbol):
    path = f"{EQUITY_DIR}/{symbol}_equity.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(ttl=300)
def load_backtest_metrics():
    if not os.path.exists(BACKTEST_CSV):
        return pd.DataFrame()
    return pd.read_csv(BACKTEST_CSV)


@st.cache_data(ttl=300)
def load_perf_csv():
    if not os.path.exists(PERF_CSV):
        return pd.DataFrame()
    return pd.read_csv(PERF_CSV)


def safe(df, col, default="—"):
    if df is None or df.empty or col not in df.columns:
        return default
    v = df[col].iloc[0]
    if pd.isna(v): return default
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 8px 0 16px 0;'>
      <div style='font-family: JetBrains Mono; font-size: 0.65rem;
                  letter-spacing: 0.16em; color: {DIM}; text-transform: uppercase;
                  margin-bottom: 4px;'>MBA Dissertation — SVNIT Surat</div>
      <div style='font-size: 1.2rem; font-weight: 700; color: {GOLD};
                  letter-spacing: 0.04em;'>Rule-Based Trading</div>
    </div>
    """, unsafe_allow_html=True)

    # Instrument select
    available = []
    for sym in ["GOLD_CFD", "SILVER_CFD"]:
        if os.path.exists(f"{INDICATORS_DIR}/{sym}.csv"):
            available.append(sym)
    if not available:
        available = ["GOLD_CFD", "SILVER_CFD"]

    instrument_labels = {
        "GOLD_CFD":   "🥇  Gold CFD   (GC=F)",
        "SILVER_CFD": "🥈  Silver CFD (SI=F)",
    }
    symbol = st.selectbox(
        "INSTRUMENT",
        options=available,
        format_func=lambda x: instrument_labels.get(x, x),
    )
    accent = GOLD if "GOLD" in symbol else SILVER

    st.markdown("<hr>", unsafe_allow_html=True)

    # Chart settings
    st.markdown("**CHART SETTINGS**")
    lookback_options = {"3 Months": 63, "6 Months": 126, "1 Year": 252,
                        "2 Years": 504, "5 Years": 1260, "All": 99999}
    lookback_label = st.selectbox("PERIOD", list(lookback_options.keys()), index=2)
    lookback = lookback_options[lookback_label]

    show_sma   = st.checkbox("SMA 20 / 50 / 200", value=True)
    show_ema   = st.checkbox("EMA 20 / 50",        value=False)
    show_bb    = st.checkbox("Bollinger Bands",     value=True)
    show_vol   = st.checkbox("Volume",              value=True)
    show_rsi   = st.checkbox("RSI (14)",            value=True)
    show_macd  = st.checkbox("MACD",                value=True)
    show_adx   = st.checkbox("ADX",                 value=False)
    show_trades= st.checkbox("Backtest Trades",     value=True)
    show_gc    = st.checkbox("Golden / Death Cross", value=True)
    show_regime= st.checkbox("Regime Zones",        value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.70rem; color:{DIM}; line-height: 1.6;'>
      Data via yfinance (GC=F / SI=F)<br>
      Refresh: every 5 min (cached)<br>
      Run <code>01 → 02 → 03</code><br>
      then <code>streamlit run 05_dashboard_cfd.py</code>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df         = load_indicators(symbol)
sig_df     = load_signal(symbol)
ml_row     = load_ml_result(symbol)
bt_df      = load_backtest(symbol)
comp_df    = load_comparative()
sig_sum    = load_signals_summary()
ml_sum     = load_ml_summary()
eq_df      = load_equity_curve(symbol)
bt_metrics = load_backtest_metrics()
perf_df    = load_perf_csv()

data_ok = not df.empty
if not data_ok:
    st.error(f"⚠️  No indicator data found for **{symbol}**. "
             f"Run the pipeline first:\n"
             f"```\npython 01_data_updater_cfd.py\n"
             f"python 02_feature_engineering_cfd.py\n"
             f"python 03_signal_generator_cfd.py\n```")
    st.stop()

# Trim to lookback
chart_df = df.tail(lookback).copy()
latest   = df.iloc[-1]
prev     = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER BAR
# ─────────────────────────────────────────────────────────────────────────────
close_price = float(latest["close"])
prev_close  = float(prev["close"])
day_chg     = close_price - prev_close
day_chg_pct = day_chg / prev_close * 100 if prev_close else 0
chg_color   = GREEN if day_chg >= 0 else RED
chg_sym     = "▲" if day_chg >= 0 else "▼"

sym_label   = "GOLD CFD" if "GOLD" in symbol else "SILVER CFD"
ticker      = "GC=F" if "GOLD" in symbol else "SI=F"

signal_from_summary = "—"
if not sig_sum.empty and "signal" in sig_sum.columns:
    row_sig = sig_sum[sig_sum["symbol"] == symbol]
    if not row_sig.empty:
        signal_from_summary = str(row_sig["signal"].iloc[0])

ml_consensus = safe(ml_row, "consensus", "—")
ml_prob      = safe(ml_row, "consensus_prob", None)
ml_votes     = safe(ml_row, "votes", "")

sig_cls_map = {
    "STRONG BUY": "sig-strong", "BUY": "sig-buy", "WATCH": "sig-watch",
    "SELL": "sig-sell", "HOLD": "sig-hold", "SKIP": "sig-skip",
    "NO_TRIGGER": "sig-skip", "—": "sig-skip",
}

h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([3, 2, 2, 2, 3])

with h_col1:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{sym_label} · {ticker}</div>
      <div class='metric-value' style='color:{accent};'>
        ${close_price:,.2f}
      </div>
      <div class='metric-delta' style='color:{chg_color};'>
        {chg_sym} ${abs(day_chg):.2f} ({day_chg_pct:+.2f}%)
      </div>
    </div>
    """, unsafe_allow_html=True)

with h_col2:
    sma200 = float(latest.get("SMA_200", np.nan))
    above_200 = "✅ ABOVE" if close_price > sma200 and not np.isnan(sma200) else "❌ BELOW"
    sma_color = GREEN if close_price > sma200 and not np.isnan(sma200) else RED
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>SMA 200 Gate</div>
      <div class='metric-value' style='color:{sma_color}; font-size:1.1rem;'>
        {above_200}
      </div>
      <div class='metric-delta' style='color:{DIM};'>
        SMA200 = ${sma200:,.2f}
      </div>
    </div>
    """, unsafe_allow_html=True)

with h_col3:
    cross = str(latest.get("Cross_Type", "NONE"))
    days_gc = latest.get("Days_Since_GC", 999)
    cross_label = cross.replace("_", " ")
    cross_color = GREEN if cross == "GOLDEN_CROSS" else (RED if cross == "DEATH_CROSS" else DIM)
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>MA Crossover</div>
      <div class='metric-value' style='color:{cross_color}; font-size:0.95rem;'>
        {cross_label}
      </div>
      <div class='metric-delta' style='color:{DIM};'>
        {f"GC {int(days_gc)}d ago" if days_gc < 999 else "No recent GC"}
      </div>
    </div>
    """, unsafe_allow_html=True)

with h_col4:
    rsi_val  = float(latest.get("RSI", 50))
    rsi_color = RED if rsi_val > 70 else (GREEN if rsi_val < 30 else ORANGE)
    adx_val  = float(latest.get("ADX", 0))
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>RSI / ADX</div>
      <div class='metric-value' style='color:{rsi_color};'>
        {rsi_val:.1f}
      </div>
      <div class='metric-delta' style='color:{DIM};'>
        ADX = {adx_val:.1f}
        {'  ✅ strong' if adx_val > 25 else '  weak trend'}
      </div>
    </div>
    """, unsafe_allow_html=True)

with h_col5:
    rule_cls  = sig_cls_map.get(signal_from_summary, "sig-skip")
    ml_cls    = sig_cls_map.get(ml_consensus, "sig-skip")
    ml_pct_str = f"  {float(ml_prob)*100:.0f}% conf" if ml_prob else ""
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Signals</div>
      <div style='display:flex; gap:8px; flex-wrap:wrap; margin-top:4px;'>
        <span class='signal-badge {rule_cls}'>
          📐 {signal_from_summary}
        </span>
        </div>
      <div class='metric-delta' style='color:{DIM}; margin-top:4px;'>
        Rule-Based Signal Only
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊  Chart",
    "📈  Backtest",
    "🔍  Signal Detail",
    "📉  Regime & Vol",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CANDLESTICK CHART
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    # How many subplots (rows) do we need?
    n_rows = 1
    row_heights = [0.55]
    subplot_titles = [f"{sym_label} — Candlestick"]

    if show_vol:
        n_rows += 1; row_heights.append(0.12); subplot_titles.append("Volume")
    if show_rsi:
        n_rows += 1; row_heights.append(0.13); subplot_titles.append("RSI (14)")
    if show_macd:
        n_rows += 1; row_heights.append(0.13); subplot_titles.append("MACD")
    if show_adx:
        n_rows += 1; row_heights.append(0.13); subplot_titles.append("ADX")

    # Normalise heights
    total = sum(row_heights)
    row_heights = [h/total for h in row_heights]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Regime zones (bull = green, bear = red) ────────────────────────────
    if show_regime and "Bullish_Regime" in chart_df.columns:
        regime_col = chart_df["Bullish_Regime"].fillna(0)
        in_bull    = False
        zone_start = None
        for i, (idx, row_d) in enumerate(chart_df.iterrows()):
            is_bull = bool(regime_col.iloc[i])
            if is_bull and not in_bull:
                zone_start = row_d["date"]
                in_bull = True
            elif not is_bull and in_bull:
                fig.add_vrect(
                    x0=zone_start, x1=row_d["date"],
                    fillcolor=f"rgba(0,230,118,0.04)",
                    layer="below", line_width=0,
                    row=1, col=1,
                )
                in_bull = False
        # Close last zone
        if in_bull and zone_start is not None:
            fig.add_vrect(
                x0=zone_start, x1=chart_df["date"].iloc[-1],
                fillcolor=f"rgba(0,230,118,0.04)",
                layer="below", line_width=0,
                row=1, col=1,
            )

    # ── Golden / Death cross markers ───────────────────────────────────────
    if show_gc and "Cross_Type" in chart_df.columns:
        gc_rows = chart_df[chart_df["Cross_Type"] == "GOLDEN_CROSS"]
        dc_rows = chart_df[chart_df["Cross_Type"] == "DEATH_CROSS"]
        if not gc_rows.empty:
            fig.add_trace(go.Scatter(
                x=gc_rows["date"], y=gc_rows["low"] * 0.997,
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=12,
                            color=GREEN, line=dict(width=1, color="white")),
                text=["GC"] * len(gc_rows), textposition="bottom center",
                textfont=dict(size=8, color=GREEN),
                name="Golden Cross", showlegend=True,
            ), row=1, col=1)
        if not dc_rows.empty:
            fig.add_trace(go.Scatter(
                x=dc_rows["date"], y=dc_rows["high"] * 1.003,
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=12,
                            color=RED, line=dict(width=1, color="white")),
                text=["DC"] * len(dc_rows), textposition="top center",
                textfont=dict(size=8, color=RED),
                name="Death Cross", showlegend=True,
            ), row=1, col=1)

    # ── Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=chart_df["date"],
        open=chart_df["open"], high=chart_df["high"],
        low=chart_df["low"],  close=chart_df["close"],
        increasing=dict(line=dict(color=GREEN, width=1),
                        fillcolor=f"rgba(0,230,118,0.75)"),
        decreasing=dict(line=dict(color=RED, width=1),
                        fillcolor=f"rgba(255,23,68,0.75)"),
        name="Price",
        showlegend=True,
    ), row=1, col=1)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    if show_bb and all(c in chart_df.columns for c in ["BB_Upper","BB_Lower","BB_Mid"]):
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["BB_Upper"],
            line=dict(color=PURPLE, width=0.8, dash="dot"),
            name="BB Upper", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["BB_Lower"],
            fill="tonexty", fillcolor="rgba(224,64,251,0.04)",
            line=dict(color=PURPLE, width=0.8, dash="dot"),
            name="BB Lower", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["BB_Mid"],
            line=dict(color=PURPLE, width=0.6, dash="dash"),
            name="BB Mid", showlegend=False,
        ), row=1, col=1)

    # ── SMAs ──────────────────────────────────────────────────────────────
    if show_sma:
        sma_config = [
            ("SMA_20",  BLUE,   "SMA 20",  1.2),
            ("SMA_50",  ORANGE, "SMA 50",  1.4),
            ("SMA_200", RED,    "SMA 200", 1.8),
        ]
        for col, color, lbl, width in sma_config:
            if col in chart_df.columns:
                fig.add_trace(go.Scatter(
                    x=chart_df["date"], y=chart_df[col],
                    line=dict(color=color, width=width),
                    name=lbl, showlegend=True,
                ), row=1, col=1)

    # ── EMAs ──────────────────────────────────────────────────────────────
    if show_ema:
        ema_config = [
            ("EMA_20", "#00BCD4", "EMA 20", 1.0),
            ("EMA_50", "#FF7043", "EMA 50", 1.0),
        ]
        for col, color, lbl, width in ema_config:
            if col in chart_df.columns:
                fig.add_trace(go.Scatter(
                    x=chart_df["date"], y=chart_df[col],
                    line=dict(color=color, width=width, dash="dash"),
                    name=lbl, showlegend=True,
                ), row=1, col=1)

    # ── Backtest trade entries / exits ─────────────────────────────────────
    if show_trades and not bt_df.empty:
        wins  = bt_df[(bt_df["outcome"] == "WIN")  & (bt_df["method"] == "ML_FILTERED")]
        losses= bt_df[(bt_df["outcome"] == "LOSS") & (bt_df["method"] == "ML_FILTERED")]
        if not wins.empty:
            fig.add_trace(go.Scatter(
                x=wins["date"], y=wins["entry"],
                mode="markers",
                marker=dict(symbol="circle", size=8, color=GREEN,
                            line=dict(width=1, color="white")),
                name="ML Entry (Win)", showlegend=True,
                hovertemplate="<b>WIN</b><br>Entry: $%{y:.2f}<br>Date: %{x}<extra></extra>",
            ), row=1, col=1)
        if not losses.empty:
            fig.add_trace(go.Scatter(
                x=losses["date"], y=losses["entry"],
                mode="markers",
                marker=dict(symbol="x", size=8, color=RED,
                            line=dict(width=1.5, color="white")),
                name="ML Entry (Loss)", showlegend=True,
                hovertemplate="<b>LOSS</b><br>Entry: $%{y:.2f}<br>Date: %{x}<extra></extra>",
            ), row=1, col=1)

    # Current row counter for sub-panels
    cur_row = 2

    # ── Volume ────────────────────────────────────────────────────────────
    if show_vol and "volume" in chart_df.columns:
        vol_colors = [GREEN if c >= o else RED
                      for c, o in zip(chart_df["close"], chart_df["open"])]
        fig.add_trace(go.Bar(
            x=chart_df["date"], y=chart_df["volume"],
            marker_color=vol_colors,
            marker_opacity=0.7,
            name="Volume", showlegend=False,
        ), row=cur_row, col=1)
        cur_row += 1

    # ── RSI ───────────────────────────────────────────────────────────────
    if show_rsi and "RSI" in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["RSI"],
            line=dict(color=GOLD, width=1.2),
            name="RSI", showlegend=False,
        ), row=cur_row, col=1)
        # Overbought / oversold zones
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,23,68,0.06)",
                      line_width=0, row=cur_row, col=1)
        fig.add_hrect(y0=0, y1=30,  fillcolor="rgba(0,230,118,0.06)",
                      line_width=0, row=cur_row, col=1)
        fig.add_hline(y=70, line=dict(color=RED,   width=0.7, dash="dot"),
                      row=cur_row, col=1)
        fig.add_hline(y=30, line=dict(color=GREEN, width=0.7, dash="dot"),
                      row=cur_row, col=1)
        fig.update_yaxes(range=[0, 100], row=cur_row, col=1)
        cur_row += 1

    # ── MACD ──────────────────────────────────────────────────────────────
    if show_macd and all(c in chart_df.columns
                         for c in ["MACD","MACD_Signal","MACD_Hist"]):
        hist_colors = [GREEN if v >= 0 else RED
                       for v in chart_df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=chart_df["date"], y=chart_df["MACD_Hist"],
            marker_color=hist_colors, marker_opacity=0.7,
            name="MACD Hist", showlegend=False,
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["MACD"],
            line=dict(color=BLUE, width=1.0),
            name="MACD", showlegend=False,
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["MACD_Signal"],
            line=dict(color=ORANGE, width=1.0),
            name="Signal", showlegend=False,
        ), row=cur_row, col=1)
        cur_row += 1

    # ── ADX ───────────────────────────────────────────────────────────────
    if show_adx and all(c in chart_df.columns for c in ["ADX","Plus_DI","Minus_DI"]):
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["ADX"],
            line=dict(color=GOLD, width=1.4),
            name="ADX", showlegend=False,
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["Plus_DI"],
            line=dict(color=GREEN, width=0.9),
            name="+DI", showlegend=False,
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["Minus_DI"],
            line=dict(color=RED, width=0.9),
            name="-DI", showlegend=False,
        ), row=cur_row, col=1)
        fig.add_hline(y=25, line=dict(color=DIM, width=0.7, dash="dot"),
                      row=cur_row, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        height=680,
        xaxis_rangeslider_visible=False,
        **PLOTLY_LAYOUT,
    )
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
        font=dict(size=10),
    ))
    fig.update_xaxes(
        showgrid=True, gridcolor=BORDER, gridwidth=0.5,
        showline=False, zeroline=False,
        rangeslider_visible=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.5)

    # Style subplot titles
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=10, color=DIM,
                           family="JetBrains Mono, monospace")

    st.plotly_chart(fig, use_container_width=True)

    # ── Quick stats row ───────────────────────────────────────────────────
    s_col1, s_col2, s_col3, s_col4, s_col5, s_col6 = st.columns(6)
    atr_val   = float(latest.get("ATR", 0))
    atr_pct   = float(latest.get("ATR_Pct", 0))
    bb_pct    = float(latest.get("BB_Pct", 0.5))
    hvol      = float(latest.get("HVol_20", 0))
    conf_cnt  = float(latest.get("Confirm_Count", 0))
    ma_gap    = float(latest.get("MA_Gap_Pct", 0))

    for col, label, val, fmt in [
        (s_col1, "ATR (14)",     atr_val,  f"${atr_val:.2f}"),
        (s_col2, "ATR %",        atr_pct,  f"{atr_pct:.2f}%"),
        (s_col3, "BB %B",        bb_pct,   f"{bb_pct:.2f}"),
        (s_col4, "HVol 20d",    hvol,     f"{hvol:.1f}%"),
        (s_col5, "Confirm Score",conf_cnt, f"{conf_cnt:.0f}/7"),
        (s_col6, "MA Gap",       ma_gap,   f"{ma_gap:+.2f}%"),
    ]:
        col.markdown(f"""
        <div class='metric-card' style='padding:12px 14px;'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value' style='font-size:1.15rem;'>{fmt}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:

    # ── Performance Matrix (Strategy vs Buy & Hold) ─────────────────────────
    st.markdown("<div class='section-header'>Performance Matrix — Strategy vs Buy & Hold</div>",
                unsafe_allow_html=True)

    if not bt_metrics.empty:
        sym_row = bt_metrics[bt_metrics["symbol"] == symbol]
        if not sym_row.empty:
            r = sym_row.iloc[0]

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            metrics_display = [
                (m_col1, "Total Return %",   f"{r.get('total_return_pct','—')}%",   GREEN if float(r.get('total_return_pct',0)) > 0 else RED),
                (m_col2, "CAGR %",           f"{r.get('cagr_pct','—')}%",           GREEN),
                (m_col3, "Sharpe Ratio",     f"{r.get('sharpe_ratio','—')}",        GREEN if float(r.get('sharpe_ratio',0)) > 1 else ORANGE),
                (m_col4, "Max Drawdown %",   f"{r.get('max_drawdown_pct','—')}%",   RED),
            ]
            for col, label, val, color in metrics_display:
                col.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>{label}</div>
                  <div class='metric-value' style='color:{color};'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

            m_col5, m_col6, m_col7, m_col8 = st.columns(4)
            metrics_display2 = [
                (m_col5, "Win Rate %",       f"{r.get('win_rate_pct', r.get('win_rate','—'))}%",       GREEN if float(r.get('win_rate_pct',0)) > 50 else ORANGE),
                (m_col6, "Profit Factor",    f"{r.get('profit_factor','—')}",       GREEN if float(r.get('profit_factor',0)) > 1 else RED),
                (m_col7, "Total Trades",     f"{int(r.get('n_trades', r.get('total_trades', r.get('n_signals',0))))}",         TEXT),
                (m_col8, "Final Equity",     f"${float(r.get('final_equity',0)):,.0f}", GREEN),
            ]
            for col, label, val, color in metrics_display2:
                col.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>{label}</div>
                  <div class='metric-value' style='color:{color};'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

            # B&H comparison row
            st.markdown("<div class='section-header'>Buy & Hold Benchmark Comparison</div>",
                        unsafe_allow_html=True)
            bh_col1, bh_col2, bh_col3, bh_col4 = st.columns(4)
            bh_metrics_display = [
                (bh_col1, "B&H Total Return", f"{r.get('bh_total_return','—')}%"),
                (bh_col2, "B&H CAGR %",       f"{r.get('bh_cagr','—')}%"),
                (bh_col3, "B&H Sharpe",        f"{r.get('bh_sharpe','—')}"),
                (bh_col4, "B&H Max Drawdown",  f"{r.get('bh_max_drawdown','—')}%"),
            ]
            for col, label, val in bh_metrics_display:
                col.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>{label}</div>
                  <div class='metric-value' style='color:{DIM};'>{val}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No backtest metrics found for {symbol}.")
    else:
        st.info("Run `python 03_signal_generator_cfd.py` first to generate backtest metrics.")

    # ── Equity Curve ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Equity Curve vs Buy & Hold</div>",
                unsafe_allow_html=True)

    if not eq_df.empty:
        # Buy & Hold equity curve from indicator data
        bh_series = None
        if not df.empty and "close" in df.columns:
            bh = df[["date","close"]].copy()
            bh["bh_equity"] = (bh["close"] / bh["close"].iloc[0]) * 10000

        fig_eq = go.Figure()

        # Strategy equity
        eq_plot = eq_df.rename(columns={"equity": "capital"}) if "capital" not in eq_df.columns else eq_df
        cap_col = "capital" if "capital" in eq_plot.columns else eq_df.columns[-1]
        fig_eq.add_trace(go.Scatter(
            x=eq_plot["date"], y=eq_plot[cap_col],
            line=dict(color=accent, width=2),
            name="Strategy Equity",
            fill="tozeroy",
            fillcolor=f"rgba({int(accent[1:3],16)},{int(accent[3:5],16)},{int(accent[5:7],16)},0.06)",
        ))

        # B&H overlay
        if not df.empty:
            fig_eq.add_trace(go.Scatter(
                x=bh["date"], y=bh["bh_equity"],
                line=dict(color=DIM, width=1.2, dash="dash"),
                name="Buy & Hold ($10,000)",
            ))

        fig_eq.add_hline(y=10000, line=dict(color=BORDER, width=0.8, dash="dot"))
        fig_eq.update_layout(
            height=380,
            **PLOTLY_LAYOUT,
            title=dict(text=f"{sym_label} — Strategy Equity Curve vs Buy & Hold",
                       font=dict(size=11, color=DIM)),
            yaxis_title="Portfolio Value ($)",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Drawdown Chart ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>Drawdown</div>",
                    unsafe_allow_html=True)
        eq_dd = eq_plot.copy()
        eq_dd[cap_col] = pd.to_numeric(eq_dd[cap_col], errors="coerce")
        eq_dd["cum_max"] = eq_dd[cap_col].cummax()
        eq_dd["drawdown"] = (eq_dd[cap_col] - eq_dd["cum_max"]) / eq_dd["cum_max"] * 100

        fig_dd = go.Figure(go.Scatter(
            x=eq_dd["date"], y=eq_dd["drawdown"],
            fill="tozeroy",
            fillcolor="rgba(255,23,68,0.10)",
            line=dict(color=RED, width=1.2),
            name="Drawdown %",
        ))
        fig_dd.add_hline(y=0, line=dict(color=BORDER, width=0.5))
        fig_dd.update_layout(
            height=220,
            **PLOTLY_LAYOUT,
            title=dict(text="Strategy Drawdown (%)",
                       font=dict(size=11, color=DIM)),
            yaxis_title="Drawdown %",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    else:
        st.info("Run `python 03_signal_generator_cfd.py` to generate equity curves.")

    # ── Win Rate comparison bar ──────────────────────────────────────────────
    if not perf_df.empty:
        st.markdown("<div class='section-header'>Win Rate — All Signals vs GC-Window</div>",
                    unsafe_allow_html=True)
        prow = perf_df[perf_df["symbol"] == symbol]
        if not prow.empty:
            pr = prow.iloc[0]
            wr_all = float(pr.get("wr_all_signals", 0))
            wr_gc  = float(pr.get("wr_in_gc_window", 0))
            fig_wr = go.Figure(go.Bar(
                x=["All Signals WR", "GC-Window WR"],
                y=[wr_all, wr_gc],
                marker_color=[BLUE, GREEN],
                marker_opacity=0.85,
                text=[f"{wr_all:.1f}%", f"{wr_gc:.1f}%"],
                textposition="outside",
                textfont=dict(color=TEXT, size=12, family="JetBrains Mono"),
            ))
            fig_wr.add_hline(y=50, line=dict(color=ORANGE, width=1, dash="dot"))
            fig_wr.update_layout(
                **PLOTLY_LAYOUT, height=280,
                title=dict(text="Win Rate: All Signals vs GC-Window Filtered",
                           font=dict(size=11, color=DIM)),
            )
            fig_wr.update_layout(yaxis=dict(range=[0, 80], gridcolor=BORDER))
            st.plotly_chart(fig_wr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SIGNAL DETAIL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    sig_col1, sig_col2 = st.columns([1, 1])

    with sig_col1:
        st.markdown("<div class='section-header'>Rule-Based Signal — Current Bar</div>",
                    unsafe_allow_html=True)

        if not sig_sum.empty:
            sym_row = sig_sum[sig_sum["symbol"] == symbol]
            if not sym_row.empty:
                sr = sym_row.iloc[0]

                # Tiers
                t1 = sr.get("t1_gate", "—")
                t2 = sr.get("t2_pa",   "—")
                t3 = sr.get("t3_gc",   "—")
                sig_main  = sr.get("signal", "—")
                gc_score  = sr.get("gc_score", "—")
                pa_trigger= sr.get("pa_trigger", "—")

                def tier_cls(v):
                    v = str(v)
                    if "PASS" in v or "ok" in v.lower(): return "tier-ok"
                    if "FAIL" in v or "SKIP" in v:       return "tier-fail"
                    if v not in ("—","NONE","nan"):       return "tier-warn"
                    return ""

                st.markdown(f"""
                <div class='tier-row'>
                  <span class='tier-label'>T1</span>
                  <span class='{tier_cls(t1)}'>{t1}</span>
                </div>
                <div class='tier-row'>
                  <span class='tier-label'>T2 PA</span>
                  <span class='tier-warn'>{pa_trigger}</span>
                </div>
                <div class='tier-row'>
                  <span class='tier-label'>T3 GC</span>
                  <span class='{tier_cls(t3)}'>{t3}</span>
                </div>
                <div class='tier-row'>
                  <span class='tier-label'>GC Score</span>
                  <span style='color:{accent};'>{gc_score}/8</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Risk plan
                entry_p  = sr.get("entry",    "—")
                sl_p     = sr.get("stop_loss", "—")
                tp_p     = sr.get("target",    "—")
                rr_p     = sr.get("rr_ratio",  "—")
                sl_pct   = sr.get("sl_pct",    "—")
                tp_pct   = sr.get("tp_pct",    "—")
                tp_meth  = sr.get("tp_method", "—")

                if entry_p != "—":
                    st.markdown(f"""
                    <div class='section-header'>Risk Plan</div>
                    <div class='tier-row'>
                      <span class='tier-label'>Entry</span>
                      <span style='color:{accent}; font-family: JetBrains Mono;'>
                        ${entry_p}
                      </span>
                    </div>
                    <div class='tier-row'>
                      <span class='tier-label'>Stop Loss</span>
                      <span style='color:{RED}; font-family: JetBrains Mono;'>
                        ${sl_p}  ({sl_pct})
                      </span>
                    </div>
                    <div class='tier-row'>
                      <span class='tier-label'>Take Profit</span>
                      <span style='color:{GREEN}; font-family: JetBrains Mono;'>
                        ${tp_p}  ({tp_pct})
                      </span>
                    </div>
                    <div class='tier-row'>
                      <span class='tier-label'>RR Ratio</span>
                      <span style='color:{ORANGE}; font-family: JetBrains Mono;'>
                        {rr_p}
                      </span>
                    </div>
                    <div class='tier-row'>
                      <span class='tier-label'>TP Method</span>
                      <span style='color:{DIM};'>{tp_meth}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Run `python 03_signal_generator_cfd.py` to generate signals.")

    with sig_col2:
        st.markdown("<div class='section-header'>Key Indicators — Latest Bar</div>",
                    unsafe_allow_html=True)

        indicator_groups = [
            ("MOVING AVERAGES", [
                ("SMA 20",  "SMA_20"),
                ("SMA 50",  "SMA_50"),
                ("SMA 200", "SMA_200"),
                ("EMA 20",  "EMA_20"),
                ("EMA 50",  "EMA_50"),
            ]),
            ("MOMENTUM", [
                ("RSI (14)",     "RSI"),
                ("RSI Slope",    "RSI_Slope"),
                ("MACD",         "MACD"),
                ("MACD Signal",  "MACD_Signal"),
                ("MACD Hist",    "MACD_Hist"),
                ("ROC 10",       "ROC_10"),
            ]),
            ("VOLATILITY", [
                ("ATR (14)",     "ATR"),
                ("ATR %",        "ATR_Pct"),
                ("BB Width",     "BB_Width"),
                ("BB %B",        "BB_Pct"),
                ("HVol 20d",     "HVol_20"),
            ]),
            ("TREND", [
                ("ADX",          "ADX"),
                ("+DI",          "Plus_DI"),
                ("-DI",          "Minus_DI"),
                ("Trend Regime", "Trend_Regime"),
            ]),
        ]

        for group_name, indicators in indicator_groups:
            st.markdown(f"<div class='section-header'>{group_name}</div>",
                        unsafe_allow_html=True)
            for label, col_name in indicators:
                val = latest.get(col_name, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    display = "—"
                elif isinstance(val, float):
                    display = f"{val:,.4f}"
                else:
                    display = str(val)
                st.markdown(f"""
                <div class='tier-row'>
                  <span style='color:{DIM}; width:110px; flex-shrink:0;
                               font-size:0.80rem;'>{label}</span>
                  <span style='font-family: JetBrains Mono; font-size:0.82rem;'>
                    {display}
                  </span>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — REGIME & VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:

    # ── Volatility regime chart ────────────────────────────────────────────
    if "HVol_20" in chart_df.columns:
        st.markdown("<div class='section-header'>Historical Volatility (20d annualised)</div>",
                    unsafe_allow_html=True)

        fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.65, 0.35],
                                vertical_spacing=0.04)

        # Vol line
        fig_vol.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["HVol_20"],
            line=dict(color=accent, width=1.4),
            fill="tozeroy", fillcolor=f"rgba({int(accent[1:3],16)},{int(accent[3:5],16)},{int(accent[5:7],16)},0.08)",
            name="HVol 20d",
        ), row=1, col=1)

        # ATR %
        if "ATR_Pct" in chart_df.columns:
            fig_vol.add_trace(go.Scatter(
                x=chart_df["date"], y=chart_df["ATR_Pct"],
                line=dict(color=PURPLE, width=1.0),
                name="ATR %",
            ), row=1, col=1)

        # Vol regime
        if "Vol_Regime" in chart_df.columns:
            regime_colors = {-1: BLUE, 0: DIM, 1: RED}
            vol_r = chart_df["Vol_Regime"].fillna(0).astype(int)
            fig_vol.add_trace(go.Bar(
                x=chart_df["date"], y=vol_r.map({-1:1, 0:1, 1:1}).fillna(1),
                marker_color=[regime_colors.get(v, DIM) for v in vol_r],
                marker_opacity=0.6,
                name="Vol Regime",
                showlegend=True,
            ), row=2, col=1)

        fig_vol.update_layout(
            height=380,
            **PLOTLY_LAYOUT,
            title=dict(text="Volatility Analysis  (Blue=Low | Grey=Normal | Red=High)",
                       font=dict(size=11, color=DIM)),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── MA Gap (conviction proxy) ──────────────────────────────────────────
    if "MA_Gap_Pct" in chart_df.columns:
        st.markdown("<div class='section-header'>MA Gap % (SMA20 vs SMA50 — momentum conviction)</div>",
                    unsafe_allow_html=True)

        gap_vals   = chart_df["MA_Gap_Pct"].fillna(0)
        gap_colors = [GREEN if v > 0 else RED for v in gap_vals]

        fig_gap = go.Figure(go.Bar(
            x=chart_df["date"], y=gap_vals,
            marker_color=gap_colors, marker_opacity=0.75,
            name="MA Gap %",
        ))
        fig_gap.add_hline(y=0, line=dict(color=DIM, width=0.8))
        fig_gap.add_hline(y=0.5,  line=dict(color=GREEN, width=0.6, dash="dot"))
        fig_gap.add_hline(y=-0.5, line=dict(color=RED,   width=0.6, dash="dot"))
        fig_gap.update_layout(
            height=260,
            **PLOTLY_LAYOUT,
            title=dict(text="Golden Cross conviction  (±0.5% threshold lines)",
                       font=dict(size=11, color=DIM)),
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    # ── Confirmation score ─────────────────────────────────────────────────
    if "Confirm_Count" in chart_df.columns:
        st.markdown("<div class='section-header'>Confirmation Score (multi-signal conviction)</div>",
                    unsafe_allow_html=True)

        conf_colors = []
        for v in chart_df["Confirm_Count"].fillna(0):
            if v >= 5:   conf_colors.append(GREEN)
            elif v >= 3: conf_colors.append(ORANGE)
            else:        conf_colors.append(RED)

        fig_conf = go.Figure()
        fig_conf.add_trace(go.Bar(
            x=chart_df["date"],
            y=chart_df["Confirm_Count"],
            marker_color=conf_colors,
            marker_opacity=0.80,
            name="Confirm Count",
        ))
        if "Confirm_MA10" in chart_df.columns:
            fig_conf.add_trace(go.Scatter(
                x=chart_df["date"], y=chart_df["Confirm_MA10"],
                line=dict(color=GOLD, width=1.4),
                name="Confirm MA10",
            ))
        fig_conf.add_hline(y=5, line=dict(color=GREEN, width=0.7, dash="dot"))
        fig_conf.add_hline(y=3, line=dict(color=ORANGE, width=0.7, dash="dot"))
        fig_conf.update_layout(
            height=260,
            **PLOTLY_LAYOUT,
            title=dict(text="Multi-signal alignment  (≥5 = STRONG BUY zone)",
                       font=dict(size=11, color=DIM)),
        )
        fig_conf.update_layout(yaxis=dict(range=[0, 8], gridcolor=BORDER))
        st.plotly_chart(fig_conf, use_container_width=True)

    # ── Setup duration ─────────────────────────────────────────────────────
    if "Setup_Duration" in chart_df.columns:
        st.markdown("<div class='section-header'>Setup Duration (bars in GC-aligned regime)</div>",
                    unsafe_allow_html=True)
        fig_sd = go.Figure(go.Scatter(
            x=chart_df["date"],
            y=chart_df["Setup_Duration"],
            fill="tozeroy",
            fillcolor=f"rgba({int(accent[1:3],16)},{int(accent[3:5],16)},{int(accent[5:7],16)},0.10)",
            line=dict(color=accent, width=1.2),
            name="Setup Duration",
        ))
        fig_sd.add_hline(y=20, line=dict(color=ORANGE, width=0.7, dash="dot"))
        fig_sd.update_layout(
            height=200,
            **PLOTLY_LAYOUT,
            title=dict(text="Consecutive bars in bullish MA regime  (>20 = mature setup)",
                       font=dict(size=11, color=DIM)),
        )
        st.plotly_chart(fig_sd, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
last_date = str(df["date"].max())[:10] if not df.empty else "—"
n_rows    = len(df)

st.markdown(f"""
<div style='text-align:center; padding: 24px 0 8px 0;
            color:{DIM}; font-size:0.72rem; letter-spacing:0.08em;
            border-top: 1px solid {BORDER}; margin-top: 16px;'>
  Design & Performance Evaluation of Rule-Based Algorithmic Trading Strategies · MBA Dissertation · SVNIT Surat ·
  {sym_label} · {n_rows:,} rows · Last bar: {last_date} ·
  Dashboard auto-refreshes every 5 min
</div>
""", unsafe_allow_html=True)
