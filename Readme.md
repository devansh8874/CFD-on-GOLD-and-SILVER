# 📊 CFD Gold & Silver — Rule-Based Algorithmic Trading System

> **MBA Dissertation Project**  
> *Design and Performance Evaluation of Rule-Based Algorithmic Trading Strategies in Gold and Silver Markets*  
> S.V. National Institute of Technology (SVNIT), Surat — Department of Management Studies

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project implements a **multi-tier, rule-based algorithmic trading system** for CFD instruments on Gold (GC=F) and Silver (SI=F), sourced via `yfinance`. The system is built around a **Golden Crossover** core strategy, enhanced by price-action pattern recognition, trend quality filters, and a comprehensive backtesting engine — all visualised through an interactive Streamlit dashboard.

The system is designed for academic research into algorithmic trading strategy design, win-rate optimisation, and risk-adjusted return comparison against a buy-and-hold benchmark.

---

## 🗂️ Project Structure

```
cfd-on-gold-and-silver/
│
├── 01_data_updater_cfd.py        # Smart incremental OHLCV data downloader
├── 02_feature_engineering_cfd.py # 60+ technical indicators & ML features
├── 03_signal_generator_cfd.py    # 3-tier signal engine + backtester
├── 04_dashboard_cfd.py           # Interactive Streamlit dashboard
├── requirements.txt              # Python dependencies
│
└── 02_data/                      # Auto-generated data directory
    ├── individual_stocks/        # Raw OHLCV CSVs (GOLD_CFD.csv, SILVER_CFD.csv)
    ├── with_indicators/          # Feature-engineered CSVs (60+ columns)
    ├── signals/                  # Per-instrument signal CSVs
    ├── equity_curves/            # Backtest equity curve time series
    ├── signals_summary.csv       # Latest signal for all instruments
    ├── strategy_performance.csv  # Win rates, CAGR, Sharpe, drawdown
    └── backtest_metrics.csv      # Strategy vs Buy & Hold comparison
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline (in order)

```bash
# Step 1 — Download & update OHLCV data (Gold & Silver since 1990)
python 01_data_updater_cfd.py

# Step 2 — Compute 60+ technical features
python 02_feature_engineering_cfd.py

# Step 3 — Generate trading signals + run backtest
python 03_signal_generator_cfd.py

# Step 4 — Launch the dashboard
streamlit run 04_dashboard_cfd.py
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
plotly
yfinance
```

> All dependencies are listed in `requirements.txt`. Streamlit Cloud installs them automatically on deployment.

---

## 🔩 Module Breakdown

### `01_data_updater_cfd.py` — Data Updater

Downloads and **incrementally updates** daily OHLCV data for:

| Instrument | yfinance Ticker | Output File |
|---|---|---|
| CFD Gold | `GC=F` (Gold Futures) | `GOLD_CFD.csv` |
| CFD Silver | `SI=F` (Silver Futures) | `SILVER_CFD.csv` |

**Key Features:**
- Historical data from 1990-01-01
- Smart incremental update — only fetches missing rows (no full re-download)
- Handles weekends and market holidays gracefully
- Auto-retry on API failures (up to 2 retries with backoff)
- Clean OHLCV normalisation — timezone stripping, type coercion, deduplication

---

### `02_feature_engineering_cfd.py` — Feature Engineering

Computes **60+ technical features** across 6 groups:

#### A. Price Action
- **Candle anatomy** — body size, upper/lower wicks, body ratio, direction flag
- **Candlestick patterns** — Bullish/Bearish Engulfing, Doji, Hammer, Shooting Star, Pin Bar, Inside Bar, Outside Bar
- **Market structure** — Higher-High (HH), Higher-Low (HL), Lower-High (LH), Lower-Low (LL), swing pivot detection (5-bar lookback)
- **Multi-timeframe returns** — 1d, 5d, 10d, 20d percentage changes

#### B. Moving Averages & Golden Crossover (Core Strategy)
- SMA 20, SMA 50, SMA 200
- EMA 20, EMA 50
- Golden Cross / Death Cross event flags (SMA20 × SMA50)
- `Cross_Type` label: `GOLDEN_CROSS` | `DEATH_CROSS` | `NONE`
- `Days_Since_GC` / `Days_Since_DC` — recency of last crossover event
- `MA_Gap_Pct` — spread between SMA20 and SMA50 as % of price
- `Bullish_MA_Stack` — full alignment: Price > EMA20 > SMA50 > SMA200

#### C. Momentum & Oscillators
- RSI (14) — with slope, regime classification (oversold/neutral/overbought), and bullish flag
- MACD (12/26/9) — line, signal, histogram, bullish flag, histogram acceleration
- Rate of Change (ROC-10)

#### D. Volatility
- ATR (14) and ATR as % of price
- Bollinger Bands (20, 2σ) — width, %B position, room-to-run flag
- Historical Volatility (20-day annualised)
- Volatility Regime — percentile-ranked: Low / Normal / High

#### E. Trend Quality (ADX)
- ADX (14), +DI, -DI (pure price-based, no volume dependency)
- `ADX_Strong` flag (ADX > 25)
- `ADX_Bullish` flag (+DI > −DI and ADX > 25)
- Trend Regime: Weak / Moderate / Strong

#### F. ML & Meta Features
- `Confirm_Count` — multi-signal confirmation score (0–7 bullish indicators aligned)
- Temporal/seasonality — Day of Week, Month, Quarter, sin/cos cyclical encoding
- `Q4_Season` flag — Gold demand seasonality (jewellery/gift demand)
- `Summer_Lull` flag — Silver industrial demand dip (June–August)
- `Setup_Duration` — consecutive bars spent in the golden-cross bullish regime

---

### `03_signal_generator_cfd.py` — Signal Generator & Backtester

Implements a **3-tier entry signal system** and a full backtesting engine.

#### Signal Generation: 3-Tier Architecture

```
T1: MACRO GATE
    └─ Price > SMA_200  →  PASS (long trades allowed)
    └─ Price < SMA_200  →  SKIP (no longs in downtrend)
    └─ Death Cross detected  →  SELL (exit all longs)

T2: PRICE ACTION TRIGGER  (must fire before T3 is evaluated)
    Pattern hierarchy (strongest to weakest):
    a) Bullish Engulfing         → PA score +3
    b) Hammer / Pin Bar          → PA score +2
    c) Inside Bar breakout       → PA score +2
    d) Bullish Outside Bar       → PA score +2
    e) Higher-High in uptrend    → PA score +2
    f) Strong bull candle (>60%) → PA score +1
    └─ No pattern  →  HOLD (no trade, regardless of MAs)

T3: GOLDEN CROSSOVER CONFIRMATION  (scored 0–8)
    • Fresh Golden Cross (SMA20 × SMA50)     +2
    • GC within last 20 bars (recency bonus) +2  ← KEY accuracy driver
    • Bullish MA Stack (P>EMA20>SMA50>SMA200)+1
    • EMA20 above EMA50 (double confirmation)+1
    • MA gap widening (MAs diverging)         +1
    • ADX_Bullish (ADX>25 and +DI>−DI)       +1
```

#### Signal Classification

| Signal | Conditions | Expected Win Rate |
|---|---|---|
| **STRONG BUY** | GC ≤ 20 bars + BullStack + Uptrend + GC score ≥ 5 | ~55–61% |
| **BUY** | GC ≤ 30 bars + BullStack + GC score ≥ 4 + PA ≥ 2 | ~40–50% |
| **WATCH** | In GC zone + GC score ≥ 1 + any PA | ~30–40% |
| **SELL** | Death Cross detected | — |
| **SKIP** | Price < SMA_200 | — |
| **HOLD** | No qualifying PA pattern | — |

#### Risk Management (Per-Trade)

| Parameter | Value / Method |
|---|---|
| **Stop Loss** | Low of entry candle − 0.5 pip buffer |
| **Take Profit** | 7% above entry OR 20-bar swing high (whichever is larger) |
| **Trailing Stop** | 3% below current price |
| **Position Size** | 10% of capital per trade |
| **Risk/Reward** | Dynamic — typically 1:4 to 1:10 (tight candle-low SL) |

#### Backtesting Engine

- Trade-by-trade portfolio simulation from full history
- Computes: Total Return %, CAGR, Sharpe Ratio, Max Drawdown, Profit Factor, Win Rate, Win/Loss Ratio
- Generates equity curves saved to `02_data/equity_curves/`
- Buy & Hold benchmark comparison (same period)
- Strategy outcome labelling for ML ground truth (7% TP target, 2.5% SL approximation)

---

### `04_dashboard_cfd.py` — Streamlit Dashboard

An interactive dark-themed dashboard with 4 tabs:

#### Tab 1 — 📊 Chart
- Interactive Plotly candlestick chart with configurable indicators
- Toggleable overlays: SMA 20/50/200, EMA 20/50, Bollinger Bands
- Volume panel with directional colouring
- RSI (14) with overbought/oversold zones
- MACD with histogram and signal line
- ADX with ±DI lines
- Golden Cross / Death Cross markers (▲ GC / ▼ DC)
- Bullish regime zone shading
- Backtest trade entry points (Win = green circle, Loss = red X)
- Quick stats: ATR, ATR%, BB%B, HVol 20d, Confirm Score, MA Gap

#### Tab 2 — 📈 Backtest
- Strategy vs Buy & Hold performance matrix
- Equity curve chart with B&H overlay and shaded fill
- Drawdown chart
- Win rate comparison bar chart (All signals vs GC-Window filtered)

#### Tab 3 — 🔍 Signal Detail
- Current bar's tier-by-tier signal breakdown (T1 / T2 PA / T3 GC)
- Full risk plan: Entry, Stop Loss, Take Profit, RR Ratio, TP Method
- Live indicator table — Moving Averages, Momentum, Volatility, Trend

#### Tab 4 — 📉 Regime & Volatility
- Historical volatility (20d annualised) + ATR% chart
- Volatility regime bar (Blue = Low / Grey = Normal / Red = High)
- MA Gap % chart (Golden Cross conviction proxy)
- Confirmation score bar chart with MA10 overlay
- Setup duration chart (consecutive bars in GC-aligned regime)

#### Sidebar Controls
- Instrument switcher: GOLD_CFD / SILVER_CFD
- Lookback period: 3M / 6M / 1Y / 2Y / 5Y / All
- Toggle: SMA, EMA, Bollinger Bands, Volume, RSI, MACD, ADX, Trades, GC markers, Regime zones

---

## 📊 Key Findings (Dissertation Summary)

| Metric | Gold — Strategy | Gold — B&H | Silver — Strategy | Silver — B&H |
|---|---|---|---|---|
| Total Return % | varies | varies | varies | varies |
| CAGR % | higher risk-adj | raw % | outperforms | baseline |
| Sharpe Ratio | ~2.35 | ~0.63 | superior | baseline |
| Max Drawdown | reduced | full exposure | reduced | full exposure |
| Win Rate | ~32% (all) | N/A | ~61% (GC window) | N/A |

> **Key Insight:** The single largest accuracy driver is trading **within 20 bars of a Golden Cross** event. Silver in the GC recency window shows a win rate of ~61%, significantly outperforming random entry. Gold is naturally choppier; risk-adjusted returns (Sharpe) justify the strategy over buy-and-hold even when raw returns are comparable.

> **Acknowledged Limitation:** Sharpe Ratio is computed on per-trade returns with fixed 7% TP / ~2.5% SL, which compresses variance and upward-biases the Sharpe vs a daily-returns calculation.

---

## ⚙️ Configuration

Key parameters in `03_signal_generator_cfd.py`:

```python
CAPITAL           = 10_000    # Total account size (USD)
CAPITAL_PER_TRADE = 0.10      # 10% of capital per trade
SL_BUFFER_PIPS    = 0.5       # Pips below candle low for stop loss
TP_FIXED_PCT      = 7.0       # Fixed take profit target (%)
TRAILING_SL_PCT   = 0.03      # Trailing stop (3% below current price)
GC_RECENCY_WINDOW = 20        # Max bars since golden cross (accuracy-optimised)
LOOKFORWARD_DAYS  = 40        # Forward window for ML outcome labelling
MIN_GC_SCORE      = 1         # Minimum GC score to take a trade
```

---

## 📁 Output Files

| File | Description |
|---|---|
| `02_data/individual_stocks/GOLD_CFD.csv` | Raw OHLCV data for Gold |
| `02_data/individual_stocks/SILVER_CFD.csv` | Raw OHLCV data for Silver |
| `02_data/with_indicators/GOLD_CFD.csv` | 60+ features for Gold |
| `02_data/with_indicators/SILVER_CFD.csv` | 60+ features for Silver |
| `02_data/signals/GOLD_CFD_signal.csv` | Signal history for Gold |
| `02_data/signals/SILVER_CFD_signal.csv` | Signal history for Silver |
| `02_data/signals_summary.csv` | Latest signal for all instruments |
| `02_data/strategy_performance.csv` | Full performance metrics |
| `02_data/backtest_metrics.csv` | Strategy vs B&H comparison |
| `02_data/equity_curves/GOLD_CFD_equity.csv` | Equity curve for Gold |
| `02_data/equity_curves/SILVER_CFD_equity.csv` | Equity curve for Silver |

---

## ☁️ Streamlit Cloud Deployment

1. Push all files (including `requirements.txt`) to your GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `04_dashboard_cfd.py`
5. Deploy

> **Note:** On Streamlit Cloud, the pipeline scripts (`01`, `02`, `03`) must be run locally first to generate the data files, which should then be committed to the repo. The dashboard reads from pre-generated CSVs.

---

## ⚠️ Disclaimer

This project is developed strictly for **academic research purposes** as part of an MBA dissertation. It does not constitute financial advice. Past performance of backtested strategies does not guarantee future results. CFD trading carries significant risk of loss.

---

## 👤 Author

**Devansh** — MBA Candidate  
Department of Management Studies  
S.V. National Institute of Technology (SVNIT), Surat

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

 
