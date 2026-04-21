"""
================================================================================
  CFD GOLD & SILVER — FEATURE ENGINEERING  [v1]
  MBA Dissertation: AI-Driven Algorithmic Trading System

  INSTRUMENTS : CFD on Gold (GOLD_CFD), CFD on Silver (SILVER_CFD)

  FEATURE GROUPS
  ──────────────
  A. PRICE ACTION
       • Candle anatomy   — body size, upper/lower wick, body ratio, direction
       • Pattern flags    — Bullish/Bearish Engulfing, Doji, Hammer, Shooting Star,
                            Inside Bar, Outside Bar, Pin Bar
       • Higher-High / Lower-Low structure (HH, HL, LH, LL)
       • Swing detection  — local pivot highs/lows (5-bar lookback)
       • Multi-timeframe returns (1d, 5d, 10d, 20d)
       • Price vs. key MAs (distance %, above/below flags)

  B. GOLDEN CROSSOVER  (20 MA & 50 MA)
       • SMA_20, SMA_50, SMA_200
       • EMA_20, EMA_50
       • Golden Cross / Death Cross event flags (20 x 50)
       • Cross_Type label:  "GOLDEN_CROSS" | "DEATH_CROSS" | "NONE"
       • Days_Since_GC / Days_Since_DC  (recency of last crossover)
       • MA_Gap_Pct   — spread between 20 & 50 MA as % of price
       • Bullish_MA_Stack  — price > EMA20 > SMA50 > SMA200
       • Price_vs_SMA20_Pct, Price_vs_SMA50_Pct (mean-reversion anchors)

  C. MOMENTUM & OSCILLATORS
       • RSI (14), RSI slope, RSI regime (oversold / neutral / overbought)
       • MACD (12/26/9): line, signal, histogram, bullish flag
       • Momentum_10 (Rate of Change)

  D. VOLATILITY
       • ATR (14), ATR % of price
       • Bollinger Bands (20, 2sigma): upper, mid, lower, width, %B position
       • Historical Volatility (20-day rolling std of log returns, annualised)
       • Volatility Regime  (-1 low / 0 normal / +1 high)

  E. TREND QUALITY  (pure-price ADX — no volume dependency)
       • ADX (14), +DI, -DI
       • ADX_Strong (>25), ADX_Bullish flag
       • Trend_Regime  (-1 weak / 0 moderate / +1 strong)

  F. SUPPORTING ML FEATURES
       • Multi-signal confirmation score (Confirm_Count)
       • Temporal features (seasonality)

  INPUT  : 02_data/individual_stocks/{SYMBOL}.csv
  OUTPUT : 02_data/with_indicators/{SYMBOL}.csv
================================================================================
"""

import pandas as pd
import numpy as np
import os, time, glob
from datetime import datetime

INPUT_DIR  = "02_data/individual_stocks"
OUTPUT_DIR = "02_data/with_indicators"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"{'='*80}")
print("  CFD GOLD & SILVER — FEATURE ENGINEERING  [v1]")
print(f"{'='*80}")
print(f"  Started : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  Input   : {INPUT_DIR}/")
print(f"  Output  : {OUTPUT_DIR}/")


# ==============================================================================
#  A.  PRICE ACTION FEATURES
# ==============================================================================

class PriceAction:

    @staticmethod
    def add_candle_anatomy(df):
        """
        Decompose each candle into body, wicks, direction.
        Normalised by high-low range to make features scale-free
        (works whether Gold is at $1800 or $2500).
        """
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]

        df["Candle_Dir"]       = np.where(c >= o, 1, -1)          # +1 bullish / -1 bearish
        df["Body"]             = (c - o).abs()
        df["Upper_Wick"]       = h - np.maximum(c, o)
        df["Lower_Wick"]       = np.minimum(c, o) - l
        df["Candle_Range"]     = h - l                             # full range

        r = df["Candle_Range"].replace(0, np.nan)
        df["Body_Ratio"]       = df["Body"]        / r             # 0-1 (1 = full body candle)
        df["Upper_Wick_Ratio"] = df["Upper_Wick"]  / r
        df["Lower_Wick_Ratio"] = df["Lower_Wick"]  / r

        df["Body_Ratio_MA5"]   = df["Body_Ratio"].rolling(5).mean()

        return df

    @staticmethod
    def add_candlestick_patterns(df):
        """
        Single- and two-candle pattern flags (1 = present, 0 = not).
        Thresholds tuned for daily commodity charts.
        """
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body   = (c - o).abs()
        rng    = (h - l).replace(0, np.nan)
        u_wick = h - np.maximum(c, o)
        l_wick = np.minimum(c, o) - l

        # ── Doji ──────────────────────────────────────────────────────────────
        # Body <= 10% of range → indecision
        df["Doji"] = (body / rng <= 0.10).astype(int)

        # ── Hammer ────────────────────────────────────────────────────────────
        # Lower wick >= 2x body, small upper wick → bullish reversal signal
        df["Hammer"] = (
            (l_wick >= 2 * body) &
            (u_wick <= 0.3 * body.replace(0, np.nan)) &
            (body / rng <= 0.35)
        ).astype(int)

        # ── Shooting Star ─────────────────────────────────────────────────────
        # Mirror of hammer — upper wick >= 2x body → bearish rejection
        df["Shooting_Star"] = (
            (u_wick >= 2 * body) &
            (l_wick <= 0.3 * body.replace(0, np.nan)) &
            (body / rng <= 0.35)
        ).astype(int)

        # ── Pin Bar (either direction) ─────────────────────────────────────────
        df["Pin_Bar"] = ((df["Hammer"] == 1) | (df["Shooting_Star"] == 1)).astype(int)

        # ── Bullish Engulfing ──────────────────────────────────────────────────
        prev_o, prev_c = o.shift(1), c.shift(1)
        df["Bullish_Engulfing"] = (
            (c > o) &
            (prev_c < prev_o) &
            (c > prev_o) &
            (o < prev_c)
        ).astype(int)

        # ── Bearish Engulfing ──────────────────────────────────────────────────
        df["Bearish_Engulfing"] = (
            (c < o) &
            (prev_c > prev_o) &
            (c < prev_o) &
            (o > prev_c)
        ).astype(int)

        # ── Inside Bar ────────────────────────────────────────────────────────
        # Today's high-low inside yesterday's → consolidation
        df["Inside_Bar"] = (
            (h <= h.shift(1)) &
            (l >= l.shift(1))
        ).astype(int)

        # ── Outside Bar ───────────────────────────────────────────────────────
        # Today's range exceeds yesterday's in both directions → expansion
        df["Outside_Bar"] = (
            (h > h.shift(1)) &
            (l < l.shift(1))
        ).astype(int)

        return df

    @staticmethod
    def add_market_structure(df, swing_bars=5):
        """
        HH / HL / LH / LL detection using rolling pivot highs/lows.
        A bar is a swing high if it is the highest in a
        (swing_bars)-bar window on each side.
        """
        h, l = df["high"], df["low"]

        swing_h = h == h.rolling(2 * swing_bars + 1, center=True).max()
        swing_l = l == l.rolling(2 * swing_bars + 1, center=True).min()
        df["Swing_High"] = swing_h.astype(int)
        df["Swing_Low"]  = swing_l.astype(int)

        # Last confirmed swing level (forward-filled)
        df["Prev_Swing_High"] = h.where(swing_h).ffill()
        df["Prev_Swing_Low"]  = l.where(swing_l).ffill()

        prev_sh = df["Prev_Swing_High"].shift(1)
        prev_sl = df["Prev_Swing_Low"].shift(1)

        df["HH"] = (swing_h & (h > prev_sh)).astype(int)   # Higher High
        df["LH"] = (swing_h & (h < prev_sh)).astype(int)   # Lower  High
        df["HL"] = (swing_l & (l > prev_sl)).astype(int)   # Higher Low
        df["LL"] = (swing_l & (l < prev_sl)).astype(int)   # Lower  Low

        # Structural regime: uptrend if HH+HL dominate over last 20 bars
        df["Uptrend_Structure"]   = (
            df["HH"].rolling(20).sum() > df["LH"].rolling(20).sum()
        ).astype(int)
        df["Downtrend_Structure"] = (
            df["LL"].rolling(20).sum() > df["HL"].rolling(20).sum()
        ).astype(int)

        return df

    @staticmethod
    def add_returns(df):
        df["Return_1d"]  = df["close"].pct_change(1)  * 100
        df["Return_5d"]  = df["close"].pct_change(5)  * 100
        df["Return_10d"] = df["close"].pct_change(10) * 100
        df["Return_20d"] = df["close"].pct_change(20) * 100
        return df

    @classmethod
    def calculate_all(cls, df):
        df = cls.add_candle_anatomy(df)
        df = cls.add_candlestick_patterns(df)
        df = cls.add_market_structure(df)
        df = cls.add_returns(df)
        return df


# ==============================================================================
#  B.  GOLDEN CROSSOVER  (20 MA x 50 MA)
# ==============================================================================

class MovingAverages:

    @staticmethod
    def add_crossover(df):
        """
        Core 20/50 MA golden-cross logic for CFD Gold & Silver.
        Includes both SMA and EMA variants, full MA stack alignment,
        and recency features (days since last crossover).
        """
        c = df["close"]

        # ── Simple MAs ────────────────────────────────────────────────────────
        df["SMA_20"]  = c.rolling(20).mean()
        df["SMA_50"]  = c.rolling(50).mean()
        df["SMA_200"] = c.rolling(200).mean()

        # ── Exponential MAs (faster response) ─────────────────────────────────
        df["EMA_20"]  = c.ewm(span=20, adjust=False).mean()
        df["EMA_50"]  = c.ewm(span=50, adjust=False).mean()

        # ── SMA crossover event ───────────────────────────────────────────────
        df["SMA20_above_SMA50"] = df["SMA_20"] > df["SMA_50"]
        cross = df["SMA20_above_SMA50"].astype(int).diff()
        df["Crossover"]  = cross
        df["Cross_Type"] = np.where(cross ==  1, "GOLDEN_CROSS",
                           np.where(cross == -1, "DEATH_CROSS",  "NONE"))

        # ── Days since last Golden / Death Cross ──────────────────────────────
        for label, event_val in [("Days_Since_GC", 1), ("Days_Since_DC", -1)]:
            df[label] = np.nan
            last_idx = None
            for i in range(len(df)):
                if df["Crossover"].iloc[i] == event_val:
                    last_idx = i
                df.iloc[i, df.columns.get_loc(label)] = (
                    i - last_idx if last_idx is not None else 999
                )

        # ── MA gap: conviction proxy (how far are the two MAs apart?) ─────────
        df["MA_Gap_Pct"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"] * 100

        # ── Price distance from key MAs ───────────────────────────────────────
        df["Price_vs_SMA20_Pct"] = (c - df["SMA_20"]) / df["SMA_20"] * 100
        df["Price_vs_SMA50_Pct"] = (c - df["SMA_50"]) / df["SMA_50"] * 100

        # ── Directional position flags ─────────────────────────────────────────
        df["Above_SMA20"]  = (c > df["SMA_20"]).astype(int)
        df["Above_SMA50"]  = (c > df["SMA_50"]).astype(int)
        df["Above_SMA200"] = (c > df["SMA_200"]).astype(int)

        # ── Full bullish MA stack: Price > EMA20 > SMA50 > SMA200 ─────────────
        df["Bullish_MA_Stack"] = (
            (c > df["EMA_20"]) &
            (df["EMA_20"] > df["SMA_50"]) &
            (df["SMA_50"] > df["SMA_200"])
        ).astype(int)

        # ── Long-term regime: SMA50 > SMA200 ──────────────────────────────────
        df["Bullish_Regime"] = (df["SMA_50"] > df["SMA_200"]).astype(int)

        # ── EMA cross (faster signal) ──────────────────────────────────────────
        df["EMA20_above_EMA50"] = (df["EMA_20"] > df["EMA_50"]).astype(int)
        ema_cross = df["EMA20_above_EMA50"].diff()
        df["EMA_Cross_Type"] = np.where(ema_cross ==  1, "EMA_GOLDEN",
                               np.where(ema_cross == -1, "EMA_DEATH",  "NONE"))

        return df

    @classmethod
    def calculate_all(cls, df):
        return cls.add_crossover(df)


# ==============================================================================
#  C.  MOMENTUM & OSCILLATORS
# ==============================================================================

class Momentum:

    @staticmethod
    def add_rsi(df, period=14):
        delta = df["close"].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        df["RSI_Slope"]  = df["RSI"].diff(3)
        df["RSI_Regime"] = np.where(df["RSI"] < 30, -1,    # oversold
                           np.where(df["RSI"] > 70,  1, 0))# overbought

        df["RSI_Rising"]  = (df["RSI"] > df["RSI"].shift(1)).astype(int)
        df["RSI_Bullish"] = (
            (df["RSI"] > 50) |
            ((df["RSI"] > 30) & (df["RSI_Rising"] == 1))
        ).astype(int)

        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        ema_fast          = df["close"].ewm(span=fast,   adjust=False).mean()
        ema_slow          = df["close"].ewm(span=slow,   adjust=False).mean()
        df["MACD"]        = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
        df["MACD_Bullish"]= (df["MACD"] > df["MACD_Signal"]).astype(int)
        df["MACD_Hist_Slope"] = df["MACD_Hist"].diff(2)    # histogram acceleration
        return df

    @staticmethod
    def add_roc(df, period=10):
        df["ROC_10"] = df["close"].pct_change(period) * 100
        return df

    @classmethod
    def calculate_all(cls, df):
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_roc(df)
        return df


# ==============================================================================
#  D.  VOLATILITY
# ==============================================================================

class Volatility:

    @staticmethod
    def add_atr(df, period=14):
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"]     = tr.rolling(period).mean()
        df["ATR_Pct"] = df["ATR"] / df["close"] * 100
        return df

    @staticmethod
    def add_bollinger(df, window=20):
        mid            = df["close"].rolling(window).mean()
        std            = df["close"].rolling(window).std()
        df["BB_Upper"] = mid + 2 * std
        df["BB_Mid"]   = mid
        df["BB_Lower"] = mid - 2 * std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / mid
        df["BB_Pct"]   = (df["close"] - df["BB_Lower"]) / (
                          df["BB_Upper"] - df["BB_Lower"])
        df["BB_Room_To_Run"] = (df["close"] < df["BB_Upper"]).astype(int)
        return df

    @staticmethod
    def add_historical_vol(df, window=20):
        """Annualised historical volatility and volatility regime."""
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["HVol_20"] = log_ret.rolling(window).std() * np.sqrt(252) * 100

        df["Vol_Regime_Pctile"] = df["ATR_Pct"].rolling(60).rank(pct=True)
        df["Vol_Regime"] = np.where(df["Vol_Regime_Pctile"] < 0.30, -1,
                           np.where(df["Vol_Regime_Pctile"] > 0.70,  1, 0))
        return df

    @classmethod
    def calculate_all(cls, df):
        df = cls.add_atr(df)
        df = cls.add_bollinger(df)
        df = cls.add_historical_vol(df)
        return df


# ==============================================================================
#  E.  TREND QUALITY  (pure-price ADX)
# ==============================================================================

class TrendQuality:

    @staticmethod
    def add_adx(df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where( (plus_dm  > minus_dm) & (plus_dm  > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm)  & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr      = tr.rolling(period).mean()
        plus_di  = 100 * (plus_dm.rolling(period).mean()  / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        dx       = 100 * ((plus_di - minus_di).abs() /
                          (plus_di + minus_di).replace(0, np.nan))

        df["ADX"]      = dx.rolling(period).mean()
        df["Plus_DI"]  = plus_di
        df["Minus_DI"] = minus_di

        df["ADX_Strong"]  = (df["ADX"] > 25).astype(int)
        df["ADX_Bullish"] = ((plus_di > minus_di) & (df["ADX"] > 25)).astype(int)

        df["Trend_Regime"] = np.where(df["ADX"] < 20, -1,
                             np.where(df["ADX"] > 35,  1, 0))
        return df

    @classmethod
    def calculate_all(cls, df):
        return cls.add_adx(df)


# ==============================================================================
#  F.  ML INTERACTION & META FEATURES
# ==============================================================================

class MLFeatures:

    @staticmethod
    def add_interaction_features(df):
        if all(c in df.columns for c in ["RSI", "ADX"]):
            df["RSI_x_ADX"] = (df["RSI"] / 100) * (df["ADX"] / 50)

        if all(c in df.columns for c in ["MACD_Hist", "BB_Pct"]):
            df["MACD_x_BB"] = df["MACD_Hist"] * df["BB_Pct"]

        if "SMA_20" in df.columns:
            roll_std = df["close"].rolling(20).std()
            df["Price_ZScore_20"] = (df["close"] - df["SMA_20"]) / roll_std.replace(0, np.nan)

        if "Return_1d" in df.columns:
            df["Return_Autocorr_10"] = df["Return_1d"].rolling(60).apply(
                lambda x: x.autocorr(lag=10) if len(x.dropna()) > 15 else 0,
                raw=False
            )
        return df

    @staticmethod
    def add_confirmation_score(df):
        """
        Multi-signal conviction score — counts how many bullish indicators
        agree at a given bar. High Confirm_Count = high-quality setup.
        """
        signal_cols = [
            "RSI_Bullish",
            "MACD_Bullish",
            "BB_Room_To_Run",
            "ADX_Bullish",
            "Bullish_MA_Stack",
            "Above_SMA200",
            "Uptrend_Structure",
        ]
        existing = [c for c in signal_cols if c in df.columns]
        if existing:
            df["Confirm_Count"]     = sum(df[c] for c in existing)
            df["Confirm_MA10"]      = df["Confirm_Count"].rolling(10).mean()
            df["Confirm_Stability"] = df["Confirm_Count"].rolling(10).std()
        return df

    @staticmethod
    def add_setup_duration(df):
        """Consecutive bars spent in the golden-cross regime."""
        if all(c in df.columns for c in ["Above_SMA200", "SMA20_above_SMA50"]):
            setup = (
                (df["Above_SMA200"] == 1) &
                (df["SMA20_above_SMA50"] == True)
            ).astype(int)
            df["Setup_Duration"] = (
                setup.groupby((setup != setup.shift()).cumsum()).cumcount() + 1
            ) * setup
        return df

    @staticmethod
    def add_temporal_features(df):
        """
        Commodity seasonality:
        Gold  — strong in Q4 (jewellery/gift demand), H1 safe-haven bids.
        Silver — industrial demand dip in summer, strong Q1 rallies.
        """
        if "date" not in df.columns:
            return df
        dt = pd.to_datetime(df["date"])
        df["DayOfWeek"] = dt.dt.dayofweek
        df["Month"]     = dt.dt.month
        df["Quarter"]   = dt.dt.quarter

        df["Month_Sin"]   = np.sin(2 * np.pi * df["Month"]     / 12)
        df["Month_Cos"]   = np.cos(2 * np.pi * df["Month"]     / 12)
        df["DoW_Sin"]     = np.sin(2 * np.pi * df["DayOfWeek"] / 5)
        df["DoW_Cos"]     = np.cos(2 * np.pi * df["DayOfWeek"] / 5)

        df["Q4_Season"]   = (df["Quarter"] == 4).astype(int)
        df["Summer_Lull"] = (df["Month"].isin([6, 7, 8])).astype(int)
        return df

    @classmethod
    def calculate_all(cls, df):
        df = cls.add_interaction_features(df)
        df = cls.add_confirmation_score(df)
        df = cls.add_setup_duration(df)
        df = cls.add_temporal_features(df)
        return df


# ==============================================================================
#  MASTER PIPELINE
# ==============================================================================

def process_instrument(df):
    df = PriceAction.calculate_all(df)
    df = MovingAverages.calculate_all(df)
    df = Momentum.calculate_all(df)
    df = Volatility.calculate_all(df)
    df = TrendQuality.calculate_all(df)
    df = MLFeatures.calculate_all(df)
    return df


# ==============================================================================
#  RUN
# ==============================================================================

print(f"\n[PROCESSING] Computing features per instrument...")
print("=" * 80)

csv_files = sorted(glob.glob(f"{INPUT_DIR}/*.csv"))
if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {INPUT_DIR}/\n"
        f"  -> Run: python 01_data_updater_cfd.py first."
    )

t0         = time.time()
ok         = 0
fail       = 0
total_rows = 0

for idx, filepath in enumerate(csv_files, 1):
    symbol = os.path.basename(filepath).replace(".csv", "")
    print(f"[{idx:2d}/{len(csv_files)}]  {symbol:20s}", end="  ")
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) < 200:
            print(f"SKIP — only {len(df)} rows (need >= 200 for SMA_200)")
            fail += 1
            continue

        df = process_instrument(df)
        out_path = f"{OUTPUT_DIR}/{symbol}.csv"
        df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        total_rows += len(df)
        ok += 1
        print(f"OK  {len(df):,} rows  ->  {df.shape[1]} features")

    except Exception as e:
        print(f"ERR  {str(e)[:80]}")
        fail += 1

elapsed = time.time() - t0
print(f"\n{'='*80}")
print(f"  DONE in {elapsed:.1f}s  |  {ok} instrument(s) OK  |  {fail} failed")
print(f"  Total rows : {total_rows:,}")
print(f"  Output     : {OUTPUT_DIR}/SYMBOL.csv")
print(f"""
  FEATURE GROUPS:
    [A] Price Action
          - Candle anatomy        (Body_Ratio, Upper/Lower_Wick_Ratio)
          - Pattern flags         (Engulfing, Doji, Hammer, Shooting Star,
                                   Pin Bar, Inside Bar, Outside Bar)
          - Market structure      (HH, HL, LH, LL, Swing pivots)
          - Returns               (1d, 5d, 10d, 20d)

    [B] Golden Crossover (20 MA x 50 MA)
          - SMA_20, SMA_50, SMA_200, EMA_20, EMA_50
          - Cross_Type            (GOLDEN_CROSS / DEATH_CROSS / NONE)
          - Days_Since_GC / DC   (recency of last crossover)
          - MA_Gap_Pct            (spread between 20 & 50 MAs)
          - Bullish_MA_Stack      (price > EMA20 > SMA50 > SMA200)
          - Bullish_Regime        (SMA50 > SMA200)
          - EMA_Cross_Type        (faster EMA cross signal)

    [C] Momentum
          - RSI (14), RSI_Slope, RSI_Regime, RSI_Bullish
          - MACD (12/26/9), MACD_Hist, MACD_Hist_Slope
          - ROC_10

    [D] Volatility
          - ATR (14), ATR_Pct
          - Bollinger Bands: BB_Width, BB_Pct, BB_Room_To_Run
          - HVol_20 (annualised historical volatility)
          - Vol_Regime (-1 low / 0 normal / +1 high)

    [E] Trend Quality
          - ADX (14), Plus_DI, Minus_DI
          - ADX_Strong, ADX_Bullish, Trend_Regime

    [F] ML Features
          - Confirm_Count (multi-signal conviction score)
          - Temporal: Q4_Season, Summer_Lull (commodity seasonality)
""")
print(f"  Next: python 03_signal_generator_cfd.py")
print(f"{'='*80}")