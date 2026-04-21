"""
================================================================================
  CFD GOLD & SILVER — SMART INCREMENTAL DATA UPDATER  [v1]
  Reference: NIFTY 50 Data Updater v6 (AI-Driven Algorithmic Trading System)

  INSTRUMENTS:
    - CFD on Gold  (GC=F  → XAUUSD proxy via yfinance)
    - CFD on Silver (SI=F → XAGUSD proxy via yfinance)

  NOTES:
    - yfinance tickers GC=F (Gold Futures) and SI=F (Silver Futures) are the
      closest freely available proxies for CFD Gold / CFD Silver price data.
    - Historical data via yfinance goes back to ~1990 for these contracts.
    - OUTPUT: 02_data/individual_stocks/{SYMBOL}.csv  (one file per instrument)
    - No master CSV — per-instrument pipeline only.
================================================================================
"""

import yfinance as yf
import pandas as pd
import os, time
from datetime import datetime, date, timedelta

START_DATE     = "1990-01-01"
INDIVIDUAL_DIR = "02_data/individual_stocks"
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

# (yfinance_ticker, output_symbol_name)
CFD_INSTRUMENTS = [
    ("GC=F",   "GOLD_CFD"),    # most reliable
    ("SI=F",   "SILVER_CFD"),  # CFD on Silver — Silver Futures front contract
    ]


def clean_ohlcv(raw_df, symbol):
    """Normalise raw yfinance DataFrame into a clean OHLCV frame."""
    df = raw_df.copy()
    df.reset_index(inplace=True)
    df.columns = [c.lower().strip() for c in df.columns]

    # yfinance may return 'datetime' or 'date' as the index name
    date_col = "datetime" if "datetime" in df.columns else "date"
    df.rename(columns={date_col: "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)
    df["date"] = df["date"].dt.normalize()

    df["symbol"] = symbol
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "symbol"]
            if c in df.columns]
    df = df[keep].copy()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["close"], inplace=True)
    df = df[df["close"] > 0].copy()

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].round(4)

    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype("int64")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def smart_update_instrument(yf_ticker, symbol, max_retries=2):
    """
    Incrementally update a single instrument CSV.

    Returns: (status, new_rows_added, total_rows, error_message_or_None)
      status: 'new' | 'updated' | 'ok' | 'error'
    """
    cache_path = f"{INDIVIDUAL_DIR}/{symbol}.csv"
    today = date.today()
    existing_df = None

    if not os.path.exists(cache_path):
        fetch_from, is_new = START_DATE, True
    else:
        existing_df = pd.read_csv(cache_path, parse_dates=["date"])
        if existing_df.empty:
            fetch_from, is_new = START_DATE, True
        else:
            last_date = existing_df["date"].max().date()
            # Already up-to-date (within last 4 calendar days covers weekends)
            if last_date >= today - timedelta(days=4):
                return ("ok", 0, len(existing_df), None)
            fetch_from = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            is_new = False

    for attempt in range(max_retries + 1):
        try:
            raw = yf.Ticker(yf_ticker).history(
                start=fetch_from,
                end=today.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

            if raw.empty:
                if is_new:
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    return ("error", 0, 0, "Empty response from yfinance")
                # Incremental update but nothing new yet (e.g. weekend run)
                return ("ok", 0, len(existing_df), None)

            new_df = clean_ohlcv(raw, symbol)

            if is_new:
                final_df = new_df
            else:
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                final_df.drop_duplicates(subset=["date"], keep="last", inplace=True)
                final_df.sort_values("date", inplace=True)
                final_df.reset_index(drop=True, inplace=True)

            final_df.to_csv(cache_path, index=False, date_format="%Y-%m-%d")
            return ("new" if is_new else "updated", len(new_df), len(final_df), None)

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return ("error", 0, 0, str(e))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  CFD GOLD & SILVER DATA UPDATER  [v1 — per-instrument CSV]")
    print(f"{'='*70}")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}  |  {len(CFD_INSTRUMENTS)} instruments")
    print(f"  History from : {START_DATE}")
    print(f"  Output dir   : {INDIVIDUAL_DIR}/")
    print(f"  Tickers used : GC=F (Gold Futures), SI=F (Silver Futures)\n")

    for idx, (yf_t, sym) in enumerate(CFD_INSTRUMENTS, 1):
        print(f"[{idx}/{len(CFD_INSTRUMENTS)}] {sym:15s}", end="  ")
        status, added, total, err = smart_update_instrument(yf_t, sym)

        if status == "new":
            print(f"NEW      {total:,} rows  (full history since {START_DATE})")
        elif status == "updated":
            print(f"UPDATED  +{added} rows  (total {total:,})")
        elif status == "ok":
            print(f"OK       {total:,} rows  (already up-to-date)")
        else:
            print(f"ERROR    {err}")

        time.sleep(0.5)   # polite delay between requests

    print(f"\n  Output files:")
    for _, sym in CFD_INSTRUMENTS:
        path = f"{INDIVIDUAL_DIR}/{sym}.csv"
        if os.path.exists(path):
            n = sum(1 for _ in open(path)) - 1   # subtract header
            print(f"    {path}  ({n:,} rows)")
        else:
            print(f"    {path}  — not created (check errors above)")

    print(f"\n  Next: python 02_feature_engineering_cfd.py")
    print(f"{'='*70}\n")