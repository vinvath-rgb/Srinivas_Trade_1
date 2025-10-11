# streamlit_app.py
# Single-file Strategy–Regime sandbox with safe Sharpe and triple fallback:
# yfinance -> Stooq -> Finnhub

import os
import time
import math
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ------------- Streamlit setup -------------
st.set_page_config(page_title="Strategy–Regime Matrix (Single File)", layout="wide")
st.title("📈 Strategy–Regime Matrix (Single File)")

# ------------- Configuration -------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

YF_IMPORT_OK = True
try:
    import yfinance as yf
except Exception:
    YF_IMPORT_OK = False

# ------------- Helpers -------------

def to_utc_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp())

def normalize_symbol_for_finnhub(sym: str) -> str:
    s = sym.strip().upper()
    if s.endswith(".TO"):
        return f"TSX:{s[:-3]}"
    if s.endswith(".NS"):
        return f"NSE:{s[:-3]}"
    return s

def map_to_stooq_symbol(sym: str) -> str:
    """
    Map common ticker forms to Stooq symbol:
      AAPL      -> aapl.us
      MSFT      -> msft.us
      SPY       -> spy.us
      SHOP.TO   -> shop.to
      ^GSPC     -> ^spx? (Stooq uses indices differently; we won't remap here)
    Notes:
      - Stooq does NOT generally support NSE tickers ('.NS'). We'll return '' to skip.
    """
    s = sym.strip()
    if not s:
        return ""
    s_up = s.upper()
    # Skip weird symbols Stooq likely won't serve
    if s_up.endswith(".NS"):     # NSE not reliably on Stooq
        return ""
    # TSX
    if s_up.endswith(".TO"):
        return s_up[:-3].lower() + ".to"
    # If already has stooq suffix, pass through
    if s_up.endswith((".US", ".TO", ".DE", ".JP", ".HK", ".PL")):
        return s_up.lower()
    # default assume US
    return s_up.lower() + ".us"

# --- Sharpe helper (PATCH 1) ----------------------
def ensure_sharpe_column(df: pd.DataFrame,
                         ret_col_candidates=("Return","StrategyReturn","ret","PnL"),
                         trading_days=252) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    alt = {"Sharpe Ratio": "Sharpe", "SR": "Sharpe", "sharpe": "Sharpe"}
    to_rename = {k: v for k, v in alt.items() if k in df.columns and "Sharpe" not in df.columns}
    if to_rename:
        df.rename(columns=to_rename, inplace=True)
    if "Sharpe" in df.columns:
        return df
    ret_col = next((c for c in ret_col_candidates if c in df.columns), None)
    if ret_col is None:
        df["Sharpe"] = np.nan
        return df
    series = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if series.empty:
        df["Sharpe"] = np.nan
        return df
    mu = series.mean()
    sigma = series.std(ddof=0)
    df["Sharpe"] = (mu / sigma * math.sqrt(trading_days)) if sigma not in (None, 0) else np.nan
    return df
# --------------------------------------------------

def compute_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve is None or equity_curve.empty:
        return np.nan
    running_max = equity_curve.cummax()
    dd = (equity_curve / running_max) - 1.0
    return dd.min()

def cagr_from_equity(equity_curve: pd.Series, periods_per_year=252) -> float:
    if equity_curve is None or equity_curve.empty:
        return np.nan
    n = len(equity_curve)
    if n < 2:
        return np.nan
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n / periods_per_year
    if years <= 0 or total_return <= 0:
        return np.nan
    return total_return ** (1/years) - 1

# ------------- Data fetchers -------------

def fetch_yfinance(symbol: str, start: datetime, end: datetime, interval="1d") -> pd.DataFrame:
    if not YF_IMPORT_OK:
        return pd.DataFrame()
    try:
        yf_ticker = yf.Ticker(symbol)
        df = yf_ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.reset_index()[["Date", "Close"]]
        if not pd.api.types.is_datetime64_any_dtype(out["Date"]):
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna()
        out["Ticker"] = symbol
        return out[["Date","Ticker","Close"]]
    except Exception:
        return pd.DataFrame()

def fetch_stooq(symbol: str, start: datetime, end: datetime, interval="1d") -> pd.DataFrame:
    """
    Pull daily candles via Stooq CSV endpoint.
    URL: https://stooq.com/q/d/l/?s={symbol}&i=d
    - Only supports daily ('i=d'). For other intervals we fall back to daily.
    - We will filter the date range client-side.
    """
    stooq_sym = map_to_stooq_symbol(symbol)
    if not stooq_sym:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(r.text))
        if df is None or df.empty:
            return pd.DataFrame()
        # Expected columns: Date,Open,High,Low,Close,Volume
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date","Close"])
        df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
        if df.empty:
            return pd.DataFrame()
        out = df[["Date","Close"]].copy()
        out["Ticker"] = symbol
        return out[["Date","Ticker","Close"]].sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def fetch_finnhub(symbol: str, start: datetime, end: datetime, interval="1d") -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        return pd.DataFrame()
    res_map = {"1d": "D", "1h": "60", "30m": "30", "15m": "15", "5m": "5"}
    resolution = res_map.get(interval, "D")
    sym = normalize_symbol_for_finnhub(symbol)
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": sym,
        "resolution": resolution,
        "from": to_utc_ts(start),
        "to": to_utc_ts(end + timedelta(days=1)),
        "token": FINNHUB_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or data.get("s") != "ok":
            return pd.DataFrame()
        df = pd.DataFrame({
            "Date": pd.to_datetime(data["t"], unit="s", utc=True).tz_convert(None),
            "Close": data["c"],
        })
        if df.empty:
            return pd.DataFrame()
        df["Ticker"] = symbol
        return df[["Date", "Ticker", "Close"]]
    except Exception:
        return pd.DataFrame()

def fetch_prices(symbol: str, start: datetime, end: datetime, interval="1d") -> pd.DataFrame:
    # 1) yfinance
    df = fetch_yfinance(symbol, start, end, interval)
    if not df.empty:
        return df
    # 2) Stooq (daily only)
    df = fetch_stooq(symbol, start, end, interval)
    if not df.empty:
        return df
    # 3) Finnhub
    df = fetch_finnhub(symbol, start, end, interval)
    return df

def load_prices_for_universe(tickers: list, start: datetime, end: datetime, interval="1d") -> pd.DataFrame:
    frames = []
    for t in tickers:
        t = t.strip()
        if not t:
            continue
        df = fetch_prices(t, start, end, interval)
        if df.empty:
            st.warning(f"⚠️ No data for {t} (Yahoo → Stooq → Finnhub).")
            continue
        frames.append(df)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["Ticker","Date"]).reset_index(drop=True)

# ------------- Strategies -------------

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change().fillna(0.0)
    return df

def strat_buy_hold(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().fillna(0.0)

def strat_sma_cross(df: pd.DataFrame, fast=10, slow=40) -> pd.Series:
    d = df.copy()
    d["fast"] = d["Close"].rolling(fast, min_periods=fast).mean()
    d["slow"] = d["Close"].rolling(slow, min_periods=slow).mean()
    d["pos"] = (d["fast"] > d["slow"]).astype(int)
    return d["pos"].shift(1, fill_value=0) * d["Close"].pct_change().fillna(0.0)

def strat_bollinger(df: pd.DataFrame, window=20, k=2.0) -> pd.Series:
    d = df.copy()
    m = d["Close"].rolling(window, min_periods=window).mean()
    s = d["Close"].rolling(window, min_periods=window).std(ddof=0)
    upper = m + k * s
    lower = m - k * s
    long_sig = (d["Close"] < lower).astype(int)
    exit_sig = (d["Close"] > upper).astype(int)
    pos = long_sig.copy()
    pos = pos.where(~exit_sig, 0)  # <-- will error in Python; change to below