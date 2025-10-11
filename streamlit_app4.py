# streamlit_app.py
# Strategy–Regime sandbox with safe Sharpe and triple fallback (Yahoo -> Stooq -> Finnhub)

import os
import io
import time
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- Streamlit setup ----------
st.set_page_config(page_title="Strategy–Regime Matrix (Single File)", layout="wide")
st.title("📈 Strategy–Regime Matrix (Single File)")

# ---------- Config / environment ----------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

YF_IMPORT_OK = True
try:
    import yfinance as yf  # noqa: F401
except Exception:
    YF_IMPORT_OK = False

# ---------- Helpers ----------
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
    Map common tickers to Stooq symbol:
      AAPL      -> aapl.us
      MSFT      -> msft.us
      SPY       -> spy.us
      SHOP.TO   -> shop.to
    Notes:
      - Stooq does NOT reliably support NSE tickers ('.NS') -> return '' to skip.
    """
    s = sym.strip()
    if not s:
        return ""
    s_up = s.upper()
    if s_up.endswith(".NS"):
        return ""  # skip NSE for Stooq
    if s_up.endswith(".TO"):
        return s_up[:-3].lower() + ".to"
    if s_up.endswith((".US", ".TO", ".DE", ".JP", ".HK", ".PL")):
        return s_up.lower()
    return s_up.lower() + ".us"  # default to US

# --- Sharpe helper (safe) ---
def ensure_sharpe_column(df: pd.DataFrame,
                         ret_col_candidates=("Return", "StrategyReturn", "ret", "PnL"),
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

def compute_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve is None or equity_curve.empty:
        return np.nan
    running_max = equity_curve.cummax()
    dd = (equity_curve / running_max) - 1.0
    return float(dd.min()) if len(dd) else np.nan

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
    return float(total_return ** (1 / years) - 1)

# ---------- Data fetchers ----------
@st.cache_data(show_spinner=False)
def fetch_yfinance(symbol: str, start: datetime, end: datetime, interval="1d") -> pd.DataFrame:
    if not YF_IMPORT_OK:
        return pd.DataFrame()
    try:
        yf_ticker = yf.Ticker(symbol)
        df = yf_ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.reset_index()
        # yfinance returns 'Date' or 'Datetime' index depending on interval
        date_col = "Date" if "Date" in out.columns else ("Datetime" if "Datetime" in out.columns else None)
        if date_col is None:
            return pd.DataFrame()
        out.rename(columns={date_col: "Date"}, inplace=True)
        if not pd.api.types.is_datetime64_any_dtype(out["Date"]):
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date", "Close"])
        out["Ticker"] = symbol
        return out[["Date", "Ticker", "Close"]]
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_stooq(symbol: str, start: datetime, end: datetime, interval="1d") -> pd.DataFrame:
    """
    Stooq supports daily via CSV:
      https://stooq.com/q/d/l/?s={symbol}&i=d
    We filter client-side by date range.
    """
    stooq_sym = map_to_stooq_symbol(symbol)
    if not stooq_sym:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        # Use io.StringIO (pd.compat is deprecated)
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty or "Date" not in df or "Close" not in df:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])
        df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
        if df.empty:
            return pd.DataFrame()
        out = df[["Date", "Close"]].copy()
        out["Ticker"] = symbol
        return out[["Date", "Ticker", "Close"]].sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
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
            "Date": pd.to_datetime(data.get("t", []), unit="s", utc=True).tz_convert(None),
            "Close": data.get("c", []),
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
        time.sleep(0.2)  # gentle on free APIs
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["Ticker", "Date"]).reset_index(drop=True)

# ---------- Strategies ----------
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Return"] = d["Close"].pct_change().fillna(0.0)
    return d

def strat_buy_hold(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().fillna(0.0)

def strat_sma_cross(df: pd.DataFrame, fast=10, slow=40) -> pd.Series:
    d = df.copy()
    d["fast"] = d["Close"].rolling(fast, min_periods=fast).mean()
    d["slow"] = d["Close"].rolling(slow, min_periods=slow).mean()
    d["pos"] = (d["fast"] > d["slow"]).astype(int)
    # use yesterday's position
    return d["pos"].shift(1, fill_value=0) * d["Close"].pct_change().fillna(0.0)

def strat_bollinger(df: pd.DataFrame, window=20, k=2.0) -> pd.Series:
    """
    Long-only mean-reversion: enter when price < lower band; exit when price > upper band.
    Stateful position with carry-forward; vectorized via ffill.
    """
    d = df.copy()
    m = d["Close"].rolling(window, min_periods=window).mean()
    s = d["Close"].rolling(window, min_periods=window).std(ddof=0)
    upper = m + k * s
    lower = m - k * s

    buy_sig = d["Close"] < lower
    sell_sig = d["Close"] > upper

    # Build position: set 1 on buy days, 0 on sell days, carry last known otherwise
    pos = pd.Series(np.nan, index=d.index)
    pos[buy_sig] = 1
    pos[sell_sig] = 0
    pos = pos.ffill().fillna(0).astype(int)

    returns = d["Close"].pct_change().fillna(0.0)
    strat_ret = pos.shift(1, fill_value=0) * returns
    return strat_ret

# ---------- Simple regime detector (volatility percentile) ----------
def compute_vol_regime(df: pd.DataFrame, window=20, low_pct=0.4, high_pct=0.6) -> pd.Series:
    """
    Percentile-based vol regime on a single series (per ticker run).
    Returns labels: 'Low', 'Neutral', 'High'
    """
    d = df.copy()
    rets = d["Close"].pct_change()
    vol = rets.rolling(window, min_periods=window).std(ddof=0)
    q_low = vol.quantile(low_pct)
    q_high = vol.quantile(high_pct)
    regime = pd.Series("Neutral", index=d.index)
    regime[vol <= q_low] = "Low"
    regime[vol >= q_high] = "High"
    return regime

# ---------- Backtest / metrics ----------
def run_strategy(df_one: pd.DataFrame, strat_key: str, params: dict) -> pd.Series:
    if strat_key == "buy_hold":
        return strat_buy_hold(df_one)
    if strat_key == "sma_cross":
        fast = int(params.get("fast", 10))
        slow = int(params.get("slow", 40))
        return strat_sma_cross(df_one, fast=fast, slow=slow)
    if strat_key == "bollinger":
        window = int(params.get("bb_window", 20))
        k = float(params.get("bb_k", 2.0))
        return strat_bollinger(df_one, window=window, k=k)
    raise ValueError(f"Unknown strategy: {strat_key}")

def summarize_performance(returns: pd.Series, periods_per_year=252) -> dict:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1 + returns).cumprod()
    # Annualized Sharpe
    mu = returns.mean()
    sigma = returns.std(ddof=0)
    sharpe = (mu / sigma * np.sqrt(periods_per_year)) if sigma not in (None, 0) else np.nan
    # CAGR & MaxDD
    cagr = cagr_from_equity(equity, periods_per_year=periods_per_year)
    maxdd = compute_drawdown(equity)
    total = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd, "TotalReturn": total, "Equity": equity}

# ---------- UI ----------
with st.sidebar:
    st.subheader("⚙️ Inputs")

    default_syms = "AAPL, MSFT, SPY"
    tickers_text = st.text_input("Tickers (comma-separated)", value=default_syms)
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Start date", value=(datetime.utcnow().date() - timedelta(days=365*3)))
    with col_dates[1]:
        end_date = st.date_input("End date", value=datetime.utcnow().date())

    interval = st.selectbox("Interval", ["1d"], index=0, help="This demo uses daily bars.")

    st.markdown("---")
    st.caption("📉 Strategy")
    strat_key = st.selectbox("Select strategy", ["sma_cross", "bollinger", "buy_hold"], index=0)
    fast = st.number_input("SMA fast", min_value=2, max_value=200, value=10, step=1)
    slow = st.number_input("SMA slow", min_value=5, max_value=400, value=40, step=1)
    bb_window = st.number_input("Bollinger window", min_value=5, max_value=200, value=20, step=1)
    bb_k = st.number_input("Bollinger k", min_value=0.5, max_value=4.0, value=2.0, step=0.1)

    st.markdown("---")
    st.caption("🧭 Regime (vol percentile)")
    vol_window = st.number_input("Vol window", min_value=10, max_value=100, value=20, step=1)
    low_pct = st.slider("Low vol percentile", 0.05, 0.5, 0.40, 0.01)
    high_pct = st.slider("High vol percentile", 0.5, 0.95, 0.60, 0.01)

    st.markdown("---")
    uploaded = st.file_uploader("Optional CSV (Date, Ticker, Close)", type=["csv"])
    st.caption("If provided, CSV rows for matching tickers override online fetch.")

# Gentle env hints (never stop the app)
if not YF_IMPORT_OK:
    st.info("ℹ️ `yfinance` import failed — will try Stooq → Finnhub.")
if not FINNHUB_API_KEY:
    st.info("ℹ️ `FINNHUB_API_KEY` not set — Finnhub fallback disabled.")

# Build data
start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.min.time())

# Prepare override DF from CSV if given
csv_override = pd.DataFrame()
if uploaded is not None:
    try:
        tmp = pd.read_csv(uploaded)
        # Normalize columns
        cols = {c.lower(): c for c in tmp.columns}
        # Expect 'Date', 'Ticker', 'Close' in any case
        # Try to find (case-insensitive)
        def find_col(name):
            for c in tmp.columns:
                if c.lower() == name:
                    return c
            return None

        c_date = find_col("date")
        c_tic = find_col("ticker")
        c_close = find_col("close")
        if c_date and c_tic and c_close:
            csv_override = tmp[[c_date, c_tic, c_close]].copy()
            csv_override.columns = ["Date", "Ticker", "Close"]
            csv_override["Date"] = pd.to_datetime(csv_override["Date"], errors="coerce")
            csv_override = csv_override.dropna(subset=["Date", "Ticker", "Close"])
        else:
            st.warning("⚠️ CSV must have columns: Date, Ticker, Close")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

def get_price_df_for_symbol(sym: str) -> pd.DataFrame:
    # Use CSV override rows if available for this symbol
    if not csv_override.empty:
        sub = csv_override[csv_override["Ticker"].astype(str).str.upper() == sym.upper()].copy()
        if not sub.empty:
            sub = sub[(sub["Date"] >= pd.to_datetime(start_dt)) & (sub["Date"] <= pd.to_datetime(end_dt))]
            sub = sub.sort_values("Date").reset_index(drop=True)
            if not sub.empty:
                return sub[["Date", "Ticker", "Close"]]

    # Otherwise fetch online with fallbacks
    return fetch_prices(sym, start_dt, end_dt, interval)

# Run
if st.button("Run Backtest", type="primary"):
    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        full_results = []
        equity_cols = {}

        params = {
            "fast": int(fast),
            "slow": int(slow),
            "bb_window": int(bb_window),
            "bb_k": float(bb_k),
        }

        for sym in tickers:
            df = get_price_df_for_symbol(sym)
            if df.empty:
                st.warning(f"⚠️ No usable data for {sym}.")
                continue

            # Strategy returns
            try:
                strat_ret = run_strategy(df, strat_key, params)
            except Exception as e:
                st.error(f"{sym}: strategy failed — {e}")
                continue

            # Baseline buy & hold
            bh_ret = strat_buy_hold(df)

            # Metrics
            strat_stats = summarize_performance(strat_ret)
            bh_stats = summarize_performance(bh_ret)

            equity_cols[f"{sym} ({strat_key})"] = strat_stats["Equity"]
            equity_cols[f"{sym} (buy&hold)"] = bh_stats["Equity"]

            full_results.append({
                "Ticker": sym,
                "Strategy": strat_key,
                "CAGR": strat_stats["CAGR"],
                "Sharpe": strat_stats["Sharpe"],
                "MaxDD": strat_stats["MaxDD"],
                "TotalReturn": strat_stats["TotalReturn"],
                "BH_CAGR": bh_stats["CAGR"],
                "BH_Sharpe": bh_stats["Sharpe"],
                "BH_MaxDD": bh_stats["MaxDD"],
                "BH_TotalReturn": bh_stats["TotalReturn"],
                "DataPoints": int(len(df)),
            })

        if not full_results:
            st.warning("No results to display.")
        else:
            # Table
            res_df = pd.DataFrame(full_results)
            # A little formatting
            fmt_cols = ["CAGR", "Sharpe", "MaxDD", "TotalReturn", "BH_CAGR", "BH_Sharpe", "BH_MaxDD", "BH_TotalReturn"]
            for c in fmt_cols:
                if c in res_df:
                    res_df[c] = res_df[c].astype(float)

            st.subheader("📊 Performance Summary")
            st.dataframe(res_df, hide_index=True, use_container_width=True)

            # Equity chart
            st.subheader("💹 Equity Curves")
            if equity_cols:
                eq_df = pd.DataFrame(equity_cols).dropna(how="all")
                # Align index to datetime for charting
                eq_df.index = pd.to_datetime(eq_df.index)
                st.line_chart(eq_df, use_container_width=True)

            # Per-ticker regime view
            st.subheader("🧭 Regime (volatility percentile)")
            for sym in tickers:
                df = get_price_df_for_symbol(sym)
                if df.empty:
                    continue
                regime = compute_vol_regime(df, window=int(vol_window), low_pct=float(low_pct), high_pct=float(high_pct))
                disp = pd.DataFrame({"Date": df["Date"], "Close": df["Close"], "Regime": regime}).set_index("Date")
                with st.expander(f"{sym} regime sample", expanded=False):
                    st.dataframe(disp.tail(40), use_container_width=True)
else:
    st.caption("👆 Configure inputs on the left and click **Run Backtest**.")