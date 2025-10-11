# streamlit_app.py
# Single-file Strategy–Regime sandbox with safe Sharpe + Finnhub fallback

import os
import time
import math
import json
import io
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

# ------------- Utilities -------------

def to_utc_ts(dt: datetime) -> int:
    """Datetime -> epoch seconds (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp())

def normalize_symbol_for_finnhub(sym: str) -> str:
    """
    Try to map common formats to Finnhub style:
      - 'SHOP.TO' -> 'TSX:SHOP'
      - 'RELIANCE.NS' -> 'NSE:RELIANCE'
      - leave US tickers as-is (AAPL, MSFT)
    """
    s = sym.strip().upper()
    if s.endswith(".TO"):
        return f"TSX:{s[:-3]}"
    if s.endswith(".NS"):
        return f"NSE:{s[:-3]}"
    # Accept already-prefixed forms and plain US
    return s

# --- PATCH 1: safe Sharpe helper -------------------
def ensure_sharpe_column(df: pd.DataFrame,
                         ret_col_candidates=("Return","StrategyReturn","ret","PnL"),
                         trading_days=252) -> pd.DataFrame:
    """
    Guarantees df has a 'Sharpe' column.
    - Tries common return column names.
    - Handles empty/constant-return cases.
    - Annualizes using sqrt(trading_days) if per-row is daily.
    """
    if df is None or len(df) == 0:
        return df

    # Normalize alternate names if present
    alt = {"Sharpe Ratio": "Sharpe", "SR": "Sharpe", "sharpe": "Sharpe"}
    to_rename = {k: v for k, v in alt.items() if k in df.columns and "Sharpe" not in df.columns}
    if to_rename:
        df.rename(columns=to_rename, inplace=True)

    if "Sharpe" in df.columns:
        return df

    # Find a usable returns column
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
    if sigma and sigma != 0:
        daily_sr = mu / sigma
        df["Sharpe"] = daily_sr * math.sqrt(trading_days)
    else:
        df["Sharpe"] = np.nan
    return df
# ---------------------------------------------------

def compute_drawdown(equity_curve: pd.Series) -> float:
    """Max drawdown (as a negative percentage)."""
    if equity_curve is None or equity_curve.empty:
        return np.nan
    running_max = equity_curve.cummax()
    dd = (equity_curve / running_max) - 1.0
    return dd.min()  # negative number

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
    """Try yfinance first; returns dataframe with Date, Close if ok, else empty."""
    if not YF_IMPORT_OK:
        return pd.DataFrame()
    try:
        yf_ticker = yf.Ticker(symbol)
        df = yf_ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "Date"})
        if not pd.api.types.is_datetime64_any_dtype(out["Date"]):
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna()
        out["Ticker"] = symbol
        return out
    except Exception:
        return pd.DataFrame()

def fetch_finnhub(symbol: str, start: datetime, end: datetime, interval="1d") -> pd.DataFrame:
    """Fallback to Finnhub /stock/candle when yfinance fails."""
    if not FINNHUB_API_KEY:
        return pd.DataFrame()
    # Map Streamlit interval to Finnhub resolution
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
    """Fetch Close prices for a symbol; try yfinance then Finnhub."""
    df = fetch_yfinance(symbol, start, end, interval)
    if df.empty:
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
            st.warning(f"⚠️ No data for {t} (yfinance+Finnhub).")
            continue
        frames.append(df)
        # polite spacing for Finnhub
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_values(["Ticker","Date"]).reset_index(drop=True)
    return out

# ------------- Strategies -------------

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change().fillna(0.0)
    return df

def strat_buy_hold(df: pd.DataFrame) -> pd.Series:
    """Daily strategy return for simple buy/hold = daily close pct-change."""
    return df["Close"].pct_change().fillna(0.0)

def strat_sma_cross(df: pd.DataFrame, fast=10, slow=40) -> pd.Series:
    d = df.copy()
    d["fast"] = d["Close"].rolling(fast, min_periods=fast).mean()
    d["slow"] = d["Close"].rolling(slow, min_periods=slow).mean()
    # position: 1 when fast>slow else 0
    d["pos"] = (d["fast"] > d["slow"]).astype(int)
    ret = d["pos"].shift(1, fill_value=0) * d["Close"].pct_change().fillna(0.0)
    return ret

def strat_bollinger(df: pd.DataFrame, window=20, k=2.0) -> pd.Series:
    d = df.copy()
    m = d["Close"].rolling(window, min_periods=window).mean()
    s = d["Close"].rolling(window, min_periods=window).std(ddof=0)
    upper = m + k * s
    lower = m - k * s
    # simple rule: long when Close < lower (mean reversion), flat when Close > upper
    long_sig = (d["Close"] < lower).astype(int)
    exit_sig = (d["Close"] > upper).astype(int)
    pos = long_sig.copy()
    pos = pos.where(~exit_sig, 0)
    pos = pos.ffill().fillna(0)
    ret = pos.shift(1, fill_value=0) * d["Close"].pct_change().fillna(0.0)
    return ret

STRATEGIES = {
    "buy_hold": lambda df, **kw: strat_buy_hold(df),
    "sma_cross": lambda df, **kw: strat_sma_cross(df, fast=kw.get("fast", 10), slow=kw.get("slow", 40)),
    "bollinger": lambda df, **kw: strat_bollinger(df, window=kw.get("window", 20), k=kw.get("k", 2.0)),
}

def run_backtest_one(df_ticker: pd.DataFrame, strategy_key: str, **params) -> dict:
    """
    Returns dict with metrics for one ticker/strategy.
    """
    if df_ticker is None or df_ticker.empty:
        return {}

    # compute strategy daily returns
    try:
        strat_fn = STRATEGIES[strategy_key]
    except KeyError:
        return {}

    strat_ret = pd.to_numeric(strat_fn(df_ticker, **params), errors="coerce").fillna(0.0)
    equity = (1.0 + strat_ret).cumprod()

    # metrics
    m = {
        "Ticker": df_ticker["Ticker"].iloc[0],
        "Strategy": strategy_key,
        "Params": json.dumps(params, sort_keys=True),
        "CAGR": cagr_from_equity(equity),
        "MaxDD": compute_drawdown(equity),
        "StrategyReturn": strat_ret,  # keep for Sharpe helper
    }

    # Place in a one-row dataframe to reuse sharpe helper
    row_df = pd.DataFrame({"StrategyReturn": strat_ret})
    row_df = ensure_sharpe_column(row_df)  # computes annualized Sharpe into 'Sharpe'

    # Convert to scalar (mean of per-row Sharpe not meaningful; use recompute)
    # We recompute properly on the vector:
    series = strat_ret
    mu = series.mean()
    sigma = series.std(ddof=0)
    m["Sharpe"] = (mu / sigma * math.sqrt(252)) if sigma not in (None, 0) else np.nan

    return m

def grid_search(prices_long: pd.DataFrame,
                selected_strategies: list,
                sma_fast=(10, 20),
                sma_slow=(40, 100),
                boll_win=(20,),
                boll_k=(2.0,)) -> pd.DataFrame:
    """
    Very small grid to keep runtime low on Render free tiers.
    """
    results = []
    if prices_long is None or prices_long.empty:
        return pd.DataFrame()

    for tkr, df_t in prices_long.groupby("Ticker", sort=False):
        df_t = df_t.sort_values("Date").reset_index(drop=True)
        df_t = add_returns(df_t)

        for s in selected_strategies:
            if s == "buy_hold":
                m = run_backtest_one(df_t, "buy_hold")
                if m:
                    results.append(m)
            elif s == "sma_cross":
                for f in sma_fast:
                    for sl in sma_slow:
                        if f >= sl:
                            continue
                        m = run_backtest_one(df_t, "sma_cross", fast=f, slow=sl)
                        if m:
                            results.append(m)
            elif s == "bollinger":
                for w in boll_win:
                    for k in boll_k:
                        m = run_backtest_one(df_t, "bollinger", window=w, k=k)
                        if m:
                            results.append(m)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # --- ensure Sharpe exists & guard ranking/sorting
    df = ensure_sharpe_column(df, ret_col_candidates=("StrategyReturn",))
    if "Sharpe" not in df.columns:
        df["Sharpe"] = np.nan

    has_sharpe_vals = df["Sharpe"].notna().any()
    if has_sharpe_vals:
        df["Sharpe_rank"] = df["Sharpe"].rank(ascending=False, method="min")
        df = df.sort_values(["Ticker","Sharpe"], ascending=[True, False], na_position="last")
    else:
        df["Sharpe_rank"] = np.nan

    # tidying
    if "StrategyReturn" in df.columns:
        df.drop(columns=["StrategyReturn"], inplace=True, errors="ignore")

    return df.reset_index(drop=True)

# ------------- UI -------------

with st.sidebar:
    st.subheader("⚙️ Settings")
    default_syms = "AAPL, MSFT, NVDA"
    syms = st.text_input("Tickers (comma-separated)", value=default_syms)
    interval = st.selectbox("Interval", ["1d"], index=0)
    days_back = st.number_input("Days back", min_value=30, max_value=3650, value=365)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=int(days_back))
    st.caption("Tip: TSX use `.TO` (e.g., SHOP.TO). NSE use `.NS` (e.g., RELIANCE.NS).")

    st.markdown("---")
    st.checkbox("Buy & Hold", value=True, key="s_buy")
    st.checkbox("SMA Cross", value=True, key="s_sma")
    st.checkbox("Bollinger", value=True, key="s_boll")

    st.markdown("**SMA params**")
    f_opts = st.multiselect("Fast (days)", [5,10,20], default=[10,20])
    s_opts = st.multiselect("Slow (days)", [20,40,100,200], default=[40,100])

    st.markdown("**Bollinger params**")
    w_opts = st.multiselect("Window", [10,20,30], default=[20])
    k_opts = st.multiselect("k (std dev)", [1.5,2.0,2.5], default=[2.0])

    run_btn = st.button("🚀 Run Backtests", type="primary")

st.write("")

if run_btn:
    tickers = [t.strip() for t in syms.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    with st.status("Fetching prices (yfinance → Finnhub fallback)…", expanded=False) as status:
        prices = load_prices_for_universe(
            tickers, start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.min.time()), interval=interval
        )
        if prices.empty:
            st.error("No price data loaded. Check symbols, dates, or API key.")
            st.stop()
        status.update(label="Running grid search…")

        selected = []
        if st.session_state.get("s_buy"): selected.append("buy_hold")
        if st.session_state.get("s_sma"): selected.append("sma_cross")
        if st.session_state.get("s_boll"): selected.append("bollinger")
        if not selected:
            selected = ["buy_hold"]

        results = grid_search(
            prices, selected,
            sma_fast=tuple(sorted(set(f_opts))),
            sma_slow=tuple(sorted(set(s_opts))),
            boll_win=tuple(sorted(set(w_opts))),
            boll_k=tuple(sorted(set(k_opts))),
        )
        status.update(label="Done", state="complete")

    if results is None or results.empty:
        st.warning("No metrics to display — maybe no valid rows/strategies?")
    else:
        # Display tidy table
        show_cols = ["Ticker","Strategy","Params","Sharpe","CAGR","MaxDD","Sharpe_rank"]
        for c in show_cols:
            if c not in results.columns:
                results[c] = np.nan
        fmt = results.copy()
        fmt["Sharpe"] = fmt["Sharpe"].map(lambda x: f"{x:0.2f}" if pd.notna(x) else "")
        fmt["CAGR"] = fmt["CAGR"].map(lambda x: f"{100*x:0.2f}%" if pd.notna(x) else "")
        fmt["MaxDD"] = fmt["MaxDD"].map(lambda x: f"{100*x:0.2f}%" if pd.notna(x) else "")
        st.subheader("Results")
        st.dataframe(fmt[show_cols], use_container_width=True, hide_index=True)

        # Per-ticker best row by Sharpe
        st.subheader("Per-Ticker Best by Sharpe")
        # guard Sharpe again
        results = ensure_sharpe_column(results, ret_col_candidates=("StrategyReturn",))
        if "Sharpe" not in results.columns or not results["Sharpe"].notna().any():
            st.info("Sharpe not available to rank winners.")
        else:
            winners = results.sort_values(["Ticker","Sharpe"], ascending=[True, False]) \
                             .groupby("Ticker", as_index=False).head(1)
            wfmt = winners.copy()
            wfmt["Sharpe"] = wfmt["Sharpe"].map(lambda x: f"{x:0.2f}")
            wfmt["CAGR"] = wfmt["CAGR"].map(lambda x: f"{100*x:0.2f}%")
            wfmt["MaxDD"] = wfmt["MaxDD"].map(lambda x: f"{100*x:0.2f}%")
            st.dataframe(wfmt[show_cols], use_container_width=True, hide_index=True)

# Footer diagnostics
with st.expander("Diagnostics"):
    st.write(f"yfinance import: {'OK' if YF_IMPORT_OK else 'Unavailable'}")
    st.write(f"Finnhub key present: {'Yes' if bool(FINNHUB_API_KEY) else 'No'}")