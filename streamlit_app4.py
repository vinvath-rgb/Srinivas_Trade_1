# streamlit_app.py
# Strategy–Regime Matrix (Single File)
# - Data fetch fallback: yfinance -> Stooq -> Finnhub (optional)
# - Regime detection: volatility percentile (configurable)
# - Strategies: SMA cross, Bollinger, RSI (long-only for simplicity)
# - Plots: Equity Curves (rebased), Regime samples with shaded bands
# - Robust plotting fixes: alignment, rebasing, state clearing, NaN-safe percentiles

import os
import math
import time
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============== Streamlit setup ==============
st.set_page_config(page_title="Strategy–Regime Matrix (Single File)", layout="wide")
st.title("📈 Strategy–Regime Matrix (Single File)")

# ============== Config / Env ==============
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# Optional auth (set APP_PASSWORD env to enable)
def _auth():
    pw_env = os.getenv("APP_PASSWORD", "").strip()
    if not pw_env:
        return True
    with st.sidebar:
        st.subheader("🔐 App Login")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            st.session_state["_ok"] = (pw == pw_env)
    return st.session_state.get("_ok", False)

if not _auth():
    st.stop()

# Try to import yfinance
YF_IMPORT_OK = True
try:
    import yfinance as yf
except Exception:
    YF_IMPORT_OK = False

# Try to import pandas_datareader for Stooq
PDR_OK = True
try:
    from pandas_datareader import data as pdr
except Exception:
    PDR_OK = False

# ============== Helpers ==============
def to_utc_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def _clean_symbol_for_stooq(sym: str) -> str:
    """
    Stooq symbols are a bit different:
    - US equities: AAPL, MSFT (OK)
    - Indices/ETFs may vary; TSX often needs .TO (not supported well on Stooq)
    We'll just pass through and let PDR throw; this is a best-effort fallback.
    """
    return sym.strip()

def _fix_prices_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # Standardize Date/Close structure, drop bad rows, sort, ensure float Close
    df = df.copy()
    if "date" in df.columns and "Date" not in df.columns:
        df.rename(columns={"date": "Date"}, inplace=True)
    if "close" in df.columns and "Close" not in df.columns:
        df.rename(columns={"close": "Close"}, inplace=True)
    # yfinance uses index as DatetimeIndex, pdr returns column "Close"
    if "Date" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        # Try to coerce index to datetime
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    if "Date" not in df.columns:
        # Last resort: create Date from any column named 'time' etc.
        for c in df.columns:
            if c.lower() in ("time", "timestamp"):
                df["Date"] = pd.to_datetime(df[c], errors="coerce")
                break
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.set_index("Date")
    if "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
    df["Ticker"] = symbol
    return df[["Ticker", "Close"]]

def fetch_yfinance(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    if not YF_IMPORT_OK:
        raise RuntimeError("yfinance not available")
    data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise RuntimeError("yfinance returned empty")
    data = data.rename(columns={"Adj Close": "Close"})
    if "Close" not in data.columns:
        if "Adj Close" in data.columns:
            data["Close"] = data["Adj Close"]
        elif "close" in data.columns:
            data["Close"] = data["close"]
        else:
            raise RuntimeError("No Close in yfinance frame")
    data = data[["Close"]].copy()
    data = data.reset_index().rename(columns={"Date": "Date"})
    return _fix_prices_df(data, symbol)

def fetch_stooq(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    if not PDR_OK:
        raise RuntimeError("pandas_datareader not available")
    s_sym = _clean_symbol_for_stooq(symbol)
    df = pdr.DataReader(s_sym, "stooq", start, end)  # Stooq returns descending index
    if df is None or df.empty:
        raise RuntimeError("stooq returned empty")
    df = df.rename(columns={"Close": "Close"})
    df = df[["Close"]].copy().sort_index()
    df = df.reset_index().rename(columns={"Date": "Date"})
    return _fix_prices_df(df, symbol)

def fetch_finnhub(symbol: str, start: datetime, end: datetime, resolution: str = "D") -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY missing")
    base = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": to_utc_ts(start),
        "to": to_utc_ts(end + timedelta(days=1)),
        "token": FINNHUB_API_KEY
    }
    r = requests.get(base, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Finnhub HTTP {r.status_code}")
    j = r.json()
    if j.get("s") != "ok":
        raise RuntimeError(f"Finnhub status: {j.get('s')}")
    df = pd.DataFrame({"Date": pd.to_datetime(np.array(j["t"], dtype="int64"), unit="s"),
                       "Close": j["c"]})
    return _fix_prices_df(df, symbol)

def fetch_prices_for_ticker(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    # Try yfinance -> stooq -> finnhub
    errors = []
    for fn in (fetch_yfinance, fetch_stooq, fetch_finnhub):
        try:
            return fn(symbol, start, end)
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
    raise RuntimeError("All fetchers failed:\n" + "\n".join(errors))

def fetch_prices_for_universe(symbols: list[str], start: datetime, end: datetime, allow_partial=True) -> pd.DataFrame:
    frames = []
    failures = []
    for s in symbols:
        try:
            df = fetch_prices_for_ticker(s, start, end)
            frames.append(df)
        except Exception as e:
            failures.append(f"{s}: {e}")
    if not frames:
        raise RuntimeError("No symbols fetched.\n" + "\n".join(failures))
    out = pd.concat(frames, axis=0).sort_index()
    if failures and not allow_partial:
        raise RuntimeError("Some symbols failed and allow_partial=False:\n" + "\n".join(failures))
    return out

# ============== Indicators & Regimes ==============
def rolling_volatility(close: pd.Series, window=20) -> pd.Series:
    rets = close.pct_change()
    vol = rets.rolling(window, min_periods=window).std()
    return vol

def compute_regime_percentile(vol: pd.Series, low_pct=0.30, high_pct=0.70) -> pd.Series:
    # Compute thresholds over non-NaN values to avoid skew
    vol_clean = vol.dropna()
    if vol_clean.empty:
        return pd.Series(index=vol.index, dtype="float64")
    lo = np.nanpercentile(vol_clean, low_pct * 100.0)
    hi = np.nanpercentile(vol_clean, high_pct * 100.0)
    regime = pd.Series(1, index=vol.index)  # 0=low,1=mid,2=high
    regime[vol <= lo] = 0
    regime[vol >= hi] = 2
    return regime.astype("int8")

def SMA(series: pd.Series, window: int):
    return series.rolling(window, min_periods=window).mean()

def BB(series: pd.Series, window=20, k=2.0):
    mid = SMA(series, window)
    std = series.rolling(window, min_periods=window).std()
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower

def RSI(series: pd.Series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window, min_periods=window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============== Strategies (long-only examples) ==============
def strat_sma_cross(close: pd.Series, fast=10, slow=40):
    f, s = SMA(close, fast), SMA(close, slow)
    sig = (f > s).astype(int)
    return sig

def strat_bollinger(close: pd.Series, window=20, k=2.0):
    mid, upper, lower = BB(close, window, k)
    # long when price above mid and rising; exit when below mid
    sig = (close > mid).astype(int)
    return sig

def strat_rsi(close: pd.Series, window=14, low=35, high=65):
    r = RSI(close, window)
    # long when RSI crosses up from below low; flat when crosses down from above high
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[(r > low) & (r.shift(1) <= low)] = 1
    sig[(r < high) & (r.shift(1) >= high)] = 0
    sig = sig.replace(to_replace=0, method="ffill").fillna(0).astype(int)
    return sig

STRATEGIES = {
    "sma_cross": ("SMA Cross", strat_sma_cross),
    "bollinger": ("Bollinger Mid", strat_bollinger),
    "rsi": ("RSI Band", strat_rsi),
}

# ============== Backtest (vectorized, no commissions/slippage) ==============
def run_backtest(close: pd.Series, signal: pd.Series) -> pd.Series:
    # Align
    df = pd.concat({"c": close, "sig": signal.astype(float)}, axis=1).dropna()
    rets = df["c"].pct_change().fillna(0.0)
    strat_rets = rets * df["sig"].shift(1).fillna(0.0)  # enter next bar
    equity = (1.0 + strat_rets).cumprod()
    return equity

def perf_summary(equity: pd.Series) -> dict:
    if equity.empty:
        return {"CAGR%": np.nan, "Sharpe": np.nan, "MaxDD%": np.nan}
    # CAGR
    n_days = (equity.index[-1] - equity.index[0]).days
    years = max(n_days / 365.25, 1e-9)
    cagr = (equity.iloc[-1] ** (1 / years) - 1) * 100.0
    # Daily returns Sharpe (approx)
    r = equity.pct_change().dropna()
    if r.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = (r.mean() / r.std(ddof=0)) * math.sqrt(252)
    # Max drawdown
    peak = equity.cummax()
    dd = equity / peak - 1.0
    maxdd = dd.min() * 100.0
    return {"CAGR%": cagr, "Sharpe": sharpe, "MaxDD%": maxdd}

# ============== Plotting (robust / NaN-safe) ==============
def _align_and_normalize(equity_dict: dict) -> pd.DataFrame:
    """
    equity_dict: {name: pd.Series}
    Returns aligned DataFrame and rebased to 1.0 at first valid point of each series.
    """
    df = pd.concat(equity_dict, axis=1, join="inner")
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index().dropna(how="all")
    # Rebase each column to 1.0
    first = df.apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
    df = df.divide(first)
    df = df.dropna(axis=1, how="all")
    return df

def plot_equity_curves(equity_dict: dict, title="Equity Curves (rebased to 1.0)"):
    import matplotlib.pyplot as plt
    aligned = _align_and_normalize(equity_dict)
    if aligned.empty:
        st.warning("No equity data to plot after alignment/normalization.")
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    aligned.plot(ax=ax, linewidth=1.75)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rebased Equity (×)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

def plot_price_with_regime(df: pd.DataFrame, title="Regime sample", low_pct=0.30, high_pct=0.70):
    import matplotlib.pyplot as plt
    d = df.copy()
    # Ensure datetime index
    if not isinstance(d.index, pd.DatetimeIndex):
        if "Date" in d.columns:
            d.index = pd.to_datetime(d["Date"], errors="coerce")
        d = d.sort_index()
    # Need Close + Volatility
    if "Close" not in d.columns or "Volatility" not in d.columns:
        st.warning(f"Regime plot skipped for {title}: missing Close/Volatility.")
        return
    vol_clean = d["Volatility"].dropna()
    if vol_clean.empty:
        st.warning(f"Regime plot skipped for {title}: volatility empty.")
        return
    lo = np.nanpercentile(vol_clean, low_pct * 100.0)
    hi = np.nanpercentile(vol_clean, high_pct * 100.0)

    # Build regime mask
    regime = pd.Series(1, index=d.index, dtype="int8")
    regime[d["Volatility"] <= lo] = 0
    regime[d["Volatility"] >= hi] = 2

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(d.index, d["Close"], linewidth=1.5, label="Close")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)

    # Shade regimes across contiguous spans
    def _shade(mask: pd.Series, alpha=0.10):
        spans = []
        in_span = False
        start = None
        prev_t = None
        for t, m in mask.items():
            if m and not in_span:
                in_span = True; start = t
            if in_span and (not m):
                spans.append((start, prev_t)); in_span = False
            prev_t = t
        if in_span:
            spans.append((start, mask.index[-1]))
        return spans

    low_spans  = _shade(regime == 0, alpha=0.10)
    high_spans = _shade(regime == 2, alpha=0.10)

    for s, e in low_spans:
        ax.axvspan(s, e, alpha=0.10, color="tab:blue", linewidth=0)
    for s, e in high_spans:
        ax.axvspan(s, e, alpha=0.10, color="tab:red", linewidth=0)

    from matplotlib.patches import Patch
    lh = [Patch(facecolor="tab:blue",  alpha=0.10, label=f"Low vol ≤ {low_pct:.0%}"),
          Patch(facecolor="tab:red",   alpha=0.10, label=f"High vol ≥ {high_pct:.0%}")]
    base_handles, base_labels = ax.get_legend_handles_labels()
    ax.legend(handles=[*base_handles, *lh], loc="upper left")

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# ============== UI ==============
with st.sidebar:
    st.header("⚙️ Settings")
    tickers = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,SPY").strip()
    default_days = 365 * 3
    lookback_days = st.number_input("Lookback (days)", min_value=200, max_value=3650, value=default_days, step=50)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=int(lookback_days))
    st.caption(f"From {start_date} to {end_date} (UTC)")
    st.markdown("---")
    st.subheader("Regime thresholds")
    low_pct  = st.slider("Low vol percentile", 0.05, 0.45, 0.30, 0.01)
    high_pct = st.slider("High vol percentile", 0.55, 0.95, 0.70, 0.01)
    vol_window = st.number_input("Volatility window", min_value=10, max_value=120, value=20, step=1)
    st.markdown("---")
    st.subheader("Strategies")
    chosen = st.multiselect("Select strategies", options=list(STRATEGIES.keys()),
                            default=["sma_cross", "bollinger", "rsi"])
    st.markdown("---")
    st.subheader("Optional CSV upload")
    st.caption("WIDE format: Date + one column per ticker (Close). LONG format: Date,Ticker,Close")
    upload = st.file_uploader("Upload CSV", type=["csv"])

# ============== Data sourcing ==============
def read_csv_any(upload_file) -> pd.DataFrame:
    if upload_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(upload_file)
    except Exception:
        upload_file.seek(0)
        df = pd.read_csv(upload_file, encoding_errors="ignore")
    # Try to detect format
    cols = [c.lower() for c in df.columns]
    if all(k in cols for k in ["date", "ticker", "close"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df[["Date", "Ticker", "Close"]]
        df = df.set_index("Date")
        return df
    # WIDE format
    if "date" in cols:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        # melt to LONG
        stacked = df.stack().reset_index()
        stacked.columns = ["Date", "Ticker", "Close"]
        stacked = stacked.dropna(subset=["Close"])
        stacked = stacked.set_index("Date").sort_index()
        return stacked
    # Unknown
    return pd.DataFrame()

def get_universe_df() -> pd.DataFrame:
    if upload is not None:
        df = read_csv_any(upload)
        if not df.empty:
            return df
    symbols = [s.strip() for s in tickers.split(",") if s.strip()]
    start = datetime.combine(start_date, datetime.min.time())
    end   = datetime.combine(end_date,   datetime.min.time())
    df = fetch_prices_for_universe(symbols, start, end, allow_partial=True)
    return df

# ============== Run ==============
run = st.button("▶️ Run Backtest")

if run:
    try:
        raw = get_universe_df()
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

    if raw.empty:
        st.warning("No data to process.")
        st.stop()

    # Pivot to WIDE by ticker
    wide = raw.reset_index().pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    wide = wide.sort_index().dropna(how="all")

    # Compute volatility & regimes per ticker
    vols = pd.DataFrame(index=wide.index)
    regimes = pd.DataFrame(index=wide.index, dtype="int8")
    for t in wide.columns:
        v = rolling_volatility(wide[t], window=int(vol_window))
        vols[t] = v
        regimes[t] = compute_regime_percentile(v, low_pct=low_pct, high_pct=high_pct)

    # Backtest each strategy per ticker
    equity_curves = {}  # { "TICKER:STRAT": series }
    perf_rows = []      # rows for summary

    for t in wide.columns:
        close = wide[t].dropna()
        if close.size < max(60, int(vol_window) + 10):
            continue
        for key in chosen:
            label, fn = STRATEGIES[key]
            # Strategy-specific defaults:
            if key == "sma_cross":
                sig = fn(close, fast=10, slow=40)
            elif key == "bollinger":
                sig = fn(close, window=20, k=2.0)
            elif key == "rsi":
                sig = fn(close, window=14, low=35, high=65)
            else:
                continue
            eq = run_backtest(close, sig)
            equity_curves[f"{t}:{key}"] = eq
            ps = perf_summary(eq)
            perf_rows.append({
                "Ticker": t,
                "Strategy": label,
                "CAGR%": round(ps["CAGR%"], 2) if not np.isnan(ps["CAGR%"]) else np.nan,
                "Sharpe": round(ps["Sharpe"], 2) if not np.isnan(ps["Sharpe"]) else np.nan,
                "MaxDD%": round(ps["MaxDD%"], 2) if not np.isnan(ps["MaxDD%"]) else np.nan,
            })

    # ======= Output: Performance Summary =======
    st.subheader("📊 Performance Summary")
    if perf_rows:
        df_perf = pd.DataFrame(perf_rows).sort_values(["Ticker", "Strategy"])
        st.dataframe(df_perf, use_container_width=True)
    else:
        st.info("No strategies produced results (insufficient data or inputs).")

    # ======= Output: Equity Curves (rebased) =======
    st.subheader("📈 Equity Curves")
    if equity_curves:
        plot_equity_curves(equity_curves, title="Equity Curves (rebased to 1.0)")
    else:
        st.info("No equity curves to display.")

    # ======= Output: Regime samples =======
    st.subheader("🧭 Regime (volatility percentile)")
    # Show up to 3 sample tickers
    sample_tickers = list(wide.columns)[:3]
    for t in sample_tickers:
        df_plot = pd.DataFrame({
            "Close": wide[t],
            "Volatility": vols[t],
        }).dropna(subset=["Close"])
        plot_price_with_regime(df_plot, title=f"{t} regime sample", low_pct=low_pct, high_pct=high_pct)

    st.caption("Done.")
else:
    st.info("Configure settings on the left, then click **Run Backtest**.")

# ============== Footer / Debug ==============
with st.expander("ℹ️ Info & Debug"):
    st.write({
        "yfinance_available": YF_IMPORT_OK,
        "pandas_datareader_available": PDR_OK,
        "finnhub_enabled": bool(FINNHUB_API_KEY),
    })