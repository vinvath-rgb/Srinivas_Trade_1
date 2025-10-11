# streamlit_app.py — Single‑file Strategy–Regime Pipeline with Optimizer, Walk‑Forward & Exports
# Srini/Giligili build — v1.1 (2025-10-11)
# 
# Features
# - Fetch prices (Yahoo → Stooq → Finnhub fallback) to WIDE format
# - Cross‑sectional volatility regimes (per‑date percentiles)
# - Strategies: SMA Cross, Bollinger, RSI (per‑asset signals)
# - Optimizers: Equal‑Weight, Min‑Variance, Risk‑Parity (rolling covariance)
# - Global grid search (Sharpe/CAGR objective) OR Walk‑Forward (3y/3m default)
# - Baseline vs Optimized vs Buy‑&‑Hold comparison with metrics
# - CSV/XLSX exports (prices, regimes, signals, weights, portfolio equity, per‑window params)
#
# Notes
# - Keep grids modest in Render for speed; expand later.
# - Finnhub is optional (set FINNHUB_API_KEY in env). Yahoo/Stooq used otherwise.
# - This is self‑contained — paste into your repo as streamlit_app.py (or any name) and deploy.

from __future__ import annotations
import io, os, math, json, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Strategy–Regime Pipeline", layout="wide")
st.title("📈 Strategy–Regime Pipeline — Universe → Regimes → Optimizer → Exports")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# Soft imports (don’t break if missing)
YF_OK = True
try:
    import yfinance as yf
except Exception:
    YF_OK = False

PDR_OK = True
try:
    import pandas_datareader.data as pdr
except Exception:
    PDR_OK = False

# -----------------------------
# Helpers: dates, safe math
# -----------------------------

def to_utc_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp())

@st.cache_data(show_spinner=False)
def _cagr(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    start, end = series.index[0], series.index[-1]
    years = max((end - start).days / 365.25, 1e-9)
    return (series.iloc[-1] / max(series.iloc[0], 1e-12)) ** (1/years) - 1

@st.cache_data(show_spinner=False)
def _sharpe(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    if returns.dropna().empty:
        return 0.0
    excess = returns - rf/periods
    mu = excess.mean() * periods
    sigma = excess.std(ddof=0) * math.sqrt(periods)
    return 0.0 if sigma == 0 or np.isnan(sigma) else mu / sigma

@st.cache_data(show_spinner=False)
def _max_dd(equity: pd.Series) -> float:
    if equity.dropna().empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    return float(dd.min())

# -----------------------------
# Fetch prices: Yahoo → Stooq → Finnhub
# -----------------------------

def _fetch_yahoo(ticker: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not YF_OK:
        raise RuntimeError("yfinance not available")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = df[["Close"]].rename(columns={"Close": ticker})
            out.index.name = "Date"
            return out
    except Exception:
        pass
    raise RuntimeError("Yahoo failed")

def _fetch_stooq(ticker: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not PDR_OK:
        raise RuntimeError("pandas_datareader not available")
    try:
        t = ticker.replace("-", ".").upper()
        df = pdr.DataReader(t, "stooq")
        if not df.empty:
            df = df.sort_index()
            if start:
                df = df[df.index >= pd.to_datetime(start)]
            if end:
                df = df[df.index <= pd.to_datetime(end)]
            out = df[["Close"]].rename(columns={"Close": ticker})
            out.index.name = "Date"
            return out
    except Exception:
        pass
    raise RuntimeError("Stooq failed")

def _fetch_finnhub(ticker: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        raise RuntimeError("Finnhub not configured")
    try:
        import requests
        resolution = "D"
        t0 = pd.to_datetime(start) if start else (pd.Timestamp.utcnow() - pd.Timedelta(days=365*5))
        t1 = pd.to_datetime(end) if end else pd.Timestamp.utcnow()
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution={resolution}&from={int(t0.timestamp())}&to={int(t1.timestamp())}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=30)
        j = r.json()
        if j.get("s") != "ok":
            raise RuntimeError("Finnhub status not ok")
        ts = pd.to_datetime(j["t"], unit="s")
        close = pd.Series(j["c"], index=ts, name=ticker)
        df = close.to_frame()
        df.index.name = "Date"
        return df
    except Exception:
        raise RuntimeError("Finnhub failed")

@st.cache_data(show_spinner=True)
def get_prices_wide(tickers: List[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        err = []
        for fn in (_fetch_yahoo, _fetch_stooq, _fetch_finnhub):
            try:
                f = fn(t, start, end)
                frames.append(f)
                break
            except Exception as e:
                err.append(str(e))
        else:
            st.warning(f"All data sources failed for {t}: {err}")
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1)
    wide = wide.asfreq("B").ffill().dropna(how="all")
    return wide

# -----------------------------
# Regimes: cross‑sectional vol percentiles per date
# -----------------------------

def _daily_vol(close: pd.Series, win: int = 20) -> pd.Series:
    ret = close.pct_change()
    vol = ret.rolling(win, min_periods=max(5, win//2)).std() * np.sqrt(252)
    return vol

@st.cache_data(show_spinner=False)
def compute_regimes_wide(wide_close: pd.DataFrame, vol_win: int = 20,
                         low_pct: float = 0.40, high_pct: float = 0.60) -> pd.DataFrame:
    if wide_close.empty:
        return pd.DataFrame()
    vols = wide_close.apply(_daily_vol, win=vol_win)
    regimes = pd.DataFrame(index=wide_close.index, columns=wide_close.columns, dtype="Int64")
    for dt, row in vols.iterrows():
        r = pd.Series(1, index=row.index)  # default MID
        valid = row.dropna()
        if not valid.empty:
            pct = valid.rank(pct=True)
            r.loc[pct.index[pct <= low_pct]] = 0
            r.loc[pct.index[pct >= high_pct]] = 2
        regimes.loc[dt] = r
    return regimes

# -----------------------------
# Strategies: per‑asset signals
# -----------------------------

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=max(3, n//3)).mean()

def strat_sma_cross(close: pd.Series, fast: int = 10, slow: int = 40) -> pd.Series:
    f, s = sma(close, fast), sma(close, slow)
    sig = (f > s).astype(float)
    sig.iloc[: max(fast, slow)] = 0.0
    return sig

def strat_bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = sma(close, n)
    std = close.rolling(n, min_periods=max(3, n//3)).std()
    upper, lower = ma + k*std, ma - k*std
    sig = ((close < lower) | (close > upper)).astype(float)  # 1 when outside bands
    sig.iloc[: n] = 0.0
    return sig

def strat_rsi(close: pd.Series, n: int = 14, lo: int = 30, hi: int = 70) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=close.index).rolling(n, min_periods=max(3, n//3)).mean()
    avg_loss = pd.Series(loss, index=close.index).rolling(n, min_periods=max(3, n//3)).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    sig = ((rsi < lo) | (rsi > hi)).astype(float)
    sig.iloc[: n] = 0.0
    return sig

STRATEGY_FNS = {
    "sma_cross": strat_sma_cross,
    "bollinger": strat_bollinger,
    "rsi": strat_rsi,
}

# -----------------------------
# Optimizers (weights across active assets)
# -----------------------------

def _min_var_weights(cov: pd.DataFrame) -> pd.Series:
    # w ∝ Σ^{-1} 1
    if cov.isna().any().any():
        cov = cov.fillna(cov.mean()).fillna(0)
    try:
        inv = np.linalg.pinv(cov.values)
        ones = np.ones((cov.shape[0], 1))
        w = inv @ ones
        w = w / (ones.T @ inv @ ones)
        w = np.clip(w.flatten(), 0, None)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        return pd.Series(w, index=cov.index)
    except Exception:
        return pd.Series(np.repeat(1/cov.shape[0], cov.shape[0]), index=cov.index)

def _risk_parity_weights(cov: pd.DataFrame, iters: int = 200) -> pd.Series:
    # Simple iterative scheme for equal risk contributions
    n = cov.shape[0]
    w = np.ones(n) / n
    target = np.ones(n) / n
    for _ in range(iters):
        mrc = cov.values @ w
        rc = w * mrc
        diff = rc - target
        grad = mrc + cov.values @ w  # rough gradient
        step = 0.01
        w = w - step * grad * diff
        w = np.clip(w, 0, None)
        s = w.sum()
        if s == 0:
            w = np.ones(n)/n
        else:
            w /= s
    return pd.Series(w, index=cov.index)

def get_optimizer(name: str):
    if name == "equal_weight":
        return lambda cov: pd.Series(np.repeat(1/cov.shape[0], cov.shape[0]), index=cov.index)
    if name == "min_variance":
        return _min_var_weights
    if name == "risk_parity":
        return _risk_parity_weights
    return lambda cov: pd.Series(np.repeat(1/cov.shape[0], cov.shape[0]), index=cov.index)

# -----------------------------
# Backtest engine
# -----------------------------

def run_backtest(wide_close: pd.DataFrame, signals: pd.DataFrame,
                 optimizer: str = "equal_weight", lookback: int = 60) -> Tuple[pd.Series, pd.DataFrame]:
    """Return (equity, weights_history). weights_history indexed by date with columns=assets (NaN if not invested)."""
    if wide_close.empty or signals.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    ret = wide_close.pct_change().fillna(0)
    opt_fn = get_optimizer(optimizer)

    dates = wide_close.index
    assets = list(wide_close.columns)
    weights_hist = []
    equity = [1.0]

    for i in range(1, len(dates)):
        dt = dates[i]
        hist_start = max(0, i - lookback)
        hist = ret.iloc[hist_start:i]

        active = [a for a in assets if signals.loc[dt, a] > 0.5]
        if len(active) == 0:
            w = pd.Series(0.0, index=assets)
        elif len(active) == 1:
            w = pd.Series(0.0, index=assets)
            w.loc[active[0]] = 1.0
        else:
            cov = hist[active].cov()
            w_active = opt_fn(cov)
            w = pd.Series(0.0, index=assets)
            w[w_active.index] = w_active

        weights_hist.append(w)
        port_ret = float((ret.iloc[i, :] * w).sum())
        equity.append(equity[-1] * (1 + port_ret))

    weights_df = pd.DataFrame(weights_hist, index=dates[1:], columns=assets)
    equity_ser = pd.Series(equity, index=dates, name="Equity")
    return equity_ser, weights_df

# -----------------------------
# Grid search (learning) across strategies/params
# -----------------------------

def build_signals(wide_close: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    fn = STRATEGY_FNS[strategy]
    sigs = {}
    for c in wide_close.columns:
        sigs[c] = fn(wide_close[c], **params)
    out = pd.DataFrame(sigs, index=wide_close.index)
    return out.fillna(0.0)

@dataclass
class BTResult:
    name: str
    params: dict
    strategy: str
    optimizer: str
    equity: pd.Series
    weights: pd.DataFrame
    sharpe: float
    cagr: float
    maxdd: float

@st.cache_data(show_spinner=True)
def evaluate_combo(wide_close: pd.DataFrame, strategy: str, params: dict, optimizer: str,
                   lookback: int, objective: str) -> BTResult:
    signals = build_signals(wide_close, strategy, params)
    equity, weights = run_backtest(wide_close, signals, optimizer=optimizer, lookback=lookback)
    rets = equity.pct_change().fillna(0)
    sharpe = _sharpe(rets)
    cagr = _cagr(equity)
    maxdd = _max_dd(equity)
    score = sharpe if objective == "Sharpe" else cagr
    name = f"{strategy} {params} | {optimizer} | {objective}={score:.3f}"
    return BTResult(name, params, strategy, optimizer, equity, weights, sharpe, cagr, maxdd)

@st.cache_data(show_spinner=True)
def grid_search(wide_close: pd.DataFrame, objective: str, optimizer: str, lookback: int) -> Tuple[BTResult, List[BTResult]]:
    grid: List[Tuple[str, dict]] = []
    # modest grid for speed; expand later
    for f,s in [(5,20), (10,40), (20,60)]:
        grid.append(("sma_cross", {"fast": f, "slow": s}))
    for n,k in [(20,2.0), (20,2.5), (30,2.0)]:
        grid.append(("bollinger", {"n": n, "k": k}))
    for n in [14, 21]:
        grid.append(("rsi", {"n": n, "lo": 30, "hi": 70}))

    results: List[BTResult] = []
    best: Optional[BTResult] = None
    for strat, params in grid:
        res = evaluate_combo(wide_close, strat, params, optimizer, lookback, objective)
        results.append(res)
        if best is None:
            best = res
        else:
            if objective == "Sharpe":
                if res.sharpe > best.sharpe:
                    best = res
            else:
                if res.cagr > best.cagr:
                    best = res
    return best, results

# -----------------------------
# Walk‑Forward Optimization
# -----------------------------

def _add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    y = ts.year + (ts.month - 1 + months) // 12
    m = (ts.month - 1 + months) % 12 + 1
    d = min(ts.day, pd.Period(f"{y}-{m}").days_in_month)
    return pd.Timestamp(year=y, month=m, day=d)

@st.cache_data(show_spinner=True)
def walk_forward(wide: pd.DataFrame, objective: str, optimizer: str, lookback: int,
                 train_years: int = 3, test_months: int = 3) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    idx = wide.index
    if len(idx) < 252:
        # too short, fall back to global
        best, _ = grid_search(wide, objective=objective, optimizer=optimizer, lookback=lookback)
        return best.equity, best.weights, pd.DataFrame([{"window": "ALL", "strategy": best.strategy, "params": json.dumps(best.params), "sharpe": best.sharpe, "cagr": best.cagr, "maxdd": best.maxdd}])

    i = 0
    eq_parts: List[pd.Series] = []
    w_parts: List[pd.DataFrame] = []
    rows: List[dict] = []

    while True:
        train_start = idx[i]
        train_end = train_start + pd.Timedelta(days=int(365.25*train_years))
        # align to available index
        if train_end > idx[-1]:
            break
        test_start = idx[idx.get_indexer([train_end], method="backfill")[0]] if train_end in idx else idx[min(len(idx)-1, idx.searchsorted(train_end))]
        test_end_time = _add_months(test_start, test_months)
        if test_end_time > idx[-1]:
            test_end = idx[-1]
        else:
            test_end = idx[min(len(idx)-1, idx.searchsorted(test_end_time))]

        # slices (include lookback history for test)
        train_mask = (idx >= train_start) & (idx <= train_end)
        # ensure at least 60 biz days
        if train_mask.sum() < max(60, lookback + 10):
            break

        best, _ = grid_search(wide.loc[train_mask], objective=objective, optimizer=optimizer, lookback=lookback)

        # test slice with history
        test_mask_core = (idx > test_start) & (idx <= test_end)
        if test_mask_core.sum() == 0:
            break
        test_start_pos = np.where(idx == test_start)[0][0]
        hist_start_pos = max(0, test_start_pos - lookback)
        test_slice = slice(hist_start_pos, np.where(idx == test_end)[0][0] + 1)
        sub_idx = idx[test_slice]
        sub_wide = wide.loc[sub_idx]

        # build signals on subwindow using best params
        sig = build_signals(sub_wide, best.strategy, best.params)
        eq, w = run_backtest(sub_wide, sig, optimizer=best.optimizer, lookback=lookback)
        # keep only the out-of-sample segment (exclude the history rows before test_start)
        eq = eq.loc[eq.index >= test_start]
        w = w.loc[w.index >= test_start]

        eq_parts.append(eq)
        w_parts.append(w)
        rows.append({
            "window": f"{train_start.date()}→{train_end.date()} | {test_start.date()}→{test_end.date()}",
            "strategy": best.strategy,
            "params": json.dumps(best.params),
            "optimizer": best.optimizer,
            "sharpe": round(best.sharpe, 3),
            "cagr": round(best.cagr, 3),
            "maxdd": round(best.maxdd, 3),
        })

        # advance pointer to end of test window for next iteration
        i = np.where(idx == test_end)[0][0] + 1
        if i >= len(idx) - 20:  # leave some tail
            break

    if not eq_parts:
        # fallback to global
        best, _ = grid_search(wide, objective=objective, optimizer=optimizer, lookback=lookback)
        return best.equity, best.weights, pd.DataFrame([{"window": "ALL", "strategy": best.strategy, "params": json.dumps(best.params), "sharpe": best.sharpe, "cagr": best.cagr, "maxdd": best.maxdd}])

    wf_eq = pd.concat(eq_parts).groupby(level=0).last()
    wf_eq.name = "Equity"
    wf_w = pd.concat(w_parts).groupby(level=0).last().sort_index()
    wf_tbl = pd.DataFrame(rows)
    return wf_eq, wf_w, wf_tbl

# -----------------------------
# Buy & Hold benchmark
# -----------------------------

def buy_hold_equity(wide_close: pd.DataFrame) -> pd.Series:
    if wide_close.empty:
        return pd.Series(dtype=float)
    rets = wide_close.pct_change().mean(axis=1).fillna(0)  # simple average across assets
    eq = (1 + rets).cumprod()
    eq.iloc[0] = 1.0
    return eq

# -----------------------------
# UI Sidebar — Inputs
# -----------------------------

with st.sidebar:
    st.subheader("🧭 Universe & Dates")
    tickers_text = st.text_area("Tickers (comma‑separated)", value="AAPL, MSFT, AMZN, GOOGL, META")
    start = st.date_input("Start", value=pd.to_datetime("2019-01-01").date())
    end = st.date_input("End", value=pd.to_datetime("today").date())

    st.subheader("⚙️ Regime Settings")
    vol_win = st.number_input("Vol window", value=20, min_value=5, max_value=120, step=5)
    low_pct = st.slider("Low vol percentile", 0.05, 0.45, 0.40, 0.05)
    high_pct = st.slider("High vol percentile", 0.55, 0.95, 0.60, 0.05)

    st.subheader("🧪 Backtest & Optimizer")
    optimizer = st.selectbox("Optimizer", ["equal_weight", "min_variance", "risk_parity"], index=0)
    lookback = st.number_input("Optimizer lookback (days)", value=60, min_value=20, max_value=252, step=10)
    objective = st.selectbox("Optimization objective", ["Sharpe", "CAGR"], index=0)

    st.subheader("🎯 Baseline Strategy")
    base_family = st.selectbox("Baseline strategy", ["sma_cross", "bollinger", "rsi"], index=0)
    if base_family == "sma_cross":
        base_params = {"fast": st.number_input("fast", value=10, min_value=3, max_value=60),
                       "slow": st.number_input("slow", value=40, min_value=5, max_value=200)}
    elif base_family == "bollinger":
        base_params = {"n": st.number_input("n", value=20, min_value=5, max_value=120),
                       "k": st.number_input("k", value=2.0, min_value=0.5, max_value=4.0, step=0.1)}
    else:
        base_params = {"n": st.number_input("n", value=14, min_value=5, max_value=60),
                       "lo": st.number_input("lo", value=30, min_value=5, max_value=50),
                       "hi": st.number_input("hi", value=70, min_value=50, max_value=95)}

    st.subheader("🚶 Walk‑Forward (optional)")
    use_wf = st.toggle("Enable walk‑forward optimization", value=False, help="Train on early window, apply to next test window, repeat.")
    train_years = st.number_input("Train window (years)", value=3, min_value=1, max_value=10, step=1, disabled=not use_wf)
    test_months = st.number_input("Test window (months)", value=3, min_value=1, max_value=12, step=1, disabled=not use_wf)

    run = st.button("▶️ Run Pipeline", use_container_width=True)

# -----------------------------
# Main execution
# -----------------------------

if run:
    tickers = [t.strip() for t in tickers_text.split(',') if t.strip()]
    with st.status("Fetching prices & building universe…", expanded=False):
        wide = get_prices_wide(tickers, str(start), str(end))
        if wide.empty:
            st.error("No price data fetched. Check tickers or data sources.")
            st.stop()
        st.write("✅ Prices shape:", wide.shape)

    with st.expander("Preview: Prices (last 5 rows)", expanded=False):
        st.dataframe(wide.tail())

    regimes = compute_regimes_wide(wide, vol_win=vol_win, low_pct=low_pct, high_pct=high_pct)
    with st.expander("Preview: Regimes (0=Low,1=Mid,2=High) — last 5", expanded=False):
        st.dataframe(regimes.tail())

    # Baseline
    st.subheader("Baseline vs Optimized vs Buy‑&‑Hold")
    base_signals = build_signals(wide, base_family, base_params)
    base_eq, base_w = run_backtest(wide, base_signals, optimizer=optimizer, lookback=lookback)
    base_ret = base_eq.pct_change().fillna(0)

    # Optimized (global or walk‑forward)
    if use_wf:
        opt_eq, opt_w, wf_tbl = walk_forward(wide, objective=objective, optimizer=optimizer, lookback=lookback,
                                             train_years=int(train_years), test_months=int(test_months))
        opt_label = "Optimized (Walk‑Forward)"
    else:
        best, all_results = grid_search(wide, objective=objective, optimizer=optimizer, lookback=lookback)
        opt_eq, opt_w, wf_tbl = best.equity, best.weights, pd.DataFrame([{ "window": "ALL", "strategy": best.strategy, "params": json.dumps(best.params), "optimizer": best.optimizer, "sharpe": round(best.sharpe,3), "cagr": round(best.cagr,3), "maxdd": round(best.maxdd,3)}])
        opt_label = "Optimized (Global)"

    # Buy & Hold
    bh_eq = buy_hold_equity(wide)

    # Metrics table
    rows = []
    rows.append({
        "Model": f"Baseline: {base_family} {base_params} | {optimizer}",
        "Sharpe": round(_sharpe(base_ret), 3),
        "CAGR": round(_cagr(base_eq), 3),
        "MaxDD": round(_max_dd(base_eq), 3),
    })
    rows.append({
        "Model": opt_label,
        "Sharpe": round(_sharpe(opt_eq.pct_change().fillna(0)), 3),
        "CAGR": round(_cagr(opt_eq), 3),
        "MaxDD": round(_max_dd(opt_eq), 3),
    })
    rows.append({
        "Model": "Buy & Hold (avg basket)",
        "Sharpe": round(_sharpe(bh_eq.pct_change().fillna(0)), 3),
        "CAGR": round(_cagr(bh_eq), 3),
        "MaxDD": round(_max_dd(bh_eq), 3),
    })
    st.dataframe(pd.DataFrame(rows))

    # Charts
    st.subheader("Equity Curves")
    chart_df = pd.concat([
        base_eq.rename("Baseline"),
        opt_eq.rename(opt_label),
        bh_eq.rename("Buy&Hold"),
    ], axis=1).dropna(how="all")
    st.line_chart(chart_df)

    with st.expander(f"Weights over time — {opt_label} (last 120 days)", expanded=False):
        st.dataframe(opt_w.tail(120))

    if use_wf:
        with st.expander("Walk‑Forward windows & chosen params", expanded=False):
            st.dataframe(wf_tbl)

    # -----------------------------
    # Exports: CSV & XLSX
    # -----------------------------
    st.subheader("⬇️ Export Outputs")

    def _to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=True).encode("utf-8")

    def _to_xlsx_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            for name, df in dfs.items():
                df.to_excel(xw, sheet_name=name[:31])
        buf.seek(0)
        return buf.read()

    exports = {
        "prices": wide,
        "regimes": regimes,
        "baseline_signals": base_signals,
        "baseline_weights": base_w,
        "baseline_equity": base_eq.to_frame(),
        "optimized_weights": opt_w,
        "optimized_equity": opt_eq.to_frame(),
        "buy_hold_equity": bh_eq.to_frame(),
        "metrics_table": pd.DataFrame(rows),
    }
    if use_wf:
        exports["walkforward_table"] = wf_tbl

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download ALL as XLSX", data=_to_xlsx_bytes(exports), file_name="pipeline_outputs.xlsx")
    with c2:
        st.download_button("Download metrics.csv", data=_to_csv_bytes(pd.DataFrame(rows)), file_name="metrics.csv")

    # Debug panel
    with st.expander("Info & Debug", expanded=False):
        st.json({
            "yfinance_available": YF_OK,
            "pandas_datareader_available": PDR_OK,
            "finnhub_enabled": bool(FINNHUB_API_KEY),
            "tickers": tickers,
            "date_range": [str(start), str(end)],
            "optimizer": optimizer,
            "objective": objective,
            "baseline": {"family": base_family, "params": base_params},
            "walk_forward": {"enabled": use_wf, "train_years": int(train_years), "test_months": int(test_months)},
            "mode": opt_label,
        })
else:
    st.info("Set your universe & click ▶️ Run Pipeline.")