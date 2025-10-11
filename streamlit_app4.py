# ============================================
# Strategy–Regime Matrix — Single-File App (Max Bells & Whistles)
# (c) Srini 2025
# ============================================
from __future__ import annotations
import io
import os
from typing import Dict, List, Callable, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional deps (installed via requirements.txt)
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Strategy–Regime Matrix — Single App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Helpers & Safety
# =============================
def _safe_df(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.Series):
        return x.replace([np.inf, -np.inf], np.nan).dropna()
    return x.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def _normalize_prices(prices_input: Dict[str, pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    """
    Accept dict[ticker->DataFrame(Date, Close)] or LONG/WIDE DF -> return WIDE DF: Date index x tickers
    """
    if isinstance(prices_input, dict):
        long_frames = []
        for t, df in prices_input.items():
            if df is None or df.empty:
                continue
            tmp = df.copy()
            # Ensure Date col
            if "Date" not in tmp.columns:
                if isinstance(tmp.index, pd.DatetimeIndex) or str(tmp.index.name).lower() == "date":
                    tmp = tmp.reset_index()
                    if "Date" not in tmp.columns:
                        tmp = tmp.rename(columns={tmp.columns[0]: "Date"})
                else:
                    raise ValueError(f"{t}: missing Date column/index")
            # Pick Close-like col
            close_col = None
            for cand in ("Close", "Adj Close", "close", "adj close", "AdjClose"):
                if cand in tmp.columns:
                    close_col = cand
                    break
            if close_col is None:
                close_col = tmp.columns[-1]
            tmp = tmp[["Date", close_col]].rename(columns={close_col: "Close"})
            tmp["Ticker"] = t
            long_frames.append(tmp)
        prices = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    else:
        prices = prices_input.copy()

    # LONG -> WIDE
    if isinstance(prices, pd.DataFrame):
        if {"Date", "Ticker", "Close"}.issubset(prices.columns):
            prices = (
                prices.assign(Date=pd.to_datetime(prices["Date"]))
                .pivot(index="Date", columns="Ticker", values="Close")
                .sort_index()
            )
        elif "Date" in prices.columns:
            prices = prices.set_index("Date").sort_index()
    return prices


# =============================
# Data Ingest
# =============================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices_for_tickers(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Yahoo first, fallback to Stooq via pandas-datareader."""
    out: Dict[str, pd.DataFrame] = {t: None for t in tickers}

    # Yahoo
    if yf is not None:
        try:
            data = yf.download(
                tickers=" ".join(tickers),
                start=start, end=end, interval=interval,
                auto_adjust=False, group_by="ticker",
                threads=True, progress=False,
            )
            for t in tickers:
                df = None
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if t in data.columns.get_level_values(0):
                            close = data[t]["Close"]
                            df = pd.DataFrame({"Date": close.index, "Close": close.values})
                    else:
                        if "Close" in data.columns:
                            df = pd.DataFrame({"Date": data.index, "Close": data["Close"].values})
                except Exception:
                    df = None
                if df is not None:
                    out[t] = df
        except Exception:
            pass

    # Stooq
    if pdr is not None:
        for t in tickers:
            if out.get(t) is None:
                try:
                    df = pdr.DataReader(t, "stooq", start=start, end=end).sort_index()
                    if not df.empty:
                        out[t] = pd.DataFrame({"Date": df.index, "Close": df["Close"].values})
                except Exception:
                    out[t] = None

    return out


def load_csv_uploaded(file) -> Dict[str, pd.DataFrame]:
    """
    Allows either:
      1) Single column 'Close' with 'Date' index and provide a ticker in UI,
      2) Long format (Date,Ticker,Close),
      3) Wide with Date + ticker columns.
    Returns dict[ticker->df(Date, Close)] for compatibility with _normalize_prices.
    """
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, parse_dates=True)
    # Heuristics
    if {"Date", "Ticker", "Close"}.issubset(df.columns):
        # long -> dict
        out = {}
        for t, g in df.groupby("Ticker"):
            out[str(t)] = g[["Date", "Close"]]
        return out
    if "Date" in df.columns and "Close" in df.columns:
        # single price, need ticker name from user later
        return {"UPLD": df[["Date", "Close"]]}
    if "Date" in df.columns:
        # wide -> dict
        out = {}
        for t in [c for c in df.columns if c != "Date"]:
            out[t] = df[["Date", t]].rename(columns={t: "Close"})
        return out
    # last resort: assume index is Date, last col is Close
    if df.index.name and df.index.name.lower() == "date":
        return {"UPLD": df.reset_index()[["Date", df.columns[-1]]].rename(columns={df.columns[-1]: "Close"})}
    return {"UPLD": pd.DataFrame(columns=["Date", "Close"])}


# =============================
# Regime Detection
# =============================
def compute_regimes(
    prices_wide: pd.DataFrame,
    mode: str = "percentile",
    low_pct: float = 0.40,
    high_pct: float = 0.60,
) -> pd.Series:
    """ {-1,0,+1} based on basket daily return """
    if prices_wide is None or prices_wide.empty:
        return pd.Series(dtype=int)

    cl = _safe_df(prices_wide)
    rets = cl.pct_change().mean(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return pd.Series(dtype=int)

    if mode == "percentile":
        lo = rets.quantile(low_pct)
        hi = rets.quantile(high_pct)
        regimes = rets.apply(lambda x: 1 if x >= hi else (-1 if x <= lo else 0))
    else:
        m, s = rets.mean(), rets.std()
        regimes = rets.apply(lambda x: 1 if x >= m + s else (-1 if x <= m - s else 0))
    return regimes.reindex(cl.index).fillna(0).astype(int)


# =============================
# Strategies (+ ATR / Vol targeting)
# =============================
def _atr_like(series: pd.Series, window: int = 14) -> pd.Series:
    """ATR proxy using close-only data: rolling mean of abs returns * price."""
    r = series.pct_change().abs()
    atr = (r.rolling(window).mean() * series.shift(1)).fillna(0.0)
    return atr.replace(0, np.nan).ffill().fillna(method="bfill")


def _vol_target_scale(series: pd.Series, target_vol: float, window: int = 20) -> pd.Series:
    """Scale positions so realized vol ≈ target_vol (annualized)."""
    if target_vol <= 0:
        return pd.Series(1.0, index=series.index)
    daily = series.pct_change().rolling(window).std()
    ann = (daily * np.sqrt(252)).replace(0, np.nan)
    scale = (target_vol / ann).clip(0, 5.0)  # cap leverage
    return scale.fillna(1.0)


def sma_cross(series: pd.Series, fast: int = 10, slow: int = 40,
              use_atr: bool=False, atr_window:int=14, atr_k:float=0.0,
              vol_target: float=0.0, vol_window:int=20) -> pd.Series:
    f = series.rolling(fast).mean()
    s = series.rolling(slow).mean()
    base = (f > s).astype(float) - (f < s).astype(float)

    if use_atr and atr_k > 0:
        atr = _atr_like(series, atr_window)
        # scale down when ATR is large (risk control)
        base = base * (series / (series + atr_k * atr)).clip(0.0, 1.0).fillna(0.0)

    if vol_target > 0:
        scale = _vol_target_scale(series, vol_target, vol_window)
        base = base * scale

    return base


def bollinger(series: pd.Series, window: int = 20, k: float = 2.0,
              use_atr: bool=False, atr_window:int=14, atr_k:float=0.0,
              vol_target: float=0.0, vol_window:int=20) -> pd.Series:
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + k * sd
    lower = ma - k * sd
    base = (series < lower).astype(float) - (series > upper).astype(float)

    if use_atr and atr_k > 0:
        atr = _atr_like(series, atr_window)
        base = base * (series / (series + atr_k * atr)).clip(0.0, 1.0).fillna(0.0)

    if vol_target > 0:
        scale = _vol_target_scale(series, vol_target, vol_window)
        base = base * scale

    return base


def rsi_trend(series: pd.Series, window: int = 14, low: float = 30, high: float = 70,
              use_atr: bool=False, atr_window:int=14, atr_k:float=0.0,
              vol_target: float=0.0, vol_window:int=20) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    dn = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (dn.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    base = (rsi < low).astype(float) - (rsi > high).astype(float)

    if use_atr and atr_k > 0:
        atr = _atr_like(series, atr_window)
        base = base * (series / (series + atr_k * atr)).clip(0.0, 1.0).fillna(0.0)

    if vol_target > 0:
        scale = _vol_target_scale(series, vol_target, vol_window)
        base = base * scale

    return base


def combo_signal(series: pd.Series, weights: Dict[str, float], params: Dict[str, Any]) -> pd.Series:
    w_sum = sum(abs(w) for w in weights.values()) or 1.0
    sig = pd.Series(0.0, index=series.index)
    if weights.get("sma", 0) != 0:
        sig += weights["sma"] * sma_cross(series,
                                          fast=params["fast_sma"], slow=params["slow_sma"],
                                          use_atr=params["use_atr"], atr_window=params["atr_window"], atr_k=params["atr_k"],
                                          vol_target=params["vol_target"], vol_window=params["vol_window"])
    if weights.get("bb", 0) != 0:
        sig += weights["bb"] * bollinger(series,
                                         window=params["bb_window"], k=params["bb_k"],
                                         use_atr=params["use_atr"], atr_window=params["atr_window"], atr_k=params["atr_k"],
                                         vol_target=params["vol_target"], vol_window=params["vol_window"])
    if weights.get("rsi", 0) != 0:
        sig += weights["rsi"] * rsi_trend(series,
                                          window=params["rsi_window"], low=params["rsi_low"], high=params["rsi_high"],
                                          use_atr=params["use_atr"], atr_window=params["atr_window"], atr_k=params["atr_k"],
                                          vol_target=params["vol_target"], vol_window=params["vol_window"])
    return (sig / w_sum).clip(-1, 1).fillna(0.0)


# Strategy registry
def make_strategy_fn(name: str, params: Dict[str, Any]) -> Callable[[pd.Series], pd.Series]:
    if name == "sma_cross":
        return lambda s: sma_cross(s, params["fast_sma"], params["slow_sma"],
                                   params["use_atr"], params["atr_window"], params["atr_k"],
                                   params["vol_target"], params["vol_window"])
    if name == "bollinger":
        return lambda s: bollinger(s, params["bb_window"], params["bb_k"],
                                   params["use_atr"], params["atr_window"], params["atr_k"],
                                   params["vol_target"], params["vol_window"])
    if name == "rsi_trend":
        return lambda s: rsi_trend(s, params["rsi_window"], params["rsi_low"], params["rsi_high"],
                                   params["use_atr"], params["atr_window"], params["atr_k"],
                                   params["vol_target"], params["vol_window"])
    if name == "combo":
        return lambda s: combo_signal(s, params["combo_weights"], params)
    # default flat
    return lambda s: pd.Series(0.0, index=s.index)


# =============================
# Backtester
# =============================
def _stats(curve: pd.Series) -> Dict[str, Any]:
    daily = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if daily.empty:
        return {}
    try:
        cagr = (curve.iloc[-1] / curve.iloc[0]) ** (252 / len(daily)) - 1
    except Exception:
        cagr = np.nan
    vol = daily.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else np.nan
    maxdd = ((1 + daily).cumprod() / ((1 + daily).cumprod().cummax()) - 1).min()
    return {
        "CAGR": float(np.round(cagr, 4)),
        "Vol": float(np.round(vol, 4)),
        "Sharpe": float(np.round(sharpe, 3)),
        "MaxDD": float(np.round(maxdd, 4)),
    }


def run_backtest(
    prices: pd.DataFrame,
    strategy_fn: Callable[[pd.Series], pd.Series],
    account_type: str = "margin",
    commission_bps: float = 1.0,
    slippage_bps: float = 1.0,
    short_borrow_apr: float = 0.03,
    max_leverage: float = 1.0,
) -> Dict[str, Any]:
    if prices is None or prices.empty:
        return {"curve": pd.Series(dtype=float), "stats": {}, "trades": []}

    cl = _safe_df(prices.copy())
    rets = cl.pct_change().fillna(0.0)

    raw_pos = cl.apply(strategy_fn, axis=0).reindex_like(cl).fillna(0.0)

    if account_type.lower() == "cash":
        raw_pos = raw_pos.clip(lower=0.0, upper=max_leverage)
    else:
        raw_pos = raw_pos.clip(lower=-max_leverage, upper=max_leverage)

    pos = raw_pos.shift(1).fillna(0.0)

    valid = cl.notna()
    denom = valid.sum(axis=1).replace(0, np.nan)
    weights = valid.div(denom, axis=0)
    signed_weights = weights * pos

    one_way_cost = (commission_bps + slippage_bps) / 1e4
    pos_change = pos.diff().abs().fillna(pos.abs())
    daily_tc = (one_way_cost * pos_change).sum(axis=1) / denom.replace(0, np.nan)
    daily_tc = daily_tc.fillna(0.0)

    borrow_daily = 0.0
    if account_type.lower() == "margin" and short_borrow_apr > 0:
        short_exposure = (-signed_weights.clip(upper=0.0)).sum(axis=1)
        borrow_daily = (short_borrow_apr / 252.0) * short_exposure

    strat_gross = (signed_weights * rets).sum(axis=1).fillna(0.0)
    strat_net = strat_gross - daily_tc - (borrow_daily if isinstance(borrow_daily, pd.Series) else 0.0)

    curve = (1.0 + strat_net).cumprod() * 100.0
    curve = _safe_df(curve)

    if curve.empty or curve.isna().all():
        return {"curve": curve, "stats": {}, "trades": []}

    return {"curve": curve, "stats": _stats(curve), "trades": []}


# =============================
# Optimizer (Grid & Walk-Forward)
# =============================
def grid_search(
    prices: pd.DataFrame, strategy_name: str, param_grid: Dict[str, List[Any]],
    costs: Dict[str, Any], top_n: int = 10
) -> pd.DataFrame:
    """Simple exhaustive grid; rank by Sharpe then CAGR."""
    rows = []
    from itertools import product
    keys = list(param_grid.keys())
    for vals in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        # Merge with default structure needed by make_strategy_fn
        filled = default_params()
        for k, v in params.items(): filled[k] = v
        fn = make_strategy_fn(strategy_name, filled)
        res = run_backtest(prices, fn, **costs)
        stats = res["stats"]
        rows.append({**params, **stats})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Sharpe_rank"] = df["Sharpe"].rank(ascending=False, na_option="bottom")
    df["CAGR_rank"] = df["CAGR"].rank(ascending=False, na_option="bottom")
    df["Score"] = (df["Sharpe_rank"] * 0.7 + df["CAGR_rank"] * 0.3)
    df = df.sort_values(["Score"]).head(top_n).reset_index(drop=True)
    return df.drop(columns=["Sharpe_rank", "CAGR_rank"], errors="ignore")


def walk_forward(
    prices: pd.DataFrame, strategy_name: str, param_grid: Dict[str, List[Any]],
    costs: Dict[str, Any], lookback_days: int = 252, forward_days: int = 63
) -> Tuple[pd.Series, pd.DataFrame]:
    """Train best params on rolling lookback, apply on next forward window."""
    idx = prices.index
    if len(idx) < lookback_days + forward_days + 5:
        return pd.Series(dtype=float), pd.DataFrame()

    from itertools import product
    equity_oos = pd.Series(dtype=float)
    decisions = []

    start_i = lookback_days
    while start_i + forward_days < len(idx):
        train = prices.iloc[start_i - lookback_days : start_i]
        test  = prices.iloc[start_i : start_i + forward_days]
        # grid on train
        best = None
        best_score = np.inf
        for vals in product(*[param_grid[k] for k in param_grid.keys()]):
            params = dict(zip(param_grid.keys(), vals))
            filled = default_params()
            for k, v in params.items(): filled[k] = v
            fn = make_strategy_fn(strategy_name, filled)
            stats = run_backtest(train, fn, **costs)["stats"]
            if not stats: 
                continue
            # rank by Sharpe then -MaxDD
            score = -(stats.get("Sharpe", 0)) - 0.2*stats.get("CAGR",0)
            if score < best_score:
                best_score = score
                best = params
        if best is None:
            start_i += forward_days
            continue
        decisions.append({"start": idx[start_i], **best})
        filled = default_params()
        for k,v in best.items(): filled[k] = v
        fn = make_strategy_fn(strategy_name, filled)
        res = run_backtest(test, fn, **costs)
        equity_oos = pd.concat([equity_oos, res["curve"]])
        start_i += forward_days

    return equity_oos, pd.DataFrame(decisions)


# =============================
# Orchestration
# =============================
def default_params() -> Dict[str, Any]:
    return {
        "fast_sma": 10, "slow_sma": 40,
        "bb_window": 20, "bb_k": 2.0,
        "rsi_window": 14, "rsi_low": 30, "rsi_high": 70,
        "use_atr": False, "atr_window": 14, "atr_k": 0.0,
        "vol_target": 0.0, "vol_window": 20,
        "combo_weights": {"sma": 1.0, "bb": 1.0, "rsi": 1.0}
    }


def run_pipeline(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    regime_mode: str,
    low_pct: float,
    high_pct: float,
    params: Dict[str, Any],
    costs: Dict[str, float],
    source: str,
    upload_file=None,
    synthetic: bool=False,
) -> Dict[str, Any]:
    # 1) Prices
    if synthetic:
        dates = pd.bdate_range("2018-01-01", pd.Timestamp.today(), freq="B")
        rng = np.random.default_rng(42)
        mat = np.cumprod(1 + rng.normal(0.0003, 0.01, size=(len(dates), len(tickers))), axis=0) * 100.0
        prices_wide = pd.DataFrame(mat, index=dates, columns=tickers)
    elif source == "Upload CSV" and upload_file is not None:
        prices_wide = _normalize_prices(load_csv_uploaded(upload_file))
    else:
        fetched = fetch_prices_for_tickers(tickers, start=start, end=end)
        prices_wide = _normalize_prices(fetched)

    prices_wide = _safe_df(prices_wide)
    if prices_wide.empty:
        return {"results": {}, "regimes": pd.Series(dtype=int), "prices": prices_wide}

    # 2) Regimes
    regimes = compute_regimes(prices_wide, regime_mode, low_pct, high_pct)

    # 3) Backtests (all strategies incl. Combo)
    results: Dict[str, Any] = {}
    for name in ["sma_cross", "bollinger", "rsi_trend", "combo"]:
        fn = make_strategy_fn(name, params)
        bt = run_backtest(prices_wide, fn, **costs)
        results[name] = bt

    return {"results": results, "regimes": regimes, "prices": prices_wide}


# =============================
# Visualization & Downloads
# =============================
def plot_equity_with_regimes(eq_df: pd.DataFrame, regimes: pd.Series) -> Optional[plt.Figure]:
    if eq_df is None or eq_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in eq_df.columns:
        ax.plot(eq_df.index, eq_df[col].values, label=col)
    reg = regimes.reindex(eq_df.index).fillna(0)
    prev = None
    seg_start = None
    for ts, v in reg.items():
        if prev is None:
            prev, seg_start = v, ts
            continue
        if v != prev:
            ax.axvspan(seg_start, ts, alpha=0.12, color={1: "green", 0: "gray", -1: "red"}[int(prev)])
            seg_start, prev = ts, v
    if seg_start is not None:
        ax.axvspan(seg_start, eq_df.index[-1], alpha=0.12, color={1: "green", 0: "gray", -1: "red"}[int(prev)])
    ax.set_title("Equity with Regime Ribbon")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    return fig


def build_download_csv(curves: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    curves.to_csv(buf)
    return buf.getvalue().encode("utf-8")


def build_excel_report(summary: pd.DataFrame, curves: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary")
        curves.to_excel(writer, sheet_name="Equity")
    return output.getvalue()


def payoff_by_regime(curve: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame()
    daily = curve.pct_change().dropna()
    r = regimes.reindex(daily.index).fillna(0)
    df = pd.DataFrame({"ret": daily, "reg": r})
    return df.groupby("reg")["ret"].agg(["mean", "std", "count"]).rename(index={-1:"Bear",0:"Neutral",1:"Bull"})


def coverage_table(prices: pd.DataFrame) -> pd.DataFrame:
    non_na = prices.notna().sum()
    total = len(prices.index)
    cov = (non_na / total * 100.0).round(2)
    return pd.DataFrame({"Non-NA Days": non_na, "Coverage%": cov}).sort_index()


def corr_heatmap(prices: pd.DataFrame) -> Optional[plt.Figure]:
    if prices is None or prices.empty:
        return None
    rets = prices.pct_change().dropna(how="any")
    if rets.empty:
        return None
    c = rets.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c.columns)))
    ax.set_xticklabels(c.columns, rotation=90)
    ax.set_yticks(range(len(c.index)))
    ax.set_yticklabels(c.index)
    ax.set_title("Return Correlation Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# =============================
# UI
# =============================
st.title("📈 Strategy–Regime Matrix — Single App (Full)")

with st.sidebar:
    st.subheader("Data Source")
    source = st.selectbox("Source", ["Fetch (Yahoo/Stooq)", "Upload CSV", "Synthetic Demo"], index=0)
    upload_file = None
    if source == "Upload CSV":
        upload_file = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()

    st.subheader("Universe & Dates")
    tickers_raw = st.text_area(
        "Tickers (comma-separated)",
        value="SPY, QQQ, IWM, EFA, EEM, TLT, LQD, HYG, GLD, USO",
        height=80
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    c1, c2 = st.columns(2)
    with c1:
        start = st.text_input("Start (YYYY-MM-DD)", value="")
    with c2:
        end = st.text_input("End (YYYY-MM-DD)", value="")

    st.subheader("Regimes")
    regime_mode = st.selectbox("Mode", ["percentile", "mean±std"], index=0)
    low_pct = st.slider("Low percentile", 0.0, 0.5, 0.40, 0.01)
    high_pct = st.slider("High percentile", 0.5, 1.0, 0.60, 0.01)

    st.subheader("Strategy Params")
    fast_sma = st.number_input("SMA Fast", 2, 200, 10, 1)
    slow_sma = st.number_input("SMA Slow", 5, 400, 40, 1)
    bb_window = st.number_input("Bollinger Window", 5, 200, 20, 1)
    bb_k = st.number_input("Bollinger k", 0.5, 5.0, 2.0, 0.1)
    rsi_window = st.number_input("RSI Window", 2, 60, 14, 1)
    rsi_low = st.number_input("RSI Low", 1, 99, 30, 1)
    rsi_high = st.number_input("RSI High", 1, 99, 70, 1)

    st.caption("ATR / Vol Targeting (optional)")
    use_atr = st.checkbox("Use ATR scaling", value=False)
    atr_window = st.number_input("ATR Window", 2, 60, 14, 1)
    atr_k = st.number_input("ATR k (risk dampener)", 0.0, 10.0, 0.0, 0.1)
    vol_target = st.number_input("Vol Target (ann., e.g. 0.15)", 0.0, 1.0, 0.0, 0.01)
    vol_window = st.number_input("Vol Window", 5, 120, 20, 1)

    st.subheader("Combo Weights")
    w_sma = st.slider("SMA weight", -2.0, 2.0, 1.0, 0.1)
    w_bb = st.slider("Bollinger weight", -2.0, 2.0, 1.0, 0.1)
    w_rsi = st.slider("RSI weight", -2.0, 2.0, 1.0, 0.1)

    st.subheader("Costs")
    account_type = st.selectbox("Account Type", ["margin", "cash"], index=0)
    commission_bps = st.number_input("Commission (bps)", 0.0, 50.0, 1.0, 0.5)
    slippage_bps = st.number_input("Slippage (bps)", 0.0, 50.0, 1.0, 0.5)
    borrow_apr = st.number_input("Short borrow APR", 0.0, 0.5, 0.03, 0.01)
    max_leverage = st.number_input("Max Leverage", 0.1, 5.0, 1.0, 0.1)

    st.divider()

    st.subheader("Optimizer")
    opt_enable = st.checkbox("Enable Optimizer", value=False)
    opt_mode = st.radio("Mode", ["Grid Search", "Walk-Forward"], index=0)
    opt_strategy = st.selectbox("Optimize Strategy", ["sma_cross", "bollinger", "rsi_trend"], index=0)
    opt_topn = st.number_input("Top N (grid)", 1, 50, 10, 1)
    opt_lookback_days = st.number_input("WF Lookback (days)", 60, 1250, 252, 1)
    opt_forward_days = st.number_input("WF Forward (days)", 21, 252, 63, 1)

    # Param grids (simple presets)
    st.caption("Param grids (used if Optimizer enabled)")
    if opt_strategy == "sma_cross":
        grid_fast = st.text_input("fast_sma grid (comma)", "5,10,20")
        grid_slow = st.text_input("slow_sma grid (comma)", "40,60,100")
        param_grid = {
            "fast_sma": [int(x) for x in grid_fast.split(",") if x.strip()],
            "slow_sma": [int(x) for x in grid_slow.split(",") if x.strip()],
        }
    elif opt_strategy == "bollinger":
        grid_win = st.text_input("bb_window grid (comma)", "10,20,40")
        grid_k = st.text_input("bb_k grid (comma)", "1.5,2.0,2.5")
        param_grid = {
            "bb_window": [int(x) for x in grid_win.split(",") if x.strip()],
            "bb_k": [float(x) for x in grid_k.split(",") if x.strip()],
        }
    else:  # rsi_trend
        grid_win = st.text_input("rsi_window grid (comma)", "7,14,21")
        grid_low = st.text_input("rsi_low grid (comma)", "25,30,35")
        grid_high = st.text_input("rsi_high grid (comma)", "65,70,75")
        param_grid = {
            "rsi_window": [int(x) for x in grid_win.split(",") if x.strip()],
            "rsi_low": [int(x) for x in grid_low.split(",") if x.strip()],
            "rsi_high": [int(x) for x in grid_high.split(",") if x.strip()],
        }

    run_btn = st.button("🚀 Run", use_container_width=True)

# Guard
if not tickers:
    st.info("Enter at least one ticker in the sidebar and click **Run**.")
    st.stop()

if run_btn:
    params = default_params()
    params.update({
        "fast_sma": fast_sma, "slow_sma": slow_sma,
        "bb_window": bb_window, "bb_k": bb_k,
        "rsi_window": rsi_window, "rsi_low": rsi_low, "rsi_high": rsi_high,
        "use_atr": use_atr, "atr_window": atr_window, "atr_k": atr_k,
        "vol_target": vol_target, "vol_window": vol_window,
        "combo_weights": {"sma": w_sma, "bb": w_bb, "rsi": w_rsi},
    })
    costs = {
        "account_type": account_type,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "short_borrow_apr": borrow_apr,
        "max_leverage": max_leverage,
    }

    with st.spinner("Running pipeline…"):
        res = run_pipeline(
            tickers=tickers,
            start=start or None,
            end=end or None,
            regime_mode=regime_mode,
            low_pct=low_pct,
            high_pct=high_pct,
            params=params,
            costs=costs,
            source=source,
            upload_file=upload_file,
            synthetic=(source=="Synthetic Demo"),
        )

    prices = res["prices"]
    regimes = res["regimes"]
    results = res["results"]

    # Summary + curves
    summary_rows, curves = [], {}
    for name, out in results.items():
        curve = out.get("curve", pd.Series(dtype=float))
        stats = out.get("stats", {})
        if isinstance(curve, pd.Series) and not curve.empty:
            curves[name] = curve.rename(name)
        row = {"Strategy": name, **stats}
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).set_index("Strategy") if summary_rows else pd.DataFrame()
    eq_df = pd.DataFrame(curves).dropna(how="all")

    st.write(
        f"Prices: {getattr(prices,'shape',None)} | Strategies: {list(results.keys())}"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Equity", "Heatmap", "Payoff", "Coverage", "Optimizer"])

    with tab1:
        st.subheader("Equity")
        fig = plot_equity_with_regimes(eq_df, regimes)
        if fig is None:
            st.warning("No equity curves to plot — check tickers/dates/data source.")
        else:
            try:
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")
        st.subheader("Summary")
        if summary.empty: st.info("No metrics.")
        else: st.dataframe(summary)

        if not eq_df.empty:
            st.download_button("⬇️ CSV: Equity Curves", build_download_csv(eq_df), "equity_curves.csv", "text/csv")
            st.download_button("⬇️ Excel: Report", build_excel_report(summary, eq_df), "report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab2:
        st.subheader("Return Correlation Heatmap")
        fig_hm = corr_heatmap(prices)
        if fig_hm is None: st.info("Not enough data for heatmap.")
        else: st.pyplot(fig_hm, use_container_width=True)

    with tab3:
        st.subheader("Payoff by Regime")
        for name, out in results.items():
            st.markdown(f"**{name}**")
            pf = payoff_by_regime(out.get("curve", pd.Series(dtype=float)), regimes)
            if pf.empty: st.info("No payoff data.")
            else: st.dataframe(pf)

    with tab4:
        st.subheader("Coverage")
        cov = coverage_table(prices)
        if cov.empty: st.info("No coverage.")
        else: st.dataframe(cov)

    with tab5:
        if not opt_enable:
            st.info("Enable optimizer in the sidebar to run.")
        else:
            with st.spinner("Optimizing…"):
                if opt_mode == "Grid Search":
                    best = grid_search(prices, opt_strategy, param_grid, costs, top_n=opt_topn)
                    if best.empty: st.warning("No results.")
                    else: st.dataframe(best)
                else:
                    equity_oos, choices = walk_forward(prices, opt_strategy, param_grid, costs,
                                                       lookback_days=opt_lookback_days, forward_days=opt_forward_days)
                    if equity_oos.empty: st.warning("No walk-forward equity.")
                    else:
                        fig2, ax2 = plt.subplots(figsize=(9,3))
                        ax2.plot(equity_oos.index, equity_oos.values, label="OOS Equity")
                        ax2.set_title("Walk-Forward OOS Equity")
                        ax2.grid(True, alpha=0.2); ax2.legend()
                        st.pyplot(fig2, use_container_width=True)
                    if not choices.empty:
                        st.subheader("Chosen Params per Window")
                        st.dataframe(choices)
else:
    st.info("Adjust parameters and press **Run**.")
