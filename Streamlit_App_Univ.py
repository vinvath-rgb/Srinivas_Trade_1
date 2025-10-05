# Srini_Integrated_Universe_Backtester.py
# One-file, filename-agnostic Streamlit app:
# - Universe Builder (ETF/Stocks/Both) with BUY/WATCH/AVOID
# - Send-to-Backtester sends BUY (default) + optional WATCH (checkbox)
# - Render-friendly: no switch_page; uses st.session_state["bt_tickers"]
# - Backtester with SMA/RSI/Composite, vol-targeting, ATR stops/TP, Excel export
# - Yahoo first, Stooq fallback (.TO/.V for Canada; .NS for India)

import os, io, time
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import streamlit as st
import matplotlib.pyplot as plt

# ───────────────────── PAGE CONFIG / OPTIONAL AUTH ─────────────────────
st.set_page_config(page_title="Srini — Universe + Backtester", layout="wide")

def _auth():
    pw = os.getenv("APP_PASSWORD", "")
    if not pw:
        return
    with st.sidebar:
        st.subheader("🔒 App Login")
        if st.text_input("Password", type="password", key="auth_pw") != pw:
            st.stop()
_auth()

# Live session monitor (helps verify Send ➜ Backtester on Render)
with st.sidebar:
    st.caption("🔎 Live session state")
    st.write({
        "last_sent": st.session_state.get("last_sent"),
        "bt_tickers": st.session_state.get("bt_tickers"),
    })

# ───────────────────── SHARED UTILS ─────────────────────
def price_col(df): return "Adj Close" if "Adj Close" in df.columns else "Close"
def _to_ts(d):     return pd.to_datetime(d).tz_localize(None)
def _to_list(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]

def annualized_return(r: pd.Series, ppy: int = 252) -> float:
    if r.empty: return 0.0
    total = float((1 + r).prod()); yrs = max(len(r)/ppy, 1e-9)
    return total**(1/yrs) - 1

def sharpe(r: pd.Series, rf: float = 0.0, ppy: int = 252) -> float:
    if r.empty: return 0.0
    ex = r - rf/ppy; sd = ex.std()
    return float(np.sqrt(ppy) * (ex.mean() / (sd + 1e-12)))

def max_drawdown(eq: pd.Series):
    if eq.empty: return 0.0, None, None
    roll = eq.cummax(); dd = (eq/roll) - 1
    t = dd.idxmin(); p = roll.loc[:t].idxmax()
    return float(dd.min()), p, t

# ───────────────────── INDICATORS ─────────────────────
def rsi(series: pd.Series, lb: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/lb, adjust=False).mean()
    rd = dn.ewm(alpha=1/lb, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(price: pd.Series, fast=12, slow=26, signal=9):
    ema_f = price.ewm(span=fast, adjust=False).mean()
    ema_s = price.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig  = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def compute_atr(df: pd.DataFrame, lb: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df[price_col(df)]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/lb, adjust=False).mean()

def realized_vol_series(returns: pd.Series, lookback: int = 20, ppy: int = 252):
    return returns.rolling(lookback).std() * np.sqrt(ppy)

def future_realized_vol(returns: pd.Series, lookahead: int = 20, ppy: int = 252):
    return returns.rolling(lookahead).std().shift(-lookahead) * np.sqrt(ppy)

# ───────────────────── SIGNALS ─────────────────────
def sma_signals(price: pd.Series, fast: int, slow: int) -> pd.Series:
    ma_f, ma_s = price.rolling(fast).mean(), price.rolling(slow).mean()
    sig = pd.Series(0.0, index=price.index)
    sig[ma_f > ma_s] = 1.0; sig[ma_f < ma_s] = -1.0
    return sig.fillna(0.0)

def rsi_signals(price: pd.Series, rsi_lb: int, rsi_buy: int, rsi_sell: int) -> pd.Series:
    rr = rsi(price, lb=rsi_lb); sig = pd.Series(0.0, index=price.index)
    sig[rr < rsi_buy] = 1.0; sig[rr > rsi_sell] = -1.0
    return sig.fillna(0.0)

def composite_signal(price: pd.Series,
                     rsi_lb=14, rsi_buy=35, rsi_sell=65,
                     sma_fast=10, sma_slow=50,
                     use_and=False, use_macd=True,
                     atr_filter=False, atr_series: pd.Series|None=None,
                     atr_cap_pct=0.05):
    ma_f = price.rolling(sma_fast).mean(); ma_s = price.rolling(sma_slow).mean()
    sma_sig = pd.Series(0.0, index=price.index); sma_sig[ma_f > ma_s] = 1.0; sma_sig[ma_f < ma_s] = -1.0
    rr = rsi(price, lb=rsi_lb); rsi_sig = pd.Series(0.0, index=price.index)
    rsi_sig[rr < rsi_buy] = 1.0; rsi_sig[rr > rsi_sell] = -1.0
    if use_macd:
        _, _, h = macd(price); macd_sig = pd.Series(0.0, index=price.index)
        macd_sig[h > 0] = 1.0; macd_sig[h < 0] = -1.0
    else:
        macd_sig = pd.Series(0.0, index=price.index)

    if use_and:
        comp = pd.Series(0.0, index=price.index)
        comp[(sma_sig==1) & (rsi_sig>=0) & ((macd_sig>=0)|(~use_macd))]  = 1.0
        comp[(sma_sig==-1)& (rsi_sig<=0) & ((macd_sig<=0)|(~use_macd))] = -1.0
    else:
        score = sma_sig + rsi_sig + macd_sig
        comp = pd.Series(0.0, index=price.index)
        comp[score >= 2]  = 1.0; comp[score <= -2] = -1.0

    if atr_filter and atr_series is not None:
        hot = (atr_series/price).fillna(0) > atr_cap_pct
        comp[hot] = 0.0
    return comp.fillna(0.0)

# ───────────────────── SIZING & EXECUTION ─────────────────────
def position_sizer(signal: pd.Series, returns: pd.Series, vol_target: float, ppy: int = 252) -> pd.Series:
    vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(ppy)
    vol.replace(0, np.nan, inplace=True)
    lev = (vol_target / (vol + 1e-12)).clip(upper=5.0).fillna(0.0)
    return signal * lev

def apply_stops(df: pd.DataFrame, pos: pd.Series, atr: pd.Series,
                atr_stop_mult: float, tp_mult: float,
                trade_cost: float = 0.0, tax_rate: float = 0.0) -> pd.Series:
    c = df[price_col(df)]; ret = c.pct_change().fillna(0.0)
    pnl = pd.Series(0.0, index=c.index)

    active_sign = 0
    entry = np.nan
    block_reentry = False

    entries = exits = flips = 0
    total_cost_paid = 0.0

    for i in range(len(c)):
        px = float(c.iloc[i])
        a  = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else np.nan
        desired_sign = int(np.sign(pos.iloc[i]))
        lev = abs(float(pos.iloc[i]))

        if block_reentry and desired_sign == 0:
            block_reentry = False

        if active_sign != 0 and desired_sign != active_sign:
            if desired_sign == 0:
                exits += 1; total_cost_paid += trade_cost; pnl.iloc[i] -= trade_cost
                active_sign = 0; entry = np.nan
            else:
                flips += 1; total_cost_paid += 2 * trade_cost; pnl.iloc[i] -= 2 * trade_cost
                active_sign = desired_sign; entry = px

        if active_sign == 0 and desired_sign != 0 and not block_reentry:
            entries += 1; total_cost_paid += trade_cost; pnl.iloc[i] -= trade_cost
            active_sign = desired_sign; entry = px

        if active_sign == 0 or np.isnan(a):
            continue

        eff_pos = active_sign * lev

        if active_sign > 0:
            stop = entry * (1 - atr_stop_mult * a / max(entry, 1e-12))
            tp   = entry * (1 + tp_mult     * a / max(entry, 1e-12))

            if px <= stop:
                exits += 1; total_cost_paid += trade_cost; pnl.iloc[i] -= trade_cost
                active_sign = 0; entry = np.nan; block_reentry = True; continue

            if px >= tp:
                realized = (tp / entry) - 1.0
                pnl.iloc[i] += eff_pos * realized
                if realized > 0: pnl.iloc[i] *= (1 - tax_rate)
                exits += 1; total_cost_paid += trade_cost; pnl.iloc[i] -= trade_cost
                active_sign = 0; entry = np.nan; block_reentry = True; continue

            pnl.iloc[i] += eff_pos * ret.iloc[i]

        else:
            stop = entry * (1 + atr_stop_mult * a / max(entry, 1e-12))
            tp   = entry * (1 - tp_mult     * a / max(entry, 1e-12))

            if px >= stop:
                exits += 1; total_cost_paid += trade_cost; pnl.iloc[i] -= trade_cost
                active_sign = 0; entry = np.nan; block_reentry = True; continue

            if px <= tp:
                realized = (entry / tp) - 1.0
                pnl.iloc[i] += eff_pos * realized
                if realized > 0: pnl.iloc[i] *= (1 - tax_rate)
                exits += 1; total_cost_paid += trade_cost; pnl.iloc[i] -= trade_cost
                active_sign = 0; entry = np.nan; block_reentry = True; continue

            pnl.iloc[i] += eff_pos * ret.iloc[i]

    pnl.attrs["entries"] = int(entries)
    pnl.attrs["exits"]   = int(exits)
    pnl.attrs["flips"]   = int(flips)
    pnl.attrs["total_cost_paid"] = round(float(total_cost_paid), 6)
    return pnl

def simple_cagr(df: pd.DataFrame) -> float:
    px = df[price_col(df)]; s, e = float(px.iloc[0]), float(px.iloc[-1])
    yrs = max((df.index[-1]-df.index[0]).days/365.25, 1e-9)
    return (e/max(s,1e-12))**(1/yrs) - 1

# ───────────────────── BACKTEST CORE ─────────────────────
def backtest(df: pd.DataFrame, strategy: str, params: dict,
             vol_target: float, long_only: bool, atr_stop: float, tp_mult: float,
             trade_cost: float=0.0, tax_rate: float=0.0,
             debug: bool=False):
    price = df[price_col(df)]
    rets  = price.pct_change().fillna(0.0)

    if strategy == "SMA Crossover":
        sig = sma_signals(price, params["fast"], params["slow"])
    elif strategy == "RSI Mean Reversion":
        sig = rsi_signals(price, params["rsi_lb"], params["rsi_buy"], params["rsi_sell"])
    elif strategy == "Composite":
        atr_s = compute_atr(df, lb=params.get("atr_lb", 14))
        sig = composite_signal(
            price,
            rsi_lb=params.get("rsi_lb",14), rsi_buy=params.get("rsi_buy",35), rsi_sell=params.get("rsi_sell",65),
            sma_fast=params.get("fast",10),  sma_slow=params.get("slow",50),
            use_and=params.get("use_and", False), use_macd=params.get("use_macd", True),
            atr_filter=params.get("atr_filter", False), atr_series=atr_s, atr_cap_pct=params.get("atr_cap_pct", 0.05),
        )
    else:
        raise ValueError("Unknown strategy")

    if long_only: sig = sig.clip(lower=0.0)

    est_vol = realized_vol_series(rets, lookback=20)
    lev = (vol_target / (est_vol + 1e-12)).clip(upper=5.0)
    pos = (sig * lev).fillna(0.0)

    atr = compute_atr(df, lb=14)
    pnl = apply_stops(df, pos, atr, atr_stop, tp_mult, trade_cost=trade_cost, tax_rate=tax_rate)
    equity = (1 + pnl).cumprod()

    act_vol = future_realized_vol(rets, lookahead=20)

    stats = {
        "CAGR": round(annualized_return(pnl), 4),
        "Sharpe": round(sharpe(pnl), 2),
        "MaxDD": round(max_drawdown(equity)[0], 4),
        "Exposure": round(float((pnl != 0).sum())/max(len(pnl),1), 3),
        "LastEquity": round(float(equity.iloc[-1]) if len(equity) else 1.0, 4),
        "Trades_Entered": pnl.attrs.get("entries", 0),
        "Trades_Exited": pnl.attrs.get("exits", 0),
        "Flips": pnl.attrs.get("flips", 0),
        "TotalCosts(%)": round(100 * pnl.attrs.get("total_cost_paid", 0.0), 3),
    }

    debug_df = None; debug_summary = None
    if debug:
        debug_df = pd.DataFrame({
            "Return": rets,
            "EstVol": est_vol,
            "ActualVol": act_vol,
            "Signal": sig,
            "Leverage": lev,
            "PreExecPosition": (sig * lev).shift(1)
        }).dropna(subset=["Return"])

        aligned = debug_df.dropna(subset=["EstVol","ActualVol"])
        vol_err = (aligned["ActualVol"] - aligned["EstVol"]).abs()
        debug_summary = [{
            "Rows": int(len(debug_df)),
            "Avg EstVol": round(float(aligned["EstVol"].mean()), 4) if len(aligned) else np.nan,
            "Avg ActualVol": round(float(aligned["ActualVol"].mean()), 4) if len(aligned) else np.nan,
            "Mean |Actual-Est|": round(float(vol_err.mean()), 4) if len(aligned) else np.nan,
            "Days Underestimated (Actual>Est)": int((aligned["ActualVol"] > aligned["EstVol"]).sum()) if len(aligned) else 0,
            "Days Overestimated (Actual<Est)": int((aligned["ActualVol"] < aligned["EstVol"]).sum()) if len(aligned) else 0,
        }]

    return equity, stats, debug_df, debug_summary

# ───────────────────── DATA LOADER (Yahoo + Stooq) ─────────────────────
@st.cache_data(show_spinner=False)
def load_prices(tickers_raw: str, start, end) -> dict:
    tickers
    