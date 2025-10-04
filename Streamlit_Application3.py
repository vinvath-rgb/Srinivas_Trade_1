# streamlit_app.py — Srini Backtester vNext
# Cash/Margin modes + Dual-Horizon Volatility Regime + 6 strategies
import os, io, time
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------- PAGE CONFIG --------------------------------
st.set_page_config(page_title="Srini Backtester vNext", layout="wide")

# -------------------------------- AUTH (optional) --------------------------------
def _auth():
    pw = os.getenv("APP_PASSWORD", "")
    if not pw:
        return
    with st.sidebar:
        st.subheader("🔒 App Login")
        if st.text_input("Password", type="password", key="auth_pw") != pw:
            st.stop()
_auth()

# -------------------------------- UTILS --------------------------------
def price_col(df): return "Adj Close" if "Adj Close" in df.columns else "Close"
def _to_ts(d):     return pd.to_datetime(d).tz_localize(None)

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

# -------------------------------- INDICATORS --------------------------------
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

def stochastic_kd(price: pd.DataFrame | pd.Series, k_lb: int = 14, d_lb: int = 3):
    # Accepts series (Close) or df with High/Low/Close
    if isinstance(price, pd.Series):
        c = price
        h = c.rolling(k_lb).max()
        l = c.rolling(k_lb).min()
    else:
        h = price["High"].rolling(k_lb).max()
        l = price["Low"].rolling(k_lb).min()
        c = price[price_col(price)]
    k = 100 * (c - l) / (h - l + 1e-12)
    d = k.rolling(d_lb).mean()
    return k, d

def bollinger(price: pd.Series, lb: int = 20, mult: float = 2.0):
    mid = price.rolling(lb).mean()
    sd  = price.rolling(lb).std()
    upper = mid + mult * sd
    lower = mid - mult * sd
    width = (upper - lower) / (mid + 1e-12)
    return mid, upper, lower, width

def compute_atr(df: pd.DataFrame, lb: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df[price_col(df)]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/lb, adjust=False).mean()

# ----- Volatility measures: EstVol (past) and ActualVol (future) -----
def realized_vol(returns: pd.Series, lookback: int = 20, ppy: int = 252):
    return returns.rolling(lookback).std() * np.sqrt(ppy)

def future_realized_vol(returns: pd.Series, lookahead: int = 20, ppy: int = 252):
    fwd = returns.rolling(lookahead).std().shift(-lookahead) * np.sqrt(ppy)
    return fwd

# -------------------------------- SIGNALS --------------------------------
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

def sma_bollinger_signal(price: pd.Series, sma_p:int=50, bb_p:int=20, bb_mult:float=2.0):
    sma = price.rolling(sma_p).mean()
    mid, upper, lower, _ = bollinger(price, bb_p, bb_mult)
    sig = pd.Series(0.0, index=price.index)
    sig[(price > upper) & (price > sma)] = 1.0
    sig[(price < lower) & (price < sma)] = -1.0
    return sig.fillna(0.0)

def sma_stoch_boll_signal(df: pd.DataFrame, sma_p:int=50, bb_p:int=20, bb_mult:float=2.0,
                          stoch_k:int=14, stoch_d:int=3, stoch_buy:int=30, stoch_sell:int=70):
    price = df[price_col(df)]
    sma = price.rolling(sma_p).mean()
    mid, upper, lower, _ = bollinger(price, bb_p, bb_mult)
    k, d = stochastic_kd(df, stoch_k, stoch_d)
    sig = pd.Series(0.0, index=price.index)

    # Trend + band + timing
    long_cond  = (price > sma) & (price > upper) & (k < stoch_sell)
    short_cond = (price < sma) & (price < lower) & (k > stoch_buy)
    sig[long_cond] = 1.0
    sig[short_cond] = -1.0
    return sig.fillna(0.0)

def stochastic_only_signal(df: pd.DataFrame, stoch_k:int=14, stoch_d:int=3,
                           buy_thr:int=20, sell_thr:int=80):
    k, d = stochastic_kd(df, stoch_k, stoch_d)
    sig = pd.Series(0.0, index=k.index)
    sig[k < buy_thr] = 1.0
    sig[k > sell_thr] = -1.0
    return sig.fillna(0.0)

# -------------------------------- VOL REGIME (Dual Horizon + Hysteresis) --------------------------------
def vol_regime(returns: pd.Series, fast:int=20, slow:int=126, epsilon:float=0.07, ppy:int=252):
    sigma_f = realized_vol(returns, fast, ppy)
    sigma_s = realized_vol(returns, slow, ppy)
    # Regime series: +1 Calm, 0 Neutral, -1 Storm with hysteresis
    regime = pd.Series(index=returns.index, dtype=float)
    state = 0
    for i, t in enumerate(returns.index):
        sf = sigma_f.iloc[i]; ss = sigma_s.iloc[i]
        if np.isnan(sf) or np.isnan(ss):
            regime.iloc[i] = 0
            continue
        if state <= 0 and sf < ss*(1-epsilon):
            state = +1  # Calm
        elif state >= 0 and sf > ss*(1+epsilon):
            state = -1  # Storm
        # else keep previous (neutral transitions keep state)
        regime.iloc[i] = state
    return sigma_f, sigma_s, regime.fillna(0.0)

# -------------------------------- EXECUTION (stateful, with stops/TP) --------------------------------
def apply_stops(df: pd.DataFrame, pos: pd.Series, atr: pd.Series,
                atr_stop_mult: float, tp_mult: float,
                trade_cost: float = 0.0, tax_rate: float = 0.0):
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

# -------------------------------- SIMPLE CAGR (buy&hold) --------------------------------
def simple_cagr(df: pd.DataFrame) -> float:
    px = df[price_col(df)]; s, e = float(px.iloc[0]), float(px.iloc[-1])
    yrs = max((df.index[-1]-df.index[0]).days/365.25, 1e-9)
    return (e/max(s,1e-12))**(1/yrs) - 1

# -------------------------------- BACKTEST CORE --------------------------------
def backtest(df: pd.DataFrame, strategy: str, params: dict,
             vol_target: float,
             account_type: str, long_only_checkbox: bool,
             atr_stop: float, tp_mult: float,
             trade_cost: float=0.0, tax_rate: float=0.0,
             debug: bool=False):
    price = df[price_col(df)]
    rets  = price.pct_change().fillna(0.0)

    # --- Strategy Signals ---
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
    elif strategy == "SMA + Bollinger":
        sig = sma_bollinger_signal(price, params["sma_p"], params["bb_p"], params["bb_mult"])
    elif strategy == "SMA + Stoch + Bollinger":
        sig = sma_stoch_boll_signal(df, params["sma_p"], params["bb_p"], params["bb_mult"],
                                    params["stoch_k"], params["stoch_d"], params["stoch_buy"], params["stoch_sell"])
    elif strategy == "Stochastic Only":
        sig = stochastic_only_signal(df, params["stoch_k"], params["stoch_d"], params["stoch_buy"], params["stoch_sell"])
    else:
        raise ValueError("Unknown strategy")

    # --- Account Mode: shorts & leverage caps ---
    if account_type == "Cash Account":
        allow_shorts = False
        lev_cap = 1.0
        long_only = True  # force long-only in cash mode
    else:
        allow_shorts = True
        lev_cap = 5.0
        long_only = bool(long_only_checkbox)

    # Enforce long-only if applicable
    if long_only:
        sig = sig.clip(lower=0.0)

    # --- Vol Regime (fast/slow + hysteresis) ---
    sigma_f, sigma_s, regime = vol_regime(rets, fast=params.get("vr_fast",20), slow=params.get("vr_slow",126),
                                          epsilon=params.get("vr_eps", 0.07), ppy=252)
    # regime: +1 Calm, 0 Neutral, -1 Storm
    # Multipliers by regime (can be tuned)
    alpha_calm   = params.get("alpha_calm", 1.25)
    alpha_storm  = params.get("alpha_storm", 0.60)

    # --- Vol targeting → position sizing ---
    est_vol = sigma_f  # use fast vol for instantaneous sizing
    lev_base = (vol_target / (est_vol + 1e-12))
    # Apply regime multipliers without double-counting
    mult = pd.Series(1.0, index=price.index)
    mult[regime > 0] = alpha_calm
    mult[regime < 0] = alpha_storm
    lev = (lev_base * mult).clip(upper=lev_cap)
    # Skip dead markets via ATR floor (optional)
    atr_series = compute_atr(df, lb=14)
    atr_floor = params.get("atr_floor", 0.0)
    if atr_floor > 0:
        active = (atr_series / price).fillna(0) >= atr_floor
        sig = sig.where(active, 0.0)

    pos = (sig * lev).fillna(0.0)

    # --- Execution with stops/TP ---
    pnl = apply_stops(df, pos, atr_series, atr_stop, tp_mult, trade_cost=trade_cost, tax_rate=tax_rate)
    equity = (1 + pnl).cumprod()

    # --- Actual future vol (lookahead window) for diagnostics ---
    act_vol = future_realized_vol(rets, lookahead=20)

    # --- Metrics ---
    time_long = float((pos >  1e-12).sum())/max(len(pos),1)
    time_short= float((pos < -1e-12).sum())/max(len(pos),1)
    time_cash = 1.0 - time_long - time_short
    stats = {
        "CAGR": round(annualized_return(pnl), 4),
        "Sharpe": round(sharpe(pnl), 2),
        "MaxDD": round(max_drawdown(equity)[0], 4),
        "Exposure": round(float((pos != 0).sum())/max(len(pos),1), 3),
        "%Time_Long": round(time_long, 3),
        "%Time_Short": round(time_short, 3),
        "%Time_Cash": round(time_cash, 3),
        "Avg_Lev": round(float(np.nanmean(np.abs(pos[pos!=0]))), 3) if len(pos[pos!=0]) else 0.0,
        "LastEquity": round(float(equity.iloc[-1]) if len(equity) else 1.0, 4),
        "Trades_Entered": pnl.attrs.get("entries", 0),
        "Trades_Exited": pnl.attrs.get("exits", 0),
        "Flips": pnl.attrs.get("flips", 0),
        "TotalCosts(%)": round(100 * pnl.attrs.get("total_cost_paid", 0.0), 3),
    }

    # --- Debug DataFrames ---
    debug_df = None; debug_summary = None
    if debug:
        debug_df = pd.DataFrame({
            "Return": rets,
            "EstVol_Fast": est_vol,
            "Vol_Slow": sigma_s,
            "Regime": regime,   # +1 Calm / 0 Neutral / -1 Storm
            "ATR": atr_series,
            "Signal": sig,
            "Lev_Base": lev_base,
            "Lev_Mult": mult,
            "Leverage": lev,
            "PreExecPosition": (sig * lev).shift(1),
            "ActualVol_Future20": act_vol
        }).dropna(subset=["Return"])

        aligned = debug_df.dropna(subset=["EstVol_Fast","ActualVol_Future20"])
        vol_err = (aligned["ActualVol_Future20"] - aligned["EstVol_Fast"]).abs()
        debug_summary = [{
            "Rows": int(len(debug_df)),
            "Avg EstVol(Fast)": round(float(aligned["EstVol_Fast"].mean()), 4) if len(aligned) else np.nan,
            "Avg ActualVol(Fut20)": round(float(aligned["ActualVol_Future20"].mean()), 4) if len(aligned) else np.nan,
            "Mean |Actual-Est|": round(float(vol_err.mean()), 4) if len(aligned) else np.nan,
            "Days Underestimated": int((aligned["ActualVol_Future20"] > aligned["EstVol_Fast"]).sum()) if len(aligned) else 0,
            "Days Overestimated": int((aligned["ActualVol_Future20"] < aligned["EstVol_Fast"]).sum()) if len(aligned) else 0,
            "Time Calm(%)": round(100*float((regime>0).sum())/max(len(regime),1), 1),
            "Time Storm(%)": round(100*float((regime<0).sum())/max(len(regime),1), 1),
        }]

    return equity, stats, debug_df, debug_summary

# -------------------------------- DATA LOADER (Yahoo + TSX/TSXV retry + Stooq) --------------------------------
@st.cache_data(show_spinner=False)
def load_prices(tickers_raw: str, start, end) -> dict:
    tickers_input = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers_input: return {}
    start = _to_ts(start); end = _to_ts(end); end_inc = end + pd.Timedelta(days=1)

    results = {}

    def try_yahoo(sym: str):
        try:
            df = yf.download(sym, start=start, end=end_inc, interval="1d",
                             auto_adjust=False, progress=False, threads=False, timeout=60)
            if df is not None and not df.empty:
                keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
                df = df[keep].sort_index().dropna(how="all")
                return df if not df.empty else None
        except Exception:
            return None
        return None

    def try_stooq(sym: str):
        try:
            df = pdr.DataReader(sym, "stooq", start=start, end=end_inc)
            if df is not None and not df.empty:
                df = df.sort_index()
                if "Adj Close" not in df.columns and "Close" in df.columns:
                    df["Adj Close"] = df["Close"]
                keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
                df = df[keep].dropna(how="all")
                return df if not df.empty else None
        except Exception:
            return None
        return None

    for orig in tickers_input:
        candidates = [orig] if "." in orig else [orig, f"{orig}.TO", f"{orig}.V"]
        df_ok = None
        for cand in candidates:
            df_ok = try_yahoo(cand)
            if df_ok is not None:
                results[orig] = df_ok; break
            time.sleep(0.2)
        if df_ok is None:
            df_ok = try_stooq(orig)
            if df_ok is not None:
                results[orig] = df_ok
            time.sleep(0.2)

    cleaned = {}
    for t, d in results.items():
        if d is None or d.empty: continue
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in d.columns]
        d = d[keep].sort_index().dropna(how="all")
        if not d.empty: cleaned[t] = d
    return cleaned

# -------------------------------- UI --------------------------------
st.title("📊 Srini Backtester vNext — Dual-Horizon Vol Regime")

with st.sidebar:
    st.header("Backtest Settings")
    tickers = st.text_input("Tickers (comma-separated)", value="SPY, XLK, ACN, XIC.TO")
    start = st.date_input("Start", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
    end   = st.date_input("End",   value=pd.Timestamp.today()).strftime("%Y-%m-%d")

    strategy = st.selectbox(
        "Strategy",
        ["SMA Crossover", "RSI Mean Reversion", "Composite",
         "SMA + Bollinger", "SMA + Stoch + Bollinger", "Stochastic Only"],
        index=0
    )

    # Strategy params
    if strategy == "SMA Crossover":
        c1, c2 = st.columns(2)
        fast = c1.number_input("Fast SMA", 2, 200, 20, 1)
        slow = c2.number_input("Slow SMA", 5, 400, 100, 5)
        params = {"fast": int(fast), "slow": int(slow)}
    elif strategy == "RSI Mean Reversion":
        c1, c2, c3 = st.columns(3)
        rsi_lb = c1.number_input("RSI lookback", 2, 100, 14, 1)
        rsi_buy = c2.number_input("RSI Buy <", 5, 50, 30, 1)
        rsi_sell = c3.number_input("RSI Sell >", 50, 95, 70, 1)
        params = {"rsi_lb": int(rsi_lb), "rsi_buy": int(rsi_buy), "rsi_sell": int(rsi_sell)}
    elif strategy == "Composite":
        f1, f2, f3 = st.columns(3)
        fast = f1.number_input("Fast SMA", 2, 200, 20, 1)
        slow = f2.number_input("Slow SMA", 5, 400, 200, 5)
        use_macd = f3.checkbox("Use MACD", value=True)

        r1, r2, r3 = st.columns(3)
        rsi_lb   = r1.number_input("RSI lookback", 2, 100, 14, 1)
        rsi_buy  = r2.number_input("RSI Buy <", 5, 50, 30, 1)
        rsi_sell = r3.number_input("RSI Sell >", 50, 95, 70, 1)

        g1, g2, g3 = st.columns(3)
        combine_logic = g1.selectbox("Combine", ["AND (strict)", "Voting (≥2)"], index=1)
        atr_filter = g2.checkbox("ATR 'too hot' filter", value=False)
        atr_cap    = g3.number_input("Max ATR/Price on entry", 0.01, 0.20, 0.05, 0.01)

        params = {"fast": int(fast), "slow": int(slow),
                  "rsi_lb": int(rsi_lb), "rsi_buy": int(rsi_buy), "rsi_sell": int(rsi_sell),
                  "use_macd": bool(use_macd), "use_and": combine_logic.startswith("AND"),
                  "atr_filter": bool(atr_filter), "atr_cap_pct": float(atr_cap)}
    elif strategy in ["SMA + Bollinger", "SMA + Stoch + Bollinger"]:
        g1, g2, g3 = st.columns(3)
        sma_p = g1.number_input("SMA Period", 5, 400, 50, 1)
        bb_p  = g2.number_input("BB Period", 5, 200, 20, 1)
        bb_m  = g3.number_input("BB σ Multiplier", 1.0, 4.0, 2.0, 0.1)
        params = {"sma_p": int(sma_p), "bb_p": int(bb_p), "bb_mult": float(bb_m)}
        if strategy == "SMA + Stoch + Bollinger":
            s1, s2, s3, s4 = st.columns(4)
            st_k = s1.number_input("%K lookback", 3, 60, 14, 1)
            st_d = s2.number_input("%D smoothing", 1, 10, 3, 1)
            st_buy  = s3.number_input("Stoch Buy <", 5, 50, 30, 1)
            st_sell = s4.number_input("Stoch Sell >", 50, 95, 70, 1)
            params.update({"stoch_k": int(st_k), "stoch_d": int(st_d),
                           "stoch_buy": int(st_buy), "stoch_sell": int(st_sell)})
    else:  # Stochastic Only
        s1, s2, s3, s4 = st.columns(4)
        st_k = s1.number_input("%K lookback", 3, 60, 14, 1)
        st_d = s2.number_input("%D smoothing", 1, 10, 3, 1)
        st_buy  = s3.number_input("Stoch Buy <", 5, 50, 20, 1)
        st_sell = s4.number_input("Stoch Sell >", 50, 95, 80, 1)
        params = {"stoch_k": int(st_k), "stoch_d": int(st_d),
                  "stoch_buy": int(st_buy), "stoch_sell": int(st_sell)}

    # Account type + regime params
    st.markdown("---")
    account_type = st.selectbox("Account Type", ["Cash Account", "Margin Account"], index=0)
    long_only = st.checkbox("Long-only (ignored in Cash mode)", value=True)

    st.subheader("Volatility Regime (Fast/Slow)")
    vr1, vr2, vr3 = st.columns(3)
    vr_fast = vr1.number_input("Fast vol window (days)", 5, 120, 20, 1)
    vr_slow = vr2.number_input("Slow vol window (days)", 20, 400, 126, 1)
    vr_eps  = vr3.number_input("Hysteresis ε (0.01–0.20)", 0.01, 0.20, 0.07, 0.01)
    params.update({"vr_fast": int(vr_fast), "vr_slow": int(vr_slow), "vr_eps": float(vr_eps)})

    vm1, vm2, vm3 = st.columns(3)
    alpha_calm  = vm1.number_input("Calm multiplier (α_calm)", 1.00, 2.00, 1.25, 0.05)
    alpha_storm = vm2.number_input("Storm multiplier (α_storm)", 0.10, 1.00, 0.60, 0.05)
    atr_floor   = vm3.number_input("ATR/Price floor (skip if <)", 0.0, 0.05, 0.0, 0.001)
    params.update({"alpha_calm": float(alpha_calm), "alpha_storm": float(alpha_storm),
                   "atr_floor": float(atr_floor)})

    st.subheader("Risk & Execution")
    vol_target = st.slider("Vol target (annualized)", 0.05, 0.40, 0.15, 0.01)
    atr_stop   = st.slider("ATR Stop (×)", 1.0, 8.0, 3.0, 0.5)
    tp_mult    = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5)

    st.subheader("Real-world frictions")
    trade_cost = st.number_input("Cost per trade (%)", 0.0, 0.50, 0.05, 0.01) / 100.0
    tax_rate   = st.number_input("Effective tax on gains (%)", 0.0, 50.0, 0.0, 1.0) / 100.0

    lot_size = st.number_input("Lot size for payoff (sh/ticker)", 1, 100000, 100)

    st.subheader("Debug / Export")
    debug_mode = st.checkbox("Run Debug Mode (log EstVol/ActualVol + Regime)", value=True)
    include_debug_in_excel = st.checkbox("Include Debug sheets in Excel", value=True)

    run_btn = st.button("Run Backtest")

# -------------------------------- RUN --------------------------------
if run_btn:
    data = load_prices(tickers, start, end)
    if not data:
        st.error("No data downloaded. Try adding exchange suffixes (.TO for Canada, .V for TSXV).")
        st.stop()

    st.subheader("🔎 Data Check")
    in_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    rows = []
    for t in in_list:
        d = data.get(t)
        if d is None or d.empty:
            hint = "" if "." in t else "Try .TO (e.g., XIC.TO) or .V."
            rows.append({"Ticker": t, "Status": "NO DATA", "Rows": 0, "First": "", "Last": "", "Hint": hint})
        else:
            rows.append({"Ticker": t, "Status": "OK", "Rows": int(len(d)),
                         "First": str(d.index.min().date()), "Last": str(d.index.max().date()), "Hint": ""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    results = []
    all_debug_rows = []
    all_dbg_summ = []

    tabs = st.tabs(list(data.keys()))
    for tab, t in zip(tabs, data.keys()):
        with tab:
            df = data[t]
            st.write(f"{t}: {len(df)} rows · {df.index.min().date()} → {df.index.max().date()}")

            equity, stats, dbg_df, dbg_sum = backtest(
                df, strategy, params, vol_target,
                account_type, long_only,
                atr_stop, tp_mult,
                trade_cost=trade_cost, tax_rate=tax_rate,
                debug=debug_mode
            )

            st.subheader(f"{t} — Equity Curve")
            st.line_chart(equity, height=300)

            raw_cagr = simple_cagr(df)
            st.markdown(
                f"**Buy & Hold CAGR:** {raw_cagr:.2%}  |  "
                f"**Strategy CAGR:** {stats['CAGR']:.2%}  |  "
                f"**Sharpe:** {stats['Sharpe']:.2f}  |  "
                f"**MaxDD:** {stats['MaxDD']:.2%}  |  "
                f"**Exposure:** {stats['Exposure']:.0%}  |  "
                f"**% Long/Short/Cash:** {stats['%Time_Long']:.0%} / {stats['%Time_Short']:.0%} / {stats['%Time_Cash']:.0%}  |  "
                f"**Avg Lev:** {stats['Avg_Lev']:.2f}  |  "
                f"**Entries/Exits/Flips:** {stats['Trades_Entered']}/{stats['Trades_Exited']}/{stats['Flips']}  |  "
                f"**Total Costs:** {stats['TotalCosts(%)']:.3f}%"
            )

            # Optional overlays for SMA/BB strategies
            if strategy in ["SMA Crossover", "SMA + Bollinger", "SMA + Stoch + Bollinger"]:
                close = df[price_col(df)]
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(df.index, close, label="Close")
                if strategy == "SMA Crossover":
                    ma_f = close.rolling(params.get("fast",20)).mean()
                    ma_s = close.rolling(params.get("slow",100)).mean()
                    ax.plot(df.index, ma_f, label=f"SMA {params.get('fast',20)}")
                    ax.plot(df.index, ma_s, label=f"SMA {params.get('slow',100)}")
                    bull = (ma_f.shift(1) <= ma_s.shift(1)) & (ma_f > ma_s)
                    bear = (ma_f.shift(1) >= ma_s.shift(1)) & (ma_f < ma_s)
                    ax.scatter(df.index[bull], close[bull], marker="^", s=60, label="Bullish Cross")
                    ax.scatter(df.index[bear], close[bear], marker="v", s=60, label="Bearish Cross")
                else:
                    sma_p = params.get("sma_p",50)
                    ma = close.rolling(sma_p).mean()
                    mid, up, lo, _ = bollinger(close, params.get("bb_p",20), params.get("bb_mult",2.0))
                    ax.plot(df.index, ma, label=f"SMA {sma_p}")
                    ax.plot(df.index, up, label="BB Upper", linestyle="--", linewidth=1)
                    ax.plot(df.index, lo, label="BB Lower", linestyle="--", linewidth=1)
                ax.legend(loc="best"); ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)

            results.append({"Ticker": t, "RawCAGR": round(raw_cagr,4), **stats})

            if debug_mode and dbg_df is not None and not dbg_df.empty:
                dcopy = dbg_df.copy(); dcopy.insert(0, "Ticker", t)
                all_debug_rows.append(dcopy)
                if dbg_sum:
                    for row in dbg_sum:
                        row["Ticker"] = t
                    all_dbg_summ += dbg_sum

    # Results table
    res_df = pd.DataFrame(results).set_index("Ticker")
    res_df["CAGR Δ (Strat − Raw)"] = (res_df["CAGR"] - res_df["RawCAGR"]).round(4)
    st.subheader("📋 Metrics Summary")
    st.dataframe(res_df, use_container_width=True)

    # Simulated Payoffs
    pay_rows = []
    for t in res_df.index:
        df_t = data.get(t)
        if df_t is None or df_t.empty: continue
        close = df_t[price_col(df_t)]
        start_px = float(close.iloc[0]); end_px = float(close.iloc[-1])
        init_cap = lot_size * start_px
        end_cap_strategy = init_cap * float(res_df.loc[t, "LastEquity"])
        pnl_strategy = end_cap_strategy - init_cap
        end_cap_bh = lot_size * end_px
        pnl_bh = end_cap_bh - init_cap
        pay_rows.append({
            "Ticker": t,
            "StartPx": round(start_px,4), "EndPx": round(end_px,4),
            "LotSize(sh)": int(lot_size), "Initial($)": round(init_cap,2),
            "Ending_Strategy($)": round(end_cap_strategy,2), "P&L_Strategy($)": round(pnl_strategy,2),
            "Ending_BuyHold($)": round(end_cap_bh,2), "P&L_BuyHold($)": round(pnl_bh,2),
            "Δ P&L (Strat − B&H)($)": round(pnl_strategy - pnl_bh,2),
        })
    pay_df = pd.DataFrame(pay_rows).set_index("Ticker")
    if not pay_df.empty:
        totals = pay_df.select_dtypes(include=[float, int]).sum(numeric_only=True)
        totals_df = pd.DataFrame(totals).T; totals_df.index = ["TOTAL"]
        pay_df = pd.concat([pay_df, totals_df], axis=0)
    st.subheader("💵 Simulated Payoffs")
    st.dataframe(pay_df, use_container_width=True)

    # Inputs table (for Excel)
    def _strategy_name_and_params(strategy, params):
        if strategy == "SMA Crossover":
            return strategy, {"Fast SMA": params.get("fast"), "Slow SMA": params.get("slow")}
        elif strategy == "RSI Mean Reversion":
            return strategy, {"RSI lookback": params.get("rsi_lb"),
                              "RSI Buy <": params.get("rsi_buy"),
                              "RSI Sell >": params.get("rsi_sell")}
        elif strategy == "Composite":
            return strategy, {
                "Fast SMA": params.get("fast"), "Slow SMA": params.get("slow"),
                "RSI lookback": params.get("rsi_lb"), "RSI Buy <": params.get("rsi_buy"),
                "RSI Sell >": params.get("rsi_sell"), "Use MACD": params.get("use_macd"),
                "Combine": "AND" if params.get("use_and", False) else "Voting (≥2)",
                "ATR hot filter": params.get("atr_filter", False),
                "Max ATR/Price": params.get("atr_cap_pct"),
            }
        elif strategy == "SMA + Bollinger":
            return strategy, {"SMA": params.get("sma_p"), "BB Period": params.get("bb_p"),
                              "BB Mult": params.get("bb_mult")}
        elif strategy == "SMA + Stoch + Bollinger":
            return strategy, {"SMA": params.get("sma_p"), "BB Period": params.get("bb_p"),
                              "BB Mult": params.get("bb_mult"), "%K": params.get("stoch_k"),
                              "%D": params.get("stoch_d"), "Stoch Buy <": params.get("stoch_buy"),
                              "Stoch Sell >": params.get("stoch_sell")}
        else:
            return strategy, {"%K": params.get("stoch_k"), "%D": params.get("stoch_d"),
                              "Stoch Buy <": params.get("stoch_buy"),
                              "Stoch Sell >": params.get("stoch_sell")}

    strat_name, strat_params = _strategy_name_and_params(strategy, params)
    inputs_rows = [("Tickers", ", ".join(sorted(data.keys()))),
                   ("Start", start), ("End", end),
                   ("Account Type", account_type), ("Long-only", long_only if account_type!="Cash Account" else True),
                   ("Strategy", strat_name)]
    for k, v in strat_params.items(): inputs_rows.append((k, v))
    inputs_rows += [("Vol target (ann.)", vol_target),
                    ("ATR Stop (×)", atr_stop), ("Take Profit (× ATR)", tp_mult),
                    ("Cost per trade (%)", round(trade_cost*100,3)),
                    ("Effective tax on gains (%)", round(tax_rate*100,3)),
                    ("ATR/Price floor", params.get("atr_floor",0.0)),
                    ("Regime fast/slow/ε", f"{params.get('vr_fast',20)}/{params.get('vr_slow',126)}/{params.get('vr_eps',0.07)}"),
                    ("α_calm / α_storm", f"{params.get('alpha_calm',1.25)} / {params.get('alpha_storm',0.60)}"),
                    ("Lot size (sh)", int(lot_size))]
    inputs_df = pd.DataFrame(inputs_rows, columns=["Parameter","Value"])
    st.subheader("⚙️ Inputs")
    st.dataframe(inputs_df, use_container_width=True)

    # ----------------------- Excel Export -----------------------
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            inputs_df.to_excel(writer, sheet_name="Inputs_&_Metrics", index=False, startrow=0)
            res_df.reset_index().to_excel(writer, sheet_name="Inputs_&_Metrics", index=False, startrow=len(inputs_df)+2)
            pay_df.reset_index().to_excel(writer, sheet_name="Simulated_Payoffs", index=False)

            if include_debug_in_excel and debug_mode:
                if all_debug_rows:
                    debug_cat = pd.concat(all_debug_rows, axis=0, ignore_index=True)
                    debug_cat.to_excel(writer, sheet_name="Debug_Log", index=False)
                if all_dbg_summ:
                    dbg_summary_df = pd.DataFrame(all_dbg_summ)
                    dbg_summary_df.to_excel(writer, sheet_name="Debug_Summary", index=False)

        data_xlsx = output.getvalue()

    st.download_button("⬇️ Download Backtest Report (Excel, Est/Actual Vol + Regime)",
                       data_xlsx, "Backtest_Report_vNext.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")