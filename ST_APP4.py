# streamlit_app_full.py (Part 1/2)
# Srini — Universe + Backtester with SMA+Bollinger, SMA+Stoch+Boll, Stoch Only
# and Volatility Regime (fast/slow) adaptive leverage caps

import os, io, time
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Srini — Universe + Backtester (Regimes)", layout="wide")

# ───────────────── Session defaults ─────────────────
for k, v in {
    "authed": False,
    "bt_tickers": "",
    "last_sent": "",
    "last_preview": [],
    "ubuild_id": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────── Optional auth ─────────────────
def _auth():
    pw = os.getenv("APP_PASSWORD", "")
    if not pw:
        return
    with st.sidebar:
        st.subheader("🔒 App Login")
        if not st.session_state["authed"]:
            entered = st.text_input("Password", type="password", key="auth_pw")
            if entered:
                if entered == pw:
                    st.session_state["authed"] = True
                    st.rerun()
                else:
                    st.error("❌ Wrong Password, please try again")
            st.stop()
_auth()

# ───────────────── Utils ─────────────────
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

# ───────────────── Indicators ─────────────────
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

def bollinger(price: pd.Series, lb: int = 20, mult: float = 2.0):
    mid = price.rolling(lb).mean()
    sd  = price.rolling(lb).std(ddof=0)
    upper = mid + mult * sd
    lower = mid - mult * sd
    return mid, upper, lower

def stoch_kd(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth: int = 3):
    high, low, close = df["High"], df["Low"], df[price_col(df)]
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    raw_k = 100.0 * (close - lowest) / (highest - lowest + 1e-12)
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d

# ───────────────── Signals ─────────────────
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

def signal_sma_bollinger(price: pd.Series, sma_slow: int = 200, bb_lb: int = 20, bb_mult: float = 2.0) -> pd.Series:
    _, upper, lower = bollinger(price, bb_lb, bb_mult)
    slow = price.rolling(sma_slow).mean()
    uptrend   = price > slow
    downtrend = price < slow
    sig = pd.Series(0.0, index=price.index)
    sig[(price < lower) & uptrend]   = 1.0
    sig[(price > upper) & downtrend] = -1.0
    return sig.fillna(0.0)

def signal_stochastic_only(df: pd.DataFrame, k_period=14, d_period=3, smooth=3,
                           os_level=20, ob_level=80) -> pd.Series:
    k, d = stoch_kd(df, k_period=k_period, d_period=d_period, smooth=smooth)
    cross_up = (k.shift(1) <= d.shift(1)) & (k > d)
    cross_dn = (k.shift(1) >= d.shift(1)) & (k < d)
    sig = pd.Series(0.0, index=k.index)
    sig[cross_up & (k < os_level)] = 1.0
    sig[cross_dn & (k > ob_level)] = -1.0
    return sig.reindex(df.index).fillna(0.0)

def signal_sma_stoch_boll(df: pd.DataFrame, sma_slow=200, bb_lb=20, bb_mult=2.0,
                          k_period=14, d_period=3, smooth=3, os_level=20, ob_level=80) -> pd.Series:
    price = df[price_col(df)]
    _, upper, lower = bollinger(price, bb_lb, bb_mult)
    slow = price.rolling(sma_slow).mean()
    uptrend   = price > slow
    downtrend = price < slow
    k, d = stoch_kd(df, k_period=k_period, d_period=d_period, smooth=smooth)
    cross_up = (k.shift(1) <= d.shift(1)) & (k > d)
    cross_dn = (k.shift(1) >= d.shift(1)) & (k < d)
    sig = pd.Series(0.0, index=price.index)
    sig[(uptrend) & (price < lower) & (cross_up) & (k < os_level)] = 1.0
    sig[(downtrend) & (price > upper) & (cross_dn) & (k > ob_level)] = -1.0
    return sig.fillna(0.0)

# ───────────────── Execution ─────────────────
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
    # streamlit_app_full.py (Part 2/2)
# ───────────────── Backtest (with Regime) ─────────────────
def backtest(df: pd.DataFrame, strategy: str, params: dict,
             vol_target: float, long_only: bool, atr_stop: float, tp_mult: float,
             trade_cost: float=0.0, tax_rate: float=0.0,
             debug: bool=False,
             fast_w: int = 20, slow_w: int = 126, eps: float = 0.07,
             cap_calm: float = 5.0, cap_neutral: float = 3.0, cap_storm: float = 1.5):
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
    elif strategy == "SMA + Bollinger":
        sig = signal_sma_bollinger(price,
                                   sma_slow=params["sma_slow"],
                                   bb_lb=params["bb_lb"],
                                   bb_mult=params["bb_mult"])
    elif strategy == "SMA + Stoch + Bollinger":
        sig = signal_sma_stoch_boll(df,
                                    sma_slow=params["sma_slow"],
                                    bb_lb=params["bb_lb"],
                                    bb_mult=params["bb_mult"],
                                    k_period=params["k_period"],
                                    d_period=params["d_period"],
                                    smooth=params["smooth"],
                                    os_level=params["os_level"],
                                    ob_level=params["ob_level"])
    elif strategy == "Stochastic Only":
        sig = signal_stochastic_only(df,
                                     k_period=params["k_period"],
                                     d_period=params["d_period"],
                                     smooth=params["smooth"],
                                     os_level=params["os_level"],
                                     ob_level=params["ob_level"])
    else:
        raise ValueError("Unknown strategy")

    if long_only:
        sig = sig.clip(lower=0.0)

    est_vol_fast = realized_vol_series(rets, lookback=fast_w)
    lev_raw = vol_target / (est_vol_fast + 1e-12)

    est_vol_slow = realized_vol_series(rets, lookback=slow_w)
    ratio = est_vol_fast / (est_vol_slow + 1e-12)

    calm    = ratio < (1.0 - eps)
    stormy  = ratio > (1.0 + eps)
    neutral = ~(calm | stormy)

    cap = pd.Series(cap_neutral, index=ratio.index)
    cap[calm]   = cap_calm
    cap[stormy] = cap_storm

    lev = np.minimum(lev_raw, cap).clip(upper=5.0).fillna(0.0)
    pos = (sig * lev).fillna(0.0)

    atr = compute_atr(df, lb=14)
    pnl = apply_stops(df, pos, atr, atr_stop, tp_mult, trade_cost=trade_cost, tax_rate=tax_rate)
    equity = (1 + pnl).cumprod()

    act_vol = future_realized_vol(rets, lookahead=fast_w)

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
        regime = pd.Series("Neutral", index=ratio.index)
        regime[calm]   = "Calm"
        regime[stormy] = "Stormy"
        debug_df = pd.DataFrame({
            "Return": rets,
            "EstVol_fast": est_vol_fast,
            "EstVol_slow": est_vol_slow,
            "RegimeRatio": ratio,
            "Regime": regime,
            "LeverageRaw": lev_raw,
            "LeverageCap": cap,
            "Leverage": lev,
            "Signal": sig,
            "PreExecPosition": (sig * lev).shift(1),
            "ActualVol_next": act_vol,
        }).dropna(subset=["Return"])

        aligned = debug_df.dropna(subset=["EstVol_fast","ActualVol_next"])
        vol_err = (aligned["ActualVol_next"] - aligned["EstVol_fast"]).abs()
        debug_summary = [{
            "Rows": int(len(debug_df)),
            "Avg EstVol_fast": round(float(aligned["EstVol_fast"].mean()), 4) if len(aligned) else np.nan,
            "Avg ActualVol_next": round(float(aligned["ActualVol_next"].mean()), 4) if len(aligned) else np.nan,
            "Mean |Actual-Est_fast|": round(float(vol_err.mean()), 4) if len(aligned) else np.nan,
            "Calm days": int((debug_df["Regime"]=="Calm").sum()),
            "Neutral days": int((debug_df["Regime"]=="Neutral").sum()),
            "Stormy days": int((debug_df["Regime"]=="Stormy").sum()),
            "Avg Lev (cap applied)": round(float(debug_df["Leverage"].mean()), 3) if len(debug_df) else np.nan,
            "Max Lev": round(float(debug_df["Leverage"].max()), 3) if len(debug_df) else np.nan,
        }]

    return equity, stats, debug_df, debug_summary

# ───────────────── Data loader (Yahoo + Stooq) ─────────────────
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
        candidates = [orig] if "." in orig else [orig, f"{orig}.TO", f"{orig}.V", f"{orig}.NS"]
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

# ───────────────── Universe scoring ─────────────────
def pct_change_lb(series: pd.Series, lookback_days: int) -> float:
    if len(series) < lookback_days + 1:
        return np.nan
    s = float(series.iloc[-(lookback_days+1)])
    e = float(series.iloc[-1])
    return (e / s - 1.0) if s > 0 else np.nan

def realized_vol_last(series: pd.Series, lb: int = 20) -> float:
    rr = series.pct_change()
    if rr.dropna().empty:
        return np.nan
    return float(rr.rolling(lb).std().iloc[-1])

def avg_dollar_vol(close: pd.Series, volume: pd.Series, lb: int = 20) -> float:
    if close.dropna().empty or volume.dropna().empty:
        return np.nan
    adv = (close * volume).rolling(lb).mean().iloc[-1]
    return float(adv) if pd.notna(adv) else np.nan

def sma_last(series: pd.Series, lb: int = 200) -> float:
    if len(series) < lb:
        return np.nan
    return float(series.rolling(lb).mean().iloc[-1])

def near_52w_high(price: pd.Series, window: int = 252) -> float:
    if len(price) < window:
        return np.nan
    high = float(price.iloc[-window:].max())
    last = float(price.iloc[-1])
    return last / high if high > 0 else np.nan

def score_universe(data: dict[str, pd.DataFrame],
                   min_price: float, min_adv: float,
                   w_mom: float, w_trend: float, w_lowvol: float, w_prox: float,
                   auto_adapt_lookbacks: bool = True,
                   base_lookbacks=(252-21, 126-21, 63-21)) -> pd.DataFrame:
    rows = []
    for t, df in data.items():
        close = df[price_col(df)].dropna()
        vol   = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(index=close.index, dtype=float)
        if close.empty:
            continue

        L = len(close)
        lb12, lb6, lb3 = base_lookbacks
        if auto_adapt_lookbacks:
            lb12 = max(min(lb12, max(L - 21, 0)), 40) if L > 60 else np.nan
            lb6  = max(min(lb6,  max(L - 21, 0)), 30) if L > 50 else np.nan
            lb3  = max(min(lb3,  max(L - 21, 0)), 20) if L > 40 else np.nan

        last_px = float(close.iloc[-1])
        adv = avg_dollar_vol(close, vol, lb=20)
        if (not np.isnan(last_px) and last_px >= min_price) and (not np.isnan(adv) and adv >= min_adv):
            m12 = pct_change_lb(close, int(lb12)) if not np.isnan(lb12) else np.nan
            m6  = pct_change_lb(close, int(lb6))  if not np.isnan(lb6)  else np.nan
            m3  = pct_change_lb(close, int(lb3))  if not np.isnan(lb3)  else np.nan

            sma200 = sma_last(close, 200)
            if np.isnan(sma200):
                sma200 = sma_last(close, 50)
            trend = (last_px / sma200 - 1.0) if (not np.isnan(sma200) and sma200 > 0) else np.nan

            vol20 = realized_vol_last(close, 20)
            inv_vol = (1.0 / vol20) if (vol20 and vol20 > 0) else np.nan

            prox_w = min(L, 252)
            prox = near_52w_high(close, prox_w) if L >= 20 else np.nan

            rows.append({
                "Ticker": t,
                "Last": round(last_px, 4),
                "ADV20": round(adv, 2) if adv is not None else np.nan,
                "Mom12_1": m12, "Mom6_1": m6, "Mom3_1": m3,
                "Trend200": trend,
                "InvVol20": inv_vol,
                "Prox52W": prox
            })

    if not rows:
        return pd.DataFrame(columns=["Ticker","Last","ADV20","Mom12_1","Mom6_1","Mom3_1","Trend200","InvVol20","Prox52W","Score","Action"])

    dfm = pd.DataFrame(rows)

    def pct_rank(col):
        return col.rank(pct=True, na_option="keep")

    comp = []
    if dfm["Mom12_1"].notna().any(): comp.append((pct_rank(dfm["Mom12_1"]), 0.5))
    if dfm["Mom6_1"].notna().any():  comp.append((pct_rank(dfm["Mom6_1"]), 0.3))
    if dfm["Mom3_1"].notna().any():  comp.append((pct_rank(dfm["Mom3_1"]), 0.2))
    if comp:
        total_w = sum(w for _, w in comp)
        mom_score = sum(z*w for z,w in comp) / (total_w if total_w>0 else 1.0)
    else:
        mom_score = pd.Series(np.nan, index=dfm.index)

    z_trend = pct_rank(dfm["Trend200"]) if dfm["Trend200"].notna().any() else pd.Series(np.nan, index=dfm.index)
    z_inv   = pct_rank(dfm["InvVol20"]) if dfm["InvVol20"].notna().any() else pd.Series(np.nan, index=dfm.index)
    z_prox  = pct_rank(dfm["Prox52W"])  if dfm["Prox52W"].notna().any()  else pd.Series(np.nan, index=dfm.index)

    scores = []
    for i in dfm.index:
        parts = []
        if not np.isnan(mom_score.iloc[i]): parts.append(w_mom   * mom_score.iloc[i])
        if not np.isnan(z_trend.iloc[i]):   parts.append(w_trend * z_trend.iloc[i])
        if not np.isnan(z_inv.iloc[i]):     parts.append(w_lowvol* z_inv.iloc[i])
        if not np.isnan(z_prox.iloc[i]):    parts.append(w_prox  * z_prox.iloc[i])
        scores.append(np.round(sum(parts), 4) if parts else np.nan)

    dfm["Score"] = scores

    valid = dfm["Score"].notna()
    if valid.any():
        p = dfm.loc[valid, "Score"].rank(pct=True)
        action = pd.Series("AVOID", index=dfm.index, dtype=object)
        action.loc[valid & (p >= 0.8)] = "BUY"
        action.loc[valid & (p >= 0.6) & (p < 0.8)] = "WATCH"
        dfm["Action"] = action
    else:
        dfm["Action"] = "AVOID"

    return dfm.sort_values(["Action","Score","Ticker"], ascending=[True, False, True])

# ───────────────── Presets & handoff ─────────────────
ETF_CORE = "SPY, QQQ, IWM, DIA, XLK, XLF, XLE, XLI, XLP, XLY, XLU, XLV, XLC, XBI, SMH, ARKK, XIC.TO, XIU.TO, ZCN.TO"
US_LARGE = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST","AMD","NFLX","CRM","ADBE","PEP","LIN","LLY","UNH","PG","V","MA","HD","KO","JNJ","BAC","XOM","WMT","CAT","MRK","ABBV","CVX","GE","CSCO","ORCL","TXN","AMAT","QCOM"]
CA_LARGE = ["RY.TO","TD.TO","BNS.TO","ENB.TO","CNQ.TO","BMO.TO","SHOP.TO","SU.TO","CNR.TO","TRP.TO","QSR.TO","BN.TO","CP.TO","NTR.TO"]
IN_LARGE = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS","SBIN.NS","KOTAKBANK.NS","WIPRO.NS","HCLTECH.NS","LT.NS"]

def preset_list(region: str) -> str:
    return {
        "US large": ", ".join(US_LARGE),
        "Canada large": ", ".join(CA_LARGE),
        "India large": ", ".join(IN_LARGE),
    }.get(region, "")

def send_to_backtester(to_send: list[str]):
    tickers_str = ", ".join(to_send)
    st.session_state["bt_tickers"] = tickers_str
    st.session_state["last_sent"] = tickers_str
    st.success(f"Sent to Backtester: {tickers_str}")
    st.rerun()

# ───────────────── Layout ─────────────────
st.title("🧭 Srini — Universe Builder + Backtester (with Vol Regimes)")
tab_univ, tab_bt = st.tabs(["Universe", "Backtester"])

# ===== Universe =====
with tab_univ:
    st.subheader("Universe Builder")
    uni_type = st.radio("Universe Type", ["ETF", "Stocks", "Both"], index=2, horizontal=True)

    with st.expander("Candidate source", expanded=True):
        src = st.selectbox("Source for Stocks", ["Manual", "US large", "Canada large", "India large"], index=1)
        default_stks = preset_list(src) if src != "Manual" else \
            "AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AMD, AVGO, COST, RY.TO, TD.TO, ENB.TO, CNQ.TO, SHOP.TO, RELIANCE.NS, TCS.NS, HDFCBANK.NS"
        default_etfs = ETF_CORE
        c1, c2 = st.columns(2)
        etf_input = c1.text_area("ETF Candidates", value=default_etfs, height=120, disabled=(uni_type=="Stocks"))
        stk_input = c2.text_area("Stock Candidates", value=default_stks, height=120, disabled=(uni_type=="ETF"))

    with st.expander("Time Window & Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        start_u = c1.date_input("History Start", value=pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")
        end_u   = c2.date_input("History End",   value=pd.Timestamp.today()).strftime("%Y-%m-%d")
        min_px  = c3.number_input("Min Price", 0.0, 10000.0, 5.0, 0.5)
        min_adv = c4.number_input("Min Avg Dollar Volume (20d)", 0.0, 1e12, 5_000_000.0, 100_000.0)
        auto_adapt = st.checkbox("Auto-adapt lookbacks when history is short", value=True)

    with st.expander("Factor Weights", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        w_mom   = c1.slider("Momentum", 0.0, 1.0, 0.5, 0.05)
        w_trend = c2.slider("Trend (SMA200)", 0.0, 1.0, 0.2, 0.05)
        w_lv    = c3.slider("Low-Vol (favor calmer)", 0.0, 1.0, 0.2, 0.05)
        w_prox  = c4.slider("52W Proximity", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")
    with st.expander("Send options", expanded=True):
        colA, colB, colC = st.columns([1,1,2])
        include_watch = colA.checkbox("Include WATCH", value=False)
        max_send = int(colB.number_input("Max tickers to send", 1, 500, 25, 1))
        preview_placeholder = colC.empty()

    run_u = st.button("🚀 Build Universe")

    etf_top = pd.DataFrame(); stk_top = pd.DataFrame()
    to_send = []
    if run_u:
        st.cache_data.clear(); st.session_state["ubuild_id"] += 1

        def _clip(df: pd.DataFrame, s, e):
            return df.loc[_to_ts(s):_to_ts(e)]

        if uni_type in ("ETF", "Both"):
            etfs = _to_list(etf_input)
            with st.spinner("Downloading ETF prices..."):
                etf_data = load_prices(", ".join(etfs), start_u, end_u)
                etf_data = {t: _clip(df, start_u, end_u) for t, df in etf_data.items()}
            with st.spinner("Scoring ETFs..."):
                etf_df = score_universe(etf_data, min_px, min_adv, w_mom, w_trend, w_lv, w_prox, auto_adapt_lookbacks=auto_adapt)
            st.subheader("📦 ETF Recommendations")
            if etf_df.empty: st.warning("No ETFs passed filters.")
            else:
                st.dataframe(etf_df, use_container_width=True)
                etf_top = etf_df.copy()

        if uni_type in ("Stocks", "Both"):
            stks = _to_list(stk_input)
            with st.spinner("Downloading Stock prices..."):
                stk_data = load_prices(", ".join(stks), start_u, end_u)
                stk_data = {t: _clip(df, start_u, end_u) for t, df in stk_data.items()}
            with st.spinner("Scoring Stocks..."):
                stk_df = score_universe(stk_data, min_px, min_adv, w_mom, w_trend, w_lv, w_prox, auto_adapt_lookbacks=auto_adapt)
            st.subheader("💎 Stock Recommendations")
            if stk_df.empty: st.warning("No stocks passed filters.")
            else:
                st.dataframe(stk_df, use_container_width=True)
                stk_top = stk_df.copy()

        def select_for_sending(df):
            if df.empty: return []
            allowed = ["BUY"] + (["WATCH"] if include_watch else [])
            df2 = df[df["Action"].isin(allowed)].sort_values("Score", ascending=False)
            return df2["Ticker"].tolist()[:max_send]

        if uni_type in ("ETF", "Both"):   to_send += select_for_sending(etf_top)
        if uni_type in ("Stocks", "Both"): to_send += select_for_sending(stk_top)

        seen = set(); dedup = []
        for t in to_send:
            if t not in seen:
                dedup.append(t); seen.add(t)
        to_send = dedup

        st.session_state["last_preview"] = to_send[:]
        if to_send:
            preview_placeholder.markdown("**Preview (to Backtester):** " + ", ".join(to_send))
        else:
            preview_placeholder.info("Nothing to send — widen dates, include WATCH, or relax filters.")

        st.caption(f"✅ Built #{st.session_state['ubuild_id']} · Window: {start_u} → {end_u} · MinPx: {min_px} · MinADV20: {min_adv:,.0f} · Preview: {len(to_send)}")

    if st.button("➡️ Send to Backtester", disabled=(len(st.session_state['last_preview']) == 0)):
        send_to_backtester(st.session_state["last_preview"])

# ===== Backtester =====
with tab_bt:
    st.subheader("Backtester — EstVol vs ActualVol + Regimes")

    if st.session_state["last_sent"]:
        st.info(f"Tickers received from Universe: {st.session_state['last_sent']}")

    if not st.session_state["bt_tickers"]:
        st.session_state["bt_tickers"] = "SPY, XLK, ACN, XIC.TO"
    tickers = st.text_input("Tickers (comma-separated)", key="bt_tickers")

    c_top2, c_top3 = st.columns(2)
    start_bt = c_top2.date_input("Start", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
    end_bt   = c_top3.date_input("End",   value=pd.Timestamp.today()).strftime("%Y-%m-%d")

    strategy = st.selectbox(
        "Strategy",
        ["SMA Crossover", "RSI Mean Reversion", "Composite",
         "SMA + Bollinger", "SMA + Stoch + Bollinger", "Stochastic Only"],
        index=0
    )

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

    elif strategy == "SMA + Bollinger":
        c1, c2, c3 = st.columns(3)
        sma_slow = c1.number_input("Slow SMA (trend)", 50, 400, 200, 5)
        bb_lb    = c2.number_input("Bollinger lookback", 5, 200, 20, 1)
        bb_mult  = c3.number_input("Band width (σ)", 1.0, 4.0, 2.0, 0.1)
        params = {"sma_slow": int(sma_slow), "bb_lb": int(bb_lb), "bb_mult": float(bb_mult)}

    elif strategy == "SMA + Stoch + Bollinger":
        r1,r2,r3 = st.columns(3)
        sma_slow = r1.number_input("Slow SMA (trend)", 50, 400, 200, 5)
        bb_lb    = r2.number_input("Bollinger lookback", 5, 200, 20, 1)
        bb_mult  = r3.number_input("Band width (σ)", 1.0, 4.0, 2.0, 0.1)
        s1,s2,s3,s4,s5 = st.columns(5)
        k_period = s1.number_input("%K period", 3, 60, 14, 1)
        d_period = s2.number_input("%D period", 1, 30, 3, 1)
        smooth   = s3.number_input("Smoothing", 1, 10, 3, 1)
        os_level = s4.number_input("Oversold ≤", 1, 50, 20, 1)
        ob_level = s5.number_input("Overbought ≥", 50, 99, 80, 1)
        params = {"sma_slow": int(sma_slow), "bb_lb": int(bb_lb), "bb_mult": float(bb_mult),
                  "k_period": int(k_period), "d_period": int(d_period), "smooth": int(smooth),
                  "os_level": int(os_level), "ob_level": int(ob_level)}

    elif strategy == "Stochastic Only":
        s1,s2,s3,s4,s5 = st.columns(5)
        k_period = s1.number_input("%K period", 3, 60, 14, 1)
        d_period = s2.number_input("%D period", 1, 30, 3, 1)
        smooth   = s3.number_input("Smoothing", 1, 10, 3, 1)
        os_level = s4.number_input("Oversold ≤", 1, 50, 20, 1)
        ob_level = s5.number_input("Overbought ≥", 50, 99, 80, 1)
        params = {"k_period": int(k_period), "d_period": int(d_period), "smooth": int(smooth),
                  "os_level": int(os_level), "ob_level": int(ob_level)}

    c2b1, c2b2, c2b3 = st.columns(3)
    long_only  = c2b1.checkbox("Long-only (ignored in Cash mode)", value=True)
    vol_target = c2b2.slider("Vol target (ann.)", 0.05, 0.40, 0.15, 0.01)
    atr_stop   = c2b3.slider("ATR Stop (×)", 1.0, 8.0, 3.0, 0.5)

    c3b1, c3b2, c3b3 = st.columns(3)
    tp_mult    = c3b1.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5)
    trade_cost = c3b2.number_input("Cost per trade (%)", 0.0, 0.50, 0.05, 0.01) / 100.0
    tax_rate   = c3b3.number_input("Effective tax on gains (%)", 0.0, 50.0, 0.0, 1.0) / 100.0

    lot_size = st.number_input("Lot size for payoff (sh/ticker)", 1, 100000, 100)

    st.markdown("### Volatility Regime (Fast/Slow)")
    r1, r2, r3 = st.columns(3)
    fast_w = r1.number_input("Fast vol window (days)", 5, 120, 20, 1)
    slow_w = r2.number_input("Slow vol window (days)", 30, 400, 126, 1)
    eps    = r3.number_input("Hysteresis ε (0.01–0.20)", 0.01, 0.20, 0.07, 0.01)

    rc1, rc2, rc3 = st.columns(3)
    cap_calm   = rc1.number_input("Cap if Calm (×)", 0.5, 10.0, 5.0, 0.1)
    cap_neutral= rc2.number_input("Cap if Neutral (×)", 0.5, 10.0, 3.0, 0.1)
    cap_storm  = rc3.number_input("Cap if Stormy (×)", 0.5, 10.0, 1.5, 0.1)

    st.markdown("---")
    debug_mode = st.checkbox("Run Debug Mode (log vols/regime/leverage)", value=True)
    include_debug_in_excel = st.checkbox("Include Debug sheets in Excel", value=True)

    run_btn = st.button("▶️ Run Backtest")

    if run_btn:
        data = load_prices(tickers, start_bt, end_bt)
        if not data:
            st.error("No data downloaded. Try adding exchange suffixes (.TO for Canada, .NS for India).")
            st.stop()

        st.subheader("🔎 Data Check")
        in_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        rows = []
        for t in in_list:
            d = data.get(t)
            if d is None or d.empty:
                hint = "" if "." in t else "Try .TO (e.g., XIC.TO) or .NS (e.g., TCS.NS)."
                rows.append({"Ticker": t, "Status": "NO DATA", "Rows": 0, "First": "", "Last": "", "Hint": hint})
            else:
                rows.append({"Ticker": t, "Status": "OK", "Rows": int(len(d)),
                             "First": str(d.index.min().date()), "Last": str(d.index.max().date()), "Hint": ""})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        results = []
        all_debug_rows = []
        all_dbg_summ = []

        subtabs = st.tabs(list(data.keys()))
        for tab, t in zip(subtabs, data.keys()):
            with tab:
                df = data[t]
                st.write(f"{t}: {len(df)} rows · {df.index.min().date()} → {df.index.max().date()}")

                equity, stats, dbg_df, dbg_sum = backtest(
                    df, strategy, params, vol_target, long_only, atr_stop, tp_mult,
                    trade_cost=trade_cost, tax_rate=tax_rate, debug=debug_mode,
                    fast_w=int(fast_w), slow_w=int(slow_w), eps=float(eps),
                    cap_calm=float(cap_calm), cap_neutral=float(cap_neutral), cap_storm=float(cap_storm)
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
                    f"**Entries/Exits/Flips:** {stats['Trades_Entered']}/{stats['Trades_Exited']}/{stats['Flips']}  |  "
                    f"**Total Costs:** {stats['TotalCosts(%)']:.3f}%"
                )

                if strategy in ("SMA Crossover", "SMA + Bollinger", "SMA + Stoch + Bollinger"):
                    close = df[price_col(df)]
                    fig, ax = plt.subplots(figsize=(9,4))
                    ax.plot(df.index, close, label="Close")
                    if strategy == "SMA Crossover":
                        ma_f = close.rolling(params["fast"]).mean()
                        ma_s = close.rolling(params["slow"]).mean()
                        bull = (ma_f.shift(1) <= ma_s.shift(1)) & (ma_f > ma_s)
                        bear = (ma_f.shift(1) >= ma_s.shift(1)) & (ma_f < ma_s)
                        ax.plot(df.index, ma_f, label=f"SMA {params['fast']}")
                        ax.plot(df.index, ma_s, label=f"SMA {params['slow']}")
                        ax.scatter(df.index[bull], close[bull], marker="^", s=60, label="Bullish Cross")
                        ax.scatter(df.index[bear], close[bear], marker="v", s=60, label="Bearish Cross")
                    else:
                        mid, up, lo = bollinger(close, lb=params["bb_lb"], mult=params["bb_mult"])
                        slow_line = close.rolling(params["sma_slow"]).mean()
                        ax.plot(df.index, mid, label="BB mid")
                        ax.plot(df.index, up,  label="BB upper")
                        ax.plot(df.index, lo,  label="BB lower")
                        ax.plot(df.index, slow_line, label=f"SMA {params['sma_slow']}")
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

        res_df = pd.DataFrame(results).set_index("Ticker")
        res_df["CAGR Δ (Strat − Raw)"] = (res_df["CAGR"] - res_df["RawCAGR"]).round(4)
        st.subheader("📋 Metrics Summary")
        st.dataframe(res_df, use_container_width=True)

        # Payoffs
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

        # Inputs for Excel
        def _strategy_name_and_params(strategy, params):
            if strategy == "SMA Crossover":
                return strategy, {"Fast SMA": params.get("fast"), "Slow SMA": params.get("slow")}
            elif strategy == "RSI Mean Reversion":
                return strategy, {"RSI lookback": params.get("rsi_lb"),
                                  "RSI Buy <": params.get("rsi_buy"),
                                  "RSI Sell >": params.get("rsi_sell")}
            elif strategy == "SMA + Bollinger":
                return strategy, {"Slow SMA": params.get("sma_slow"),
                                  "BB lookback": params.get("bb_lb"),
                                  "BB σ": params.get("bb_mult")}
            elif strategy == "SMA + Stoch + Bollinger":
                return strategy, {"Slow SMA": params.get("sma_slow"),
                                  "BB lookback": params.get("bb_lb"), "BB σ": params.get("bb_mult"),
                                  "%K": params.get("k_period"), "%D": params.get("d_period"),
                                  "Smooth": params.get("smooth"),
                                  "OS≤": params.get("os_level"), "OB≥": params.get("ob_level")}
            elif strategy == "Stochastic Only":
                return strategy, {"%K": params.get("k_period"), "%D": params.get("d_period"),
                                  "Smooth": params.get("smooth"),
                                  "OS≤": params.get("os_level"), "OB≥": params.get("ob_level")}
            else:
                return strategy, {}
        strat_name, strat_params = _strategy_name_and_params(strategy, params)

        inputs_rows = [("Tickers", ", ".join(sorted(data.keys()))),
                       ("Start", start_bt), ("End", end_bt), ("Strategy", strat_name)]
        for k, v in strat_params.items(): inputs_rows.append((k, v))
        inputs_rows += [("Long-only", long_only), ("Vol target (ann.)", vol_target),
                        ("ATR Stop (×)", atr_stop), ("Take Profit (× ATR)", tp_mult),
                        ("Cost per trade (%)", round(trade_cost*100,3)),
                        ("Effective tax on gains (%)", round(tax_rate*100,3)),
                        ("Lot size (sh)", int(lot_size)),
                        ("FastVol window", int(fast_w)), ("SlowVol window", int(slow_w)),
                        ("Hysteresis ε", float(eps)),
                        ("Cap Calm", float(cap_calm)), ("Cap Neutral", float(cap_neutral)), ("Cap Stormy", float(cap_storm))]
        inputs_df = pd.DataFrame(inputs_rows, columns=["Parameter","Value"])
        st.subheader("⚙️ Inputs")
        st.dataframe(inputs_df, use_container_width=True)

        # Excel export
        with io.BytesIO() as output:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                inputs_df.to_excel(writer, sheet_name="Inputs_&_Metrics", index=False, startrow=0)
                res_df.reset_index().to_excel(writer, sheet_name="Inputs_&_Metrics", index=False, startrow=len(inputs_df)+2)
                pay_df.reset_index().to_excel(writer, sheet_name="Simulated_Payoffs", index=False)
                if debug_mode:
                    if all_debug_rows:
                        debug_cat = pd.concat(all_debug_rows, axis=0, ignore_index=True)
                        debug_cat.to_excel(writer, sheet_name="Debug_Log", index=False)
                    if all_dbg_summ:
                        dbg_summary_df = pd.DataFrame(all_dbg_summ)
                        dbg_summary_df.to_excel(writer, sheet_name="Debug_Summary", index=False)
            data_xlsx = output.getvalue()

        st.download_button("⬇️ Download Backtest Report (Excel)",
                           data_xlsx, "Backtest_Report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")