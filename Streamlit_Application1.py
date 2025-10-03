# streamlit_app.py — Global Backtester with Yahoo/Stooq + Finnhub (search), Est/Actual Vol, Bollinger & SMA+BB, Capital logs
import os, io, time, requests
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(page_title="Srini Backtester (Global + Symbol Search)", layout="wide")

# ----------------------- OPTIONAL AUTH -----------------------
def _auth():
    pw = os.getenv("APP_PASSWORD", "")
    if not pw:
        return
    with st.sidebar:
        st.subheader("🔒 App Login")
        if st.text_input("Password", type="password", key="auth_pw") != pw:
            st.stop()
_auth()

# ----------------------- UTILS -----------------------
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

# ----------------------- INDICATORS -----------------------
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

def bollinger(price: pd.Series, window: int = 20, k: float = 2.0):
    mid = price.rolling(window).mean()
    sd  = price.rolling(window).std(ddof=0)
    up  = mid + k * sd
    lo  = mid - k * sd
    return mid, up, lo

# ----- Volatility measures: EstVol (past) and ActualVol (future) -----
def realized_vol(returns: pd.Series, lookback: int = 20, ppy: int = 252):
    return returns.rolling(lookback).std() * np.sqrt(ppy)

def future_realized_vol(returns: pd.Series, lookahead: int = 20, ppy: int = 252):
    return returns.rolling(lookahead).std().shift(-lookahead) * np.sqrt(ppy)

# ----------------------- SIGNALS -----------------------
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

def bollinger_mr_signals(price: pd.Series, window: int = 20, k: float = 2.0) -> pd.Series:
    mid, up, lo = bollinger(price, window, k)
    long_ent  = (price.shift(1) < lo.shift(1)) & (price >= lo)
    long_exit = (price.shift(1) > mid.shift(1)) & (price <= mid)
    short_ent = (price.shift(1) > up.shift(1)) & (price <= up)
    short_exit= (price.shift(1) < mid.shift(1)) & (price >= mid)
    sig = pd.Series(0.0, index=price.index); state = 0
    for i in range(1, len(price)):
        if state == 0:
            if long_ent.iloc[i]: state = 1
            elif short_ent.iloc[i]: state = -1
        elif state == 1 and long_exit.iloc[i]: state = 0
        elif state == -1 and short_exit.iloc[i]: state = 0
        sig.iloc[i] = state
    return sig.fillna(0.0)

def sma_with_bb_filter(price: pd.Series, fast: int, slow: int, bb_win: int, bb_k: float) -> pd.Series:
    """
    Keep SMA crossover signal but require:
      Long  only if price >= BB_Mid and price < BB_Upper
      Short only if price <= BB_Mid and price > BB_Lower
    """
    base = sma_signals(price, fast, slow)
    mid, up, lo = bollinger(price, bb_win, bb_k)
    filt = base.copy()
    long_mask  = base > 0
    short_mask = base < 0
    filt[long_mask & ~((price >= mid) & (price < up))] = 0.0
    filt[short_mask & ~((price <= mid) & (price > lo))] = 0.0
    return filt.fillna(0.0)

# ----------------------- SIZING -----------------------
def position_sizer(signal: pd.Series, returns: pd.Series, vol_target: float, ppy: int = 252) -> pd.Series:
    vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(ppy)
    vol.replace(0, np.nan, inplace=True)
    lev = (vol_target / (vol + 1e-12)).clip(upper=5.0).fillna(0.0)
    return signal * lev

# ----------------------- EXECUTION (stateful) -----------------------
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

# ----------------------- RAW CAGR -----------------------
def simple_cagr(df: pd.DataFrame) -> float:
    px = df[price_col(df)]; s, e = float(px.iloc[0]), float(px.iloc[-1])
    yrs = max((df.index[-1]-df.index[0]).days/365.25, 1e-9)
    return (e/max(s,1e-12))**(1/yrs) - 1

# ----------------------- DATA ADAPTERS (Finnhub) -----------------------
def finnhub_symbol_search(query: str, api_key: str, limit: int = 50) -> list[dict]:
    """Return a list of symbols with fields: symbol, description, exchange, type, currency."""
    if not api_key or not query.strip():
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/search",
            params={"q": query.strip(), "token": api_key},
            timeout=20,
        )
        r.raise_for_status()
        j = r.json() or {}
        res = j.get("result", [])[:limit]
        # Filter to common equity-like types & known exchanges
        cleaned = []
        for x in res:
            sym = x.get("symbol", "")
            desc = x.get("description", "")
            ex  = x.get("exchange", "")
            t   = x.get("type", "")
            cur = x.get("currency", "")
            if sym and desc:
                cleaned.append({
                    "symbol": sym,
                    "description": desc,
                    "exchange": ex,
                    "type": t,
                    "currency": cur
                })
        return cleaned
    except Exception:
        return []

def finnhub_download_daily(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, api_key: str):
    """Finnhub stock/candle -> OHLCV DF (1D) with Adj Close copied from Close."""
    if not api_key:
        return None
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": int(pd.Timestamp(start_ts).timestamp()),
        "to":   int(pd.Timestamp(end_ts).timestamp()),
        "adjusted": "true",
        "token": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if not j or j.get("s") != "ok":
            return None
        df = pd.DataFrame({
            "Date": pd.to_datetime(j["t"], unit="s"),
            "Open": j["o"],
            "High": j["h"],
            "Low":  j["l"],
            "Close":j["c"],
            "Volume": j["v"],
        }).set_index("Date").sort_index()
        df["Adj Close"] = df["Close"]
        return df
    except Exception:
        return None

# ----------------------- DATA LOADER (Yahoo/Stooq + Finnhub) -----------------------
@st.cache_data(show_spinner=False)
def load_prices(tickers_raw: str, start, end, provider: str = "Yahoo/Stooq (free)", finnhub_key: str | None = None) -> dict:
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
        candidates = [orig] if "." in orig else [orig, f"{orig}.TO", f"{orig}.V", f"{orig}.NS", f"{orig}.BO"]
        df_ok = None

        if provider == "Yahoo/Stooq (free)":
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

        elif provider == "Finnhub (free tier)":
            # Finnhub first
            for cand in candidates:
                df_ok = finnhub_download_daily(cand, start, end, finnhub_key or "")
                if df_ok is not None:
                    results[orig] = df_ok; break
                time.sleep(0.2)
            # Fallback to Yahoo/Stooq
            if df_ok is None:
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

# ----------------------- BACKTEST CORE -----------------------
def backtest(df: pd.DataFrame, strategy: str, params: dict,
             vol_target: float, long_only: bool, atr_stop: float, tp_mult: float,
             trade_cost: float=0.0, tax_rate: float=0.0,
             debug: bool=False, debug_lb: int=20):
    price = df[price_col(df)]
    rets  = price.pct_change().fillna(0.0)

    # signals
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
    elif strategy == "Bollinger Mean Reversion":
        sig = bollinger_mr_signals(price, params["bb_window"], params["bb_k"])
    elif strategy == "SMA + Bollinger Filter":
        sig = sma_with_bb_filter(price, params["fast"], params["slow"], params["bb_window"], params["bb_k"])
    else:
        raise ValueError("Unknown strategy")

    if long_only:
        sig = sig.clip(lower=0.0)

    # sizing (gross leverage)
    est_vol = realized_vol(rets, lookback=20)
    lev = (vol_target / (est_vol + 1e-12)).clip(upper=5.0)
    pos = (sig * lev).fillna(0.0)

    # execution
    atr = compute_atr(df, lb=14)
    pnl = apply_stops(df, pos, atr, atr_stop, tp_mult, trade_cost=trade_cost, tax_rate=tax_rate)
    equity = (1 + pnl).cumprod()

    # future vol
    act_vol = future_realized_vol(rets, lookahead=20)

    # leverage & invested capital logs
    gross_lev = pos.abs()
    invested_cap = equity.shift(1).fillna(equity.iloc[0]) * gross_lev
    min_cap = float(invested_cap.min()) if len(invested_cap) else 0.0
    max_cap = float(invested_cap.max()) if len(invested_cap) else 0.0

    stats = {
        "CAGR": round(annualized_return(pnl), 4),
        "Sharpe": round(sharpe(pnl), 2),
        "MaxDD": round(max_drawdown(equity)[0], 4),
        "Exposure": round(float((pnl != 0).sum())/max(len(pnl),1), 3),
        "LastEquity": round(float(equity.iloc[-1]) if len(equity) else 1.0, 4),
        "MinCapital": round(min_cap, 4),
        "MaxCapital": round(max_cap, 4),
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
            "GrossLeverage": gross_lev,
            "InvestedCapital": invested_cap,
            "PreExecPosition": (sig * lev).shift(1),
            "Equity": equity
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
            "MinCapital": round(min_cap, 4),
            "MaxCapital": round(max_cap, 4),
        }]

    return equity, stats, debug_df, debug_summary

# ----------------------- UI -----------------------
st.title("📊 Srini Backtester — Global (US/Canada/India) + Symbol Search")

with st.sidebar:
    st.header("Backtest Settings")
    # Data provider & API key
    st.subheader("Data Provider")
    data_provider = st.selectbox(
        "Provider",
        ["Yahoo/Stooq (free)", "Finnhub (free tier)"],
        index=0
    )
    finnhub_key = None
    if data_provider == "Finnhub (free tier)":
        finnhub_key = st.text_input("FINNHUB_API_KEY", value=os.getenv("FINNHUB_API_KEY",""), type="password")

    # Symbol search (Finnhub)
    st.subheader("Symbol Search (Finnhub)")
    search_q = st.text_input("Search company/ticker (e.g., TCS, Reliance, SHOP, XIC)", "")
    search_btn = st.button("🔍 Search (Finnhub)")
    search_results = []
    if search_btn:
        if not finnhub_key:
            st.warning("Enter FINNHUB_API_KEY to search.")
        else:
            search_results = finnhub_symbol_search(search_q, finnhub_key, limit=50)
            if not search_results:
                st.info("No results (or rate-limited). Try a different query/symbol.")
    if search_results:
        # render results and let user add one to the tickers string
        opts = [f"{r['symbol']} — {r['description']} ({r.get('exchange','')}, {r.get('currency','')})" for r in search_results]
        pick = st.selectbox("Select symbol to add", ["--"] + opts, index=0)
        if pick != "--":
            pick_idx = opts.index(pick)
            picked_symbol = search_results[pick_idx]["symbol"]
            st.session_state.setdefault("picked_symbols", set())
            if picked_symbol not in st.session_state["picked_symbols"]:
                st.session_state["picked_symbols"].add(picked_symbol)
                st.success(f"Added {picked_symbol} to your session picks.")
    picks_text = ", ".join(sorted(list(st.session_state.get("picked_symbols", set())))) if "picked_symbols" in st.session_state else ""

    # Base tickers
    default_tickers = "SPY, XLK, ACN, XIC.TO"
    tickers = st.text_input("Tickers (comma-separated)", value=default_tickers)
    if picks_text:
        st.caption(f"Picked from search: {picks_text}")
        # allow one-click append
        if st.button("➕ Append picked to tickers"):
            combined = [t.strip() for t in (tickers + "," + picks_text).split(",") if t.strip()]
            tickers = ", ".join(dict.fromkeys(combined))  # de-dup preserve order
            st.success("Appended to tickers input. You can edit it above.")

    start = st.date_input("Start", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
    end   = st.date_input("End",   value=pd.Timestamp.today()).strftime("%Y-%m-%d")

    strategy = st.selectbox(
        "Strategy",
        ["SMA Crossover", "RSI Mean Reversion", "Composite", "Bollinger Mean Reversion", "SMA + Bollinger Filter"],
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

    elif strategy == "Bollinger Mean Reversion":
        c1, c2 = st.columns(2)
        bb_window = c1.number_input("BB Window", 5, 200, 20, 1)
        bb_k      = c2.number_input("BB Std Dev (k)", 1.0, 4.0, 2.0, 0.5)
        params = {"bb_window": int(bb_window), "bb_k": float(bb_k)}

    else:  # SMA + BB filter
        c1, c2, c3 = st.columns(3)
        fast = c1.number_input("Fast SMA", 2, 200, 20, 1)
        slow = c2.number_input("Slow SMA", 5, 400, 100, 5)
        bb_window = c3.number_input("BB Window", 5, 200, 20, 1)
        bb_k      = st.number_input("BB Std Dev (k)", 1.0, 4.0, 2.0, 0.5)
        params = {"fast": int(fast), "slow": int(slow), "bb_window": int(bb_window), "bb_k": float(bb_k)}

    long_only  = st.checkbox("Long-only", value=True)
    vol_target = st.slider("Vol target (ann.)", 0.05, 0.40, 0.15, 0.01)
    atr_stop   = st.slider("ATR Stop (×)", 1.0, 8.0, 3.0, 0.5)
    tp_mult    = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5)

    st.subheader("Real-world frictions")
    trade_cost = st.number_input("Cost per trade (%)", 0.0, 0.50, 0.05, 0.01) / 100.0
    tax_rate   = st.number_input("Effective tax on gains (%)", 0.0, 50.0, 0.0, 1.0) / 100.0

    lot_size = st.number_input("Lot size for payoff (sh/ticker)", 1, 100000, 100)

    st.subheader("Debug / Export")
    debug_mode = st.checkbox("Run Debug Mode (log vols, leverage, capital)", value=True)
    debug_lookback = st.number_input("Debug vol lookback (days)", 5, 100, 20, 1)
    include_debug_in_excel = st.checkbox("Include Debug sheets in Excel", value=True)

    run_btn = st.button("Run Backtest")

# ----------------------- RUN -----------------------
if run_btn:
    data = load_prices(tickers, start, end, provider=data_provider, finnhub_key=finnhub_key)
    if not data:
        st.error("No data downloaded. Try adding exchange suffixes (.TO for Canada, .NS/.BO for India) or switch provider.")
        st.stop()

    st.subheader("🔎 Data Check")
    in_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    rows = []
    for t in in_list:
        d = data.get(t)
        if d is None or d.empty:
            hint = "" if "." in t else "Try .TO (Canada), .NS or .BO (India)."
            rows.append({"Ticker": t, "Status": "NO DATA", "Rows": 0, "First": "", "Last": "", "Hint": hint})
        else:
            rows.append({"Ticker": t, "Status": "OK", "Rows": int(len(d)),
                         "First": str(d.index.min().date()), "Last": str(d.index.max().date()), "Hint": ""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Per-ticker tabs & results
    results = []
    all_debug_rows = []
    all_dbg_summ = []

    tabs = st.tabs(list(data.keys()))
    for tab, t in zip(tabs, data.keys()):
        with tab:
            df = data[t]
            st.write(f"{t}: {len(df)} rows · {df.index.min().date()} → {df.index.max().date()}")

            equity, stats, dbg_df, dbg_sum = backtest(
                df, strategy, params, vol_target, long_only, atr_stop, tp_mult,
                trade_cost=trade_cost, tax_rate=tax_rate,
                debug=debug_mode, debug_lb=int(debug_lookback)
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
                f"**Min Capital:** {stats['MinCapital']:.2f}×  |  "
                f"**Max Capital:** {stats['MaxCapital']:.2f}×  |  "
                f"**Entries/Exits/Flips:** {stats['Trades_Entered']}/{stats['Trades_Exited']}/{stats['Flips']}  |  "
                f"**Total Costs:** {stats['TotalCosts(%)']:.3f}%"
            )

            # Optional visuals per strategy
            if strategy == "SMA Crossover":
                close = df[price_col(df)]
                ma_f = close.rolling(params["fast"]).mean()
                ma_s = close.rolling(params["slow"]).mean()
                bull = (ma_f.shift(1) <= ma_s.shift(1)) & (ma_f > ma_s)
                bear = (ma_f.shift(1) >= ma_s.shift(1)) & (ma_f < ma_s)
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(df.index, close, label="Close")
                ax.plot(df.index, ma_f, label=f"SMA {params['fast']}")
                ax.plot(df.index, ma_s, label=f"SMA {params['slow']}")
                ax.scatter(df.index[bull], close[bull], marker="^", s=60, label="Bullish Cross")
                ax.scatter(df.index[bear], close[bear], marker="v", s=60, label="Bearish Cross")
                ax.legend(loc="best"); ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)

            if strategy in ("Bollinger Mean Reversion", "SMA + Bollinger Filter"):
                close = df[price_col(df)]
                bbw = params.get("bb_window", 20); bbk = params.get("bb_k", 2.0)
                mid, up, lo = bollinger(close, bbw, bbk)
                if strategy == "Bollinger Mean Reversion":
                    sig = bollinger_mr_signals(close, bbw, bbk)
                else:
                    sig = sma_with_bb_filter(close, params["fast"], params["slow"], bbw, bbk)
                diff = sig.diff().fillna(0)
                long_ent_idx  = diff[diff == 1].index
                long_exit_idx = diff[diff == -1].index
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(df.index, close, label="Close")
                ax.plot(df.index, mid,   label=f"BB Mid ({bbw})")
                ax.plot(df.index, up,    label=f"BB Upper (k={bbk})")
                ax.plot(df.index, lo,    label=f"BB Lower (k={bbk})")
                ax.scatter(long_ent_idx,  close.reindex(long_ent_idx),  marker="^", s=60, label="Long Entry")
                ax.scatter(long_exit_idx, close.reindex(long_exit_idx), marker="v", s=60, label="Long Exit")
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

    # Metrics Summary
    res_df = pd.DataFrame(results).set_index("Ticker")
    res_df["CAGR Δ (Strat − Raw)"] = (res_df["CAGR"] - res_df["RawCAGR"]).round(4)
    st.subheader("📋 Metrics Summary")
    cols_order = ["RawCAGR","CAGR","CAGR Δ (Strat − Raw)","Sharpe","MaxDD","Exposure","LastEquity","MinCapital","MaxCapital","Trades_Entered","Trades_Exited","Flips","TotalCosts(%)"]
    st.dataframe(res_df[cols_order], use_container_width=True)

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
        min_inv_dollars = init_cap * float(res_df.loc[t, "MinCapital"])
        max_inv_dollars = init_cap * float(res_df.loc[t, "MaxCapital"])
        pay_rows.append({
            "Ticker": t,
            "StartPx": round(start_px,4), "EndPx": round(end_px,4),
            "LotSize(sh)": int(lot_size), "Initial($)": round(init_cap,2),
            "Ending_Strategy($)": round(end_cap_strategy,2), "P&L_Strategy($)": round(pnl_strategy,2),
            "Ending_BuyHold($)": round(end_cap_bh,2), "P&L_BuyHold($)": round(pnl_bh,2),
            "MinInvested($)": round(min_inv_dollars,2), "MaxInvested($)": round(max_inv_dollars,2),
            "Δ P&L (Strat − B&H)($)": round(pnl_strategy - pnl_bh,2),
        })
    pay_df = pd.DataFrame(pay_rows).set_index("Ticker")
    if not pay_df.empty:
        totals = pay_df.select_dtypes(include=[float, int]).sum(numeric_only=True)
        totals_df = pd.DataFrame(totals).T; totals_df.index = ["TOTAL"]
        pay_df = pd.concat([pay_df, totals_df], axis=0)
    st.subheader("💵 Simulated Payoffs (incl. Min/Max Invested $)")
    st.dataframe(pay_df, use_container_width=True)

    # Inputs (for Excel)
    def _strategy_name_and_params(strategy, params):
        if strategy == "SMA Crossover":
            return strategy, {"Fast SMA": params.get("fast"), "Slow SMA": params.get("slow")}
        elif strategy == "RSI Mean Reversion":
            return strategy, {"RSI lookback": params.get("rsi_lb"),
                              "RSI Buy <": params.get("rsi_buy"),
                              "RSI Sell >": params.get("rsi_sell")}
        elif strategy == "Bollinger Mean Reversion":
            return strategy, {"BB Window": params.get("bb_window"), "BB k": params.get("bb_k")}
        elif strategy == "SMA + Bollinger Filter":
            return strategy, {"Fast SMA": params.get("fast"), "Slow SMA": params.get("slow"),
                              "BB Window": params.get("bb_window"), "BB k": params.get("bb_k")}
        else:
            return strategy, {
                "Fast SMA": params.get("fast"), "Slow SMA": params.get("slow"),
                "RSI lookback": params.get("rsi_lb"), "RSI Buy <": params.get("rsi_buy"),
                "RSI Sell >": params.get("rsi_sell"), "Use MACD": params.get("use_macd"),
                "Combine": "AND" if params.get("use_and", False) else "Voting (≥2)",
                "ATR hot filter": params.get("atr_filter", False),
                "Max ATR/Price": params.get("atr_cap_pct"),
            }
    strat_name, strat_params = _strategy_name_and_params(strategy, params)

    inputs_rows = [("Provider", data_provider)]
    if data_provider == "Finnhub (free tier)":
        inputs_rows.append(("FINNHUB_API_KEY present", bool(finnhub_key)))
    inputs_rows += [("Tickers", ", ".join(sorted(data.keys()))),
                   ("Start", start), ("End", end), ("Strategy", strat_name)]
    for k, v in strat_params.items(): inputs_rows.append((k, v))
    inputs_rows += [("Long-only", long_only), ("Vol target (ann.)", vol_target),
                    ("ATR Stop (×)", atr_stop), ("Take Profit (× ATR)", tp_mult),
                    ("Cost per trade (%)", round(trade_cost*100,3)),
                    ("Effective tax on gains (%)", round(tax_rate*100,3)),
                    ("Lot size (sh)", int(lot_size))]
    inputs_df = pd.DataFrame(inputs_rows, columns=["Parameter","Value"])
    st.subheader("⚙️ Inputs")
    st.dataframe(inputs_df, use_container_width=True)

    # Excel Export
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

    st.download_button("⬇️ Download Backtest Report (Excel)", data_xlsx, "Backtest_Report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")