# streamlit_app.py — Backtester (US/CA) with Debug Mode + Excel export
import os, io, time
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import streamlit as st
import matplotlib.pyplot as plt

# ============== PAGE CONFIG (must be first Streamlit call) ==============
st.set_page_config(page_title="Srini Backtester (with Debug Mode)", layout="wide")

# ============== AUTH (optional) =====================
def _auth():
    pw = os.getenv("APP_PASSWORD", "")
    if not pw:
        return
    with st.sidebar:
        st.subheader("🔒 App Login")
        if st.text_input("Password", type="password", key="auth_pw") != pw:
            st.stop()
_auth()

# ============== UTILS ==============================
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

# ============== INDICATORS =========================
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

def compute_realized_vol_and_lev(returns: pd.Series, vol_target: float, lookback: int = 20, ppy: int = 252):
    """Rolling realized vol (stdev * sqrt(252)) and implied leverage = target / realized."""
    realized = returns.rolling(lookback).std() * np.sqrt(ppy)
    lev = (vol_target / (realized + 1e-12)).clip(lower=0.0, upper=5.0)
    return realized, lev

# ============== SIGNALS ============================
def sma_signals(price: pd.Series, fast: int, slow: int) -> pd.Series:
    ma_f, ma_s = price.rolling(fast).mean(), price.rolling(slow).mean()
    sig = pd.Series(0.0, index=price.index)
    sig[ma_f > ma_s] = 1.0; sig[ma_f < ma_s] = -1.0
    return sig.fillna(0.0)

def rsi_signals(price: pd.Series, rsi_lb: int, rsi_buy: int, rsi_sell: int) -> pd.Series:
    r = rsi(price, lb=rsi_lb); sig = pd.Series(0.0, index=price.index)
    sig[r < rsi_buy] = 1.0; sig[r > rsi_sell] = -1.0
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

# ============== SIZING =============================
def position_sizer(signal: pd.Series, returns: pd.Series, vol_target: float, ppy: int = 252) -> pd.Series:
    vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(ppy)
    vol.replace(0, np.nan, inplace=True)
    lev = (vol_target / (vol + 1e-12)).clip(upper=5.0).fillna(0.0)
    return signal * lev

# ============== EXECUTION (stateful, stops/TP) =====
def apply_stops(df: pd.DataFrame, pos: pd.Series, atr: pd.Series,
                atr_stop_mult: float, tp_mult: float,
                trade_cost: float = 0.0, tax_rate: float = 0.0) -> pd.Series:
    """
    Stateful execution engine:
      - Separate active position sign from raw signal to avoid instant re-entry after stop/TP.
      - Re-entry blocked until signal passes through 0.
      - Flip (long->short) counts as exit+entry (2 costs).
    """
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

# ============== RAW CAGR (buy & hold) ===============
def simple_cagr(df: pd.DataFrame) -> float:
    px = df[price_col(df)]; s, e = float(px.iloc[0]), float(px.iloc[-1])
    yrs = max((df.index[-1]-df.index[0]).days/365.25, 1e-9)
    return (e/max(s,1e-12))**(1/yrs) - 1

# ============== BACKTEST CORE (returns debug optionally) ===
def backtest(df: pd.DataFrame, strategy: str, params: dict,
             vol_target: float, long_only: bool, atr_stop: float, tp_mult: float,
             trade_cost: float=0.0, tax_rate: float=0.0,
             debug: bool=False, debug_lb: int=20):
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

    # Production sizer
    pos = position_sizer(sig, rets, vol_target)

    # Debug realized vol / leverage
    dbg_vol, dbg_lev = compute_realized_vol_and_lev(rets, vol_target, lookback=debug_lb)
    dbg_pos = (sig * dbg_lev).shift(1)

    # Execute with stops/TP
    atr = compute_atr(df, lb=14)
    pnl = apply_stops(df, pos, atr, atr_stop, tp_mult, trade_cost=trade_cost, tax_rate=tax_rate)
    equity = (1 + pnl).cumprod()

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
            "RealizedVol": dbg_vol,
            "Leverage": dbg_lev,
            "Signal": sig,
            "PreExecPosition": dbg_pos,
        }).dropna().copy()
        levered_days  = int((debug_df["Leverage"] > 1.0).sum())
        derisk_days   = int((debug_df["Leverage"] < 1.0).sum())
        neutral_days  = int((debug_df["PreExecPosition"] == 0).sum())
        avg_lev       = float(debug_df["Leverage"].mean())
        max_lev       = float(debug_df["Leverage"].max())
        debug_summary = {
            "LeveredDays(>1x)": levered_days,
            "DeriskDays(<1x)": derisk_days,
            "NeutralDays(Pos=0)": neutral_days,
            "AvgLeverage": round(avg_lev, 3),
            "MaxLeverage": round(max_lev, 3),
            "Rows": int(len(debug_df)),
        }

    return equity, stats, debug_df, debug_summary

# ============== DATA LOADER (Yahoo + .TO/.V retry + Stooq fallback) ===
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
            time.sleep(0.3)
        if df_ok is None:
            df_ok = try_stooq(orig)
            if df_ok is not None:
                results[orig] = df_ok
            time.sleep(0.3)

    cleaned = {}
    for t, d in results.items():
        if d is None or d.empty: continue
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in d.columns]
        d = d[keep].sort_index().dropna(how="all")
        if not d.empty: cleaned[t] = d
    return cleaned

# ============== UI =================================
st.title("📊 Srini Backtester — SMA / RSI / Composite + Debug Mode")
st.caption("Vol targeting, stateful ATR stops/TP, costs/tax, multi-ticker, metrics, payoffs, and optional Debug sheets in the Excel export.")

with st.sidebar:
    st.header("Backtest Settings")
    tickers = st.text_input("Tickers (comma-separated)", value="SPY, XLK, ACN, XIC.TO")
    start = st.date_input("Start", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
    end   = st.date_input("End",   value=pd.Timestamp.today()).strftime("%Y-%m-%d")

    strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion", "Composite"], index=2)

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
    else:
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

    long_only  = st.checkbox("Long-only", value=True)
    vol_target = st.slider("Vol target (ann.)", 0.05, 0.40, 0.15, 0.01)
    atr_stop   = st.slider("ATR Stop (×)", 1.0, 8.0, 3.0, 0.5)
    tp_mult    = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5)

    st.subheader("Real-world frictions")
    trade_cost = st.number_input("Cost per trade (%)", 0.0, 0.50, 0.05, 0.01) / 100.0
    tax_rate   = st.number_input("Effective tax on gains (%)", 0.0, 50.0, 0.0, 1.0) / 100.0

    lot_size = st.number_input("Lot size for payoff (sh/ticker)", 1, 100000, 100)

    st.subheader("Debug / Export")
    debug_mode = st.checkbox("Run Debug Mode (log vol/leverage/position)", value=False)
    debug_lookback = st.number_input("Debug vol lookback (days)", 5, 100, 20, 1)
    include_debug_in_excel = st.checkbox("Include Debug sheets in Excel", value=True)

    run_btn = st.button("Run Backtest")

with st.expander("🧪 Diagnostics", expanded=False):
    st.write("yfinance:", getattr(yf, "__version__", "unknown"))
    if st.button("Clear caches"):
        load_prices.clear(); st.success("Caches cleared. Rerun.")

# ============== RUN =================================
if run_btn:
    data = load_prices(tickers, start, end)
    if not data:
        st.error("No data downloaded. Try adding exchange suffixes (.TO for Canada).")
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

            # Plot equity
            st.subheader(f"{t} — Equity Curve")
            st.line_chart(equity, height=300)

            # Metrics line
            raw_cagr = simple_cagr(df)
            st.markdown(
                f"**Buy & Hold CAGR:** {raw_cagr:.2%}  |  "
                f"**Strategy CAGR:** {stats['CAGR']:.2%}  |  "
                f"**Sharpe:** {stats['Sharpe']:.2f}  |  "
                f"**MaxDD:** {stats['MaxDD']:.2%}  |  "
                f"**Exposure (time in mkt):** {stats['Exposure']:.0%}  |  "
                f"**Entries/Exits/Flips:** {stats['Trades_Entered']}/{stats['Trades_Exited']}/{stats['Flips']}  |  "
                f"**Total Costs:** {stats['TotalCosts(%)']:.3f}%"
            )

            # Optional SMA cross markers
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

            results.append({"Ticker": t, "RawCAGR": round(raw_cagr,4), **stats})

            # Collect debug
            if debug_mode and dbg_df is not None and not dbg_df.empty:
                dcopy = dbg_df.copy()
                dcopy.insert(0, "Ticker", t)
                all_debug_rows.append(dcopy)
                if dbg_sum:
                    all_dbg_summ.append({"Ticker": t, **dbg_sum})

    # Results table
    res_df = pd.DataFrame(results).set_index("Ticker")
    res_df["CAGR Δ (Strategy - Raw)"] = (res_df["CAGR"] - res_df["RawCAGR"]).round(4)
    st.subheader("📋 Metrics Summary")
    st.dataframe(res_df, use_container_width=True)

    # Simulated Payoffs (per ticker + total)
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

    # Inputs table
    def _strategy_name_and_params(strategy, params):
        if strategy == "SMA Crossover":
            return strategy, {"Fast SMA": params.get("fast"), "Slow SMA": params.get("slow")}
        elif strategy == "RSI Mean Reversion":
            return strategy, {"RSI lookback": params.get("rsi_lb"),
                              "RSI Buy <": params.get("rsi_buy"),
                              "RSI Sell >": params.get("rsi_sell")}
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

    inputs_rows = [("Tickers", ", ".join(sorted(data.keys()))),
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

    # ========== Excel Export (adds Debug sheets if enabled) ==========
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Tab 1: Inputs + Metrics
            inputs_df.to_excel(writer, sheet_name="Inputs_&_Metrics", index=False, startrow=0)
            res_df.reset_index().to_excel(writer, sheet_name="Inputs_&_Metrics", index=False, startrow=len(inputs_df)+2)
            # Tab 2: Payoffs
            pay_df.reset_index().to_excel(writer, sheet_name="Simulated_Payoffs", index=False)

            # Debug tabs (optional)
            if include_debug_in_excel and debug_mode:
                if all_debug_rows:
                    debug_cat = pd.concat(all_debug_rows, axis=0, ignore_index=True)
                    debug_cat.to_excel(writer, sheet_name="Debug_Log", index=False)
                if all_dbg_summ:
                    dbg_summary_df = pd.DataFrame(all_dbg_summ)
                    dbg_summary_df.to_excel(writer, sheet_name="Debug_Summary", index=False)

        data_xlsx = output.getvalue()

    st.download_button("⬇️ Download Backtest Report (Excel, 2–4 tabs)",
                       data_xlsx, "Backtest_Report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")