# US Backtester (Srini) — SMA / RSI / COMPOSITE + ATR stops, Raw vs Strategy, costs/tax
# IMPORTANT: Must be the first Streamlit call
import os, time, numpy as np, pandas as pd, yfinance as yf
from pandas_datareader import data as pdr
import streamlit as st, matplotlib.pyplot as plt

st.set_page_config(page_title="US Backtester (Srini)", layout="wide")

# -------- Auth (env APP_PASSWORD) --------
def _auth():
    pw_env = os.getenv("APP_PASSWORD", "")
    if not pw_env:
        return
    with st.sidebar:
        st.subheader("🔒 App Login")
        pw = st.text_input("Password", type="password", key="auth_password")
        if pw != pw_env:
            st.stop()
_auth()

# --------- Helpers / Indicators ----------
def price_col(df): return "Adj Close" if "Adj Close" in df.columns else "Close"
def _to_ts(d):     return pd.to_datetime(d).tz_localize(None)

def rsi(series: pd.Series, lb: int = 14) -> pd.Series:
    d = series.diff()
    up, down = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/lb, adjust=False).mean()
    rd = down.ewm(alpha=1/lb, adjust=False).mean()
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
    high, low, close = df["High"], df[price_col(df)], df[price_col(df)]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/lb, adjust=False).mean()

def annualized_return(returns: pd.Series, ppy: int = 252) -> float:
    if returns.empty: return 0.0
    total = float((1 + returns).prod())
    years = len(returns) / ppy
    return total ** (1 / max(years, 1e-9)) - 1

def sharpe(returns: pd.Series, rf: float = 0.0, ppy: int = 252) -> float:
    if returns.std() == 0 or returns.empty: return 0.0
    excess = returns - rf / ppy
    return float(np.sqrt(ppy) * excess.mean() / (excess.std() + 1e-12))

def max_drawdown(equity: pd.Series):
    if equity.empty: return 0.0, None, None
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    trough = dd.idxmin()
    peak = roll_max.loc[:trough].idxmax()
    return float(dd.min()), peak, trough

# --------- Signals ----------
def sma_signals(price: pd.Series, fast: int, slow: int) -> pd.Series:
    ma_f, ma_s = price.rolling(fast).mean(), price.rolling(slow).mean()
    sig = pd.Series(0.0, index=price.index)
    sig[ma_f > ma_s] = 1.0
    sig[ma_f < ma_s] = -1.0
    return sig.fillna(0.0)

def rsi_signals(price: pd.Series, rsi_lb: int, rsi_buy: int, rsi_sell: int) -> pd.Series:
    r = rsi(price, lb=rsi_lb)
    sig = pd.Series(0.0, index=price.index)
    sig[r < rsi_buy] = 1.0
    sig[r > rsi_sell] = -1.0
    return sig.fillna(0.0)

def composite_signal(
    price: pd.Series,
    rsi_lb=14, rsi_buy=35, rsi_sell=65,
    sma_fast=20, sma_slow=100,
    use_and=True,
    use_macd=True,
    atr_filter=False,
    atr_series: pd.Series | None = None,
    atr_cap_pct=0.05,
):
    # 1) Trend (SMA)
    ma_f = price.rolling(sma_fast).mean()
    ma_s = price.rolling(sma_slow).mean()
    sma_sig = pd.Series(0.0, index=price.index)
    sma_sig[ma_f > ma_s] = 1.0
    sma_sig[ma_f < ma_s] = -1.0

    # 2) RSI tilt
    r = rsi(price, lb=rsi_lb)
    rsi_sig = pd.Series(0.0, index=price.index)
    rsi_sig[r < rsi_buy] = 1.0
    rsi_sig[r > rsi_sell] = -1.0

    # 3) MACD (optional)
    if use_macd:
        _, _, h = macd(price)
        macd_sig = pd.Series(0.0, index=price.index)
        macd_sig[h > 0] = 1.0
        macd_sig[h < 0] = -1.0
    else:
        macd_sig = pd.Series(0.0, index=price.index)

    # Combine
    if use_and:
        comp = pd.Series(0.0, index=price.index)
        # require alignment of signs (non-negative for long regime / non-positive for short)
        agree_long  = ((sma_sig == 1) & (rsi_sig >= 0) & ((macd_sig >= 0) | (~use_macd)))
        agree_short = ((sma_sig == -1) & (rsi_sig <= 0) & ((macd_sig <= 0) | (~use_macd)))
        comp[agree_long]  = 1.0
        comp[agree_short] = -1.0
    else:
        # voting: need at least 2 agreements
        score = sma_sig + rsi_sig + macd_sig
        comp = pd.Series(0.0, index=price.index)
        comp[score >= 2]  = 1.0
        comp[score <= -2] = -1.0

    # ATR hot-vol filter
    if atr_filter and atr_series is not None:
        hot = (atr_series / price).fillna(0) > atr_cap_pct
        comp[hot] = 0.0

    return comp.fillna(0.0)

# --------- Sizing / Stops / P&L ----------
def position_sizer(signal: pd.Series, returns: pd.Series, vol_target: float, ppy: int = 252) -> pd.Series:
    vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(ppy)
    vol.replace(0, np.nan, inplace=True)
    lev = (vol_target / (vol + 1e-12)).clip(upper=5.0).fillna(0.0)
    return signal * lev

def apply_stops(df: pd.DataFrame, pos: pd.Series, atr: pd.Series,
                atr_stop_mult: float, tp_mult: float,
                trade_cost: float = 0.0, tax_rate: float = 0.0) -> pd.Series:
    c = df[price_col(df)]
    ret = c.pct_change().fillna(0.0)
    pnl = pd.Series(0.0, index=c.index)
    current_pos, entry = 0.0, np.nan
    last_pos = 0.0

    for i in range(len(c)):
        s, px = float(pos.iloc[i]), float(c.iloc[i])
        a = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else np.nan

        # detect trade/change of position (entry/exit/flip)
        signal_change = (np.sign(s) != np.sign(last_pos))
        last_pos = s

        if i == 0 or signal_change:
            entry = px
            # cost on any entry/exit (one side); treat flip as exit+entry
            pnl.iloc[i] -= trade_cost

        current_pos = s

        if current_pos == 0 or np.isnan(a):
            # flat day
            continue

        # active position: compute stops/TP
        if current_pos > 0:
            stop = entry * (1 - atr_stop_mult * a / max(entry, 1e-12))
            tp   = entry * (1 + tp_mult     * a / max(entry, 1e-12))
            if px <= stop or px >= tp:
                # realized exit on this bar -> apply cost & (simplified) tax on gains
                # approximate realized P&L magnitude when TP hits; at stop we assume 0 extra here
                realized = (tp/entry - 1.0) if px >= tp else 0.0
                if realized > 0:
                    pnl.iloc[i] += current_pos * realized
                    pnl.iloc[i] *= (1 - tax_rate)
                pnl.iloc[i] -= trade_cost
                current_pos = 0.0
            else:
                pnl.iloc[i] += current_pos * ret.iloc[i]
        else:
            stop = entry * (1 + atr_stop_mult * a / max(entry, 1e-12))
            tp   = entry * (1 - tp_mult     * a / max(entry, 1e-12))
            if px >= stop or px <= tp:
                realized = (entry/tp - 1.0) if px <= tp else 0.0
                if realized > 0:
                    pnl.iloc[i] += current_pos * realized
                    pnl.iloc[i] *= (1 - tax_rate)
                pnl.iloc[i] -= trade_cost
                current_pos = 0.0
            else:
                pnl.iloc[i] += current_pos * ret.iloc[i]
    return pnl

# --------- Backtest core ----------
def backtest(df: pd.DataFrame, strategy: str, params: dict,
             vol_target: float, long_only: bool, atr_stop: float, tp_mult: float,
             trade_cost: float = 0.0, tax_rate: float = 0.0):
    price = df[price_col(df)]
    rets = price.pct_change().fillna(0.0)

    if strategy == "SMA Crossover":
        sig = sma_signals(price, params["fast"], params["slow"])
    elif strategy == "RSI Mean Reversion":
        sig = rsi_signals(price, params["rsi_lb"], params["rsi_buy"], params["rsi_sell"])
    elif strategy == "Composite":
        atr = compute_atr(df, lb=params.get("atr_lb", 14))
        sig = composite_signal(
            price,
            rsi_lb=params.get("rsi_lb",14),
            rsi_buy=params.get("rsi_buy",35),
            rsi_sell=params.get("rsi_sell",65),
            sma_fast=params.get("fast",20),
            sma_slow=params.get("slow",100),
            use_and=params.get("use_and", True),
            use_macd=params.get("use_macd", True),
            atr_filter=params.get("atr_filter", False),
            atr_series=atr,
            atr_cap_pct=params.get("atr_cap_pct", 0.05),
        )
    else:
        raise ValueError("Unknown strategy")

    if long_only:
        sig = sig.clip(lower=0.0)

    pos = position_sizer(sig, rets, vol_target)
    atr = compute_atr(df, lb=14)
    pnl = apply_stops(df, pos, atr, atr_stop, tp_mult, trade_cost=trade_cost, tax_rate=tax_rate)
    equity = (1 + pnl).cumprod()

    stats = {
        "CAGR": round(annualized_return(pnl), 4),
        "Sharpe": round(sharpe(pnl), 2),
        "MaxDD": round(max_drawdown(equity)[0], 4),
        "Exposure": round(float((pnl != 0).sum()) / max(len(pnl), 1), 3),
        "LastEquity": round(float(equity.iloc[-1]) if not equity.empty else 1.0, 4),
    }
    return equity, stats

def simple_cagr(df: pd.DataFrame) -> float:
    px = df[price_col(df)]
    start_price, end_price = float(px.iloc[0]), float(px.iloc[-1])
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    return (end_price / max(start_price, 1e-12)) ** (1 / years) - 1

def evaluate_ticker(df: pd.DataFrame, ticker: str, strategy: str, params: dict,
                    vol_target: float, long_only: bool, atr_stop: float, tp_mult: float,
                    trade_cost: float = 0.0, tax_rate: float = 0.0):
    equity, stats = backtest(df, strategy, params, vol_target, long_only, atr_stop, tp_mult,
                             trade_cost=trade_cost, tax_rate=tax_rate)
    raw = simple_cagr(df)
    stats_out = {
        "RawCAGR": round(raw, 4),
        "StrategyCAGR": stats["CAGR"],
        "Sharpe": stats["Sharpe"],
        "MaxDD": stats["MaxDD"],
        "Exposure": stats["Exposure"],
        "LastEquity": stats["LastEquity"],
    }
    return equity, stats_out

# --------- Data loading (yfinance + Stooq fallback) ----------
@st.cache_data(show_spinner=False)
def load_prices(tickers_raw: str, start, end) -> dict:
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers: return {}
    start = _to_ts(start); end = _to_ts(end); end_inc = end + pd.Timedelta(days=1)
    results = {}

    # batch
    try:
        df = yf.download(tickers=tickers, start=start, end=end_inc, interval="1d",
                         auto_adjust=False, progress=False, group_by="ticker",
                         threads=False, timeout=60)
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            for t in tickers:
                if t in lvl0:
                    sub = df[t].dropna(how="all").copy()
                    if not sub.empty: results[t] = sub
        else:
            if not df.empty and len(tickers) == 1:
                results[tickers[0]] = df.dropna(how="all").copy()
    except Exception:
        pass

    # per-ticker retry
    remaining = [t for t in tickers if t not in results]
    for t in remaining:
        try:
            dft = yf.download(t, start=start, end=end_inc, interval="1d",
                              auto_adjust=False, progress=False, threads=False, timeout=60
                             ).dropna(how="all")
            if not dft.empty: results[t] = dft
        except Exception:
            pass
        time.sleep(0.4)

    # Stooq fallback
    still = [t for t in tickers if t not in results]
    for t in still:
        try:
            dft = pdr.DataReader(t, "stooq", start=start, end=end_inc)
            if dft is not None and not dft.empty:
                dft = dft.sort_index()
                if "Adj Close" not in dft.columns and "Close" in dft.columns:
                    dft["Adj Close"] = dft["Close"]
                keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in dft.columns]
                dft = dft[keep].dropna(how="all")
                if not dft.empty: results[t] = dft
        except Exception:
            pass

    # clean
    cleaned = {}
    for t, d in results.items():
        if d is None or d.empty: continue
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in d.columns]
        d = d[keep].sort_index().dropna(how="all")
        if not d.empty: cleaned[t] = d
    return cleaned

# --------- SMA cross events (for chart/inspection) ----------
def compute_sma_cross_events(df: pd.DataFrame, fast: int, slow: int):
    close = df[price_col(df)]
    ma_f = close.rolling(fast).mean()
    ma_s = close.rolling(slow).mean()
    bull = (ma_f.shift(1) <= ma_s.shift(1)) & (ma_f > ma_s)
    bear = (ma_f.shift(1) >= ma_s.shift(1)) & (ma_f < ma_s)
    mask = bull | bear
    ev_idx = close.index[mask]
    ev_type = np.where(bull[mask], "Bullish Cross", "Bearish Cross")
    events = pd.DataFrame({
        "Type": ev_type,
        "Price": close.loc[ev_idx].values,
        "FastSMA": ma_f.loc[ev_idx].values,
        "SlowSMA": ma_s.loc[ev_idx].values,
    }, index=ev_idx)
    events.index.name = "Date"
    return ma_f, ma_s, events

# --------- UI ---------
st.title("🇺🇸 US Backtester — SMA / RSI / Composite (Srini)")
st.caption("Raw vs Strategy, ATR stops/TP, optional costs/taxes, SMA cross events, yfinance (Stooq fallback).")

with st.sidebar:
    st.header("Backtest Settings")
    tickers = st.text_input("Tickers (comma-separated)", value="AAPL, ACN, SPY, XLK", key="sb_tickers")
    start = st.date_input("Start", value=pd.to_datetime("2015-01-01"), key="sb_start").strftime("%Y-%m-%d")
    end   = st.date_input("End", value=pd.Timestamp.today(), key="sb_end").strftime("%Y-%m-%d")

    strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion", "Composite"], key="main_strategy")

    c1, c2 = st.columns(2)
    if strategy == "SMA Crossover":
        fast = c1.number_input("Fast SMA", 2, 200, 20, 1, key="sb_fast")
        slow = c2.number_input("Slow SMA", 5, 400, 100, 5, key="sb_slow")
        params = {"fast": int(fast), "slow": int(slow)}
    elif strategy == "RSI Mean Reversion":
        rsi_lb = c1.number_input("RSI lookback", 2, 100, 14, 1, key="sb_rsi_lb")
        rsi_buy = c2.number_input("RSI Buy <", 5, 50, 30, 1, key="sb_rsi_buy")
        rsi_sell = st.number_input("RSI Sell >", 50, 95, 70, 1, key="sb_rsi_sell")
        params = {"rsi_lb": int(rsi_lb), "rsi_buy": int(rsi_buy), "rsi_sell": int(rsi_sell)}
    else:
        f1, f2, f3 = st.columns(3)
        fast = f1.number_input("Fast SMA", 2, 200, 20, 1, key="cmp_fast")
        slow = f2.number_input("Slow SMA", 5, 400, 100, 5, key="cmp_slow")
        use_macd = f3.checkbox("Use MACD", value=True, key="cmp_use_macd")
        r1, r2, r3 = st.columns(3)
        rsi_lb = r1.number_input("RSI lookback", 2, 100, 14, 1, key="cmp_rsi_lb")
        rsi_buy = r2.number_input("RSI Buy <", 5, 50, 35, 1, key="cmp_rsi_buy")
        rsi_sell = r3.number_input("RSI Sell >", 50, 95, 65, 1, key="cmp_rsi_sell")
        g1, g2, g3 = st.columns(3)
        combine_logic = g1.selectbox("Combine", ["AND (strict)", "Voting (≥2)"], index=0, key="cmp_logic")
        atr_filter = g2.checkbox("ATR 'too hot' filter", value=False, key="cmp_atr_filter")
        atr_cap = g3.number_input("Max ATR/Price on entry", 0.01, 0.20, 0.05, 0.01, key="cmp_atr_cap")
        params = {
            "fast": int(fast), "slow": int(slow),
            "rsi_lb": int(rsi_lb), "rsi_buy": int(rsi_buy), "rsi_sell": int(rsi_sell),
            "use_macd": bool(use_macd),
            "use_and": combine_logic.startswith("AND"),
            "atr_filter": bool(atr_filter),
            "atr_cap_pct": float(atr_cap),
        }

    long_only  = st.checkbox("Long-only", value=True, key="sb_long_only")
    vol_target = st.slider("Vol target (ann.)", 0.05, 0.40, 0.12, 0.01, key="sb_vol")
    atr_stop   = st.slider("ATR Stop (×)", 1.0, 6.0, 3.0, 0.5, key="sb_atr_stop")
    tp_mult    = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5, key="sb_tp")

    st.subheader("Real-world frictions")
    trade_cost = st.number_input("Cost per trade (%)", 0.0, 0.50, 0.05, 0.01, key="sb_cost") / 100.0
    tax_rate   = st.number_input("Effective tax on gains (%)", 0.0, 50.0, 0.0, 1.0, key="sb_tax") / 100.0
    run_btn    = st.button("Run Backtest", key="btn_run")

with st.expander("🔧 Diagnostics"):
    st.write("yfinance:", getattr(yf, "__version__", "unknown"))
    if st.button("Clear caches", key="btn_cache_clear"):
        load_prices.clear()
        st.success("Caches cleared.")

# --------- Run backtests ---------
if run_btn:
    data = load_prices(tickers, start, end)
    if not data:
        st.error("No data downloaded. Try other tickers/dates.")
        st.stop()

    st.caption("Loaded → " + ", ".join(sorted(data.keys())))
    results = []
    tabs = st.tabs(list(data.keys()))
    for tab, t in zip(tabs, data.keys()):
        with tab:
            df = data[t]
            st.write(f"{t}: {len(df)} rows · {df.index.min().date()} → {df.index.max().date()}")

            equity, stats = evaluate_ticker(
                df, t, strategy, params, vol_target, long_only, atr_stop, tp_mult,
                trade_cost=trade_cost, tax_rate=tax_rate
            )

            st.subheader(f"{t} — Equity Curve")
            st.line_chart(equity, height=320)
            st.markdown(
                f"**Buy & Hold CAGR:** {stats['RawCAGR']:.2%}  |  "
                f"**Strategy CAGR:** {stats['StrategyCAGR']:.2%}  |  "
                f"**Sharpe:** {stats['Sharpe']:.2f}  |  "
                f"**MaxDD:** {stats['MaxDD']:.2%}  |  "
                f"**Exposure:** {stats['Exposure']:.0%}"
            )

            if strategy == "SMA Crossover":
                ma_f, ma_s, events = compute_sma_cross_events(df, params["fast"], params["slow"])
                close = df[price_col(df)]
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(df.index, close, label="Close")
                ax.plot(df.index, ma_f, label=f"SMA {params['fast']}")
                ax.plot(df.index, ma_s, label=f"SMA {params['slow']}")
                bull_ix = events.index[events["Type"] == "Bullish Cross"]
                bear_ix = events.index[events["Type"] == "Bearish Cross"]
                ax.scatter(bull_ix, close.loc[bull_ix], marker="^", s=60, label="Bullish Cross")
                ax.scatter(bear_ix, close.loc[bear_ix], marker="v", s=60, label="Bearish Cross")
                ax.set_title(f"{t} — Price with SMA Crossovers")
                ax.legend(loc="best"); ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)

                st.subheader("SMA Crossover Events")
                if events.empty:
                    st.write("No crossovers in the selected window.")
                else:
                    st.dataframe(events.tail(20).round({"Price":2,"FastSMA":2,"SlowSMA":2}), use_container_width=True)

            results.append({"Ticker": t, **stats})

    if results:
        res_df = pd.DataFrame(results).set_index("Ticker")
        res_df["CAGR Δ (Strategy-Raw)"] = (res_df["StrategyCAGR"] - res_df["RawCAGR"]).round(4)
        st.subheader("📋 Summary (Raw vs Strategy)")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("Download Results CSV", res_df.to_csv().encode(), "results_summary.csv", key="dl_summary")

# --------- Simple comparator for ACN vs ETFs ---------
with st.expander("🆚 Compare: Accenture (ACN) vs ETFs"):
    default_start = "2015-01-01"
    default_end   = pd.Timestamp.today().strftime("%Y-%m-%d")
    c1, c2 = st.columns(2)
    start_cmp = c1.text_input("Start (YYYY-MM-DD)", value=default_start, key="cmp_start")
    end_cmp   = c2.text_input("End (YYYY-MM-DD)",   value=default_end,   key="cmp_end")
    universe = st.text_input("Tickers to compare", value="ACN, SPY, XLK, VT", key="cmp_universe")

    strat = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion", "Composite"], index=2, key="compare_strategy")
    cc1, cc2, cc3 = st.columns(3)
    if strat == "SMA Crossover":
        fast_cmp = cc1.number_input("Fast SMA", 2, 200, 20, 1, key="cmp_fast")
        slow_cmp = cc2.number_input("Slow SMA", 5, 400, 100, 5, key="cmp_slow")
        params_cmp = {"fast": int(fast_cmp), "slow": int(slow_cmp)}
    elif strat == "RSI Mean Reversion":
        rsi_lb_cmp  = cc1.number_input("RSI lookback", 2, 100, 14, 1, key="cmp_rsi_lb")
        rsi_buy_cmp = cc2.number_input("RSI Buy <", 5, 50, 30, 1, key="cmp_rsi_buy")
        rsi_sell_cmp= cc3.number_input("RSI Sell >", 50, 95, 70, 1, key="cmp_rsi_sell")
        params_cmp = {"rsi_lb": int(rsi_lb_cmp), "rsi_buy": int(rsi_buy_cmp), "rsi_sell": int(rsi_sell_cmp)}
    else:
        fast_cmp = cc1.number_input("Fast SMA", 2, 200, 20, 1, key="cmp2_fast")
        slow_cmp = cc2.number_input("Slow SMA", 5, 400, 100, 5, key="cmp2_slow")
        use_macd_cmp = cc3.checkbox("Use MACD", value=True, key="cmp2_use_macd")
        rr1, rr2, rr3 = st.columns(3)
        rsi_lb_cmp  = rr1.number_input("RSI lookback", 2, 100, 14, 1, key="cmp2_rsi_lb")
        rsi_buy_cmp = rr2.number_input("RSI Buy <", 5, 50, 35, 1, key="cmp2_rsi_buy")
        rsi_sell_cmp= rr3.number_input("RSI Sell >", 50, 95, 65, 1, key="cmp2_rsi_sell")
        gg1, gg2, gg3 = st.columns(3)
        combine_logic_cmp = gg1.selectbox("Combine", ["AND (strict)", "Voting (≥2)"], index=0, key="cmp2_logic")
        atr_filter_cmp = gg2.checkbox("ATR 'too hot' filter", value=False, key="cmp2_atr_filter")
        atr_cap_cmp = gg3.number_input("Max ATR/Price", 0.01, 0.20, 0.05, 0.01, key="cmp2_atr_cap")
        params_cmp = {
            "fast": int(fast_cmp), "slow": int(slow_cmp),
            "rsi_lb": int(rsi_lb_cmp), "rsi_buy": int(rsi_buy_cmp), "rsi_sell": int(rsi_sell_cmp),
            "use_macd": bool(use_macd_cmp),
            "use_and": combine_logic_cmp.startswith("AND"),
            "atr_filter": bool(atr_filter_cmp),
            "atr_cap_pct": float(atr_cap_cmp),
        }

    long_only_cmp  = st.checkbox("Long-only (ETFs)", value=True, key="cmp_longonly")
    vol_target_cmp = st.slider("Vol target (ann.)", 0.05, 0.40, 0.12, 0.01, key="cmp_vol")
    atr_stop_cmp   = st.slider("ATR Stop (×)", 1.0, 6.0, 3.0, 0.5, key="cmp_atr")
    tp_mult_cmp    = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5, key="cmp_tp")
    cost_cmp       = st.number_input("Cost per trade (%)", 0.0, 0.50, 0.05, 0.01, key="cmp_cost") / 100.0
    tax_cmp        = st.number_input("Effective tax on gains (%)", 0.0, 50.0, 0.0, 1.0, key="cmp_tax") / 100.0

    if st.button("Run comparison", key="btn_run_cmp"):
        tickers_cmp = [t.strip().upper() for t in universe.split(",") if t.strip()]
        data_cmp = load_prices(",".join(tickers_cmp), start_cmp, end_cmp)
        if not data_cmp:
            st.error("No data downloaded for the selected tickers/range.")
            st.stop()

        rows, equities = [], []
        for t in tickers_cmp:
            df = data_cmp.get(t)
            if df is None or df.empty: continue
            eq, stats = evaluate_ticker(
                df, t, strat, params_cmp, vol_target_cmp, long_only_cmp, atr_stop_cmp, tp_mult_cmp,
                trade_cost=cost_cmp, tax_rate=tax_cmp
            )
            rows.append({"Ticker": t, **stats})
            equities.append(eq.rename(t) / float(eq.iloc[0]))

        if rows:
            res = pd.DataFrame(rows).set_index("Ticker").sort_values("StrategyCAGR", ascending=False)
            res["CAGR Δ (Strategy-Raw)"] = (res["StrategyCAGR"] - res["RawCAGR"]).round(4)
            st.subheader("Summary (Strategy vs Raw)")
            st.dataframe(res, use_container_width=True)
            if equities:
                combined = pd.concat(equities, axis=1).dropna(how="all")
                st.subheader("Normalized Equity Curves (start = 1.0)")
                st.line_chart(combined, height=360)