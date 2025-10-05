import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Srini Universe + Backtester", layout="wide")

# ────────────────────── Helper ──────────────────────
def push_to_backtester(tickers_str: str):
    st.session_state["final_tickers"] = tickers_str
    st.session_state["bt_tickers"] = tickers_str      # <- critical for Render
    st.session_state["last_sent"] = tickers_str
    st.toast("Sent to Backtester ✔ — switch to the Backtester tab and hit Run.")

# ────────────────────── Tabs ──────────────────────
tab_univ, tab_bt = st.tabs(["🌌 Universe Builder", "📊 Backtester"])

# ========== UNIVERSE BUILDER TAB ==========
with tab_univ:
    st.header("Universe Builder")

    st.write("Example mock data for testing the Universe–Backtester connection.")
    df = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA"],
        "Reco": ["BUY", "WATCH", "AVOID", "BUY", "WATCH"]
    })
    st.dataframe(df)

    include_watch = st.checkbox("Include WATCH tickers", value=True)
    reco_filter = ["BUY"]
    if include_watch:
        reco_filter.append("WATCH")

    to_send = df.loc[df["Reco"].isin(reco_filter), "Ticker"].tolist()
    st.markdown(f"**Preview (to Backtester):** {', '.join(to_send) if to_send else '— nothing to send —'}")

    if st.button("➡️ Send to Backtester", disabled=(len(to_send) == 0)):
        push_to_backtester(", ".join(to_send))

# ========== BACKTESTER TAB ==========
with tab_bt:
    st.header("Backtester")

    if "last_sent" in st.session_state:
        st.info(f"Tickers received from Universe: {st.session_state['last_sent']}")

    default_tickers = st.session_state.get("final_tickers", "SPY, XLK, ACN, XIC.TO")
    tickers = st.text_input(
        "Tickers (comma-separated)",
        value=st.session_state.get("bt_tickers", default_tickers),
        key="bt_tickers"
    )

    st.write(f"✅ Active tickers: {tickers}")

    if st.button("Run Backtest"):
        st.success(f"Pretending to backtest: {tickers}")