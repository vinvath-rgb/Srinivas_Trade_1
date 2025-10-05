# app.py  (filename can be anything)
import os, time, requests, pandas as pd, streamlit as st

API_KEY = os.getenv("FINNHUB_API_KEY")
BASE = "https://finnhub.io/api/v1"

st.set_page_config(page_title="Finnhub TSX Quick Tester", layout="centered")
st.title("🇨🇦 Finnhub Quick Tester (TSX & US)")

if not API_KEY:
    st.error("Missing FINNHUB_API_KEY environment variable. Set it in Render → Environment.")
    st.stop()

st.write("Use this to quickly verify Finnhub connectivity and fetch recent candles.")

col1, col2 = st.columns([2,1])
with col1:
    raw_symbol = st.text_input("Symbol", value="SHOP")  # example: SHOP (TSX), AAPL (US)
with col2:
    is_tsx = st.checkbox("Canada / TSX (.TO)", value=True)

resolution = st.selectbox("Resolution", ["D","60","30","15","5","1"], index=0,
                          help="Finnhub resolutions: 1,5,15,30,60 (minutes) or D (daily)")
lookback_days = st.slider("Lookback (days)", 30, 1000, 365)

def normalize_symbol(sym: str, tsx: bool) -> str:
    s = sym.strip().upper()
    if tsx and "." not in s:
        s = f"{s}.TO"      # Finnhub TSX format
    return s

def ping_finnhub():
    try:
        r = requests.get(f"{BASE}/quote",
                         params={"symbol": normalize_symbol(raw_symbol, is_tsx), "token": API_KEY},
                         timeout=20)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        j = r.json()
        if "c" in j:
            return True, f"OK. Current price = {j['c']}"
        return False, f"Unexpected payload: {j}"
    except Exception as e:
        return False, str(e)

def fetch_candles(sym: str, res: str, days: int) -> pd.DataFrame | None:
    now = int(time.time())
    _from = now - days*24*60*60
    params = {
        "symbol": sym,
        "resolution": res,
        "from": _from,
        "to": now,
        "token": API_KEY,
    }
    r = requests.get(f"{BASE}/stock/candle", params=params, timeout=30)
    try:
        j = r.json()
    except Exception:
        st.error(f"Bad response: {r.status_code} {r.text[:200]}")
        return None

    if j.get("s") != "ok":
        st.error(f"Finnhub error for {sym}: {j}")
        return None

    df = pd.DataFrame({
        "Open":  j["o"],
        "High":  j["h"],
        "Low":   j["l"],
        "Close": j["c"],
        "Volume": j.get("v", []),
    }, index=pd.to_datetime(j["t"], unit="s", utc=True).tz_convert("UTC"))
    df.index.name = "Date"
    return df

st.divider()
if st.button("🔌 Test Finnhub Connectivity"):
    ok, msg = ping_finnhub
    st.success(msg) if ok else st.error(msg)

symbol = normalize_symbol(raw_symbol, is_tsx)
if st.button("📈 Fetch Prices"):
    df = fetch_candles(symbol, resolution, lookback_days)
    if df is not None and not df.empty:
        st.success(f"Retrieved {len(df)} rows for {symbol} at {resolution} resolution.")
        st.dataframe(df.tail(10))
        st.line_chart(df["Close"])
    else:
        st.warning("No data returned. Check symbol, resolution, or lookback.")

st.caption("Tip: TSX symbols usually need the .TO suffix (e.g., RY.TO, ENB.TO, SHOP.TO).")