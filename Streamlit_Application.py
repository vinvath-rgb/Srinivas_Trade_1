# streamlit_app.py  (diagnostic harness)
import os, sys, time, pathlib, platform
import streamlit as st

st.set_page_config(page_title="SRM – Diagnostics", layout="centered")
st.title("🧪 Strategy–Regime Matrix: Diagnostics")

st.subheader("1) Basic widgets should show below")
name = st.text_input("Your name", value="Srini")
num  = st.number_input("Pick a number", value=42, step=1)
clicked = st.button("Click me")
if clicked:
    st.success(f"Hello {name}! Button works. Number={num}")

st.subheader("2) Runtime info")
col1, col2 = st.columns(2)
with col1:
    st.write("**Python**", sys.version)
    st.write("**Streamlit**", st.__version__)
    st.write("**Platform**", platform.platform())
with col2:
    st.write("**CWD**", os.getcwd())
    st.write("**File**", __file__)
    st.write("**Dir list**", sorted(os.listdir("."))[:20])

with st.expander("3) Environment keys (names only)"):
    st.write(sorted([k for k in os.environ.keys()]))

st.subheader("4) Liveness")
ph = st.empty()
for i in range(3):
    ph.info(f"Rendering tick {i+1}/3")
    time.sleep(0.3)
ph.success("UI loop progressed; app is not stuck.")

st.subheader("5) Main() sanity check")
def main():
    st.write("✅ main() ran successfully.")
main()  # ensure it’s actually called

st.caption("If you can see inputs above, the framework is fine; the original app is exiting early or not calling main().")
