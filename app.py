"""Streamlit web app: upload a broker position CSV, get the risk tearsheet.

Run locally:
    streamlit run app.py

Private deploy (Render / Railway / Fly / VPS): set an APP_PASSWORD secret to
gate access. See DEPLOY.md.
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import date
from pathlib import Path

import pdfplumber
import streamlit as st

from riskreport.pipeline import generate_report

st.set_page_config(page_title="Portfolio Risk Report", page_icon="📊",
                   layout="wide")

CACHE_DIR = os.environ.get("RISK_CACHE_DIR", "cache")
OUT_DIR = os.environ.get("RISK_OUT_DIR", "reports")


# ----------------------------------------------------------------------
# Password gate (only enforced when APP_PASSWORD is set — local dev is open)
# ----------------------------------------------------------------------
def _expected_password() -> str | None:
    env = os.environ.get("APP_PASSWORD")
    if env:
        return env
    try:  # st.secrets raises if no secrets file is configured
        return st.secrets.get("APP_PASSWORD")
    except Exception:
        return None


def _check_password() -> bool:
    expected = _expected_password()
    if not expected:
        return True  # no password configured -> open (local use)
    if st.session_state.get("authed"):
        return True
    st.title("📊 Portfolio Risk Report")
    pw = st.text_input("Password", type="password")
    if pw and pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()


# ----------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------
st.title("📊 Portfolio Risk Report")
st.caption(
    "Upload a broker position export (the 'Intraday Position' CSV — account, "
    "symbol, and quantities). The tool prices the book with free market data "
    "and generates a two-page risk tearsheet. Not investment advice."
)

with st.sidebar:
    st.header("Options")
    name = st.text_input("Portfolio name", value="",
                         help="Defaults to the account number in the file.")
    aum_m = st.number_input("AUM ($M, optional)", min_value=0.0, value=0.0,
                            step=1.0, help="Enables % AUM columns. 0 = use % gross.")
    asof_override = st.date_input("As-of date (optional)", value=None,
                                  help="Defaults to the date in the filename.")
    with_factors = st.toggle("Factor model, stress & VaR (page 2)", value=True)
    with_hedge = st.toggle("Hedge-basket suggestion", value=True,
                           disabled=not with_factors)
    alerts_file = st.file_uploader("Risk-limit config (optional JSON)",
                                   type=["json"])
    st.divider()
    st.caption("First run for a new book takes a few minutes (market-data "
               "fetch); repeat runs are cached and fast.")

uploaded = st.file_uploader("Position CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a position CSV to begin.")
    st.stop()

if not st.button("Generate report", type="primary"):
    st.stop()

# persist the upload under its original name so the as-of date parses from it
work_dir = Path(tempfile.mkdtemp(prefix="riskreport_"))
csv_path = work_dir / uploaded.name
csv_path.write_bytes(uploaded.getbuffer())

alerts_path = None
if alerts_file is not None:
    alerts_path = work_dir / "alerts.json"
    alerts_path.write_bytes(alerts_file.getbuffer())

log_lines: list[str] = []
status = st.status("Running…", expanded=True)
log_box = status.empty()


def _progress(msg: str) -> None:
    log_lines.append(msg)
    log_box.code("\n".join(log_lines))


try:
    result = generate_report(
        csv_path,
        aum=(aum_m * 1e6) if aum_m else None,
        name=name or None,
        asof=asof_override if isinstance(asof_override, date) else None,
        out_dir=OUT_DIR,
        cache_dir=CACHE_DIR,
        alerts_path=alerts_path,
        no_factors=not with_factors,
        no_hedge=not with_hedge,
        progress=_progress,
    )
    status.update(label=f"Done in {result.elapsed_s:.0f}s", state="complete",
                  expanded=False)
except Exception as exc:  # surface the failure instead of a blank screen
    status.update(label="Failed", state="error")
    st.error(f"Report generation failed: {exc}")
    st.stop()

# --------------------------------------------------------------- results
h = result.headline
if result.alert_hits:
    st.error("⚠ **Risk limit breach(es):**\n\n"
             + "\n".join(f"- {x}" for x in result.alert_hits))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net Δ-adj exposure", f"${h['exp_net']/1e6:,.1f}M")
c2.metric("Gross Δ-adj exposure", f"${h['exp_gross']/1e6:,.1f}M")
c3.metric("Predicted vol (ann.)",
          f"${h['vol_total']/1e6:,.1f}M" if h.get("vol_total") else "—")
c4.metric("1-day 95% VaR",
          f"${h['var_95']/1e6:,.2f}M" if h.get("var_95") else "—")

pdf_bytes = Path(result.pdf_path).read_bytes()
st.download_button("⬇ Download PDF", data=pdf_bytes,
                   file_name=Path(result.pdf_path).name,
                   mime="application/pdf", type="primary")

with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
    for i, page in enumerate(pdf.pages):
        img = page.to_image(resolution=150)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.image(buf.getvalue(), caption=f"Page {i + 1}", use_container_width=True)

if result.issues:
    with st.expander(f"Data-quality notes ({len(result.issues)})"):
        for msg in result.issues:
            st.write(f"- {msg}")
