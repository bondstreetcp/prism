"""Streamlit web app for the portfolio risk tool.

Views: on-screen Report (with a cash / delta-adjusted / beta-adjusted toggle),
Trends over time, Benchmark-relative (active) risk, and an Optimizer.

Run locally:  streamlit run app.py
Private deploy: set APP_PASSWORD.  See DEPLOY.md.
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from riskreport.analytics import (
    BASIS_COLUMNS, CAP_ORDER, basis_category_table, basis_summary,
    basis_top_issuers,
)
from riskreport.benchmark import BENCHMARK_CHOICES, active_risk
from riskreport.macro import compute_macro
from riskreport.narrative import (
    build_facts, build_reference, chat as ai_chat, generate_narrative,
    is_available,
)
from riskreport.optimizer import OBJECTIVES, optimize_overlay
from riskreport.pipeline import generate_report
from riskreport import remote_store, theme
from riskreport.screener import build_screen_frame, screen
from riskreport.tags import parse_tags, theme_exposure
from riskreport.trends import TREND_METRICS, load_trend_series

st.set_page_config(page_title="Prism — Portfolio Risk", page_icon="📊",
                   layout="wide")
theme.inject()

CACHE_DIR = os.environ.get("RISK_CACHE_DIR", "cache")
OUT_DIR = os.environ.get("RISK_OUT_DIR", "reports")
SNAP_DIR = os.environ.get("RISK_SNAP_DIR", "snapshots")

# On ephemeral hosts (Streamlit Cloud) the snapshot dir is wiped on reboot.
# If a remote mirror is configured, pull the accumulated history once per
# session so Trends/attribution see it. No-op when unconfigured (NAS/local).
if remote_store.enabled() and not st.session_state.get("snap_pulled"):
    remote_store.pull(SNAP_DIR)
    st.session_state["snap_pulled"] = True


def _m(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    # sub-million values (small books, VaR, sector nets) read better in $k than
    # as "$0.0M"; $1M+ stays in $M
    if abs(v) < 1e6:
        k = v / 1e3
        return f"(${abs(k):,.0f}k)" if k < 0 else f"${k:,.0f}k"
    v = v / 1e6
    return f"(${abs(v):,.1f}M)" if v < 0 else f"${v:,.1f}M"


def _kd(v) -> str:
    """Signed $ in thousands — for greeks (gamma/vega/theta) and small P&L."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    k = v / 1e3
    return f"-${abs(k):,.0f}k" if k < 0 else f"${k:,.0f}k"


# ----------------------------------------------------------------- auth
def _expected_password():
    env = os.environ.get("APP_PASSWORD")
    if env:
        return env
    try:
        return st.secrets.get("APP_PASSWORD")
    except Exception:
        return None


def _gate() -> bool:
    expected = _expected_password()
    if not expected or st.session_state.get("authed"):
        return True
    theme.brand_header()
    pw = st.text_input("Password", type="password")
    if pw and pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    return False


if not _gate():
    st.stop()

theme.brand_header()
st.warning(
    "**Not investment advice · Internal use only.** Figures are model estimates "
    "from free, best-effort market data (Yahoo Finance) that may be delayed, "
    "incomplete, or wrong. Verify before acting.",
    icon="⚠️",
)

# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Run")
    name = st.text_input("Portfolio name", value="")
    cash_m = st.number_input("Add cash ($M)", value=0.0, step=1.0,
                             help="AUM = net market value + cash; % columns are "
                                  "% of AUM. IBKR files include their cash "
                                  "automatically — enter only cash your files "
                                  "omit (e.g. the Goldman account's balance). It "
                                  "ADDS to file-reported cash. Use negative for a "
                                  "margin debit. 0 = add nothing.")
    asof_override = st.date_input("As-of date (optional)", value=None)
    with_factors = st.toggle("Factor model, stress & VaR", value=True)
    with_hedge = st.toggle("Hedge suggestion", value=True, disabled=not with_factors)
    with_scenarios = st.toggle("Crisis scenarios in PDF", value=False,
                               disabled=not with_factors,
                               help="Adds the crisis-scenario replays to the PDF. "
                                    "First run fetches multi-year history "
                                    "(slower); cached afterward.")
    alerts_file = st.file_uploader("Risk-limit config (JSON)", type=["json"])
    uploaded = st.file_uploader("Position CSV(s)", type=["csv"],
                                accept_multiple_files=True,
                                help="Upload one file, or several to aggregate "
                                     "multiple accounts (Goldman + IBKR) into one book.")
    run = st.button("Generate report", type="primary", disabled=not uploaded)
    st.caption("First run for a new book takes a few minutes; repeats are cached.")


def _do_run():
    work = Path(tempfile.mkdtemp(prefix="riskreport_"))
    paths = []
    for uf in uploaded:
        p = work / uf.name
        p.write_bytes(uf.getbuffer())
        paths.append(p)
    csv_arg = paths if len(paths) > 1 else paths[0]
    alerts_path = None
    if alerts_file is not None:
        alerts_path = work / "alerts.json"
        alerts_path.write_bytes(alerts_file.getbuffer())

    lines: list[str] = []
    status = st.status("Running…", expanded=True)
    box = status.empty()

    def prog(msg):
        lines.append(msg)
        box.code("\n".join(lines))

    try:
        res = generate_report(
            csv_arg,
            cash=(cash_m * 1e6) if cash_m else None,
            name=name or None,
            asof=asof_override if isinstance(asof_override, date) else None,
            out_dir=OUT_DIR, cache_dir=CACHE_DIR, snap_dir=SNAP_DIR,
            alerts_path=alerts_path,
            no_factors=not with_factors, no_hedge=not with_hedge,
            include_scenarios=with_scenarios, progress=prog,
        )
        status.update(label=f"Done in {res.elapsed_s:.0f}s", state="complete",
                      expanded=False)
        st.session_state["result"] = res
        # invalidate derived caches from any prior book so the MC / Scenarios /
        # AI tabs never show a previous report's numbers (these caches are
        # keyed by as-of date, which can collide across different uploads)
        for k in [k for k in st.session_state
                  if k.startswith(("mc_", "scenlib_"))]:
            del st.session_state[k]
        st.session_state.pop("ai_summary", None)
        st.session_state["ai_chat"] = []
        # Mirror this run's snapshot off-box so the history survives reboots.
        remote_store.push_date(SNAP_DIR, res.asof, log=prog)
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.error(f"Report generation failed: {exc}")


if run:
    _do_run()

result = st.session_state.get("result")


# =====================================================================
# Tab renderers
# =====================================================================
def _render_risk_greeks(res):
    """1-day VaR (vol-aware) + portfolio option greeks."""
    risk = res.summary.get("risk") or {}
    greeks = res.summary.get("greeks") or {}
    if risk.get("var_95") is None and not greeks:
        return  # factor model / scenarios were turned off

    st.subheader("Risk — 1-day VaR & greeks")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("VaR 95%", _m(risk.get("var_95")))
    v2.metric("VaR 99%", _m(risk.get("var_99")))
    v3.metric("Expected shortfall 95%", _m(risk.get("es_95")))
    addon = None
    if risk.get("var_95") is not None and risk.get("var_95_spot") is not None:
        addon = risk["var_95"] - risk["var_95_spot"]
    v4.metric("Vol add-on to VaR95", _m(addon),
              help="Extra VaR from the historical vol spike that accompanies a "
                   "sell-off, beyond the spot move alone — the short-vega / "
                   "short-gamma tail a delta-only VaR misses.")
    if risk.get("vol_aware"):
        st.caption("Vol-aware historical VaR — option IV co-shocks with the VIX "
                   f"path each day. Spot-only VaR95 would read "
                   f"{_m(risk.get('var_95_spot'))}.")
    elif risk.get("var_95") is not None:
        st.caption("Spot-only VaR (implied vol held constant — VIX history "
                   "unavailable this run).")

    if greeks:
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Net delta", _m(greeks.get("net_delta")),
                  help="Delta-adjusted net exposure (equity + option delta).")
        g2.metric("Net gamma · P&L per ±1%", _kd(greeks.get("net_gamma_1pct")),
                  help="Convexity P&L from a 1% underlying move in either "
                       "direction. Negative = short gamma (put-writing).")
        g3.metric("Net vega · per +1 vol pt", _kd(greeks.get("net_vega_1pt")),
                  help="P&L if implied vol rises one point. Negative = short "
                       "vol — loses when vol spikes.")
        g4.metric("Net theta · per day", _kd(greeks.get("net_theta_day")),
                  help="Time decay per calendar day. Positive = long theta "
                       "(collecting premium).")

    _render_risk_contrib(res)
    st.divider()


def _render_risk_contrib(res):
    """Component / marginal VaR — which names drive the tail loss."""
    sc = getattr(res, "scenarios", None)
    rc = getattr(sc, "risk_contrib", None) if sc is not None else None
    if rc is None or rc.empty:
        return
    st.markdown("**Top risk contributors** — share of expected tail loss (ES95). "
                "Contributions sum to ES95; a negative value means the name "
                "*cushions* the tail (a diversifier or hedge).")
    top = rc.head(12)
    disp = pd.DataFrame({
        "Ticker": top["underlying"],
        "Name": top["name"].astype(str).str.slice(0, 24),
        "Sector": top["sector"].astype(str).str.slice(0, 16),
        "Exposure": top["exposure"].map(_m),
        "Contrib to ES95": top["contrib_es95"].map(_m),
        "% of tail": top["pct_of_es95"].map(lambda x: f"{x:.1%}"),
        "Risk / $1M expo": top["risk_per_1m"].map(
            lambda v: _kd(v) if pd.notna(v) else "—"),
    })
    st.dataframe(disp, hide_index=True, width="stretch")
    sec = (rc.groupby("sector")["contrib_es95"].sum()
           .sort_values(ascending=False) / 1e6)
    st.markdown("**Risk contribution by sector** ($M of ES95)")
    st.bar_chart(sec, horizontal=True, height=220)


def _render_factor_decomp(res):
    """Ex-ante (factor-model) decomposition of predicted volatility: which
    factors and which names drive forecast risk — the forward-looking
    complement to the tail-based component VaR above."""
    fr = getattr(res, "factor_risk", None)
    if fr is None or not getattr(fr, "vol_total", 0):
        return
    st.divider()
    st.subheader("Predicted-volatility drivers (factor model)")
    st.caption(f"Ex-ante: how the {_m(fr.vol_total)} of predicted annual "
               "volatility decomposes by risk factor and by name. Shares are "
               "of predicted variance and sum to 100% (factors + stock-"
               "specific).")

    # factor + specific contribution to predicted variance
    fc = (fr.factor_risk_contrib * 100)  # % of variance per factor
    fc = fc[fc.abs() > 1e-9]
    specific_share = max(0.0, 1.0 - float(fr.factor_var_share)) * 100
    contrib = fc.copy()
    contrib["Stock-specific"] = specific_share
    contrib = contrib.sort_values(ascending=False).rename("% of variance")
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        st.markdown("**By risk factor** (% of predicted variance)")
        st.bar_chart(contrib, horizontal=True, height=280)
    with cc2:
        st.markdown("**Top names by predicted-vol contribution**")
        pr = fr.position_risk
        if pr is not None and len(pr):
            byname = (pr.groupby("underlying")
                      .agg(sector=("sector", "first"),
                           exposure=("exposure", "sum"),
                           rc=("risk_contrib", "sum"))
                      .sort_values("rc", ascending=False).head(10)
                      .reset_index())
            disp = pd.DataFrame({
                "Ticker": byname["underlying"],
                "Sector": byname["sector"].astype(str).str.slice(0, 14),
                "Exposure": byname["exposure"].map(_m),
                "% of var": byname["rc"].map(lambda x: f"{x:.1%}"),
            })
            st.dataframe(disp, hide_index=True, width="stretch")
    st.caption("This is ex-ante model risk (factor covariance × exposures); the "
               "component-VaR table above is realized-tail risk. They answer "
               "different questions — forecast vs. historical stress.")


def _render_montecarlo(res):
    """Parametric Monte Carlo VaR (factor-model), run on demand."""
    if getattr(res, "model", None) is None or res.analytics is None:
        return
    st.divider()
    with st.expander("Monte Carlo VaR — parametric (factor model), run on demand"):
        st.caption("Draws a large synthetic sample from the fitted factor "
                   "covariance (+ stock-specific noise), reprices the book with "
                   "full Black-Scholes option revaluation and a vol co-shock. "
                   "Complements the historical-sim VaR with a smooth, model-based "
                   "tail not limited to the last ~250 days.")
        c1, c2 = st.columns([1, 3])
        n_sims = c1.select_slider("Simulations", [5000, 10000, 25000, 50000],
                                  value=10000)
        run = c1.button("▶ Run Monte Carlo", type="primary")
        key = f"mc_{res.analytics.asof}_{n_sims}"
        if run or key in st.session_state:
            if key not in st.session_state:
                with st.spinner(f"Simulating {n_sims:,} scenarios…"):
                    from riskreport.montecarlo import monte_carlo_var
                    st.session_state[key] = monte_carlo_var(
                        res.analytics.positions, res.model, res.closes,
                        res.analytics.asof, n_sims=n_sims)
            mc = st.session_state[key]
            if mc is None:
                st.info("Monte Carlo needs the factor model.")
                return
            hist = res.summary.get("risk") or {}
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MC VaR 95%", _m(mc.var_95), help="1-day, vol-aware")
            m2.metric("MC VaR 99%", _m(mc.var_99))
            m3.metric("MC ES 95%", _m(mc.es_95))
            m4.metric("Hist-sim VaR95", _m(hist.get("var_95")),
                      help="The historical-simulation VaR, for comparison.")
            addon = mc.var_95 - mc.var_95_spot
            st.caption(f"{mc.n_sims:,} sims · vol co-shock β={mc.vol_beta:.1f} "
                       f"(IV rises when the market falls) · spot-only VaR95 "
                       f"{_m(mc.var_95_spot)} → vol adds {_m(addon)} · "
                       f"coverage {mc.coverage:.0%}.")
            import numpy as np
            pnl_m = mc.pnl / 1e6
            counts, edges = np.histogram(pnl_m, bins=60)
            centers = (edges[:-1] + edges[1:]) / 2
            # label bins but keep a unique index (rounded centers can collide
            # on a small-range book, which would make Altair merge bars)
            hist_df = pd.DataFrame({"P&L $M": np.round(centers, 3),
                                    "scenarios": counts})
            st.markdown("**Simulated 1-day P&L distribution ($M)**")
            st.bar_chart(hist_df, x="P&L $M", y="scenarios", height=220)


def _render_concentration(res):
    """Concentration & diversification — is the risk really concentrated?"""
    fr = getattr(res, "factor_risk", None)
    if fr is None or getattr(res, "model", None) is None:
        return
    from riskreport.concentration import compute_concentration
    con = compute_concentration(fr, res.model)
    if con is None:
        return
    st.divider()
    st.subheader("Concentration & diversification")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effective bets (risk)", f"{con['effective_bets_risk']:.1f}",
              help="Number of *independent* risk bets the book behaves like "
                   "(Herfindahl of risk contributions). Far below the issuer "
                   "count means correlated, concentrated risk.")
    c2.metric("Issuers held", f"{con['n_issuers']}",
              help="For contrast with effective bets.")
    c3.metric("Diversification ratio", f"{con['diversification_ratio']:.2f}×",
              help="Σ standalone name vol ÷ portfolio vol. >1 means "
                   "correlations reduce risk; near 1 means little benefit.")
    c4.metric("Top-5 share of risk", f"{con['top5_risk_share']:.0%}",
              help="Share of total risk in the 5 largest contributors.")
    st.caption(f"The book holds {con['n_issuers']} issuers but its risk behaves "
               f"like ~{con['effective_bets_risk']:.0f} independent bets "
               f"(exposure-weighted: ~{con['effective_bets_exposure']:.0f}). "
               "See the risk-contributor tables above for the names.")


def render_report(res):
    if res.alert_hits:
        st.error("⚠ **Risk limit breach(es):**\n\n"
                 + "\n".join(f"- {x}" for x in res.alert_hits))

    basis_label = st.radio("Exposure basis", list(BASIS_COLUMNS), horizontal=True,
                           help="Cash = market value · Delta-adjusted = economic "
                                "exposure · Beta-adjusted = delta-adj × beta vs SPY")
    col = BASIS_COLUMNS[basis_label]
    iss = res.analytics.issuers
    summ = basis_summary(iss, col)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Long ({basis_label})", _m(summ["long"]))
    c2.metric("Short", _m(summ["short"]))
    c3.metric("Gross", _m(summ["gross"]))
    c4.metric("Net", _m(summ["net"]))
    if summ["coverage"] < 0.999:
        st.caption(f"{basis_label} covers {summ['coverage']:.0%} of gross "
                   "delta-adjusted exposure (names without a beta are excluded).")

    _render_risk_greeks(res)
    _render_factor_decomp(res)
    _render_concentration(res)
    _render_montecarlo(res)

    st.subheader("Exposure breakdown")
    cats = [("By sector", "sector", None), ("By market cap", "cap_bucket", CAP_ORDER),
            ("By region", "region", None)]
    cols = st.columns(3)
    for (title, cat_col, order), c in zip(cats, cols):
        tbl = basis_category_table(iss, cat_col, col, order=order)
        with c:
            st.markdown(f"**{title}**")
            chart_df = tbl.set_index(cat_col)["net"] / 1e6
            st.bar_chart(chart_df, horizontal=True, height=240)
            show = tbl[[cat_col, "long", "short", "net"]].copy()
            for k in ("long", "short", "net"):
                show[k] = show[k].map(_m)
            st.dataframe(show, hide_index=True, width="stretch")

    st.subheader("Top issuers")
    aum = res.summary.get("aum")
    if aum is not None and aum <= 0:      # a negative/zero AUM can't scale %
        aum = None
    pct_col = "% AUM" if aum else "% side"
    lc, sc = st.columns(2)
    for side, c in (("long", lc), ("short", sc)):
        sel, total = basis_top_issuers(iss, col, side, n=10)
        rows = []
        for _, r in sel.iterrows():
            pct = (r[col] / aum) if aum else (abs(r[col]) / total if total else 0)
            rows.append({
                "Ticker": r["underlying"],
                "Issuer": str(r["name"] or r["underlying"])[:26],
                "Sector": str(r["sector"])[:18], "$": _m(r[col]),
                pct_col: f"{pct:.1%}",
            })
        with c:
            st.markdown(f"**Top {side}s**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.download_button("⬇ Download PDF", data=Path(res.pdf_path).read_bytes(),
                       file_name=Path(res.pdf_path).name, mime="application/pdf")
    if res.issues:
        with st.expander(f"Data-quality notes ({len(res.issues)})"):
            for msg in res.issues:
                st.write(f"- {msg}")


def render_trends():
    ts = load_trend_series(SNAP_DIR)
    if ts.empty:
        st.info("No snapshots yet — run a report to start the history.")
        return
    st.caption(f"{len(ts)} snapshot(s): {ts['date'].min()} → {ts['date'].max()}. "
               "Each report run archives one; history builds as you run daily.")
    if len(ts) < 2:
        st.warning("Only one snapshot so far — trends need at least two runs on "
                   "different as-of dates.")
    picks = st.multiselect("Metrics", list(TREND_METRICS),
                           default=["Net exposure", "Gross exposure",
                                    "Predicted vol (ann.)"])
    dollar = {k: v for k, v in TREND_METRICS.items()
              if k in picks and v[2] == "$M"}
    other = {k: v for k, v in TREND_METRICS.items()
             if k in picks and v[2] != "$M"}
    idx = pd.to_datetime(ts["date"])
    if dollar:
        df = pd.DataFrame({k: ts[c] / s for k, (c, s, _u) in dollar.items()})
        df.index = idx
        st.markdown("**$M metrics**")
        st.line_chart(df, height=280)
    for k, (c, s, _u) in other.items():
        df = pd.DataFrame({k: ts[c] / s}); df.index = idx
        st.markdown(f"**{k}**")
        st.line_chart(df, height=200)


def render_brinson(res):
    """Brinson-Fachler realized-return attribution vs the S&P 500."""
    from riskreport.attribution_brinson import (
        brinson_attribution, WINDOWS, SP500_WEIGHT_ASOF)

    st.subheader("Performance attribution (Brinson–Fachler vs S&P 500)")
    if res.analytics is None or res.closes is None:
        st.info("Attribution needs holdings and price history — re-run a report.")
        return
    win = st.radio("Window", list(WINDOWS), index=1, horizontal=True,
                   key="brinson_win")
    try:
        br = brinson_attribution(res.analytics.issuers, res.closes,
                                 res.analytics.asof, window=win)
    except Exception as exc:
        st.error(f"Attribution failed: {exc}")
        return
    if br is None:
        st.info("Not enough price history in the window for attribution.")
        return

    st.caption(f"Current holdings held {br.start} → {br.end} · portfolio return "
               f"{br.r_port:+.2%} vs benchmark {br.r_bench:+.2%} · S&P sector "
               f"weights approx as of {SP500_WEIGHT_ASOF} · {br.coverage:.0%} of "
               "gross exposure has a usable return.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active return", f"{br.active:+.2%}")
    m2.metric("Allocation", f"{br.allocation:+.2%}",
              help="From over/under-weighting sectors vs the index.")
    m3.metric("Selection", f"{br.selection:+.2%}",
              help="From name picks within sectors.")
    m4.metric("Interaction", f"{br.interaction:+.2%}",
              help="Cross term of active weight × active return.")

    t = br.table.copy()
    disp = pd.DataFrame({
        "Sector": t["sector"],
        "Wt port": (t["w_port"]).map(lambda x: f"{x:.0%}"),
        "Wt bench": (t["w_bench"]).map(lambda x: f"{x:.0%}"),
        "Ret port": (t["r_port"]).map(lambda x: f"{x:+.1%}"),
        "Ret bench": (t["r_bench"]).map(lambda x: f"{x:+.1%}"),
        "Allocation": (t["allocation"]).map(lambda x: f"{x:+.2%}"),
        "Selection": (t["selection"]).map(lambda x: f"{x:+.2%}"),
        "Total": (t["total"]).map(lambda x: f"{x:+.2%}"),
    })
    st.dataframe(disp, hide_index=True, width="stretch")
    st.markdown("**Active contribution by sector** (allocation + selection + "
                "interaction, %)")
    st.bar_chart(t.set_index("sector")["total"], horizontal=True, height=280)
    if br.issues:
        for msg in br.issues:
            st.caption(f"⚠ {msg}")
    st.caption("Current-holdings attribution; option legs are attributed at "
               "their underlying's return (theta/vega/gamma P&L is covered on "
               "the Report risk panel). Net-exposure weighted — a long/short "
               "book stretches the classic long-only Brinson frame.")
    st.divider()


def render_factor_attr(res):
    """Factor-based (Barra) return attribution — realized P&L by factor."""
    from riskreport.attribution_factor import factor_return_attribution, WINDOWS

    st.subheader("Factor-based return attribution")
    if (res.factor_risk is None or res.model is None
            or getattr(res, "factor_returns", None) is None):
        st.info("Needs the factor model — re-run with it enabled.")
        return
    win = st.radio("Window", list(WINDOWS), index=1, horizontal=True,
                   key="factorattr_win")
    try:
        fa = factor_return_attribution(
            res.factor_risk, res.model, res.closes, res.factor_returns,
            res.analytics.issuers, res.analytics.asof,
            res.summary.get("aum"), window=win)
    except Exception as exc:
        st.error(f"Factor attribution failed: {exc}")
        return
    if fa is None:
        st.info("Not enough history in the window.")
        return

    aum = fa.aum if fa.aum else None
    st.caption(f"Realized book P&L {fa.start} → {fa.end}, decomposed into "
               "systematic factor P&L (net exposure × factor return) and a "
               "stock-specific remainder. Parts sum to realized.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Realized P&L", _m(fa.realized_pnl),
              (f"{fa.realized_pnl/aum:+.1%} of AUM" if aum else None))
    m2.metric("From factors", _m(fa.factor_pnl))
    m3.metric("Stock-specific", _m(fa.specific_pnl))

    t = fa.table.reindex(fa.table["pnl"].abs().sort_values(ascending=False).index)
    disp = pd.DataFrame({
        "Factor": t["factor"],
        "Net exposure": t["exposure"].map(_m),
        "Factor return": t["factor_return"].map(lambda x: f"{x:+.1%}"),
        "P&L": t["pnl"].map(_m),
    })
    disp.loc[len(disp)] = ["Stock-specific", "", "", _m(fa.specific_pnl)]
    st.dataframe(disp, hide_index=True, width="stretch")
    chart = pd.concat([t.set_index("factor")["pnl"],
                       pd.Series({"Stock-specific": fa.specific_pnl})]) / 1e6
    st.markdown("**P&L contribution by factor** ($M)")
    st.bar_chart(chart, horizontal=True, height=280)
    for msg in fa.issues:
        st.caption(f"⚠ {msg}")
    st.divider()


def render_benchmark(res):
    render_brinson(res)
    render_factor_attr(res)
    if res.factor_risk is None or res.model is None:
        st.info("Benchmark-relative risk needs the factor model — re-run with "
                "the factor model enabled.")
        return
    c1, c2 = st.columns([2, 1])
    bench_label = c1.selectbox("Benchmark", list(BENCHMARK_CHOICES))
    bench = BENCHMARK_CHOICES[bench_label]
    notional_m = c2.number_input("Benchmark notional ($M, 0 = beta-match)",
                                 min_value=0.0, value=0.0, step=5.0)
    try:
        br = active_risk(res.factor_risk, res.model, bench,
                         notional=(notional_m * 1e6) if notional_m else None)
    except Exception as exc:
        st.error(f"Could not compute active risk: {exc}")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracking error (ann.)", _m(br.tracking_error),
              help="Annualized $ volatility of the book minus the benchmark.")
    m2.metric("TE (% of notional)", f"{br.te_pct:.1%}")
    m3.metric("Beta to benchmark", f"{br.beta_to_benchmark:.2f}")
    m4.metric("Bench notional", _m(br.notional))
    st.caption(f"Book vol {_m(br.port_vol)} vs benchmark vol {_m(br.bench_vol)} · "
               f"{br.active_specific_share:.0%} of active variance is stock-specific · "
               f"default notional beta-matches the benchmark to the book's market exposure.")

    st.subheader("Active factor exposure (portfolio − benchmark)")
    ae = (br.active_exposures / 1e6)
    chart = ae["active"].rename("Active $M")
    st.bar_chart(chart, horizontal=True, height=280)
    show = ae.copy()
    for k in show.columns:
        show[k] = show[k].map(lambda v: f"{v:,.1f}")
    show.columns = [f"{c} $M" for c in show.columns]
    st.dataframe(show, width="stretch")

    st.subheader("Active risk decomposition (what drives tracking error)")
    st.caption("Each active bet's share of tracking-error variance, and its $ "
               "contribution to TE. Factor shares + stock-specific sum to 100%.")
    afc = (br.active_factor_contrib * 100)
    afc = afc[afc.abs() > 1e-9]
    decomp = afc.copy()
    decomp["Stock-specific"] = br.active_specific_share * 100
    decomp = decomp.sort_values(ascending=False)
    dc1, dc2 = st.columns([3, 2])
    with dc1:
        st.bar_chart(decomp.rename("% of active variance"), horizontal=True,
                     height=300)
    with dc2:
        tbl = pd.DataFrame({
            "Driver": decomp.index,
            "% of TE var": decomp.map(lambda x: f"{x:+.0f}%").values,
            "$ contrib to TE": [
                _m(x / 100 * br.tracking_error) for x in decomp.values],
        })
        st.dataframe(tbl, hide_index=True, width="stretch")

    pac = getattr(br, "position_active_contrib", None)
    if pac is not None and not pac.empty:
        st.markdown("**Names driving tracking error** — each holding's "
                    "contribution to TE (these sum to TE, including a benchmark "
                    "line). A negative value means the name *reduces* tracking "
                    "error — a diversifying active bet.")
        held = pac[~pac["underlying"].str.startswith("[benchmark")]
        top = pd.concat([held.head(10), held.tail(4)]).drop_duplicates("underlying")
        disp = pd.DataFrame({
            "Ticker": top["underlying"],
            "Name": top["name"].astype(str).str.slice(0, 22),
            "Sector": top["sector"].astype(str).str.slice(0, 14),
            "Exposure": top["exposure"].map(_m),
            "Contrib to TE": top["ctr_te"].map(_m),
            "% of TE": top["pct_of_te"].map(lambda x: f"{x:+.1%}"),
            "Marginal / $1M": top["mcte_per_$"].map(
                lambda v: _kd(v * 1e6) if pd.notna(v) else "—"),
        })
        st.dataframe(disp, hide_index=True, width="stretch")


def render_optimizer(res):
    if res.factor_risk is None or res.model is None:
        st.info("The optimizer needs the factor model — re-run with it enabled.")
        return
    obj = st.selectbox("Objective", OBJECTIVES)
    c1, c2, c3 = st.columns(3)
    turnover_m = c1.number_input("Max turnover ($M)", min_value=0.0, value=15.0, step=5.0)
    per_name_m = c2.number_input("Max per name ($M)", min_value=0.0, value=6.0, step=1.0)
    mkt_cap_m = c3.number_input("Max |market| exposure ($M, 0 = none)",
                                min_value=0.0, value=0.0, step=1.0)
    bench = None
    if obj == "Minimize tracking error":
        bench_label = st.selectbox("Benchmark", list(BENCHMARK_CHOICES), key="opt_bench")
        bench = BENCHMARK_CHOICES[bench_label]

    if not st.button("Optimize", type="primary"):
        return
    caps = {"Mkt-RF": mkt_cap_m * 1e6} if mkt_cap_m else None
    try:
        opt = optimize_overlay(
            res.factor_risk, res.model, res.stats, objective=obj,
            turnover_max=(turnover_m * 1e6) if turnover_m else None,
            max_per_name=(per_name_m * 1e6) if per_name_m else None,
            factor_caps=caps, benchmark_ticker=bench,
        )
    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
        return

    if not opt.success:
        st.warning("Solver did not fully satisfy the constraints — they may be "
                   "infeasible together (e.g. a tight market cap with a low "
                   "turnover cap). Loosen one and retry. Showing best effort.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted vol", _m(opt.vol_after),
              delta=_m(opt.vol_after - opt.vol_before), delta_color="inverse")
    m2.metric("Turnover", _m(opt.turnover))
    m3.metric("Trades", f"{len(opt.trades)}")
    if opt.binding:
        st.caption("Binding constraints: " + "; ".join(opt.binding))

    st.subheader("Proposed trades")
    if len(opt.trades):
        show = opt.trades.copy()
        show["notional"] = show["notional"].map(_m)
        show["shares"] = show["shares"].map(lambda v: f"{v:+,}" if v is not None else "—")
        st.dataframe(show.rename(columns={"etf": "ETF", "notional": "$ notional",
                                          "shares": "~shares"}),
                     hide_index=True, width="stretch")
    else:
        st.write("No trades — the book already meets the objective within constraints.")

    st.subheader("Factor exposure: before → after ($M)")
    ex = pd.DataFrame({"before": opt.exposures_before / 1e6,
                       "after": opt.exposures_after / 1e6})
    st.bar_chart(ex, height=280)


def _pct1(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:+.1%}"


def render_scenarios(res):
    """Named scenario library — replay historical crises against this book."""
    from riskreport import scenario_library as scl

    st.markdown("**Scenario library** — reprice *today's* book under historical "
                "crises (full option revaluation, implied vol shocked by the "
                "episode's actual VIX move) and hypothetical shocks. An "
                "instantaneous shock: time to expiry is held fixed.")
    if res.analytics is None:
        st.info("Run a report first.")
        return

    aum = res.summary.get("aum")
    key = f"scenlib_{res.analytics.asof}"
    if st.button("▶ Run scenario library", type="primary") or key in st.session_state:
        if key not in st.session_state:
            status = st.status("Fetching multi-year history and repricing…",
                               expanded=True)
            box = status.empty()
            lines = []

            def prog(m):
                lines.append(m); box.code("\n".join(lines))

            try:
                unders = sorted(res.analytics.positions["underlying"].unique())
                betas = {t: s.beta for t, s in (res.stats or {}).items()}
                closes_long = scl.fetch_long_history(
                    unders, res.analytics.asof, CACHE_DIR, log=prog)
                results = scl.run_library(
                    res.analytics.positions, closes_long, betas,
                    res.analytics.asof, aum, log=prog)
                st.session_state[key] = results
                status.update(label=f"Ran {len(results)} scenarios", state="complete",
                              expanded=False)
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(f"Scenario library failed: {exc}")
                return

        results = st.session_state[key]
        if not results:
            st.warning("No scenarios could be run (no usable history for this "
                       "book's names in the crisis windows).")
            return
        hist = [r for r in results if r.kind == "historical"]
        hypo = [r for r in results if r.kind == "hypothetical"]

        def _table(rows):
            data = []
            for r in rows:
                data.append({
                    "Scenario": r.name,
                    "Book P&L": _m(r.pnl),
                    "% AUM": _pct1(r.pnl_pct_aum),
                    "S&P move": _pct1(r.spx_move),
                    "VIX move": _pct1(r.vix_move),
                    "Coverage": f"{r.coverage:.0%}",
                })
            return pd.DataFrame(data)

        st.subheader("Historical replays")
        st.dataframe(_table(hist), hide_index=True, width="stretch")
        # P&L bar across scenarios
        pnl_series = pd.Series({r.name: r.pnl / 1e6 for r in hist})
        st.markdown("**Book P&L by scenario** ($M)")
        st.bar_chart(pnl_series, horizontal=True, height=260)

        st.subheader("Hypothetical shocks")
        st.dataframe(_table(hypo), hide_index=True, width="stretch")

        st.subheader("Worst names by scenario")
        pick = st.selectbox("Scenario", [r.name for r in results])
        chosen = next(r for r in results if r.name == pick)
        if chosen.note:
            st.caption(chosen.note)
        if chosen.n_proxied or chosen.n_missing:
            st.caption(f"{chosen.n_proxied} name(s) moved via beta (no history in "
                       f"the window); {chosen.n_missing} dropped (no history, no "
                       "beta).")
        c = chosen.contributors
        if not c.empty:
            worst = c.head(10).copy()
            best = c.tail(5).copy()
            disp = pd.concat([worst, best]).drop_duplicates("underlying")
            show = pd.DataFrame({
                "Ticker": disp["underlying"],
                "Name": disp["name"].astype(str).str.slice(0, 24),
                "Sector": disp["sector"].astype(str).str.slice(0, 16),
                "P&L": disp["pnl"].map(_m),
            })
            st.dataframe(show, hide_index=True, width="stretch")
    else:
        st.caption("Fetches several years of daily history for your names the "
                   "first time (cached afterward), so it runs on demand rather "
                   "than on every report.")


def _parse_trade_lines(text):
    """Parse 'Symbol, Qty' lines into Position trades (broker symbol format)."""
    from riskreport.parse import build_position
    trades, issues = [], []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            issues.append(f"line {i}: expected 'Symbol, Qty'")
            continue
        sym, qtxt = line.split(",", 1)
        sym, qtxt = sym.strip(), qtxt.strip().replace(",", "")
        if sym.lower() in ("symbol", "ticker"):
            continue
        try:
            qty = float(qtxt)
        except ValueError:
            issues.append(f"line {i}: unreadable quantity {qtxt!r}")
            continue
        if qty == 0:
            continue
        pos, err = build_position("WHATIF", sym, qty)
        if err:
            issues.append(f"line {i}: {err}")
            continue
        trades.append(pos)
    return trades, issues


def render_pretrade(res):
    """Pre-trade compliance & risk-impact: reprice the book with proposed
    trades and show how risk metrics and limit status would change."""
    from riskreport.whatif import apply_trades
    from riskreport.analytics import build_analytics
    from riskreport.alerts import evaluate_limits

    st.markdown("**Pre-trade compliance & risk-impact** — enter proposed trades "
                "(signed quantities, broker symbol format, options too) and see "
                "how the book's risk and limit status *would* change before you "
                "execute.")
    if (res.analytics is None or res.model is None
            or not getattr(res, "base_positions", None)):
        st.info("Run a report first (with the factor model enabled).")
        return
    st.code("# Symbol, Quantity  (+ buys, - sells/shorts)\n"
            "SPY,-2000\nMDT,1500\nAAPL  16JAN26 200 P,-25", language="text")
    txt = st.text_area("Proposed trades", height=130,
                       placeholder="SPY,-2000\nNVDA,1000")
    if not st.button("⚖ Check trades", type="primary"):
        return
    trades, issues = _parse_trade_lines(txt)
    for m in issues:
        st.caption(f"⚠ {m}")
    if not trades:
        st.warning("No valid trades parsed.")
        return

    with st.spinner("Repricing the proposed book…"):
        proposed_pos = apply_trades(res.base_positions, trades)
        asof = res.analytics.asof
        try:
            prop = build_analytics(proposed_pos, res.stats, res.profiles, {},
                                   asof=asof, cash=res.summary.get("cash"))
        except Exception as exc:
            st.error(f"Could not reprice proposed book: {exc}")
            return
        prop_fr = prop_sc = prop_cr = None
        recompute_warns = []
        try:
            from riskreport.factors import compute_factor_risk
            prop_fr = compute_factor_risk(prop.positions, res.model)
        except Exception as exc:
            recompute_warns.append(f"predicted-vol limit not re-checked: {exc}")
        try:
            from riskreport.scenarios import run_scenarios
            betas = {t: s.beta for t, s in (res.stats or {}).items()}
            prop_sc = run_scenarios(prop.positions, res.closes, betas, asof)
        except Exception as exc:
            recompute_warns.append(f"VaR limit not re-checked: {exc}")
        try:
            from riskreport.crowding import compute_crowding
            prop_cr = compute_crowding(prop.issuers, prop.positions)
        except Exception:
            prop_cr = res.crowding  # fall back to base crowding

    cfg = getattr(res, "alert_config", None)
    before = {r["key"]: r for r in evaluate_limits(
        res.analytics, res.factor_risk, res.scenarios, cfg, res.crowding)}
    after = {r["key"]: r for r in evaluate_limits(
        prop, prop_fr, prop_sc, cfg, prop_cr)}
    # a failed recompute would silently read as "not breached" — surface it so a
    # compliance clear is never based on a limit that wasn't actually evaluated
    for w in recompute_warns:
        st.warning(f"⚠ {w} — treat the verdict for that limit as unknown.")

    st.caption(f"{len(trades)} trade(s) applied · {len(proposed_pos)} positions "
               f"after. {len(prop.issues)} data note(s) on the proposed book.")

    # verdict on configured limits
    limited = [k for k in after if after[k]["limit"] is not None]
    new_breaches = [k for k in limited
                    if after[k]["breached"] and not before.get(k, {}).get("breached")]
    cured = [k for k in limited
             if before.get(k, {}).get("breached") and not after[k]["breached"]]
    if cfg is None:
        st.info("No risk-limit config loaded (upload a JSON in the sidebar to "
                "check limits) — showing risk-metric deltas only.")
    elif new_breaches:
        st.error("⛔ **Would breach:** "
                 + ", ".join(after[k]["label"] for k in new_breaches))
    else:
        st.success("✓ No new limit breaches from these trades.")
    if cured:
        st.success("✓ Would cure: "
                   + ", ".join(before[k]["label"] for k in cured))

    # limit table
    if cfg is not None and limited:
        rows = []
        for k in limited:
            a = after[k]
            b = before.get(k)
            fmt = a["fmt"]
            rows.append({
                "Check": a["label"],
                "Before": fmt(b["value"]) if b and b["value"] is not None else "—",
                "After": fmt(a["value"]) if a["value"] is not None else "—",
                "Cap": fmt(a["limit"]),
                "Status": "⛔ BREACH" if a["breached"] else "✓ OK",
            })
        st.markdown("**Limit checks**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # headline risk deltas (always)
    bs, as_ = res.analytics.summary, prop.summary
    def ng(su):
        return abs(su["exp_net"]) / (su["exp_gross"] or 1.0)
    metrics = [
        ("Gross exposure", bs["exp_gross"], as_["exp_gross"], _m),
        ("Net exposure", bs["exp_net"], as_["exp_net"], _m),
        ("Net / gross", ng(bs), ng(as_), lambda v: f"{v:.0%}"),
        ("Beta-adjusted net", bs["beta_net"], as_["beta_net"], _m),
    ]
    if res.factor_risk is not None and prop_fr is not None:
        metrics.append(("Predicted vol (ann.)", res.factor_risk.vol_total,
                        prop_fr.vol_total, _m))
    if res.scenarios is not None and prop_sc is not None:
        metrics.append(("1-day VaR 95%", res.scenarios.var_95,
                        prop_sc.var_95, _m))
    bg, ag = bs.get("greeks") or {}, as_.get("greeks") or {}
    if ag:
        metrics += [
            ("Net vega (/ +1 vol pt)", bg.get("net_vega_1pt"),
             ag.get("net_vega_1pt"), _kd),
            ("Net theta (/ day)", bg.get("net_theta_day"),
             ag.get("net_theta_day"), _kd),
        ]
    drows = []
    for label, bval, aval, fmt in metrics:
        chg = (aval - bval) if (bval is not None and aval is not None) else None
        drows.append({
            "Metric": label,
            "Before": fmt(bval) if bval is not None else "—",
            "After": fmt(aval) if aval is not None else "—",
            "Change": fmt(chg) if chg is not None else "—",
        })
    st.markdown("**Risk-metric impact**")
    st.dataframe(pd.DataFrame(drows), hide_index=True, width="stretch")


def render_fixedincome(res):
    """Interest-rate risk for the bond-ETF sleeve (duration/DV01/key-rate)."""
    from riskreport.fixedincome import compute_fi_risk, BOND_ETF, BUCKETS

    st.markdown("**Fixed-income risk** — interest-rate and credit-spread risk "
                "for the bond/rate ETFs in the book, which the equity factor "
                "model does not capture.")
    if res.analytics is None:
        st.info("Run a report first.")
        return
    fi = compute_fi_risk(res.analytics.issuers)
    if fi is None:
        st.info("No recognised fixed-income ETFs in the book. This tab covers "
                "Treasury, aggregate, IG/HY credit, TIPS, muni, MBS and EM bond "
                "ETFs — e.g. TLT, IEF, SHY, AGG, LQD, HYG, TIP, MUB, EMB.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FI market value", _m(fi.total_mv))
    m2.metric("Total DV01 (/ +1bp)", _kd(fi.total_dv01),
              help="$ P&L per +1bp parallel rate move. Negative = loses when "
                   "rates rise (net long duration).")
    m3.metric("Dollar duration (/ +1%)", _m(fi.dollar_duration),
              help="$ P&L per +100bp parallel move.")
    m4.metric("Total CS01 (/ +1bp)", _kd(fi.total_cs01),
              help="$ P&L per +1bp credit-spread widening (credit ETFs).")

    st.subheader("Key-rate DV01 (curve exposure)")
    st.caption("$ P&L per +1bp at each point on the curve — the shape of your "
               "rate exposure, so a steepener/flattener can be priced.")
    krd = pd.Series({b: fi.krd_dv01[b] for b in BUCKETS}) / 1e3
    st.bar_chart(krd.rename("DV01 $k / +1bp"), height=220)

    st.subheader("Holdings")
    h = fi.holdings.copy()
    disp = pd.DataFrame({
        "Ticker": h["underlying"],
        "Type": h["kind"].str.upper(),
        "Market value": h["mv"].map(_m),
        "Eff. duration": h["duration"].map(lambda x: f"{x:.1f}"),
        "DV01 / +1bp": h["dv01"].map(_kd),
        "CS01 / +1bp": h["cs01"].map(lambda x: _kd(x) if abs(x) > 1e-9 else "—"),
    })
    st.dataframe(disp, hide_index=True, width="stretch")

    st.subheader("Rate scenarios")
    st.caption("Book P&L under parallel shifts and curve twists (duration "
               "approximation, no convexity).")
    sc = fi.scenarios.copy()
    scd = pd.DataFrame({
        "Scenario": sc["scenario"],
        "Book P&L": sc["pnl"].map(_m),
        "% of FI MV": sc["pnl"].map(
            lambda x: f"{x/fi.total_mv:+.1%}" if fi.total_mv else "—"),
    })
    st.dataframe(scd, hide_index=True, width="stretch")
    st.bar_chart(sc.set_index("scenario")["pnl"] / 1e6, horizontal=True,
                 height=300)
    for msg in fi.issues:
        st.caption(f"⚠ {msg}")


def render_options_ladder(res):
    """Options term structure — greeks and premium by time to expiry."""
    from riskreport.options_ladder import options_ladder

    st.markdown("**Options expiry ladder** — the term structure of the option "
                "book: where premium rolls off, where theta is earned, and where "
                "the short gamma/vega sits. Near-dated options carry the sharpest "
                "gamma, so a gap into a near expiry hurts most.")
    if res.analytics is None:
        st.info("Run a report first.")
        return
    lad = options_ladder(res.analytics.positions, res.analytics.asof)
    if lad is None:
        st.info("No option positions in this book.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net premium", _m(lad.net_premium),
              help="Option market value; negative = net short premium collected.")
    m2.metric("Theta / day", _kd(lad.total_theta_day),
              help="Total time decay per calendar day (positive = collecting).")
    m3.metric("Vega / +1 vol pt", _kd(lad.total_vega_1pt))
    m4.metric("Gamma P&L / ±1%", _kd(lad.total_gamma_1pct))
    st.caption(f"**{lad.near_theta_share:.0%}** of theta and "
               f"**{lad.near_gamma_share:.0%}** of gamma risk sit in options "
               "expiring within 30 days — the near-dated income/pin-risk trade-off.")

    b = lad.by_bucket
    st.subheader("By time to expiry")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Theta / day ($k)**")
        st.bar_chart((b.set_index("bucket")["theta_day"] / 1e3), height=240)
    with cc2:
        st.markdown("**Vega per +1 vol pt ($k)**")
        st.bar_chart((b.set_index("bucket")["vega_1pt"] / 1e3), height=240)
    disp = pd.DataFrame({
        "Bucket": b["bucket"],
        "Contracts": b["n_contracts"].map(lambda x: f"{x:,.0f}"),
        "Net premium": b["net_premium"].map(_m),
        "Delta exp": b["delta_exp"].map(_m),
        "Gamma ±1%": b["gamma_1pct"].map(_kd),
        "Vega /1pt": b["vega_1pt"].map(_kd),
        "Theta /day": b["theta_day"].map(_kd),
    })
    st.dataframe(disp, hide_index=True, width="stretch")

    st.subheader("Expiration calendar (premium roll-off)")
    e = lad.by_expiry.copy()
    prem = pd.Series((e["net_premium"] / 1e3).values,
                     index=pd.to_datetime(e["expiry"]).dt.date.astype(str))
    st.markdown("**Net premium by expiry ($k)** — when the short book rolls off")
    st.bar_chart(prem, height=240)
    ecal = pd.DataFrame({
        "Expiry": pd.to_datetime(e["expiry"]).dt.date.astype(str),
        "Days": e["dte"].map(lambda x: f"{int(x)}"),
        "Contracts": e["n_contracts"].map(lambda x: f"{x:,.0f}"),
        "Net premium": e["net_premium"].map(_m),
        "Theta /day": e["theta_day"].map(_kd),
        "Vega /1pt": e["vega_1pt"].map(_kd),
    })
    st.dataframe(ecal, hide_index=True, width="stretch")


def render_macro(res):
    if res.factor_risk is None or res.closes is None:
        st.info("The macro overlay needs the factor model and price history — "
                "re-run with the factor model enabled.")
        return
    try:
        mac = compute_macro(res.analytics.positions, res.closes, res.asof)
    except Exception as exc:
        st.error(f"Macro regression failed: {exc}")
        return
    st.caption("Book P&L regressed on macro-proxy ETF returns, controlling for "
               "the equity market — each beta is the incremental $ P&L per +1% "
               f"move, holding the market fixed. {res.asof} · {mac.window}d · "
               f"R²={mac.r2:.0%}.")
    m1, m2 = st.columns(2)
    m1.metric("Market beta ($ per +1% SPY)", _m(mac.market_beta))
    m2.metric("Variance explained (mkt+macro)", f"{mac.r2:.0%}")
    show = mac.betas.copy()
    show["$ P&L per +1%"] = show["beta_per_1pct"].map(_m)
    show["t-stat"] = show["t_stat"].map(lambda v: f"{v:+.1f}")
    st.dataframe(show[["factor", "$ P&L per +1%", "t-stat"]]
                 .rename(columns={"factor": "Macro factor"}),
                 hide_index=True, width="stretch")
    chart = mac.betas.set_index("factor")["beta_per_1pct"] / 1e6
    st.markdown("**$M P&L per +1% move (net of market)**")
    st.bar_chart(chart, horizontal=True, height=240)
    st.caption("|t| ≳ 2 ≈ statistically meaningful. Macro factors use liquid "
               "ETF proxies (IEF, LQD, HYG, TIP, USO, UUP, GLD).")


def render_screener(res):
    if res.model is None:
        st.info("The screener needs the factor model — re-run with it enabled.")
        return
    frame = build_screen_frame(res.model, res.stats, res.profiles,
                               res.analytics.positions)
    fnames = res.model.factor_names
    st.caption(f"Screening {len(frame)} fitted names (your book + hedge/macro "
               "ETF candidates) by factor profile. Find hedges or replacements "
               "with a target loading.")
    c1, c2, c3 = st.columns(3)
    sort_by = c1.selectbox("Sort by", fnames + ["beta", "r2"], index=0)
    held = c2.selectbox("Holdings", ["all", "held", "not_held"])
    min_r2 = c3.slider("Min R²", 0.0, 0.8, 0.0, 0.05)
    with st.expander("Factor loading filters"):
        ranges = {}
        for f in fnames:
            lo, hi = st.slider(f, -3.0, 3.0, (-3.0, 3.0), 0.1, key=f"scr_{f}")
            if (lo, hi) != (-3.0, 3.0):
                ranges[f] = (lo, hi)
    out = screen(frame, res.model, factor_ranges=ranges or None,
                 min_r2=min_r2 or None, held=held, sort_by=sort_by,
                 ascending=False, limit=40)
    disp = out[["ticker", "name", "sector", "beta", "r2", "held"] + fnames].copy()
    # pd.notna (not `is not None`): a mixed None/float column coerces to float64,
    # turning None into NaN, which `is not None` would wrongly render as "nan".
    disp["beta"] = disp["beta"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    disp["r2"] = disp["r2"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    for f in fnames:
        disp[f] = disp[f].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
    st.dataframe(disp, hide_index=True, width="stretch")


def _llm_key():
    key = os.environ.get("LLM_API_KEY") or os.environ.get("ZHIPUAI_API_KEY") \
        or os.environ.get("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("LLM_API_KEY")
        except Exception:
            key = None
    return key


def render_themes(res):
    st.caption("Group the book by your own themes. Upload a CSV mapping tickers "
               "to themes — `Ticker,Theme` per row; a ticker may appear on "
               "several rows to sit in several themes. Exposure is delta-adjusted "
               "and overlaps across themes by design.")
    up = st.file_uploader("Theme map CSV", type=["csv"], key="tags_csv")
    with st.expander("Format example"):
        st.code("Ticker,Theme\nNVDA,AI\nNVDA,Semis\nMSFT,AI\nXOM,Energy",
                language="text")
    if up is None:
        st.info("Upload a theme map to see grouped exposure.")
        return
    try:
        with tempfile.NamedTemporaryFile(
                "wb", suffix=".csv", delete=False) as tf:
            tf.write(up.getvalue())
            tags_path = tf.name
        tags = parse_tags(tags_path)
    except Exception as exc:
        st.error(f"Could not read theme map: {exc}")
        return
    finally:
        try:
            os.unlink(tags_path)
        except (OSError, NameError):
            pass
    if not tags:
        st.warning("No ticker→theme rows found in that file.")
        return

    table, coverage = theme_exposure(res.analytics.issuers, tags)
    st.caption(f"{len(tags)} tickers tagged across {table.shape[0]} themes · "
               f"{coverage:.0%} of gross exposure is tagged.")
    if table.empty:
        st.warning("None of the tagged tickers are held in this book.")
        return

    disp = pd.DataFrame({
        "Theme": table["theme"],
        "Long": table["long"].map(_m),
        "Short": table["short"].map(_m),
        "Net": table["net"].map(_m),
        "Gross": table["gross"].map(_m),
        "Names": table["n_issuers"],
        "% Gross": table["pct_gross"].map(lambda v: f"{v:.0%}"),
    })
    st.dataframe(disp, hide_index=True, width="stretch")
    st.markdown("**Net exposure by theme ($M)**")
    st.bar_chart(table.set_index("theme")["net"] / 1e6,
                 horizontal=True, height=max(240, 32 * len(table)))


def render_ai(res):
    st.caption("An AI risk analyst reads the computed report — ask it questions "
               "or generate a commentary. Uses GLM (Zhipu / z.ai) by default. "
               "Describes risk, not investment advice.")
    key = _llm_key()
    model = os.environ.get("LLM_MODEL", "glm-4.6")
    with st.expander("LLM settings", expanded=not key):
        if not key:
            key = st.text_input("API key (LLM_API_KEY — used only this session)",
                                type="password") or None
        model = st.text_input("Model", value=model,
                              help="e.g. glm-4.6 or glm-5.2, per your z.ai account")
    if not is_available(key):
        st.info("Set an `LLM_API_KEY` (env var, secret, or above) to enable the "
                "AI features. Configure `LLM_MODEL` / `LLM_BASE_URL` for your "
                "provider (defaults to GLM on z.ai).")
        return

    facts = build_facts(res)
    reference = build_reference(res.model)

    # one-shot commentary
    if st.button("Generate risk commentary", type="primary"):
        with st.spinner("Writing…"):
            try:
                text = generate_narrative(facts, api_key=key, model=model)
                st.session_state["ai_summary"] = text
            except Exception as exc:
                st.error(f"Failed: {exc}")
    if st.session_state.get("ai_summary"):
        st.markdown(st.session_state["ai_summary"])

    st.divider()
    st.markdown("**Ask about the book** — e.g. *how would I lower the net short "
                "momentum exposure?*")
    hist = st.session_state.setdefault("ai_chat", [])
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask a risk question…")
    if q:
        hist.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    ans = ai_chat(hist, facts, reference, api_key=key, model=model)
                except Exception as exc:
                    ans = f"⚠ {exc}"
            st.markdown(ans)
        hist.append({"role": "assistant", "content": ans})


# =====================================================================
if result is None:
    st.info("Upload a position CSV in the sidebar and click **Generate report**.")
    st.stop()

(tab_report, tab_trends, tab_scen, tab_pretrade, tab_bench, tab_fi, tab_optladder,
 tab_opt, tab_macro, tab_screen, tab_themes, tab_narr) = st.tabs(
    ["📄 Report", "📈 Trends", "🌩 Scenarios", "⚖ Pre-trade", "🎯 Benchmark",
     "🏦 Fixed Income", "🗓 Options", "🛠 Optimizer", "📉 Macro", "🔎 Screener",
     "🏷 Themes", "🤖 AI"])
with tab_report:
    render_report(result)
with tab_trends:
    render_trends()
with tab_scen:
    render_scenarios(result)
with tab_pretrade:
    render_pretrade(result)
with tab_bench:
    render_benchmark(result)
with tab_fi:
    render_fixedincome(result)
with tab_optladder:
    render_options_ladder(result)
with tab_opt:
    render_optimizer(result)
with tab_macro:
    render_macro(result)
with tab_screen:
    render_screener(result)
with tab_themes:
    render_themes(result)
with tab_narr:
    render_ai(result)
