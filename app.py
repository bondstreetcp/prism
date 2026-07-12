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
from riskreport.screener import build_screen_frame, screen
from riskreport.tags import parse_tags, theme_exposure
from riskreport.trends import TREND_METRICS, load_trend_series

st.set_page_config(page_title="Portfolio Risk", page_icon="📊", layout="wide")

CACHE_DIR = os.environ.get("RISK_CACHE_DIR", "cache")
OUT_DIR = os.environ.get("RISK_OUT_DIR", "reports")
SNAP_DIR = os.environ.get("RISK_SNAP_DIR", "snapshots")


def _m(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    v = v / 1e6
    return f"(${abs(v):,.1f}M)" if v < 0 else f"${v:,.1f}M"


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
    st.title("📊 Portfolio Risk")
    pw = st.text_input("Password", type="password")
    if pw and pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    return False


if not _gate():
    st.stop()

st.title("📊 Portfolio Risk")
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
    cash_m = st.number_input("Cash ($M)", value=0.0, step=1.0,
                             help="AUM = net market value + cash; % columns are "
                                  "% of AUM. IBKR files carry cash automatically; "
                                  "for Goldman, enter it here (0 = none/unknown).")
    asof_override = st.date_input("As-of date (optional)", value=None)
    with_factors = st.toggle("Factor model, stress & VaR", value=True)
    with_hedge = st.toggle("Hedge suggestion", value=True, disabled=not with_factors)
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
            out_dir=OUT_DIR, cache_dir=CACHE_DIR, alerts_path=alerts_path,
            no_factors=not with_factors, no_hedge=not with_hedge, progress=prog,
        )
        status.update(label=f"Done in {res.elapsed_s:.0f}s", state="complete",
                      expanded=False)
        st.session_state["result"] = res
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.error(f"Report generation failed: {exc}")


if run:
    _do_run()

result = st.session_state.get("result")


# =====================================================================
# Tab renderers
# =====================================================================
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


def render_benchmark(res):
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
        disp[f] = disp[f].map(lambda v: f"{v:+.2f}")
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

(tab_report, tab_trends, tab_bench, tab_opt, tab_macro, tab_screen,
 tab_themes, tab_narr) = st.tabs(
    ["📄 Report", "📈 Trends", "🎯 Benchmark", "🛠 Optimizer", "📉 Macro",
     "🔎 Screener", "🏷 Themes", "🤖 AI"])
with tab_report:
    render_report(result)
with tab_trends:
    render_trends()
with tab_bench:
    render_benchmark(result)
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
