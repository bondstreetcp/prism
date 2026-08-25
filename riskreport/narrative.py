"""AI risk narrative & chat — Omega Point's 'AI Teammate', free-data version.

Feeds the report's structured facts to an LLM and gets back a plain-English
risk commentary, plus an interactive chat you can ask things like "how would I
lower the net short momentum exposure?".

Provider is GLM (Zhipu / z.ai) by default, via its OpenAI-compatible API — so
this uses the `openai` SDK pointed at the GLM endpoint. Everything is
configurable by env var / secret so you can point it at any OpenAI-compatible
model:
    LLM_API_KEY   (or ZHIPUAI_API_KEY / OPENAI_API_KEY)
    LLM_BASE_URL  (default https://api.z.ai/api/paas/v4 — z.ai international;
                   China platform: https://open.bigmodel.cn/api/paas/v4)
    LLM_MODEL     (default glm-4.6; set to glm-5.2 or whatever your account
                   exposes)

Without a key the feature reports itself unavailable and the rest of the tool
is unaffected.
"""

from __future__ import annotations

import os

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.z.ai/api/paas/v4")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "glm-4.6")

SYSTEM_PROMPT = (
    "You are a buy-side risk analyst embedded in a portfolio risk tool. You are "
    "given a JSON snapshot of a book's computed risk metrics (all dollar figures "
    "in $M unless a key says otherwise). Write a tight risk commentary a PM would "
    "actually read: headline positioning (net/gross, market direction, biggest "
    "factor and sector tilts); what drives predicted risk and where concentration "
    "sits; how realized risk compares to predicted (trailing-1y realized vol vs "
    "predicted vol, max drawdown, realized vs model VaR); concentration "
    "(effective number of risk bets vs issuers held, "
    "diversification ratio, top-5 share of risk); the option-greek profile "
    "(net gamma/vega/theta — a short-premium book is short gamma, short vega, "
    "long theta), the options term structure (how much theta income and gamma "
    "risk sit in near-dated <30d expiries), and how much implied-vol moves add to "
    "VaR (the vol add-on vs spot-only) plus the Monte Carlo VaR cross-check; which "
    "names drive the expected tail loss; active risk "
    "vs the S&P (tracking error, which factor bets and which names drive it); "
    "what attribution says drove return — sector "
    "allocation vs selection (Brinson) and which factor bets paid vs stock-"
    "specific (factor attribution); crisis-scenario exposures; interest-rate "
    "risk of any bond-ETF sleeve (DV01, dollar duration, +100bp P&L); macro "
    "sensitivities (rates/credit/oil/USD/gold betas); and "
    "notable liquidity (days-to-liquidate, estimated liquidation cost and the "
    "liquidity-adjusted VaR) or crowding/squeeze exposures and any limit "
    "breaches; then "
    "one or two concrete things to watch. Reference the actual numbers. Be "
    "specific, not generic. Do not invent data. Describe risk and positioning; do "
    "not give buy/sell investment advice. Write prose, 3-5 short paragraphs."
)

CHAT_SYSTEM_PROMPT = (
    "You are a buy-side risk analyst chatting with the PM about their book. You "
    "have a JSON risk snapshot and a reference table of factor loadings for a menu "
    "of liquid hedge ETFs. The snapshot may include option greeks (net gamma/vega/"
    "theta), vol-aware VaR (with the vol add-on vs spot-only), component VaR (each "
    "name's share of expected tail loss), Brinson performance attribution "
    "(allocation vs selection vs the S&P), factor-based return attribution "
    "(factor P&L vs stock-specific), active risk vs the S&P (tracking error and "
    "its drivers), Monte Carlo VaR, fixed-income rate risk (DV01/duration), and "
    "macro sensitivities (rates/credit/oil/USD/gold betas), and crisis-scenario "
    "P&L. Use them when relevant. Answer the PM's questions grounded in these "
    "numbers. "
    "When they ask how to change an exposure (e.g. reduce net short momentum), "
    "reason from the factor loadings: name specific instruments and rough dollar "
    "sizes that would move the exposure the right way, and point out side effects "
    "on other factors. Suggest they use the tool's Optimizer or What-If to size it "
    "precisely. Be concrete and quantitative, cite the snapshot's numbers, and do "
    "not invent data. Frame everything as risk/exposure management, not investment "
    "advice. Keep answers focused and not longer than needed."
)


def _resolve_key(api_key: str | None) -> str | None:
    return (api_key or os.environ.get("LLM_API_KEY")
            or os.environ.get("ZHIPUAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY"))


def is_available(api_key: str | None = None) -> bool:
    return bool(_resolve_key(api_key))


def _client(api_key: str | None, base_url: str | None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(f"openai SDK not installed: {exc}")
    key = _resolve_key(api_key)
    if not key:
        raise RuntimeError("No LLM API key configured (LLM_API_KEY).")
    return OpenAI(api_key=key, base_url=base_url or DEFAULT_BASE_URL)


def _complete(messages, api_key, model, base_url, max_tokens, temperature=0.4):
    client = _client(api_key, base_url)
    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# ----------------------------------------------------------------------
# Facts and reference the LLM sees
# ----------------------------------------------------------------------
def build_facts(result, benchmark=None, macro=None) -> dict:
    """Compact, model-friendly facts dict from a ReportResult."""
    # the pipeline can attach the macro overlay to the result; use it when the
    # caller didn't pass one so the AI sees macro betas even in the app path
    if macro is None:
        macro = getattr(result, "macro", None)
    s = result.summary
    facts: dict = {
        "as_of": str(result.asof),
        "name": result.name,
        "aum_$M": round(s["aum"] / 1e6, 1) if s.get("aum") else None,
        "cash_$M": round(s["cash"] / 1e6, 1) if s.get("cash") is not None else None,
        "exposure_$M": {
            "long": round(s["exp_long"] / 1e6, 1),
            "short": round(s["exp_short"] / 1e6, 1),
            "gross": round(s["exp_gross"] / 1e6, 1),
            "net": round(s["exp_net"] / 1e6, 1),
            "beta_adj_net": round(s["beta_net"] / 1e6, 1),
        },
        "instruments": s["n_instruments"],
        "issuers": s["n_issuers"],
        "options_delta_adj_net_$M": round(s["opt_exp_net"] / 1e6, 1),
    }
    risk = s.get("risk") or {}
    if risk.get("vol_total") is not None:
        facts["predicted_vol_ann_$M"] = round(risk["vol_total"] / 1e6, 2)
        facts["factor_share_of_variance"] = round(risk.get("factor_var_share", 0), 2)
        facts["factor_exposures_net_$M"] = {
            k: round(v / 1e6, 1) for k, v in (risk.get("factor_exposures_net") or {}).items()
        }
    if risk.get("var_95") is not None:
        facts["var_1d_95_$M"] = round(risk["var_95"] / 1e6, 2)
        facts["var_1d_99_$M"] = round(risk["var_99"] / 1e6, 2)
        # vol-aware VaR detail: how much the historical vol spike adds
        if risk.get("var_95_spot") is not None:
            facts["var_1d_95_spot_only_$M"] = round(risk["var_95_spot"] / 1e6, 2)
            facts["var_vol_addon_$M"] = round(
                (risk["var_95"] - risk["var_95_spot"]) / 1e6, 2)
            facts["var_is_vol_aware"] = bool(risk.get("vol_aware"))
    # option greeks (dollar terms) — a short-premium book reads short gamma,
    # short vega, long theta
    if risk.get("net_vega_1pt") is not None:
        facts["option_greeks"] = {
            "net_delta_$M": round(risk.get("net_delta", 0) / 1e6, 1),
            "net_gamma_pnl_per_1pct_$K": round(risk.get("net_gamma_1pct", 0) / 1e3, 1),
            "net_vega_per_+1volpt_$K": round(risk.get("net_vega_1pt", 0) / 1e3, 1),
            "net_theta_per_day_$K": round(risk.get("net_theta_day", 0) / 1e3, 1),
        }
    # component VaR: which names drive the expected tail loss (share of ES95)
    if risk.get("top_contributors"):
        facts["top_tail_risk_contributors_pct_of_ES95"] = {
            c["ticker"]: round(c["pct"], 3) for c in risk["top_contributors"][:6]
        }
    for tbl, key in [("sector_table", "by_sector_net_$M"),
                     ("cap_table", "by_market_cap_net_$M"),
                     ("region_table", "by_region_net_$M")]:
        t = getattr(result.analytics, tbl, None)
        if t is not None and len(t):
            col = t.columns[0]
            facts[key] = {str(r[col]): round(r["net"] / 1e6, 1)
                          for _, r in t.head(8).iterrows()}
    liq = s.get("liquidity") or {}
    if liq:
        facts["liquidity"] = {
            "pct_gross_over_50pct_adv": round(liq.get("pct_gross_over_50adv", 0), 3),
            "days_to_liquidate_p95": liq.get("days_to_liq_p95"),
        }
    if result.crowding is not None:
        cr = result.crowding
        si = cr.wavg_si_float_short
        facts["crowding"] = {
            "short_book_avg_short_pct_float": round(si, 3) if si is not None else None,
            "num_squeeze_risk_names": int(len(cr.squeeze_names)),
            "short_exposure_in_crowded_names_$M": round(cr.n_crowded_short_exposure / 1e6, 1),
        }
    if result.hedge is not None and len(result.hedge.trades):
        facts["suggested_hedge"] = {
            "vol_before_$M": round(result.hedge.vol_before / 1e6, 2),
            "vol_after_$M": round(result.hedge.vol_after / 1e6, 2),
            "basket": [f"{r['etf']} {r['notional']/1e6:+.1f}M"
                       for _, r in result.hedge.trades.iterrows()],
        }
    if benchmark is not None:
        facts["benchmark_relative"] = {
            "benchmark": benchmark.benchmark,
            "tracking_error_$M": round(benchmark.tracking_error / 1e6, 2),
            "beta_to_benchmark": round(benchmark.beta_to_benchmark, 2),
        }
    if macro is not None:
        # macro betas are $ P&L per +1% move; report in $K to keep units clear
        facts["macro_betas_$K_per_1pct_move"] = {
            r["factor"]: round(r["beta_per_1pct"] / 1e3, 1)
            for _, r in macro.betas.iterrows()
        }
    # Brinson performance attribution vs the S&P 500 (from the pipeline's 3M run)
    br = getattr(result, "brinson", None)
    if br is not None:
        top = br.table.reindex(
            br.table["total"].abs().sort_values(ascending=False).index).head(5)
        facts["performance_attribution_vs_sp500"] = {
            "window": f"{br.start}..{br.end}",
            "active_return_pct": round(br.active * 100, 2),
            "allocation_pct": round(br.allocation * 100, 2),
            "selection_pct": round(br.selection * 100, 2),
            "interaction_pct": round(br.interaction * 100, 2),
            "top_sector_contributions_pct": {
                str(r.sector): round(r.total * 100, 2) for r in top.itertuples()
            },
        }
    # factor-based (Barra) return attribution
    fa = getattr(result, "factor_attr", None)
    if fa is not None:
        top = fa.table.reindex(
            fa.table["pnl"].abs().sort_values(ascending=False).index).head(6)
        facts["factor_return_attribution"] = {
            "window": f"{fa.start}..{fa.end}",
            "realized_pnl_$M": round(fa.realized_pnl / 1e6, 2),
            "factor_pnl_$M": round(fa.factor_pnl / 1e6, 2),
            "specific_pnl_$M": round(fa.specific_pnl / 1e6, 2),
            "top_factor_pnl_$M": {str(r.factor): round(r.pnl / 1e6, 2)
                                  for r in top.itertuples()},
        }
    # realized risk & drawdown (current-holdings backtest, trailing ~1y)
    ps = getattr(result, "perf_stats", None)
    if ps is not None:
        facts["realized_backtest_1y"] = {
            "realized_vol_ann_$M": round(ps.ann_vol / 1e6, 2),
            "realized_vol_pct": round(ps.ann_vol_pct, 3),
            "max_drawdown_pct": round(ps.max_drawdown_pct, 3),
            "realized_1d_var95_$M": round(ps.realized_var95 / 1e6, 2),
            "window_return_pct": round(ps.window_return_pct, 3),
        }
    # liquidation cost & liquidity-adjusted VaR
    lq = getattr(result, "liquidity_cost", None)
    if lq is not None:
        facts["liquidity"] = facts.get("liquidity") or {}
        facts["liquidity"].update({
            "est_liquidation_cost_$M": round(lq.total_cost / 1e6, 2),
            "liquidation_cost_pct_gross": round(lq.cost_pct_gross, 4),
            "liquidity_adjusted_var95_$M": (round(lq.lvar / 1e6, 2)
                                            if lq.lvar is not None else None),
        })
    # concentration / diversification
    con = getattr(result, "concentration", None)
    if con is not None:
        facts["concentration"] = {
            "issuers_held": con["n_issuers"],
            "effective_risk_bets": round(con["effective_bets_risk"], 1),
            "diversification_ratio": round(con["diversification_ratio"], 2),
            "top5_share_of_risk": round(con["top5_risk_share"], 3),
        }
    # options term structure (expiry / theta ladder)
    lad = getattr(result, "options_ladder", None)
    if lad is not None:
        facts["options_term_structure"] = {
            "net_premium_$M": round(lad.net_premium / 1e6, 2),
            "theta_per_day_$K": round(lad.total_theta_day / 1e3, 1),
            "vega_per_+1volpt_$K": round(lad.total_vega_1pt / 1e3, 1),
            "pct_theta_within_30d": round(lad.near_theta_share, 2),
            "pct_gamma_risk_within_30d": round(lad.near_gamma_share, 2),
        }
    # Monte Carlo VaR (parametric, factor model)
    mc = getattr(result, "mc_var", None)
    if mc is not None:
        facts["monte_carlo_var"] = {
            "n_sims": mc.n_sims,
            "var_1d_95_$M": round(mc.var_95 / 1e6, 2),
            "var_1d_99_$M": round(mc.var_99 / 1e6, 2),
            "es_95_$M": round(mc.es_95 / 1e6, 2),
            "vol_addon_$M": round((mc.var_95 - mc.var_95_spot) / 1e6, 2),
        }
    # active risk vs the S&P 500 (tracking error + its factor / name drivers)
    br = getattr(result, "benchmark_risk", None)
    if br is not None:
        afc = br.active_factor_contrib
        top_fac = afc.reindex(afc.abs().sort_values(ascending=False).index).head(5)
        blk = {
            "tracking_error_$M": round(br.tracking_error / 1e6, 2),
            "te_pct_of_notional": round(br.te_pct, 4),
            "beta_to_benchmark": round(br.beta_to_benchmark, 2),
            "active_specific_share": round(br.active_specific_share, 2),
            "top_active_factor_contrib_pct_of_var": {
                str(k): round(float(v) * 100, 1) for k, v in top_fac.items()},
        }
        pac = getattr(br, "position_active_contrib", None)
        if pac is not None and not pac.empty:
            held = pac[~pac["underlying"].astype(str).str.startswith("[benchmark")]
            blk["top_names_driving_TE_pct"] = {
                str(r.underlying): round(float(r.pct_of_te) * 100, 1)
                for r in held.head(6).itertuples()}
        facts["active_risk_vs_sp500"] = blk
    # fixed-income (interest-rate) risk of the bond-ETF sleeve
    fi = getattr(result, "fi_risk", None)
    if fi is not None:
        try:
            p100 = float(fi.scenarios.set_index("scenario")
                         .loc["+100bp parallel", "pnl"])
        except Exception:
            p100 = None
        facts["fixed_income_rate_risk"] = {
            "fi_market_value_$M": round(fi.total_mv / 1e6, 1),
            "dv01_per_1bp_$K": round(fi.total_dv01 / 1e3, 1),
            "dollar_duration_per_1pct_$M": round(fi.dollar_duration / 1e6, 2),
            "cs01_per_1bp_$K": round(fi.total_cs01 / 1e3, 1),
            "pnl_up_100bp_$M": round(p100 / 1e6, 2) if p100 is not None else None,
        }
    # crisis-scenario replays (only when the run included them)
    lib = getattr(result, "scenario_lib", None)
    if lib:
        facts["crisis_scenario_book_pnl_pct_aum"] = {
            r.name: (round(r.pnl_pct_aum * 100, 1)
                     if r.pnl_pct_aum is not None else None)
            for r in lib
        }
    if result.alert_hits:
        facts["limit_breaches"] = result.alert_hits
    return facts


def build_reference(model) -> dict:
    """Hedge-ETF factor loadings so the chat can suggest concrete instruments."""
    if model is None:
        return {}
    from .hedge import HEDGE_MENU
    fn = model.factor_names
    ref = {}
    for etf in HEDGE_MENU:
        if etf in model.loadings.index:
            ref[etf] = {f: round(float(model.loadings.loc[etf, f]), 2) for f in fn}
    return {"hedge_etf_factor_loadings": ref, "factor_names": fn}


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------
def generate_narrative(facts: dict, api_key=None, model=None, base_url=None,
                       max_tokens: int = 1200) -> str:
    import json
    return _complete(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Portfolio risk snapshot:\n\n"
          + json.dumps(facts, indent=1) + "\n\nWrite the risk commentary."}],
        api_key, model, base_url, max_tokens, temperature=0.4,
    )


def chat(history: list[dict], facts: dict, reference: dict, api_key=None,
         model=None, base_url=None, max_tokens: int = 1500,
         max_turns: int = 16) -> str:
    """One assistant turn given the running chat `history` (role/content dicts).

    Only the last `max_turns` messages are sent so a long session can't grow
    the request unbounded (the risk snapshot below already carries the state)."""
    import json
    context = (CHAT_SYSTEM_PROMPT + "\n\nRISK SNAPSHOT:\n" + json.dumps(facts, indent=1)
               + "\n\nFACTOR REFERENCE:\n" + json.dumps(reference, indent=1))
    messages = [{"role": "system", "content": context}] + history[-max_turns:]
    return _complete(messages, api_key, model, base_url, max_tokens, temperature=0.4)
