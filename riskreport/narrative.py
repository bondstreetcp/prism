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
    "sits; notable tail/scenario, liquidity, or crowding/squeeze exposures and any "
    "limit breaches; and one or two concrete things to watch. Reference the actual "
    "numbers. Be specific, not generic. Do not invent data. Describe risk and "
    "positioning; do not give buy/sell investment advice. Write prose, 3-5 short "
    "paragraphs."
)

CHAT_SYSTEM_PROMPT = (
    "You are a buy-side risk analyst chatting with the PM about their book. You "
    "have a JSON risk snapshot and a reference table of factor loadings for a menu "
    "of liquid hedge ETFs. Answer the PM's questions grounded in these numbers. "
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
         model=None, base_url=None, max_tokens: int = 1500) -> str:
    """One assistant turn given the running chat `history` (role/content dicts)."""
    import json
    context = (CHAT_SYSTEM_PROMPT + "\n\nRISK SNAPSHOT:\n" + json.dumps(facts, indent=1)
               + "\n\nFACTOR REFERENCE:\n" + json.dumps(reference, indent=1))
    messages = [{"role": "system", "content": context}] + history
    return _complete(messages, api_key, model, base_url, max_tokens, temperature=0.4)
