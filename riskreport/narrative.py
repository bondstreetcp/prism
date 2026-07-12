"""AI risk narrative — Omega Point's 'AI Teammate', free-data version.

Feeds the report's structured facts to Claude and gets back a concise,
plain-English risk commentary. Uses the official Anthropic SDK; defaults to
claude-opus-4-8. Requires an ANTHROPIC_API_KEY (passed in or from the
environment) — without one the feature reports itself unavailable and the
rest of the tool is unaffected.

A single narrative is a short single-shot call: ~2K input + ~800 output
tokens, roughly $0.02-0.03 on Opus 4.8.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = os.environ.get("RISK_NARRATIVE_MODEL", "claude-opus-4-8")

SYSTEM_PROMPT = (
    "You are a buy-side risk analyst writing a short internal risk commentary "
    "on a portfolio for the portfolio manager. You are given a JSON snapshot of "
    "the book's computed risk metrics. Write 3-5 tight paragraphs in plain "
    "English that a PM would actually read:\n"
    "1. The headline positioning (net/gross, market direction, biggest factor "
    "and sector tilts).\n"
    "2. What is driving predicted risk and where the concentration sits.\n"
    "3. Notable tail/scenario, liquidity, or crowding/squeeze exposures worth "
    "watching, and any limit breaches.\n"
    "4. One or two concrete, specific things to keep an eye on.\n\n"
    "Rules: reference the actual numbers from the JSON (dollar amounts in $M, "
    "percentages). Be specific, not generic. Do not invent data not present. "
    "Do not give buy/sell recommendations or investment advice — describe risk. "
    "No preamble, headings optional, no bullet-point dump — write prose."
)


def is_available(api_key: str | None = None) -> bool:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    return bool(key)


def build_facts(result, benchmark=None, macro=None) -> dict:
    """Assemble a compact, model-friendly facts dict from a ReportResult."""
    s = result.summary
    facts: dict = {
        "as_of": str(result.asof),
        "name": result.name,
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
        facts["crowding"] = {
            "short_book_avg_short_pct_float": round(cr.wavg_si_float_short, 3) if cr.wavg_si_float_short else None,
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
        facts["macro_betas_$per_1pct"] = {
            r["factor"]: round(r["beta_per_1pct"], 0)
            for _, r in macro.betas.iterrows()
        }
    if result.alert_hits:
        facts["limit_breaches"] = result.alert_hits
    return facts


def generate_narrative(
    facts: dict, api_key: str | None = None, model: str | None = None,
    max_tokens: int = 1200,
) -> str:
    """Return the risk commentary text. Raises RuntimeError if unavailable."""
    import json

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY configured.")
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(f"anthropic SDK not installed: {exc}")

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": "Here is the portfolio risk snapshot:\n\n"
                       + json.dumps(facts, indent=1)
                       + "\n\nWrite the risk commentary.",
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
