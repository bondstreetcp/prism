"""Liquidation cost & liquidity-adjusted VaR (LVaR).

Standard VaR assumes you can exit at the mid price instantly. In a stress you
can't — unwinding a large position moves the market against you. This estimates
that cost with a square-root market-impact model (Almgren-style) and adds it to
VaR to give a liquidity-adjusted VaR:

    impact_i (return)  = k · σ_daily,i · sqrt(|shares_i| / ADV_i)
    liquidation cost_i = impact_i · |delta-adjusted notional_i|
    LVaR               = VaR95 + Σ_i liquidation cost_i

σ is the name's realized daily vol and ADV its 60-day average volume — both
already computed. `k` is the impact coefficient (~0.5). Names without a usable
vol or ADV are excluded and their weight disclosed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class LiquidationResult:
    by_name: pd.DataFrame        # per issuer: exposure, days_to_liq, impact_bps, cost
    total_cost: float            # $ cost to liquidate the book
    cost_pct_gross: float
    var_95: float | None
    lvar: float | None           # VaR95 + total liquidation cost
    days_to_liq_p95: float | None
    coverage: float              # share of gross exposure with a cost estimate
    impact_k: float
    issues: list = field(default_factory=list)


def liquidation_analysis(
    issuers: pd.DataFrame, stats: dict, var_95: float | None = None,
    days_to_liq_p95: float | None = None, impact_k: float = 0.5,
) -> LiquidationResult | None:
    """Per-name liquidation cost and liquidity-adjusted VaR."""
    if issuers is None or issuers.empty:
        return None
    rows = []
    for _, r in issuers.iterrows():
        u = r["underlying"]
        expo = abs(float(r.get("exposure", 0.0)))
        adv = r.get("adv_shares")
        shares = abs(float(r.get("net_shares") or 0.0))
        st = stats.get(u) if stats else None
        sig_ann = getattr(st, "realized_vol", None) if st else None
        cost = impact = None
        if adv and adv > 0 and sig_ann and shares > 0 and expo > 0:
            sig_d = sig_ann / np.sqrt(TRADING_DAYS)
            impact = impact_k * sig_d * np.sqrt(shares / adv)   # return terms
            cost = impact * expo
        rows.append({
            "underlying": u, "name": r.get("name", u), "sector": r.get("sector"),
            "exposure": float(r.get("exposure", 0.0)),
            "days_to_liq": r.get("days_to_liq"),
            "impact_bps": (impact * 1e4) if impact is not None else np.nan,
            "liq_cost": cost if cost is not None else np.nan,
        })
    by_name = pd.DataFrame(rows).sort_values(
        "liq_cost", ascending=False, na_position="last").reset_index(drop=True)

    gross = float(issuers["exposure"].abs().sum()) or 1.0
    total_cost = float(by_name["liq_cost"].dropna().sum())
    covered = float(by_name.loc[by_name["liq_cost"].notna(), "exposure"]
                    .abs().sum())
    coverage = covered / gross
    lvar = (var_95 + total_cost) if var_95 is not None else None

    issues = []
    if coverage < 0.9:
        issues.append(f"Liquidation cost covers {coverage:.0%} of gross exposure "
                      "(names without a usable vol/ADV are excluded).")

    return LiquidationResult(
        by_name=by_name, total_cost=total_cost,
        cost_pct_gross=total_cost / gross, var_95=var_95, lvar=lvar,
        days_to_liq_p95=days_to_liq_p95, coverage=coverage, impact_k=impact_k,
        issues=issues,
    )
