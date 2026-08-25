"""Concentration & diversification analytics.

Answers the Aladdin/PORT question "is my risk actually concentrated, or am I
genuinely diversified?" — separate from raw position count. Built from the
factor-model risk decomposition the tool already computes:

  * Effective number of bets (risk)  — 1 / Σ (risk-contribution share)²; the
    count of *independent* risk bets your book really behaves like. A 300-name
    book whose risk sits in 8 correlated themes has ~8 effective bets.
  * Effective number of bets (exposure) — the same Herfindahl on gross dollar
    exposure, for contrast (names, not risk).
  * Diversification ratio — Σ standalone name vol / portfolio vol. >1 means the
    correlations are working for you; near 1 means little diversification.
  * Risk Herfindahl and the top-5 share of risk — how top-heavy the book is.

Standalone name vol uses the same factor covariance and residual vols as the
predicted-vol engine, so the diversification ratio is internally consistent.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def compute_concentration(factor_risk, model) -> dict | None:
    """Concentration/diversification metrics from the factor risk decomposition.

    Returns None when there is no usable factor risk (model off or zero vol).
    """
    pr = getattr(factor_risk, "position_risk", None)
    if pr is None or len(pr) == 0 or getattr(factor_risk, "vol_total", 0) <= 0:
        return None

    fn = model.factor_names
    F = model.fcov.loc[fn, fn].to_numpy()
    g = pr.groupby("underlying").agg(exposure=("exposure", "sum"),
                                     rc=("risk_contrib", "sum"))
    names = list(g.index)
    x = g["exposure"].to_numpy()
    B = model.loadings.loc[names, fn].to_numpy()
    s = model.resid_vol.reindex(names).fillna(0.0).to_numpy()

    # standalone annualized $ vol of each name (factor + specific), same math as
    # the predicted-vol engine but per single name
    name_var = (x ** 2) * (np.einsum("ij,jk,ik->i", B, F, B) + s ** 2) * TRADING_DAYS
    name_vol = np.sqrt(np.clip(name_var, 0.0, None))
    sum_standalone = float(name_vol.sum())
    div_ratio = sum_standalone / factor_risk.vol_total if factor_risk.vol_total else 0.0

    # risk-contribution shares (sum to 1; a hedge can be slightly negative).
    # Concentration measures use the positive part, renormalized.
    rc = g["rc"].to_numpy()
    pos = np.clip(rc, 0.0, None)
    ps = pos / pos.sum() if pos.sum() > 0 else pos
    herf_risk = float((ps ** 2).sum())
    enb_risk = 1.0 / herf_risk if herf_risk > 0 else float(len(names))
    top5_risk = float(np.sort(ps)[::-1][:5].sum())

    # exposure concentration (gross $), for contrast with the risk view
    ax = np.abs(x)
    axs = ax / ax.sum() if ax.sum() > 0 else ax
    herf_exp = float((axs ** 2).sum())
    enb_exp = 1.0 / herf_exp if herf_exp > 0 else float(len(names))

    return {
        "n_issuers": len(names),
        "effective_bets_risk": enb_risk,
        "effective_bets_exposure": enb_exp,
        "diversification_ratio": div_ratio,
        "herfindahl_risk": herf_risk,
        "top5_risk_share": top5_risk,
    }
