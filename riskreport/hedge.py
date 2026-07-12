"""Hedge-basket suggestion — a simplified "Smart Trade".

Given the book's net dollar factor exposures and a menu of liquid hedge
instruments (broad-market and style/sector ETFs), solve for the dollar
notional in each hedge that minimizes residual factor risk:

    min_h  (x + H h)' F (x + H h) + lambda * ||h||^2 (ridge, keeps it sparse-ish)

where x is the book's net factor-exposure vector, H the hedge instruments'
factor-exposure-per-dollar matrix (their own loadings), F the factor
covariance, and h the hedge dollar notionals. A no-short-the-hedge sign
convention is NOT imposed; negative h means short that ETF.

The result is reported as a trade list (ETF, $ notional, ~shares) plus the
predicted-vol reduction, so it can be dropped straight into the what-if tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .factors import TRADING_DAYS, FactorModel

# liquid, cheap-to-trade hedge menu: broad market + style + a few sectors
HEDGE_MENU = [
    "SPY", "IWM", "MDY", "QQQ",       # market / size
    "IWD", "IWF",                      # value / growth
    "MTUM", "USMV", "QUAL",            # momentum / low-vol / quality
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU",  # sectors
]
RIDGE_LAMBDA = 1e-3


@dataclass
class HedgeSuggestion:
    trades: pd.DataFrame  # etf, notional, shares, factor exposure removed
    vol_before: float
    vol_after: float
    residual_before: pd.Series  # net factor exposure before hedge
    residual_after: pd.Series
    menu_available: list[str]
    issues: list[str] = field(default_factory=list)


def suggest_hedge(
    net_factor_exposure: pd.Series,
    model: FactorModel,
    stats: dict,
    specific_var: float = 0.0,
    max_names: int = 6,
) -> HedgeSuggestion:
    """Solve for the ETF basket that best neutralizes factor risk.

    net_factor_exposure: book net $ exposure per factor (index=FACTORS).
    model: fitted FactorModel (must contain loadings for the hedge ETFs).
    stats: {ticker: TickerStats} for share-count conversion.
    specific_var: annualized book specific variance, added back so the
                  reported vol_after is apples-to-apples with the risk page.
    """
    issues: list[str] = []
    available = [t for t in HEDGE_MENU if t in model.loadings.index]
    missing = [t for t in HEDGE_MENU if t not in model.loadings.index]
    if missing:
        issues.append(
            f"{len(missing)} hedge ETF(s) lacked loadings and were dropped: "
            + ", ".join(missing)
        )
    if not available:
        raise ValueError("No hedge instruments had factor loadings available.")

    fnames = model.factor_names
    x = net_factor_exposure.reindex(fnames).fillna(0.0).to_numpy()
    F = model.fcov.reindex(index=fnames, columns=fnames).to_numpy()
    # H: columns are per-$ factor exposure of each hedge (its own loadings)
    H = model.loadings.loc[available, fnames].to_numpy().T  # k x m
    # each hedge ETF's own daily residual (idiosyncratic) vol
    s_hedge = model.resid_vol.reindex(available).fillna(0.0).to_numpy()

    def solve(cols: list[int]) -> np.ndarray:
        """Ridge solve restricted to the given hedge columns; residual
        idiosyncratic variance of the shorted ETFs is part of the objective
        (adds lambda-like ridge on the diagonal) so the basket does not lean
        on a name purely for its idiosyncratic offset."""
        Hc = H[:, cols]
        sc = s_hedge[cols]
        A = Hc.T @ F @ Hc + np.diag(sc**2) + RIDGE_LAMBDA * np.eye(len(cols))
        b = -Hc.T @ F @ x
        return np.linalg.solve(A, b)

    # first pass over the full menu, then re-solve on the kept names so the
    # reported basket is optimal for exactly the names traded (not a truncation)
    h_full = solve(list(range(H.shape[1])))
    keep_cols = sorted(np.argsort(-np.abs(h_full))[:max_names])
    h_keep = solve(keep_cols)
    h_sparse = np.zeros(H.shape[1])
    for j, col in enumerate(keep_cols):
        h_sparse[col] = h_keep[j]
    order = keep_cols

    resid_before = x
    resid_after = x + H @ h_sparse
    # shorted hedge ETFs add their own idiosyncratic variance to the book
    hedge_specific_var = float(((h_sparse * s_hedge) ** 2).sum()) * TRADING_DAYS

    def ann_vol(resid, extra_specific=0.0):
        return float(np.sqrt(max(
            resid @ F @ resid * TRADING_DAYS + specific_var + extra_specific,
            0.0,
        )))

    rows = []
    for i in sorted(order, key=lambda j: -abs(h_sparse[j])):
        if abs(h_sparse[i]) < 1.0:
            continue
        etf = available[i]
        spot = getattr(stats.get(etf), "spot", None)
        shares = h_sparse[i] / spot if spot else None
        rows.append({
            "etf": etf,
            "notional": float(h_sparse[i]),
            "shares": None if shares is None else int(round(shares)),
        })
    trades = pd.DataFrame(rows)

    return HedgeSuggestion(
        trades=trades,
        vol_before=ann_vol(resid_before),
        vol_after=ann_vol(resid_after, extra_specific=hedge_specific_var),
        residual_before=pd.Series(resid_before, index=fnames),
        residual_after=pd.Series(resid_after, index=fnames),
        menu_available=available,
        issues=issues,
    )
