"""Factor exposure by sector — a risk map of where the factor bets sit.

The Risk tab shows the book's net factor exposures and the risk contribution by
sector separately. This crosses them: for each (sector, factor) it sums the
delta-adjusted dollar exposure times the loading, so you can see that (say) the
momentum tilt lives in Tech while the value tilt sits in Financials — structure
neither the factor totals nor the sector totals reveal on their own.
"""

from __future__ import annotations

import pandas as pd


def factor_sector_exposure(factor_risk, model) -> pd.DataFrame | None:
    """Sector × factor net dollar exposure matrix (rows=sector, cols=factor)."""
    pr = getattr(factor_risk, "position_risk", None)
    if pr is None or len(pr) == 0:
        return None
    fn = list(model.factor_names)
    g = pr.groupby("underlying").agg(exposure=("exposure", "sum"),
                                     sector=("sector", "first"))
    names = [u for u in g.index if u in model.loadings.index]
    if not names:
        return None
    g = g.loc[names]
    B = model.loadings.loc[names, fn].to_numpy()          # (U, K) loadings
    x = g["exposure"].to_numpy()                          # (U,) $ exposure
    fe = pd.DataFrame(x[:, None] * B, index=names, columns=fn)  # $ factor exp
    fe["sector"] = g["sector"].to_numpy()
    matrix = fe.groupby("sector")[fn].sum()
    # order sectors by gross factor exposure (most active first)
    order = matrix.abs().sum(axis=1).sort_values(ascending=False).index
    return matrix.reindex(order)


def top_cells(matrix: pd.DataFrame, n: int = 6) -> list[dict]:
    """Largest |sector × factor| exposures, for the AI facts."""
    if matrix is None or matrix.empty:
        return []
    stacked = matrix.stack()
    top = stacked.reindex(stacked.abs().sort_values(ascending=False).index).head(n)
    return [{"sector": str(idx[0]), "factor": str(idx[1]),
             "exposure_$M": round(float(v) / 1e6, 1)} for idx, v in top.items()]
