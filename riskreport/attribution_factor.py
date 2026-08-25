"""Factor-based (Barra-style) return attribution.

Decomposes the book's realized dollar return over a trailing window into the
part explained by each systematic factor and a stock-specific remainder:

    book P&L  =  Σ_k  (net $ exposure to factor k) · (factor k return)  +  specific

where the net factor exposures are the same delta-adjusted dollar exposures the
risk page uses (Σ_i exposure_i · loading_{i,k}), and factor returns are the
Ken French daily series cumulated over the window. Specific P&L is the residual
(true idiosyncratic return plus any model error), so the three parts always sum
to the realized book P&L.

This is the factor-lens complement to the sector-based Brinson attribution:
Brinson asks "which sectors drove active return vs the S&P"; this asks "which
factor bets (market, size, value, momentum, …) drove the book's own return".

Same self-contained, current-holdings, option-at-underlying caveats as Brinson.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}


@dataclass
class FactorAttrResult:
    table: pd.DataFrame       # per factor: exposure, factor_return, pnl
    factor_pnl: float         # total explained by factors ($)
    specific_pnl: float       # residual ($)
    realized_pnl: float       # realized book P&L over the window ($)
    aum: float | None
    start: date
    end: date
    coverage: float
    issues: list


def _winret(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    pre = s.loc[s.index <= start]
    post = s.loc[s.index <= end]
    if pre.empty or post.empty or pre.iloc[-1] <= 0:
        return None
    return float(post.iloc[-1] / pre.iloc[-1] - 1.0)


def factor_return_attribution(
    factor_risk, model, closes: pd.DataFrame, factor_returns: pd.DataFrame,
    issuers: pd.DataFrame, asof: date, aum: float | None, window: str = "3M",
) -> FactorAttrResult | None:
    if factor_risk is None or model is None or factor_returns is None:
        return None
    issues: list = []
    ndays = WINDOWS.get(window, 63)
    end_ts = pd.Timestamp(asof)
    idx = closes.index[closes.index <= end_ts]
    if len(idx) <= ndays:
        return None
    start_ts, end_ts = idx[-(ndays + 1)], idx[-1]

    # ---- factor cumulative returns over the window ------------------------
    fnames = [f for f in model.factor_names if f in factor_returns.columns]
    fwin = factor_returns.loc[(factor_returns.index > start_ts)
                              & (factor_returns.index <= end_ts), fnames]
    if fwin.empty:
        return None
    cumf = ((1.0 + fwin).prod() - 1.0)  # per-factor compounded return
    xf = factor_risk.exposures["net"].reindex(fnames).fillna(0.0)
    factor_pnl_k = xf * cumf                        # $ P&L per factor

    # ---- realized book P&L over the window --------------------------------
    rets = {}
    for u in issuers["underlying"].unique():
        if u in closes.columns:
            r = _winret(closes[u], start_ts, end_ts)
            if r is not None:
                rets[u] = r
    iss = issuers.copy()
    iss["ret"] = iss["underlying"].map(rets)
    gross = float(iss["exposure"].abs().sum()) or 1.0
    coverage = float(iss.loc[iss["ret"].notna(), "exposure"].abs().sum()) / gross
    iss = iss.dropna(subset=["ret"])
    realized = float((iss["exposure"] * iss["ret"]).sum())
    factor_total = float(factor_pnl_k.sum())
    specific = realized - factor_total
    if coverage < 0.9:
        issues.append(f"Return coverage {coverage:.0%} of gross — uncovered "
                      "names fall into the specific remainder.")

    table = pd.DataFrame({
        "factor": fnames,
        "exposure": xf.to_numpy(),
        "factor_return": cumf.to_numpy(),
        "pnl": factor_pnl_k.to_numpy(),
    }).sort_values("pnl").reset_index(drop=True)

    return FactorAttrResult(
        table=table, factor_pnl=factor_total, specific_pnl=specific,
        realized_pnl=realized, aum=aum, start=start_ts.date(),
        end=end_ts.date(), coverage=coverage, issues=issues,
    )
