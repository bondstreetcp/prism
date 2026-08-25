"""Realized performance & drawdown statistics (current-holdings backtest).

Reconstructs what *today's* book would have done over a trailing window by
applying each underlying's realized daily returns to its current delta-adjusted
exposure (buy-and-hold), then computes realized risk stats:

  * realized annualized volatility — a reality check on the factor model's
    predicted vol,
  * maximum drawdown and its length,
  * empirical 1-day VaR/ES — a backtest of the parametric VaR,
  * best / worst day and the window return.

Same current-holdings, option-at-underlying caveats as the attribution views:
options move at their underlying's return times delta-adjusted exposure, so
greek P&L (theta/vega/gamma) is out of scope here — the risk tabs cover that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class PerfStats:
    daily_pnl: pd.Series         # reconstructed daily $ P&L over the window
    cum_return: pd.Series        # cumulative return path
    drawdown: pd.Series          # drawdown path (<= 0)
    ann_vol: float               # realized annualized $ vol
    ann_vol_pct: float           # realized annualized vol as % of base
    max_drawdown_pct: float
    max_dd_days: int
    realized_var95: float        # empirical 1-day 95% VaR ($, positive = loss)
    realized_es95: float
    best_day: float
    worst_day: float
    window_return_pct: float
    base: float                  # AUM or gross used to scale returns
    n_days: int
    coverage: float
    issues: list = field(default_factory=list)


def performance_stats(
    issuers: pd.DataFrame, closes: pd.DataFrame, asof: date,
    aum: float | None = None, window_days: int = 252,
) -> PerfStats | None:
    if issuers is None or issuers.empty or closes is None:
        return None
    end = pd.Timestamp(asof)
    idx = closes.index[closes.index <= end]
    if len(idx) < 40:
        return None
    win = idx[-(window_days + 1):]
    held = [u for u in issuers["underlying"].unique() if u in closes.columns]
    if not held:
        return None
    rets = closes.loc[win, held].pct_change(fill_method=None).iloc[1:]

    expo = (issuers.set_index("underlying")["exposure"].reindex(held)
            .fillna(0.0))
    gross = float(issuers["exposure"].abs().sum()) or 1.0
    covered = float(expo.abs().sum())
    coverage = covered / gross

    # daily $ P&L = Σ_u exposure_u × return_{u,t}
    daily_pnl = (rets.fillna(0.0) * expo).sum(axis=1)
    daily_pnl = daily_pnl[np.isfinite(daily_pnl)]
    if daily_pnl.empty:
        return None

    base = float(aum) if aum and aum > 0 else gross
    daily_ret = daily_pnl / base
    cum = (1.0 + daily_ret).cumprod()
    cum_return = cum - 1.0
    peak = cum.cummax()
    drawdown = cum / peak - 1.0

    # longest drawdown (days from a peak until recovery/its trough end)
    max_dd_pct = float(drawdown.min())
    dd_days = 0
    cur = 0
    for v in drawdown.to_numpy():
        cur = cur + 1 if v < 0 else 0
        dd_days = max(dd_days, cur)

    losses = -daily_pnl.to_numpy()
    var95 = float(np.percentile(losses, 95))
    tail = losses[losses >= var95]
    es95 = float(tail.mean()) if len(tail) else var95

    issues = []
    if coverage < 0.9:
        issues.append(f"Backtest covers {coverage:.0%} of gross exposure "
                      "(names without return history are excluded).")

    return PerfStats(
        daily_pnl=daily_pnl, cum_return=cum_return, drawdown=drawdown,
        ann_vol=float(daily_ret.std() * np.sqrt(252) * base),
        ann_vol_pct=float(daily_ret.std() * np.sqrt(252)),
        max_drawdown_pct=max_dd_pct, max_dd_days=int(dd_days),
        realized_var95=var95, realized_es95=es95,
        best_day=float(daily_pnl.max()), worst_day=float(daily_pnl.min()),
        window_return_pct=float(cum_return.iloc[-1]),
        base=base, n_days=int(len(daily_pnl)), coverage=coverage,
        issues=issues,
    )
