"""Options expiry / theta ladder — the term structure of the option book.

The greeks on the Report tab are aggregate; this breaks them out by time to
expiry, which is what actually matters for a short-premium (put-writing) book:

  * where premium rolls off (the income schedule),
  * where theta is earned (decay concentrates in the near expiries),
  * and where the dangerous short gamma / vega sits — near-dated options have
    the sharpest gamma, so a gap into a near expiry hurts most.

Built entirely from the per-position dollar greeks already on the analytics
positions frame (gamma_pnl_1pct, vega_dollar, theta_dollar) plus market value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

# (min_days, max_days, label) — inclusive day buckets by time to expiry
DTE_BUCKETS = [
    (0, 7, "0–7d"),
    (8, 30, "8–30d"),
    (31, 90, "31–90d"),
    (91, 180, "91–180d"),
    (181, 10 ** 7, "180d+"),
]

_AGG = {
    "n_contracts": ("qty", lambda s: float(s.abs().sum())),
    "net_premium": ("mv", "sum"),          # short premium is negative (collected)
    "delta_exp": ("exposure", "sum"),
    "gamma_1pct": ("gamma_pnl_1pct", "sum"),
    "vega_1pt": ("vega_dollar", "sum"),
    "theta_day": ("theta_dollar", "sum"),
}


@dataclass
class LadderResult:
    by_expiry: pd.DataFrame     # one row per expiry date
    by_bucket: pd.DataFrame     # one row per DTE bucket (ordered)
    total_theta_day: float
    total_vega_1pt: float
    total_gamma_1pct: float
    net_premium: float          # negative = net short premium collected
    near_theta_share: float     # share of |theta| in <=30 days
    near_gamma_share: float     # share of |gamma P&L| in <=30 days


def _bucket(dte: int) -> str:
    for lo, hi, lbl in DTE_BUCKETS:
        if lo <= dte <= hi:
            return lbl
    return "expired"


def options_ladder(positions: pd.DataFrame, asof: date) -> LadderResult | None:
    opt = positions[positions["kind"] == "option"].copy()
    if opt.empty:
        return None
    opt["dte"] = (pd.to_datetime(opt["expiry"]) - pd.Timestamp(asof)).dt.days
    opt["bucket"] = opt["dte"].map(_bucket)

    by_expiry = (opt.groupby("expiry")
                 .agg(dte=("dte", "first"), **_AGG)
                 .reset_index().sort_values("expiry").reset_index(drop=True))

    order = [b[2] for b in DTE_BUCKETS] + ["expired"]
    by_bucket = opt.groupby("bucket").agg(**_AGG)
    by_bucket = by_bucket.reindex([b for b in order if b in by_bucket.index])
    by_bucket = by_bucket.reset_index()

    theta = float(opt["theta_dollar"].sum())
    gamma = float(opt["gamma_pnl_1pct"].sum())
    near = opt[opt["dte"] <= 30]
    theta_abs = float(opt["theta_dollar"].abs().sum()) or 1.0
    gamma_abs = float(opt["gamma_pnl_1pct"].abs().sum()) or 1.0

    return LadderResult(
        by_expiry=by_expiry, by_bucket=by_bucket,
        total_theta_day=theta,
        total_vega_1pt=float(opt["vega_dollar"].sum()),
        total_gamma_1pct=gamma,
        net_premium=float(opt["mv"].sum()),
        near_theta_share=float(near["theta_dollar"].abs().sum()) / theta_abs,
        near_gamma_share=float(near["gamma_pnl_1pct"].abs().sum()) / gamma_abs,
    )
