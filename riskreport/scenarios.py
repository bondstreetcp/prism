"""Stress tests and historical-simulation VaR with full option revaluation.

Both engines reprice the actual book rather than scaling delta:
  * equities move linearly with their shocked spot;
  * options are repriced with Black-Scholes at the shocked spot (and shocked
    implied vol for the stress grid), using each position's stored IV.

Stress grid: market moves are propagated to each underlying through its beta
(spot_i' = spot_i * (1 + beta_i * mkt_move)); vol shocks scale IV
multiplicatively. P&L is instantaneous (same time to expiry).

Historical-simulation VaR: each of the last ~250 daily cross-sectional
return vectors from the price history is applied to today's book (IV held
constant), giving a P&L distribution whose percentiles are the VaR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from scipy.stats import norm

from .analytics import DEFAULT_IV, RISK_FREE_RATE, bs_price_delta

MKT_MOVES = [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]
VOL_SHOCKS = [0.0, 0.25]  # relative IV changes per grid row
VAR_WINDOW = 250
VAR_MIN_OBS = 120


def _bs_price_vec(spots: np.ndarray, strike: float, t: float, iv: float, cp: str) -> np.ndarray:
    """Vectorized Black-Scholes price over an array of spots (q=0)."""
    spots = np.maximum(spots, 1e-8)
    if t <= 0:
        return np.maximum(spots - strike, 0.0) if cp == "C" else np.maximum(strike - spots, 0.0)
    sq = iv * np.sqrt(t)
    d1 = (np.log(spots / strike) + (RISK_FREE_RATE + 0.5 * iv * iv) * t) / sq
    d2 = d1 - sq
    disc = np.exp(-RISK_FREE_RATE * t)
    if cp == "C":
        return spots * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    return strike * disc * norm.cdf(-d2) - spots * norm.cdf(-d1)


def _reval_position(row, new_spot: float, iv_scale: float, asof: date) -> float:
    """Instantaneous P&L of one position under a shocked spot / IV."""
    if row.kind == "equity":
        return row.qty * (new_spot - row.spot)
    t_years = (row.expiry - asof).days / 365.0
    iv = (row.iv if row.iv and np.isfinite(row.iv) else DEFAULT_IV) * iv_scale
    base_price, _ = bs_price_delta(row.spot, row.strike, t_years, row.iv or DEFAULT_IV, row.cp)
    new_price, _ = bs_price_delta(new_spot, row.strike, t_years, iv, row.cp)
    return row.qty * 100.0 * (new_price - base_price)


@dataclass
class ScenarioResults:
    stress_grid: pd.DataFrame  # index=vol shock label, cols=market move label
    var_95: float  # 1-day, positive number = loss
    var_99: float
    es_95: float  # expected shortfall beyond VaR95
    var_obs: int
    pnl_best: float
    pnl_worst: float
    worst_date: str
    issues: list[str] = field(default_factory=list)


def run_scenarios(
    positions: pd.DataFrame,
    closes: pd.DataFrame,
    betas: dict[str, float | None],
    asof: date,
) -> ScenarioResults:
    issues: list[str] = []
    pos = positions.copy()

    # ---------------------------------------------------------- stress grid
    grid = pd.DataFrame(
        index=[f"Vol {v:+.0%}" for v in VOL_SHOCKS],
        columns=[f"{m:+.0%}" for m in MKT_MOVES],
        dtype=float,
    )
    # names without a computed beta are held FLAT in the grid (beta 0), not
    # defaulted to 1 — a bond ETF or inverse product marked down 15% with the
    # market would misstate the stress P&L worse than excluding it
    no_beta = pos["underlying"].map(lambda u: betas.get(u) is None)
    if no_beta.any():
        weight = pos.loc[no_beta, "exposure"].abs().sum()
        gross_all = pos["exposure"].abs().sum()
        if gross_all and weight / gross_all > 0.01:
            issues.append(
                f"{weight / gross_all:.0%} of gross exposure has no computed "
                "beta and is held flat in the stress grid."
            )
    beta_series = pos["underlying"].map(
        lambda u: betas.get(u) if betas.get(u) is not None else 0.0
    )
    for vi, vol_shock in enumerate(VOL_SHOCKS):
        for mi, mkt in enumerate(MKT_MOVES):
            total = 0.0
            for row, b in zip(pos.itertuples(), beta_series):
                new_spot = row.spot * (1.0 + b * mkt)
                if new_spot <= 0:
                    new_spot = row.spot * 0.01
                total += _reval_position(row, new_spot, 1.0 + vol_shock, asof)
            grid.iloc[vi, mi] = total

    # --------------------------------------------------- historical-sim VaR
    # bridge short price gaps (halts, missed prints) so the resume-day move
    # survives pct_change instead of being deleted along with the gap
    filled = closes.ffill(limit=5)
    rets = filled.pct_change(fill_method=None)
    rets = rets.loc[rets.index <= pd.Timestamp(asof)].tail(VAR_WINDOW)
    tickers = [t for t in pos["underlying"].unique() if t in rets.columns]
    r = rets[tickers]
    # keep days where most of the book has data (young listings drop out)
    r = r.loc[r.notna().mean(axis=1) >= 0.7]

    # names with sparse coverage inside the window would be silently muted by
    # fillna(0) — treat them as uncovered instead and disclose their weight
    name_cov = r.notna().mean()
    sparse = set(name_cov[name_cov < 0.9].index)
    if sparse:
        r = r.drop(columns=sparse)
    obs = len(r)
    if obs < VAR_MIN_OBS:
        issues.append(
            f"Only {obs} usable days of joint history for VaR "
            f"(minimum {VAR_MIN_OBS}) — VaR not reported."
        )
        return ScenarioResults(
            stress_grid=grid, var_95=float("nan"), var_99=float("nan"),
            es_95=float("nan"), var_obs=obs, pnl_best=float("nan"),
            pnl_worst=float("nan"), worst_date="", issues=issues,
        )

    r = r.fillna(0.0)
    pnl = np.zeros(obs)
    for row in pos.itertuples():
        u = row.underlying
        if u not in r.columns:
            continue
        shocked = row.spot * (1.0 + r[u].to_numpy())
        if row.kind == "equity":
            pnl += row.qty * (shocked - row.spot)
        else:
            t_years = (row.expiry - asof).days / 365.0
            iv = row.iv if row.iv and np.isfinite(row.iv) else DEFAULT_IV
            base_price, _ = bs_price_delta(row.spot, row.strike, t_years, iv, row.cp)
            new_prices = _bs_price_vec(shocked, row.strike, t_years, iv, row.cp)
            pnl += row.qty * 100.0 * (new_prices - base_price)

    losses = -pnl  # positive = loss
    var_95 = float(np.percentile(losses, 95))
    var_99 = float(np.percentile(losses, 99))
    tail = losses[losses >= var_95]
    es_95 = float(tail.mean()) if len(tail) else var_95
    worst_i = int(np.argmax(losses))

    uncovered = pos.loc[~pos["underlying"].isin(r.columns), "exposure"].abs().sum()
    gross = pos["exposure"].abs().sum()
    if gross and uncovered / gross > 0.02:
        issues.append(
            f"{uncovered / gross:.0%} of gross exposure lacks return history "
            "and is excluded from VaR."
        )

    return ScenarioResults(
        stress_grid=grid,
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        var_obs=obs,
        pnl_best=float(pnl.max()),
        pnl_worst=float(pnl.min()),
        worst_date=str(r.index[worst_i].date()),
        issues=issues,
    )
