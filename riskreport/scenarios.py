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
    base_iv = row.iv if row.iv and np.isfinite(row.iv) else DEFAULT_IV
    iv = base_iv * iv_scale
    base_price, _ = bs_price_delta(row.spot, row.strike, t_years, base_iv, row.cp)
    new_price, _ = bs_price_delta(new_spot, row.strike, t_years, iv, row.cp)
    return row.qty * 100.0 * (new_price - base_price)


@dataclass
class ScenarioResults:
    stress_grid: pd.DataFrame  # index=vol shock label, cols=market move label
    var_95: float  # 1-day, positive number = loss (vol-aware when available)
    var_99: float
    es_95: float  # expected shortfall beyond VaR95
    var_obs: int
    pnl_best: float
    pnl_worst: float
    worst_date: str
    # spot-only comparison (IV held constant) — the delta to the headline VaR
    # is the contribution of the historical vol move (short-vega/gamma risk)
    var_95_spot: float = float("nan")
    var_99_spot: float = float("nan")
    es_95_spot: float = float("nan")
    vol_aware: bool = False
    # per-underlying tail-risk decomposition (contribution to ES95); sums to es_95
    risk_contrib: pd.DataFrame | None = None
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

    # ---- vol path: on each historical day, scale option IV by that day's
    # actual VIX move. In a historical sim this is the faithful co-shock — a
    # 2020-03 style crash applies BOTH the spot crash and the vol spike, so a
    # short-premium book shows its true short-vega/gamma tail. Falls back to
    # constant IV (spot-only) when VIX history is unavailable.
    iv_scale = np.ones(obs)
    vol_aware = False
    vix_col = next((c for c in ("^VIX", "VIX") if c in closes.columns), None)
    if vix_col is not None:
        vix = closes[vix_col].ffill(limit=5).reindex(r.index)
        vix_prev = closes[vix_col].ffill(limit=5).shift(1).reindex(r.index)
        ratio = (vix / vix_prev).to_numpy()
        good = np.isfinite(ratio) & (ratio > 0)
        if good.mean() >= 0.9:
            # clamp to a sane band so a bad print can't blow up the reval
            iv_scale = np.clip(np.where(good, ratio, 1.0), 0.5, 3.0)
            vol_aware = True
    if not vol_aware:
        issues.append(
            "VaR holds implied vol constant (no usable VIX history) — for a "
            "short-option book this understates the tail."
        )

    pnl_spot = np.zeros(obs)   # IV held constant (legacy method)
    pnl_vol = np.zeros(obs)    # IV co-shocked with VIX (vol-aware)
    # per-underlying head-method P&L, for component/marginal VaR decomposition
    pnl_by_u: dict[str, np.ndarray] = {}
    for row in pos.itertuples():
        u = row.underlying
        if u not in r.columns:
            continue
        shocked = row.spot * (1.0 + r[u].to_numpy())
        if row.kind == "equity":
            eq_pnl = row.qty * (shocked - row.spot)
            pnl_spot += eq_pnl
            pnl_vol += eq_pnl
            head_pnl = eq_pnl
        else:
            t_years = (row.expiry - asof).days / 365.0
            iv = row.iv if row.iv and np.isfinite(row.iv) else DEFAULT_IV
            base_price, _ = bs_price_delta(row.spot, row.strike, t_years, iv, row.cp)
            new_spot_only = _bs_price_vec(shocked, row.strike, t_years, iv, row.cp)
            new_vol = _bs_price_vec(shocked, row.strike, t_years, iv * iv_scale, row.cp)
            p_spot = row.qty * 100.0 * (new_spot_only - base_price)
            p_vol = row.qty * 100.0 * (new_vol - base_price)
            pnl_spot += p_spot
            pnl_vol += p_vol
            head_pnl = p_vol if vol_aware else p_spot
        if u in pnl_by_u:
            pnl_by_u[u] += head_pnl
        else:
            pnl_by_u[u] = head_pnl.copy()

    def _var(pnl):
        losses = -pnl
        v95 = float(np.percentile(losses, 95))
        v99 = float(np.percentile(losses, 99))
        tail = losses[losses >= v95]
        return v95, v99, (float(tail.mean()) if len(tail) else v95)

    v95_spot, v99_spot, es_spot = _var(pnl_spot)
    # headline VaR is vol-aware when we have the vol path, else identical to spot
    pnl_head = pnl_vol if vol_aware else pnl_spot
    v95, v99, es = _var(pnl_head)
    worst_i = int(np.argmax(-pnl_head))

    # ---- component / marginal VaR ------------------------------------------
    # Allocate tail risk to underlyings by their average P&L on the tail days
    # (total loss >= VaR95). This "contribution to ES95" is the Euler/additive
    # decomposition: the parts sum exactly to ES95. Marginal = contribution per
    # $1M of gross exposure (which names are risk-dense vs risk-cheap).
    losses_head = -pnl_head
    tail_mask = losses_head >= v95
    exp_by_u = pos.groupby("underlying")["exposure"].sum()
    meta = pos.groupby("underlying").agg(name=("name", "first"),
                                         sector=("sector", "first"))
    crows = []
    for u, arr in pnl_by_u.items():
        comp = float(np.mean(-arr[tail_mask])) if tail_mask.any() else 0.0
        expo = float(exp_by_u.get(u, 0.0))
        crows.append({
            "underlying": u,
            "name": meta.loc[u, "name"] if u in meta.index else u,
            "sector": meta.loc[u, "sector"] if u in meta.index else "Unknown",
            "exposure": expo,
            "contrib_es95": comp,                       # sums to es_95
            "pct_of_es95": comp / es if es else 0.0,
            # marginal: tail-loss $ per $1M of gross exposure on this name
            "risk_per_1m": comp / (abs(expo) / 1e6) if abs(expo) > 0 else float("nan"),
        })
    risk_contrib = (
        pd.DataFrame(crows).sort_values("contrib_es95", ascending=False)
        .reset_index(drop=True)
        if crows else None
    )

    uncovered = pos.loc[~pos["underlying"].isin(r.columns), "exposure"].abs().sum()
    gross = pos["exposure"].abs().sum()
    if gross and uncovered / gross > 0.02:
        issues.append(
            f"{uncovered / gross:.0%} of gross exposure lacks return history "
            "and is excluded from VaR."
        )

    return ScenarioResults(
        stress_grid=grid,
        var_95=v95,
        var_99=v99,
        es_95=es,
        var_obs=obs,
        pnl_best=float(pnl_head.max()),
        pnl_worst=float(pnl_head.min()),
        worst_date=str(r.index[worst_i].date()),
        var_95_spot=v95_spot,
        var_99_spot=v99_spot,
        es_95_spot=es_spot,
        vol_aware=vol_aware,
        risk_contrib=risk_contrib,
        issues=issues,
    )
