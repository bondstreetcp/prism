"""Monte Carlo VaR — parametric, factor-model based, full option revaluation.

The historical-simulation VaR replays the last ~250 actual days. This draws a
large synthetic sample from the fitted factor model instead, so the tail is not
limited to what recently happened and the percentiles are smooth:

  1. Draw daily factor returns  f ~ N(0, F)   (F = fitted factor covariance).
  2. Add idiosyncratic returns  eps_u ~ N(0, s_u^2)  per underlying.
  3. Each underlying's return is  r_u = B_u·f + eps_u.
  4. Reprice the book: equities move linearly; options are fully repriced with
     Black-Scholes at the shocked spot. Implied vol is co-shocked from the
     simulated market move (calibrated to the VIX/market relationship), so a
     short-premium book shows its gamma/vega tail — same idea as the vol-aware
     historical VaR, here inside the simulation.

Deterministic (fixed seed) so the number doesn't jitter between reruns. Runs on
demand — it's heavier than the historical engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .analytics import DEFAULT_IV, bs_price_delta
from .scenarios import _bs_price_vec

_SEED = 20260825
_IV_CLAMP = (0.5, 4.0)


@dataclass
class MCVaRResult:
    var_95: float
    var_99: float
    es_95: float
    var_95_spot: float          # IV held constant (for the vol add-on)
    es_95_spot: float
    n_sims: int
    vol_beta: float             # dln(VIX)/d(market return), used for the IV shock
    pnl: np.ndarray             # simulated P&L sample (for a distribution chart)
    coverage: float
    issues: list = field(default_factory=list)


def _calibrate_vol_beta(closes: pd.DataFrame) -> float:
    """dln(VIX) per unit market return (negative: vol rises when market falls)."""
    if closes is None or "^VIX" not in closes or "SPY" not in closes:
        return -5.0
    v = np.log(closes["^VIX"]).diff()
    m = closes["SPY"].pct_change()
    d = pd.concat([v, m], axis=1).dropna()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 60 or float(d.iloc[:, 1].var()) <= 0:
        return -5.0
    k = float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var())
    return float(np.clip(k, -12.0, -1.0))


def monte_carlo_var(
    positions: pd.DataFrame, model, closes: pd.DataFrame, asof: date,
    n_sims: int = 10000,
) -> MCVaRResult | None:
    """Parametric MC VaR with full option revaluation and a vol co-shock."""
    fn = model.factor_names
    F = model.fcov.loc[fn, fn].to_numpy()
    # ensure PSD for the Cholesky (fcov is Ledoit-Wolf shrunk, but be safe)
    try:
        L = np.linalg.cholesky(F)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(F)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-16, None)))

    unders = [u for u in positions["underlying"].unique()
              if u in model.loadings.index]
    if not unders:
        return None
    uidx = {u: i for i, u in enumerate(unders)}
    B = model.loadings.loc[unders, fn].to_numpy()          # (U, K)
    s = model.resid_vol.reindex(unders).fillna(0.0).to_numpy()  # (U,)

    rng = np.random.default_rng(_SEED)
    fsim = rng.standard_normal((n_sims, len(fn))) @ L.T     # (M, K) ~ N(0, F)
    eps = rng.standard_normal((n_sims, len(unders))) * s    # (M, U)
    r = fsim @ B.T + eps                                    # (M, U) returns

    # implied-vol co-shock driven by the simulated market factor. Without a
    # market factor we can't identify the driver, so skip the co-shock (spot-
    # only) rather than shock off an unrelated factor.
    vol_beta = _calibrate_vol_beta(closes)
    if "Mkt-RF" in fn:
        iv_scale = np.clip(np.exp(vol_beta * fsim[:, fn.index("Mkt-RF")]),
                           *_IV_CLAMP)  # (M,)
    else:
        iv_scale = np.ones(n_sims)

    pnl_spot = np.zeros(n_sims)
    pnl_vol = np.zeros(n_sims)
    gross = float(positions["exposure"].abs().sum()) or 1.0
    covered = 0.0
    for row in positions.itertuples():
        u = row.underlying
        if u not in uidx:
            continue
        covered += abs(float(getattr(row, "exposure", 0.0)))
        ru = r[:, uidx[u]]
        if row.kind == "equity":
            eq = row.qty * row.spot * ru
            pnl_spot += eq
            pnl_vol += eq
        else:
            t_years = (row.expiry - asof).days / 365.0
            iv = row.iv if row.iv and np.isfinite(row.iv) else DEFAULT_IV
            base, _ = bs_price_delta(row.spot, row.strike, t_years, iv, row.cp)
            shocked = np.maximum(row.spot * (1.0 + ru), row.spot * 0.01)
            new_spot_only = _bs_price_vec(shocked, row.strike, t_years, iv, row.cp)
            new_vol = _bs_price_vec(shocked, row.strike, t_years,
                                    iv * iv_scale, row.cp)
            pnl_spot += row.qty * 100.0 * (new_spot_only - base)
            pnl_vol += row.qty * 100.0 * (new_vol - base)

    def _tail(pnl):
        losses = -pnl
        v95 = float(np.percentile(losses, 95))
        v99 = float(np.percentile(losses, 99))
        tail = losses[losses >= v95]
        return v95, v99, (float(tail.mean()) if len(tail) else v95)

    v95, v99, es = _tail(pnl_vol)
    v95s, _, ess = _tail(pnl_spot)
    return MCVaRResult(
        var_95=v95, var_99=v99, es_95=es, var_95_spot=v95s, es_95_spot=ess,
        n_sims=n_sims, vol_beta=vol_beta, pnl=pnl_vol,
        coverage=covered / gross,
    )
