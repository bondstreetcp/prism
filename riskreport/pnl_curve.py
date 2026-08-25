"""Portfolio P&L curve — the book's payoff across a range of market moves.

Reprices the whole book (equities linear, options full Black-Scholes) as the
market moves from −30% to +30%, propagating to each name through its beta.
Shows the *shape* of the book's exposure: a short-premium book is concave
(short gamma) — it makes a little as the market drifts and loses fast in a big
move, and worse still once implied vol is co-shocked with the drop.

Two curves: spot-only, and vol-aware (implied vol scaled with the market move
via the same VIX-calibrated beta as the Monte Carlo VaR).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .analytics import DEFAULT_IV, bs_price_delta
from .scenarios import _bs_price_vec


@dataclass
class PnlCurve:
    table: pd.DataFrame          # move, pnl_spot, pnl_vol ($)
    vol_beta: float
    pnl_down20: float            # vol-aware P&L at −20% market
    pnl_down10: float
    pnl_up10: float
    max_gain: float              # best P&L in the range
    worst_loss: float            # worst P&L in the range (vol-aware)


def pnl_curve(
    positions: pd.DataFrame, betas: dict, closes: pd.DataFrame, asof: date,
    lo: float = -0.30, hi: float = 0.30, n: int = 61,
) -> PnlCurve | None:
    if positions is None or positions.empty:
        return None
    moves = np.linspace(lo, hi, n)
    from .montecarlo import _calibrate_vol_beta
    vol_beta = _calibrate_vol_beta(closes)
    iv_scale = np.clip(np.exp(vol_beta * moves), 0.5, 4.0)   # (n,)

    pnl_spot = np.zeros(n)
    pnl_vol = np.zeros(n)
    for row in positions.itertuples():
        b = betas.get(row.underlying)
        b = 0.0 if b is None else float(b)          # no-beta names held flat
        new_spots = np.maximum(row.spot * (1.0 + b * moves), row.spot * 0.01)
        if row.kind == "equity":
            eq = row.qty * (new_spots - row.spot)
            pnl_spot += eq
            pnl_vol += eq
        else:
            t = (row.expiry - asof).days / 365.0
            iv = row.iv if row.iv and np.isfinite(row.iv) else DEFAULT_IV
            base, _ = bs_price_delta(row.spot, row.strike, t, iv, row.cp)
            new_spot_only = _bs_price_vec(new_spots, row.strike, t, iv, row.cp)
            new_v = _bs_price_vec(new_spots, row.strike, t, iv * iv_scale, row.cp)
            mult = getattr(row, "multiplier", 100) or 100
            pnl_spot += row.qty * mult * (new_spot_only - base)
            pnl_vol += row.qty * mult * (new_v - base)

    table = pd.DataFrame({"move": moves, "pnl_spot": pnl_spot,
                          "pnl_vol": pnl_vol})

    def at(m):
        return float(np.interp(m, moves, pnl_vol))

    return PnlCurve(
        table=table, vol_beta=vol_beta,
        pnl_down20=at(-0.20), pnl_down10=at(-0.10), pnl_up10=at(0.10),
        max_gain=float(pnl_vol.max()), worst_loss=float(pnl_vol.min()),
    )
