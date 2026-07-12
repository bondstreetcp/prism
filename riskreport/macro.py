"""Macro factor overlay — Omega Point's Quant Insight lens, free-data version.

Beyond equity style factors, this measures the book's sensitivity to macro
drivers using liquid ETF proxies: rates (duration), credit (IG and HY),
inflation breakevens, oil, the US dollar, and gold. The book's daily P&L
(current delta-adjusted exposures applied to underlying returns) is regressed
on the macro factor returns with the equity market (SPY) as a control, so each
macro beta is *incremental* to market — "$ P&L per +1% move in oil, holding
the market fixed."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MACRO_WINDOW = 252
MIN_OBS = 120

# ETF proxies. Spreads (a − b) isolate the factor from duration; e.g. IG credit
# excess return ≈ LQD − IEF strips the Treasury-rate component out of LQD.
MACRO_ETFS = ["SPY", "IEF", "LQD", "HYG", "TIP", "USO", "UUP", "GLD"]

# factor -> (label, long leg, short leg or None)
MACRO_FACTORS = [
    ("Rates (10y UST)", "IEF", None),
    ("IG credit", "LQD", "IEF"),
    ("HY credit", "HYG", "IEF"),
    ("Inflation breakeven", "TIP", "IEF"),
    ("Oil", "USO", None),
    ("US dollar", "UUP", None),
    ("Gold", "GLD", None),
]


@dataclass
class MacroExposure:
    betas: pd.DataFrame          # factor -> beta ($ P&L per 1% move), t-stat
    market_beta: float           # incremental market ($ per 1% SPY)
    r2: float                    # regression R^2
    window: int
    coverage: float              # share of gross exposure with return history
    issues: list[str] = field(default_factory=list)


def _factor_returns(closes: pd.DataFrame) -> pd.DataFrame:
    rets = closes.pct_change(fill_method=None)
    out = {}
    for label, lo, sh in MACRO_FACTORS:
        if lo not in rets.columns:
            continue
        series = rets[lo]
        if sh is not None and sh in rets.columns:
            series = series - rets[sh]
        out[label] = series
    fac = pd.DataFrame(out)
    if "SPY" in rets.columns:
        fac.insert(0, "Market (SPY)", rets["SPY"])
    return fac


def compute_macro(
    positions: pd.DataFrame, closes: pd.DataFrame, asof
) -> MacroExposure:
    """Regress the book's daily P&L on macro factor returns (market-controlled)."""
    issues: list[str] = []
    x_by_u = positions.groupby("underlying")["exposure"].sum()
    rets = closes.pct_change(fill_method=None)
    rets = rets.loc[rets.index <= pd.Timestamp(asof)].tail(MACRO_WINDOW)

    names = [u for u in x_by_u.index if u in rets.columns]
    R = rets[names]
    R = R.loc[R.notna().mean(axis=1) >= 0.8]
    name_cov = R.notna().mean()
    keep = list(name_cov[name_cov >= 0.9].index)
    if len(keep) < 2:
        raise ValueError("Too few names with history for a macro regression.")
    R = R[keep].fillna(0.0)
    x = x_by_u.reindex(keep).to_numpy()
    pnl = pd.Series(R.to_numpy() @ x, index=R.index)  # daily $ P&L

    fac = _factor_returns(closes)
    fac = fac.reindex(pnl.index).dropna()
    pnl = pnl.reindex(fac.index)
    if len(fac) < MIN_OBS:
        raise ValueError(f"Only {len(fac)} joint days for macro regression "
                         f"(minimum {MIN_OBS}).")
    y = pnl.to_numpy()
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0

    mkt = fac["Market (SPY)"].to_numpy() if "Market (SPY)" in fac.columns \
        else np.zeros(len(fac))
    macro_cols = [c for c in fac.columns if not c.startswith("Market")]

    def bivariate(factor_col: np.ndarray) -> tuple[float, float]:
        """OLS of pnl on [1, market, factor]; return (factor coef, t-stat)."""
        X = np.column_stack([np.ones(len(y)), mkt, factor_col])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        dof = max(len(y) - X.shape[1], 1)
        sigma2 = float((resid ** 2).sum()) / dof
        se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
        b = float(coef[2])
        t = b / se[2] if se[2] > 0 else float("nan")
        return b, t

    # market beta: univariate pnl ~ market (standard book beta)
    Xm = np.column_stack([np.ones(len(y)), mkt])
    cm, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    market_beta = float(cm[1]) / 100.0

    # each macro beta is market-controlled but collinearity-free (one macro
    # factor per regression) — "$ P&L per +1% move in <factor>, net of market"
    rows = []
    for label in macro_cols:
        b, t = bivariate(fac[label].to_numpy())
        rows.append({"factor": label, "beta_per_1pct": b / 100.0, "t_stat": t})
    betas = pd.DataFrame(rows)

    # overall variance explained by market + all macro factors together
    Xall = np.column_stack([np.ones(len(y)), fac.to_numpy()])
    call, *_ = np.linalg.lstsq(Xall, y, rcond=None)
    r2 = 1.0 - float(((y - Xall @ call) ** 2).sum()) / ss_tot

    gross = float(x_by_u.abs().sum()) or 1.0
    coverage = float(x_by_u.reindex(keep).abs().sum()) / gross
    if coverage < 0.9:
        issues.append(f"Macro regression covers {coverage:.0%} of gross "
                      "exposure (names without history excluded).")

    return MacroExposure(
        betas=betas, market_beta=market_beta, r2=r2,
        window=len(fac), coverage=coverage, issues=issues,
    )
