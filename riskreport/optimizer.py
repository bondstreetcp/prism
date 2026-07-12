"""Portfolio optimizer — a focused version of Omega Point's 'Construct'.

Solves for a dollar overlay in a liquid, tradable universe (the hedge-ETF menu
by default) that best meets an objective under a constraint library, and
returns the trade list. This is the constrained generalization of the hedge
suggester: choose what to minimize, then bound turnover, per-name size,
factor exposures, and market neutrality.

Formulation (quadratic program, solved with SLSQP):

    variables  t          dollar trade per tradable instrument
    post-trade factor exposure  x(t) = x0 + B t     (B = instrument loadings)
    objective   minimize   Var(t) + mu * ||t||^2    (mu = small turnover ridge)
      - total risk:   (x0+Bt)'F(x0+Bt)*252 + port_specific + sum(t_i^2 s_i^2)*252
      - factor risk:  drop the specific terms
      - track error:  replace x0 with (x0 - benchmark exposure)
    constraints
      - gross turnover:  sum|t_i| <= turnover_max
      - per-name size:   |t_i| <= max_per_name
      - factor caps:     |x(t)_f| <= cap_f      (e.g. market-neutral band)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .benchmark import beta_match_notional
from .factors import TRADING_DAYS, FactorModel, FactorRisk
from .hedge import HEDGE_MENU

TURNOVER_RIDGE = 1e-6   # keeps the solver well-behaved / trades tidy
_ABS_EPS = 1.0          # smoothing for the |t| turnover constraint ($)

OBJECTIVES = ["Minimize total risk", "Minimize factor risk",
              "Minimize tracking error"]


@dataclass
class OptimizeResult:
    trades: pd.DataFrame                 # etf, notional, shares
    objective: str
    vol_before: float
    vol_after: float
    exposures_before: pd.Series          # net factor $ before
    exposures_after: pd.Series           # net factor $ after
    turnover: float                      # sum |t_i|
    turnover_max: float
    binding: list[str] = field(default_factory=list)
    success: bool = True
    message: str = ""
    issues: list[str] = field(default_factory=list)


def optimize_overlay(
    factor_risk: FactorRisk,
    model: FactorModel,
    stats: dict,
    *,
    objective: str = "Minimize total risk",
    universe: list[str] | None = None,
    turnover_max: float | None = None,
    max_per_name: float | None = None,
    factor_caps: dict[str, float] | None = None,
    benchmark_ticker: str | None = None,
    benchmark_notional: float | None = None,
) -> OptimizeResult:
    fn = model.factor_names
    universe = [t for t in (universe or HEDGE_MENU) if t in model.loadings.index]
    if not universe:
        raise ValueError("No tradable instruments have factor loadings.")
    m = len(universe)

    F = model.fcov.loc[fn, fn].to_numpy()
    B = model.loadings.loc[universe, fn].to_numpy().T          # k x m
    s_univ = model.resid_vol.reindex(universe).fillna(0.0).to_numpy()
    x0 = factor_risk.exposures["net"].reindex(fn).fillna(0.0).to_numpy()
    port_specific_var = factor_risk.vol_specific ** 2

    # baseline exposure for the objective (shifted for tracking error)
    x_base = x0.copy()
    bench_specific_var = 0.0
    if objective == "Minimize tracking error":
        if not benchmark_ticker or benchmark_ticker not in model.loadings.index:
            raise ValueError("Tracking-error objective needs a benchmark with loadings.")
        bench_load = model.loadings.loc[benchmark_ticker, fn].to_numpy()
        # beta-match the benchmark to the book's market exposure (same as the
        # Benchmark view) unless the caller supplies an explicit notional
        notional = benchmark_notional or beta_match_notional(x0, bench_load, fn)
        bench_exp = notional * bench_load
        bench_resid = float(model.resid_vol.get(benchmark_ticker, 0.0))
        bench_specific_var = (notional * bench_resid) ** 2 * TRADING_DAYS
        x_base = x0 - bench_exp

    include_specific = objective != "Minimize factor risk"

    # quadratic objective  f(t) = t'Q t + c't + const
    Q = (B.T @ F @ B) * TRADING_DAYS
    if include_specific:
        Q = Q + np.diag(s_univ ** 2) * TRADING_DAYS
    Q = Q + TURNOVER_RIDGE * np.eye(m)
    c = 2.0 * (B.T @ F @ x_base) * TRADING_DAYS

    def f(t):
        return float(t @ Q @ t + c @ t)

    def grad(t):
        return 2.0 * Q @ t + c

    # ---- constraints -------------------------------------------------
    cons = []
    if turnover_max is not None:
        def turn_g(t):
            return turnover_max - np.sum(np.sqrt(t ** 2 + _ABS_EPS ** 2))

        def turn_jac(t):
            return -t / np.sqrt(t ** 2 + _ABS_EPS ** 2)
        cons.append({"type": "ineq", "fun": turn_g, "jac": turn_jac})

    for fac, cap in (factor_caps or {}).items():
        if fac not in fn:
            continue
        j = fn.index(fac)
        Bj = B[j]
        x0j = x0[j]
        # cap - (x0j + Bj·t) >= 0  and  cap + (x0j + Bj·t) >= 0
        cons.append({"type": "ineq",
                     "fun": (lambda t, Bj=Bj, x0j=x0j, cap=cap: cap - (x0j + Bj @ t)),
                     "jac": (lambda t, Bj=Bj: -Bj)})
        cons.append({"type": "ineq",
                     "fun": (lambda t, Bj=Bj, x0j=x0j, cap=cap: cap + (x0j + Bj @ t)),
                     "jac": (lambda t, Bj=Bj: Bj)})

    cap_per = max_per_name if max_per_name is not None else np.inf
    bounds = [(-cap_per, cap_per)] * m

    res = minimize(f, np.zeros(m), jac=grad, bounds=bounds,
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-9})
    t = res.x

    # ---- assemble result --------------------------------------------
    x_after = x0 + B @ t

    def ann_vol(x, extra_specific):
        v = float(x @ F @ x) * TRADING_DAYS + port_specific_var + extra_specific
        return float(np.sqrt(max(v, 0.0)))

    trade_specific = float(np.sum((t * s_univ) ** 2)) * TRADING_DAYS
    if objective == "Minimize tracking error":
        vol_before = np.sqrt(max(
            float(x_base @ F @ x_base) * TRADING_DAYS + port_specific_var + bench_specific_var, 0.0))
        xb_after = x_base + B @ t
        vol_after = np.sqrt(max(
            float(xb_after @ F @ xb_after) * TRADING_DAYS + port_specific_var
            + bench_specific_var + trade_specific, 0.0))
    else:
        vol_before = ann_vol(x0, 0.0)
        vol_after = ann_vol(x_after, trade_specific)

    rows = []
    for i in np.argsort(-np.abs(t)):
        if abs(t[i]) < 1.0:
            continue
        etf = universe[i]
        spot = getattr(stats.get(etf), "spot", None)
        rows.append({
            "etf": etf, "notional": float(t[i]),
            "shares": None if not spot else int(round(t[i] / spot)),
        })
    trades = pd.DataFrame(rows)

    binding = []
    turnover = float(np.sum(np.abs(t)))
    if turnover_max is not None and turnover >= turnover_max - 1e3:
        binding.append(f"turnover ≈ ${turnover_max/1e6:,.1f}M cap")
    for fac, cap in (factor_caps or {}).items():
        if fac in fn and abs(x_after[fn.index(fac)]) >= cap - 1e3:
            binding.append(f"{fac} exposure at ±${cap/1e6:,.1f}M cap")

    issues = []
    if not res.success:
        issues.append(f"Solver did not fully converge: {res.message}")

    return OptimizeResult(
        trades=trades, objective=objective,
        vol_before=vol_before, vol_after=vol_after,
        exposures_before=pd.Series(x0, index=fn),
        exposures_after=pd.Series(x_after, index=fn),
        turnover=turnover,
        turnover_max=turnover_max if turnover_max is not None else float("nan"),
        binding=binding, success=bool(res.success), message=str(res.message),
        issues=issues,
    )
