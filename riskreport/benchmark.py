"""Benchmark-relative (active) risk — Omega Point's active-space analytics.

Everything else in the tool is absolute. This expresses the book *relative to
a benchmark*: active factor exposures (portfolio minus benchmark), tracking
error (active risk), what drives it, and the book's beta to the benchmark.

The benchmark is a single ETF held at a chosen notional (default: the book's
net delta-adjusted exposure). Its factor exposure is notional x its own
loadings; its specific risk is notional x its residual vol. Portfolio and
benchmark specific risks are treated as independent, so active variance is:

    active_exp = port_factor_exp - notional * bench_loadings
    active_var = active_exp' F active_exp * 252
               + port_specific_var + (notional * bench_resid_vol)^2 * 252
    tracking_error = sqrt(active_var)                     (annualized $)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .factors import TRADING_DAYS, FactorModel, FactorRisk

# convenient default benchmarks (must be in the fitted loadings universe —
# they are part of the hedge menu, so they get loadings on every run)
BENCHMARK_CHOICES = {
    "S&P 500 (SPY)": "SPY",
    "Russell 2000 (IWM)": "IWM",
    "Nasdaq 100 (QQQ)": "QQQ",
    "S&P MidCap 400 (MDY)": "MDY",
}


def beta_match_notional(
    port_exp: np.ndarray, bench_load: np.ndarray, factor_names: list[str]
) -> float:
    """Signed benchmark notional that neutralizes the book's market exposure:
    notional * bench_market_beta = book_market_exposure. Signed on purpose —
    a net-short-market book needs a negative (short-benchmark) notional to
    zero the active market bet; taking abs() would double it instead."""
    mkt = factor_names.index("Mkt-RF") if "Mkt-RF" in factor_names else 0
    bench_mkt = bench_load[mkt]
    n = (port_exp[mkt] / bench_mkt) if abs(bench_mkt) > 1e-9 else float(port_exp.sum())
    return n if n != 0 else 1.0


@dataclass
class BenchmarkRisk:
    benchmark: str
    notional: float
    active_exposures: pd.DataFrame       # per factor: portfolio, benchmark, active
    tracking_error: float                # annualized $ active risk
    te_pct: float                        # tracking error / notional
    active_factor_contrib: pd.Series     # per-factor share of active variance
    active_specific_share: float
    position_active_contrib: pd.DataFrame  # per-underlying contribution to TE
    port_vol: float
    bench_vol: float
    beta_to_benchmark: float
    coverage: float                      # factor-model coverage of the book
    issues: list[str] = field(default_factory=list)


def active_risk(
    factor_risk: FactorRisk, model: FactorModel,
    benchmark_ticker: str, notional: float | None = None,
) -> BenchmarkRisk:
    fn = model.factor_names
    if benchmark_ticker not in model.loadings.index:
        raise ValueError(
            f"No factor loadings for benchmark {benchmark_ticker}; "
            "pick one of the fitted ETFs."
        )

    port_exp = factor_risk.exposures["net"].reindex(fn).fillna(0.0).to_numpy()
    port_specific_var = factor_risk.vol_specific ** 2  # already annualized $^2

    bench_load = model.loadings.loc[benchmark_ticker, fn].to_numpy()
    # default: beta-match the benchmark to the book's market exposure, so the
    # active market bet is ~0 and tracking error reflects the style/residual
    # bets. (Caller can override with the book's actual benchmark notional.)
    if notional is None:
        notional = beta_match_notional(port_exp, bench_load, fn)
    bench_exp = notional * bench_load
    bench_resid = float(model.resid_vol.get(benchmark_ticker, 0.0))
    bench_specific_var = (notional * bench_resid) ** 2 * TRADING_DAYS

    F = model.fcov.loc[fn, fn].to_numpy()
    active_exp = port_exp - bench_exp

    active_factor_var = float(active_exp @ F @ active_exp) * TRADING_DAYS
    active_specific_var = port_specific_var + bench_specific_var
    active_var = active_factor_var + active_specific_var
    te = float(np.sqrt(max(active_var, 0.0)))

    # per-factor contribution to active variance
    contrib = pd.Series(active_exp * (F @ active_exp) * TRADING_DAYS, index=fn)
    active_factor_contrib = contrib / active_var if active_var > 0 else contrib * 0.0
    active_specific_share = active_specific_var / active_var if active_var > 0 else 0.0

    # ---- position-level contribution to tracking error (MCTE) -------------
    # Each name's contribution to TE (a volatility, degree-1 homogeneous, so
    # contributions sum to TE): CTR_u = x_u·(B_u·F·a + x_u·s_u²)·252 / TE.
    # The single-ETF benchmark contributes the rest as one line, so the held
    # names plus the benchmark line sum exactly to TE.
    Fa = F @ active_exp
    pr = factor_risk.position_risk
    pac_rows = []
    if te > 0 and pr is not None and len(pr):
        x_by_u = pr.groupby("underlying")["exposure"].sum()
        meta = pr.groupby("underlying").agg(name=("name", "first"),
                                            sector=("sector", "first"))
        names_u = list(x_by_u.index)
        Bu = model.loadings.loc[names_u, fn].to_numpy()
        su = model.resid_vol.reindex(names_u).fillna(0.0).to_numpy()
        xu = x_by_u.to_numpy()
        # marginal TE per $ of each name, then $ contribution
        mcte = ((Bu @ Fa) + xu * su ** 2) * TRADING_DAYS / te   # per $1 exposure
        ctr = xu * mcte                                          # $ contrib to TE
        for i, u in enumerate(names_u):
            pac_rows.append({
                "underlying": u,
                "name": meta.loc[u, "name"], "sector": meta.loc[u, "sector"],
                "exposure": float(xu[i]),
                "ctr_te": float(ctr[i]),
                "pct_of_te": float(ctr[i] / te),
                "mcte_per_$": float(mcte[i]),
            })
        # benchmark line: -N·B_bench·F·a·252 + bench_specific_var, over TE
        bench_ctr = (-notional * float(bench_load @ Fa) * TRADING_DAYS
                     + bench_specific_var) / te
        pac_rows.append({
            "underlying": f"[benchmark {benchmark_ticker}]", "name": "Benchmark",
            "sector": "Benchmark", "exposure": -float(notional),
            "ctr_te": float(bench_ctr), "pct_of_te": float(bench_ctr / te),
            "mcte_per_$": float("nan"),
        })
    position_active_contrib = (
        pd.DataFrame(pac_rows).sort_values("ctr_te", ascending=False)
        .reset_index(drop=True) if pac_rows else pd.DataFrame(
            columns=["underlying", "name", "sector", "exposure", "ctr_te",
                     "pct_of_te", "mcte_per_$"]))

    # portfolio and benchmark standalone predicted vol
    port_vol = factor_risk.vol_total
    bench_factor_var = float(bench_exp @ F @ bench_exp) * TRADING_DAYS
    bench_vol = float(np.sqrt(max(bench_factor_var + bench_specific_var, 0.0)))

    # book beta to the benchmark: cov(port, bench) / var(bench), factor-model
    cov_pb = float(port_exp @ F @ bench_exp) * TRADING_DAYS  # specifics independent
    var_b = bench_factor_var + bench_specific_var
    beta_to_bench = cov_pb / var_b if var_b > 0 else float("nan")

    active_exposures = pd.DataFrame({
        "portfolio": port_exp,
        "benchmark": bench_exp,
        "active": active_exp,
    }, index=fn)

    return BenchmarkRisk(
        benchmark=benchmark_ticker,
        notional=notional,
        active_exposures=active_exposures,
        tracking_error=te,
        te_pct=te / abs(notional) if notional else float("nan"),
        active_factor_contrib=active_factor_contrib,
        active_specific_share=active_specific_share,
        position_active_contrib=position_active_contrib,
        port_vol=port_vol,
        bench_vol=bench_vol,
        beta_to_benchmark=beta_to_bench,
        coverage=factor_risk.coverage,
    )
