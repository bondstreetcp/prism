"""Reusable report-generation pipeline shared by the CLI and the web app.

`generate_report` runs the whole flow — parse, fetch market data, analytics,
factor model, scenarios, hedge, crowding, alerts, PDF — and returns a small
result object. Progress is streamed through an optional callback so a UI can
show it live; the CLI passes ``print``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from .analytics import build_analytics
from .marketdata import MarketData
from .parse import parse_positions_csv
from .snapshot import save_snapshot
from .tearsheet import render_tearsheet


@dataclass
class ReportResult:
    pdf_path: Path
    asof: date
    name: str
    summary: dict
    alert_hits: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    headline: dict = field(default_factory=dict)  # a few key figures for a UI
    # in-memory objects so the app can build interactive views (basis toggle,
    # benchmark-relative risk, optimizer) without recomputing anything heavy
    analytics: object = None
    factor_risk: object = None
    model: object = None
    scenarios: object = None
    hedge: object = None
    crowding: object = None
    bias: object = None
    stats: dict = field(default_factory=dict)


def generate_report(
    csv_path: str | Path,
    *,
    aum: float | None = None,
    name: str | None = None,
    asof: date | None = None,
    out_dir: str | Path = "reports",
    cache_dir: str | Path = "cache",
    alerts_path: str | Path | None = None,
    no_factors: bool = False,
    no_hedge: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ReportResult:
    log = progress or (lambda _msg: None)
    t0 = time.time()

    parsed = parse_positions_csv(csv_path)
    asof = asof or parsed.asof or date.today()
    n_opt = sum(1 for p in parsed.positions if p.kind == "option")
    log(f"Parsed {len(parsed.positions)} positions "
        f"({len(parsed.positions) - n_opt} equity, {n_opt} option) as of {asof}; "
        f"{len(parsed.issues)} parse issue(s).")

    md = MarketData(cache_dir)
    tickers = sorted({p.underlying for p in parsed.positions})
    fetch_tickers = tickers
    if not no_factors and not no_hedge:
        from .hedge import HEDGE_MENU
        fetch_tickers = sorted(set(tickers) | set(HEDGE_MENU))

    log(f"Fetching price history for {len(fetch_tickers)} tickers…")
    closes, vols = md.fetch_history(fetch_tickers, asof)
    stats = md.compute_stats(closes, vols, asof)
    log(f"  priced {len(stats)}/{len(fetch_tickers)} tickers")

    log("Fetching company profiles…")
    profiles = md.fetch_profiles(tickers)

    contracts = [
        {"key": p.contract_key, "underlying": p.underlying, "root": p.root,
         "expiry": p.expiry, "strike": p.strike, "cp": p.cp}
        for p in parsed.positions if p.kind == "option"
    ]
    log(f"Fetching option chains for {len(contracts)} contracts…")
    quotes = md.fetch_option_quotes(contracts)
    log(f"  matched {len(quotes)}/{len(contracts)} contracts in live chains")

    analytics = build_analytics(
        parsed.positions, stats, profiles, quotes,
        asof=asof, aum=aum, issues=parsed.issues,
    )

    factor_risk = scenarios = model = bias = None
    if not no_factors:
        from .factors import (
            compute_factor_risk, factor_bias_test, fetch_factor_returns,
            fit_loadings,
        )
        from .scenarios import run_scenarios

        log("Fitting factor model (Ken French daily library)…")
        try:
            factor_returns = fetch_factor_returns(cache_dir)
            sectors = dict(zip(analytics.positions["underlying"],
                               analytics.positions["sector"]))
            model = fit_loadings(closes, factor_returns, asof, sectors=sectors)
            factor_risk = compute_factor_risk(analytics.positions, model)
            bias = factor_bias_test(
                analytics.positions, closes, factor_returns, asof, sectors=sectors
            )
            log(f"  {len(model.factor_names)} factors, loadings for "
                f"{len(model.loadings)} names ({model.n_shrunk} shrunk), "
                f"avg R²={model.avg_r2:.2f}, data through {model.data_end}")
            if bias is not None:
                log(f"  OOS bias test: realized/predicted vol = {bias.ratio:.2f}")
        except Exception as exc:
            log(f"  factor model unavailable: {exc}")
            analytics.issues.append(f"Factor model unavailable: {str(exc)[:120]}")

        log("Running stress grid and historical-simulation VaR…")
        scenarios = run_scenarios(
            analytics.positions, closes,
            {t: st.beta for t, st in stats.items()}, asof,
        )
        risk_summary = {
            "var_95": None if scenarios.var_95 != scenarios.var_95 else scenarios.var_95,
            "var_99": None if scenarios.var_99 != scenarios.var_99 else scenarios.var_99,
            "es_95": None if scenarios.es_95 != scenarios.es_95 else scenarios.es_95,
        }
        if factor_risk is not None:
            risk_summary.update({
                "vol_total": factor_risk.vol_total,
                "vol_factor": factor_risk.vol_factor,
                "vol_specific": factor_risk.vol_specific,
                "factor_var_share": factor_risk.factor_var_share,
                "model_coverage": factor_risk.coverage,
                "factor_set": model.factor_names,
                "avg_r2": model.avg_r2,
                "cov_shrinkage": model.cov_shrinkage,
                "cond_number": model.cond_number,
                "n_shrunk": model.n_shrunk,
                "bias_ratio": None if bias is None else bias.ratio,
                "factor_exposures_net": {
                    f: float(v) for f, v in factor_risk.exposures["net"].items()
                },
            })
        analytics.summary["risk"] = risk_summary

    hedge = None
    if not no_factors and not no_hedge and factor_risk is not None:
        from .hedge import suggest_hedge
        try:
            hedge = suggest_hedge(
                factor_risk.exposures["net"], model, stats,
                specific_var=(factor_risk.vol_specific ** 2),
            )
            log(f"Hedge suggestion: {len(hedge.trades)} ETF(s), predicted vol "
                f"${hedge.vol_before/1e6:,.1f}M -> ${hedge.vol_after/1e6:,.1f}M")
        except Exception as exc:
            log(f"  hedge suggestion unavailable: {exc}")

    crowding = None
    try:
        from .crowding import compute_crowding
        crowding = compute_crowding(analytics.issuers, analytics.positions)
    except Exception as exc:
        log(f"  crowding unavailable: {exc}")
    crowding_obj = crowding

    from .alerts import check_alerts, load_config
    alert_hits: list[str] = []
    try:
        cfg = load_config(alerts_path)
        alert_hits = check_alerts(analytics, factor_risk, scenarios, cfg, crowding)
        if cfg is not None:
            log(f"{len(alert_hits)} limit breach(es)." if alert_hits
                else "All configured risk limits OK.")
    except Exception as exc:
        log(f"  alerts unavailable: {exc}")

    snap_dir = save_snapshot(analytics)
    log(f"Snapshot archived to {snap_dir}")

    name = name or ", ".join(parsed.accounts) or "Portfolio"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"Risk Report {name} {asof.isoformat()}.pdf"
    render_tearsheet(
        analytics, name, pdf_path,
        factor_risk=factor_risk, scenarios=scenarios, hedge=hedge,
        alert_hits=alert_hits, crowding=crowding,
        model=model if not no_factors else None,
        bias=bias if not no_factors else None,
    )

    s = analytics.summary
    headline = {
        "exp_long": s["exp_long"], "exp_short": s["exp_short"],
        "exp_net": s["exp_net"], "exp_gross": s["exp_gross"],
        "beta_net": s["beta_net"], "n_instruments": s["n_instruments"],
        "n_issuers": s["n_issuers"],
        "vol_total": factor_risk.vol_total if factor_risk else None,
        "var_95": scenarios.var_95 if scenarios and scenarios.var_95 == scenarios.var_95 else None,
    }
    elapsed = time.time() - t0
    log(f"Report written to {pdf_path}  ({elapsed:.0f}s)")

    return ReportResult(
        pdf_path=pdf_path, asof=asof, name=name, summary=s,
        alert_hits=alert_hits, issues=analytics.issues,
        elapsed_s=elapsed, headline=headline,
        analytics=analytics, factor_risk=factor_risk, model=model,
        scenarios=scenarios, hedge=hedge, crowding=crowding_obj, bias=bias,
        stats=stats,
    )
