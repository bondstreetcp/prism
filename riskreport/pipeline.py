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
from .parse import merge_parse_results, parse_positions
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
    closes: object = None      # price history (for macro overlay)
    profiles: dict = field(default_factory=dict)  # for the screener
    brinson: object = None     # Brinson attribution (PDF + app)
    factor_attr: object = None  # factor-based return attribution (PDF + app)
    factor_returns: object = None  # Ken French daily factors (app re-slices windows)
    scenario_lib: list = field(default_factory=list)  # crisis replays (opt-in)
    base_positions: list = field(default_factory=list)  # for pre-trade what-if
    alert_config: object = None  # risk-limit config, for pre-trade checks
    fi_risk: object = None       # fixed-income (rate) risk of the bond-ETF sleeve


def generate_report(
    csv_path,  # str | Path | list of them (multiple files are aggregated)
    *,
    aum: float | None = None,
    cash: float | None = None,
    name: str | None = None,
    asof: date | None = None,
    out_dir: str | Path = "reports",
    cache_dir: str | Path = "cache",
    snap_dir: str | Path = "snapshots",
    alerts_path: str | Path | None = None,
    no_factors: bool = False,
    no_hedge: bool = False,
    include_scenarios: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ReportResult:
    log = progress or (lambda _msg: None)
    t0 = time.time()

    if isinstance(csv_path, (list, tuple)):
        parsed = merge_parse_results([parse_positions(p) for p in csv_path])
    else:
        parsed = parse_positions(csv_path)
    asof = asof or parsed.asof or date.today()
    # AUM/cash priority: explicit --aum > explicit --cash (net MV + cash) >
    # broker-reported NAV (IBKR) > broker file cash (net MV + file cash).
    # `cash` (from --cash / the app's Cash field) is ADDITIONAL cash for
    # accounts whose files omit it (e.g. Goldman) — it ADDS to any cash the
    # broker files already report (e.g. IBKR), it does not replace it. So the
    # total cash of a consolidated book = file-reported cash + this input.
    user_add = cash
    file_cash = parsed.cash
    total_cash = None
    if file_cash is not None or user_add:
        total_cash = (file_cash or 0.0) + (user_add or 0.0)

    if aum is not None:
        cash = total_cash                      # explicit AUM wins; keep cash for display
    elif user_add:
        cash = total_cash                      # build_analytics derives MV + total cash
    elif parsed.nav is not None:
        aum = parsed.nav                       # single broker book: trust its own NAV
        cash = file_cash
    else:
        cash = total_cash                      # net MV + file-reported cash (may be None)
    n_opt = sum(1 for p in parsed.positions if p.kind == "option")
    log(f"Parsed {len(parsed.positions)} {parsed.source.upper()} positions "
        f"({len(parsed.positions) - n_opt} equity, {n_opt} option) as of {asof}; "
        f"{len(parsed.issues)} parse issue(s)."
        + (f" Cash ${cash/1e6:,.1f}M." if cash is not None else ""))

    md = MarketData(cache_dir)
    tickers = sorted({p.underlying for p in parsed.positions})
    fetch_tickers = tickers
    if not no_factors:
        # hedge-basket ETFs (for the hedge suggester + optimizer) and macro
        # proxy ETFs (for the macro overlay) get loadings/returns too
        extra = set()
        if not no_hedge:
            from .hedge import HEDGE_MENU
            extra |= set(HEDGE_MENU)
        from .macro import MACRO_ETFS
        extra |= set(MACRO_ETFS)
        # VIX drives the vol-aware VaR (option IV co-shocks with its history)
        extra.add("^VIX")
        # SPDR sector ETFs = benchmark sector returns for Brinson attribution
        from .attribution_brinson import SECTOR_ETFS
        extra |= set(SECTOR_ETFS)
        fetch_tickers = sorted(set(tickers) | extra)

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
        asof=asof, aum=aum, cash=cash, issues=parsed.issues,
    )

    factor_risk = scenarios = model = bias = None
    factor_returns = None
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
        def _nn(x):  # NaN -> None for clean JSON snapshots
            return None if x != x else x

        risk_summary = {
            "var_95": _nn(scenarios.var_95),
            "var_99": _nn(scenarios.var_99),
            "es_95": _nn(scenarios.es_95),
            "var_95_spot": _nn(scenarios.var_95_spot),
            "var_99_spot": _nn(scenarios.var_99_spot),
            "es_95_spot": _nn(scenarios.es_95_spot),
            "vol_aware": scenarios.vol_aware,
        }
        # portfolio greeks (from the option book) travel with the risk block so
        # Trends can plot them and the AI narrative can reason about them
        risk_summary.update(analytics.summary.get("greeks", {}))
        # top tail-risk contributors (component VaR) — small list for the
        # snapshot/PDF/AI; the full frame lives on scenarios.risk_contrib
        if scenarios.risk_contrib is not None:
            risk_summary["top_contributors"] = [
                {"ticker": row.underlying, "sector": row.sector,
                 "contrib_es95": float(row.contrib_es95),
                 "pct": float(row.pct_of_es95)}
                for row in scenarios.risk_contrib.head(8).itertuples()
            ]
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
    cfg = None
    try:
        cfg = load_config(alerts_path)
        alert_hits = check_alerts(analytics, factor_risk, scenarios, cfg, crowding)
        if cfg is not None:
            log(f"{len(alert_hits)} limit breach(es)." if alert_hits
                else "All configured risk limits OK.")
    except Exception as exc:
        log(f"  alerts unavailable: {exc}")

    # Brinson performance attribution (cheap: sector ETFs already fetched)
    brinson = None
    factor_attr = None
    if not no_factors:
        try:
            from .attribution_brinson import brinson_attribution
            brinson = brinson_attribution(analytics.issuers, closes, asof,
                                          window="3M")
        except Exception as exc:
            log(f"  Brinson attribution unavailable: {exc}")
        try:
            from .attribution_factor import factor_return_attribution
            factor_attr = factor_return_attribution(
                factor_risk, model, closes, factor_returns, analytics.issuers,
                asof, analytics.summary.get("aum"), window="3M")
        except Exception as exc:
            log(f"  factor attribution unavailable: {exc}")

    # Crisis-scenario replays (opt-in: needs a multi-year history fetch)
    scenario_lib: list = []
    if include_scenarios:
        try:
            from .scenario_library import fetch_long_history, run_library
            log("Fetching multi-year history for crisis scenarios…")
            unders = sorted(analytics.positions["underlying"].unique())
            betas = {t: s.beta for t, s in stats.items()}
            closes_long = fetch_long_history(unders, asof, cache_dir, log=log)
            scenario_lib = run_library(analytics.positions, closes_long, betas,
                                       asof, analytics.summary.get("aum"), log=log)
            log(f"  ran {len(scenario_lib)} scenarios")
        except Exception as exc:
            log(f"  scenario library unavailable: {exc}")

    # fixed-income (rate) risk for the bond-ETF sleeve — cheap, issuer-level
    fi_risk = None
    try:
        from .fixedincome import compute_fi_risk
        fi_risk = compute_fi_risk(analytics.issuers)
        if fi_risk is not None:
            log(f"Fixed-income sleeve: DV01 ${fi_risk.total_dv01/1e3:,.1f}K/bp "
                f"across {len(fi_risk.holdings)} bond ETF(s)")
    except Exception as exc:
        log(f"  fixed-income risk unavailable: {exc}")

    snap_written = save_snapshot(analytics, base_dir=snap_dir)
    log(f"Snapshot archived to {snap_written}")

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
        brinson=brinson, scenario_lib=scenario_lib, factor_attr=factor_attr,
        fi_risk=fi_risk,
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
        stats=stats, closes=closes, profiles=profiles,
        brinson=brinson, scenario_lib=scenario_lib,
        factor_attr=factor_attr, factor_returns=factor_returns,
        base_positions=parsed.positions, alert_config=cfg, fi_risk=fi_risk,
    )
