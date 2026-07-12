"""What-if simulator: apply trades to a position export (or diff two exports)
and compare exposures, factor loadings, predicted vol, and VaR before/after.

Usage:
    python run_whatif.py "Intraday Position_2026-07-07_0109PM.csv" --trades trades.csv
    python run_whatif.py base.csv --proposed proposed_book.csv
    python run_whatif.py base.csv --trades trades.csv --aum 25000000

Trade file format (Symbol,Quantity — signed deltas, broker symbol format):
    Symbol,Quantity
    SPY,-2000
    IWM    DEC 18 2026   200.000 P,-50
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from riskreport.analytics import build_analytics
from riskreport.marketdata import MarketData
from riskreport.parse import asof_from_filename
from riskreport.whatif import load_proposed_book


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_csv", help="Base broker position export CSV")
    ap.add_argument("--trades", default=None,
                    help="Trade list CSV (Symbol,Quantity signed deltas)")
    ap.add_argument("--proposed", default=None,
                    help="Full proposed position export to diff against")
    ap.add_argument("--aum", type=float, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--asof", default=None)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()

    t0 = time.time()
    base, proposed, trades, issues = load_proposed_book(
        args.base_csv, args.trades, args.proposed
    )
    asof = (
        date.fromisoformat(args.asof) if args.asof
        else asof_from_filename(args.base_csv) or date.today()
    )
    print(f"Base book: {len(base)} positions; proposed: {len(proposed)}; "
          f"{len(trades)} trade(s); as of {asof}")

    md = MarketData(args.cache)
    tickers = sorted({p.underlying for p in base} | {p.underlying for p in proposed})
    print(f"Fetching market data for {len(tickers)} tickers…")
    closes, vols = md.fetch_history(tickers, asof)
    stats = md.compute_stats(closes, vols, asof)
    profiles = md.fetch_profiles(tickers)

    seen, contracts = set(), []
    for p in list(base) + list(proposed):
        if p.kind == "option" and p.contract_key not in seen:
            seen.add(p.contract_key)
            contracts.append({
                "key": p.contract_key, "underlying": p.underlying,
                "root": p.root, "expiry": p.expiry,
                "strike": p.strike, "cp": p.cp,
            })
    quotes = md.fetch_option_quotes(contracts)

    a_before = build_analytics(base, stats, profiles, quotes, asof=asof,
                               aum=args.aum, issues=issues)
    a_after = build_analytics(proposed, stats, profiles, quotes, asof=asof,
                              aum=args.aum)

    fr_b = fr_a = sc_b = sc_a = None
    try:
        from riskreport.factors import (
            compute_factor_risk, fetch_factor_returns, fit_loadings,
        )
        sectors = dict(zip(a_before.positions["underlying"],
                           a_before.positions["sector"]))
        model = fit_loadings(closes, fetch_factor_returns(args.cache), asof,
                             sectors=sectors)
        fr_b = compute_factor_risk(a_before.positions, model)
        fr_a = compute_factor_risk(a_after.positions, model)
    except Exception as exc:
        a_before.issues.append(f"Factor model unavailable: {str(exc)[:120]}")

    from riskreport.scenarios import run_scenarios
    betas = {t: st.beta for t, st in stats.items()}
    sc_b = run_scenarios(a_before.positions, closes, betas, asof)
    sc_a = run_scenarios(a_after.positions, closes, betas, asof)

    from riskreport.tearsheet import render_whatif
    name = args.name or ", ".join(
        sorted({p.account for p in base if p.account})
    ) or "Portfolio"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"WhatIf {name} {asof.isoformat()}.pdf"
    # the after book collects its own issues (e.g. an unpriceable trade being
    # excluded) — merge them so a dropped trade is never silently invisible
    merged_issues = a_before.issues + [
        f"(after) {i}" for i in a_after.issues if i not in a_before.issues
    ]
    for i in a_after.issues:
        if i not in a_before.issues:
            print(f"  after-book issue: {i}")
    render_whatif(
        a_before, a_after, name, pdf_path, trades,
        fr_before=fr_b, fr_after=fr_a, sc_before=sc_b, sc_after=sc_a,
        issues=merged_issues,
    )

    def line(label, b, a, scale=1e6):
        d = a - b
        print(f"  {label:<22} {b/scale:>10,.1f}  ->  {a/scale:>10,.1f}   "
              f"(Δ {d/scale:+,.1f})")

    sb, sa = a_before.summary, a_after.summary
    print("\nBefore -> After ($M):")
    line("Δ-adj net", sb["exp_net"], sa["exp_net"])
    line("Δ-adj gross", sb["exp_gross"], sa["exp_gross"])
    line("Beta-adj net", sb["beta_net"], sa["beta_net"])
    if fr_b is not None and fr_a is not None:
        line("Predicted vol (ann.)", fr_b.vol_total, fr_a.vol_total)
    if sc_b.var_95 == sc_b.var_95 and sc_a.var_95 == sc_a.var_95:
        line("VaR 95% (1-day)", sc_b.var_95, sc_a.var_95)
    print(f"\nReport written to {pdf_path}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
