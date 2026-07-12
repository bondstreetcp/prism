"""Performance attribution across archived snapshots.

Attributes each day's model P&L (from the latest prior snapshot's book) into
market, style-factor, and stock-specific components. Runs over everything in
snapshots/ up to the latest close.

Usage:
    python run_attribution.py
    python run_attribution.py --name "My Fund" --end 2026-07-10
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from riskreport.attribution import compute_attribution, load_snapshots
from riskreport.factors import fetch_factor_returns, fit_loadings
from riskreport.marketdata import MarketData


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", default="snapshots")
    ap.add_argument("--name", default=None)
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: latest close)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()

    t0 = time.time()
    snapshots = load_snapshots(args.snapshots)
    if not snapshots:
        print("No snapshots found — run run_report.py at least once first.")
        return 1
    print(f"Loaded {len(snapshots)} snapshot(s): "
          f"{snapshots[0][0]} … {snapshots[-1][0]}")

    end = date.fromisoformat(args.end) if args.end else date.today()
    tickers = sorted({
        u for _, df in snapshots for u in df["underlying"].unique()
    })

    md = MarketData(args.cache)
    print(f"Fetching price history for {len(tickers)} tickers…")
    closes, _ = md.fetch_history(tickers, end)

    factor_returns = model = None
    try:
        factor_returns = fetch_factor_returns(args.cache)
        sectors = {}
        for _, df in snapshots:
            if "sector" in df.columns:
                sectors.update(dict(zip(df["underlying"], df["sector"])))
        model = fit_loadings(closes, factor_returns, end, sectors=sectors)
        print(f"Factor model: {len(model.factor_names)} factors, "
              f"{len(model.loadings)} names, data through {model.data_end}")
    except Exception as exc:
        print(f"Factor model unavailable ({exc}); market/style split skipped.")

    result = compute_attribution(snapshots, closes, factor_returns, model, end)

    from riskreport.tearsheet import render_attribution
    name = args.name or "AWKF1209"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"Attribution {name} {result.start} to {result.end}.pdf"
    render_attribution(result, name, pdf_path)

    cum = result.daily[["total", "market", "style", "specific"]].sum()
    print(f"\nAttribution {result.start} → {result.end} "
          f"({result.n_days} trading days):")
    for label, key in [("Total", "total"), ("Market", "market"),
                       ("Style", "style"), ("Specific", "specific")]:
        print(f"  {label:<10} ${cum[key]/1e6:+,.2f}M")
    if result.n_proxy_days:
        print(f"  ({result.n_proxy_days} day(s) market-only — factor data "
              f"through {result.factor_data_end})")
    print(f"\nReport written to {pdf_path}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
