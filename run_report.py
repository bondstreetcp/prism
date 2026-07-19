"""Generate a portfolio risk tearsheet from a broker position CSV.

Usage:
    python run_report.py "Intraday Position_2026-07-07_0109PM.csv"
    python run_report.py positions.csv --aum 25000000 --name "My Fund"
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from riskreport.pipeline import generate_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", nargs="+",
                    help="Broker position export CSV(s); multiple files are "
                         "aggregated into one book")
    ap.add_argument("--aum", type=float, default=None,
                    help="Fund AUM in dollars (overrides net-MV + cash)")
    ap.add_argument("--cash", type=float, default=None,
                    help="Additional cash in dollars for accounts whose files "
                         "omit it (e.g. Goldman); ADDS to broker-reported cash "
                         "(IBKR files carry theirs). AUM = net MV + total cash")
    ap.add_argument("--name", default=None,
                    help="Portfolio display name (default: account number)")
    ap.add_argument("--asof", default=None,
                    help="Override as-of date, YYYY-MM-DD (default: from filename)")
    ap.add_argument("--out", default="reports", help="Output directory for the PDF")
    ap.add_argument("--cache", default="cache", help="Market data cache directory")
    ap.add_argument("--no-factors", action="store_true",
                    help="Skip the factor model / stress / VaR page")
    ap.add_argument("--no-hedge", action="store_true",
                    help="Skip the hedge-basket suggestion")
    ap.add_argument("--alerts", default=None,
                    help="Alert config JSON (default: alerts.json if present)")
    args = ap.parse_args()

    result = generate_report(
        args.csv_path if len(args.csv_path) > 1 else args.csv_path[0],
        aum=args.aum,
        cash=args.cash,
        name=args.name,
        asof=date.fromisoformat(args.asof) if args.asof else None,
        out_dir=args.out,
        cache_dir=args.cache,
        alerts_path=args.alerts,
        no_factors=args.no_factors,
        no_hedge=args.no_hedge,
        progress=print,
    )

    s = result.summary
    print(f"\nDelta-adj exposure: long ${s['exp_long']/1e6:,.1f}M, "
          f"short ${s['exp_short']/1e6:,.1f}M, net ${s['exp_net']/1e6:,.1f}M")
    if result.alert_hits:
        print(f"⚠  {len(result.alert_hits)} limit breach(es):")
        for h in result.alert_hits:
            print(f"   - {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
