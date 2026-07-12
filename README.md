# Portfolio Risk Report

Generates a one-page PDF risk tearsheet (Omega Point-style) from a broker
"Intraday Position" CSV export, using free market data from Yahoo Finance.

## Web app

For a no-terminal, upload-a-CSV experience:

```
pip install -r requirements.txt
streamlit run app.py
```

Opens in your browser with four tabs:

* **Report** — on-screen exposures/concentration with a **cash / delta-adjusted
  / beta-adjusted** basis toggle, plus the downloadable PDF.
* **Trends** — net/gross exposure, predicted vol, VaR, factor share, and bias
  ratio over time (builds as you run daily; each run archives a snapshot).
* **Benchmark** — active factor exposure vs a benchmark ETF, tracking error,
  and beta to the benchmark.
* **Optimizer** — minimize total/factor risk or tracking error over a tradable
  ETF universe under a constraint library (turnover, per-name size, factor
  caps, market-neutral); returns a trade list and before/after.
* **Macro** — the book's sensitivity to macro drivers (rates, IG/HY credit,
  inflation breakevens, oil, USD, gold) via liquid ETF proxies, market-
  controlled — "$ P&L per +1% move."
* **Screener** — screen the fitted universe (your names + hedge/macro ETFs) by
  factor loading, beta, and model fit to find hedges or replacements.
* **Narrative** — an AI risk analyst (Claude) reads the computed report and
  writes a plain-English commentary. Needs `ANTHROPIC_API_KEY`; ~$0.02–0.03
  per commentary. Optional — the rest of the tool works without it.

To host it privately for colleagues, see [DEPLOY.md](DEPLOY.md) (Render /
Railway / Fly / a VPS behind Tailscale — a few dollars a month). Set an
`APP_PASSWORD` env var to gate access.

## Command line

```
python run_report.py "Intraday Position_2026-07-07_0109PM.csv"
python run_report.py positions.csv --aum 25000000 --name "My Fund"
```

What-if simulation (before/after exposures, factor loadings, vol, VaR):

```
python run_whatif.py "Intraday Position_2026-07-07_0109PM.csv" --trades trades_example.csv
python run_whatif.py base.csv --proposed proposed_book.csv
```

Trade files are `Symbol,Quantity` CSVs with signed deltas (positive buys,
negative sells) using the broker's symbol format, so options trade too —
see [trades_example.csv](trades_example.csv).

Performance attribution (market / style / stock-specific model P&L across
all archived snapshots):

```
python run_attribution.py
```

Each `run_report.py` run archives a snapshot, so attribution coverage grows
automatically as you run the daily report. Note: Ken French factor returns
publish with a ~5-week lag; more recent days use a market-only decomposition
(disclosed on the report) until the style data catches up.

Options:

| Flag | Meaning |
|------|---------|
| `--aum 25000000` | Fund AUM in dollars; enables % AUM columns (default: percentages are % of gross exposure) |
| `--name "My Fund"` | Display name on the report (default: account number from the CSV) |
| `--asof 2026-07-07` | Override the as-of date (default: parsed from the filename) |
| `--out reports` | Output directory for the PDF |

## What it computes

Page 1 — exposures & concentration:

* **Exposure summary** — long/short/gross/net market value and delta-adjusted
  exposure, beta-adjusted net (betas vs SPY, 250 trading days).
* **Options overlay** — the options book's raw MV and delta-adjusted exposure
  vs the equity book.
* **Breakdowns** — delta-adjusted long/short/net exposure by sector, market
  cap bucket, and region (issuer-level, so options net against stock in the
  same name).
* **Top 10 long / short issuers** with sector and % of side.
* **Data-quality notes** — anything unpriced, IV fallbacks, adjusted
  contracts, blank symbols; printed in the report footer.

Page 2 — factor model, scenarios, liquidity & hedge (skip with `--no-factors`):

* **Style factor exposures** — EWMA-weighted loadings of each name's daily
  excess returns on 8 factors (Fama-French 5 + momentum + short/long-term
  reversal, Ken French daily library, free), rolled up to dollar factor
  exposures. Names with 20–60 days of history (recent spins/IPOs) are shrunk
  toward their sector's median loadings instead of dropped; the factor
  covariance is EWMA-weighted with Ledoit-Wolf shrinkage. The factor set is
  selectable (`ff3` / `ff5` / `ff5mom` / `ff5mom_rev`, default the last).
* **Model diagnostics + bias test** — avg R², covariance shrinkage δ,
  condition number, and a realized-vs-predicted-vol calibration ratio
  (≈1.0 = well-scaled).
* **Predicted volatility** — annualized, decomposed factor vs stock-specific,
  with per-factor, per-sector, and top-10 issuer risk contributions.
* **Stress grid** — market moves −15%…+15% (propagated per name via beta)
  × IV shocks, with the options book fully repriced under Black-Scholes.
* **Historical-simulation VaR** — 1-day 95%/99% and expected shortfall from
  applying the last 250 daily joint return vectors to the current book.
* **Liquidity** — % of gross in names above 25/50/100% of 60-day ADV, days
  to liquidate (median + 95th %ile at 20% participation), least-liquid table.
* **Hedge suggestion** — the liquid-ETF basket that best neutralizes the
  book's residual factor risk (`--no-hedge` to skip); output is a trade list
  you can feed straight to `run_whatif.py`.

Risk limits (optional `--alerts config.json`, see
[alerts_example.json](alerts_example.json)): breaches print to console and
show as a red banner atop page 1. Limits cover net/gross ratio, gross $,
issuer & sector concentration, VaR, predicted vol, beta-adj net, and
days-to-liquidate; any limit set to `null` is skipped.

## Methodology notes

* Equity prices: last close on or before the as-of date (auto-adjusted).
* Option market values: live chain bid/ask mid, falling back to last trade,
  falling back to Black-Scholes theoretical value.
* Option deltas: Black-Scholes using chain implied vol, falling back to 60-day
  realized vol of the underlying (flagged in the footer when it happens).
* Adjusted option roots (e.g. `APTV1`) from corporate actions are mapped to
  the base underlying and treated as standard 100-multiplier contracts — an
  approximation flagged in the footer.
* Yahoo option chains are live-only, so chain quotes reflect the run date, not
  the as-of date. Equity spots are as-of.

## Files

```
riskreport/
  parse.py       broker CSV -> Position records (equities + OCC-style options)
  marketdata.py  yfinance fetch layer with on-disk cache (cache/)
  analytics.py   pricing, delta-adjusted exposures, aggregations
  tearsheet.py   matplotlib + reportlab one-page PDF
  snapshot.py    archives each run under snapshots/<date>/
run_report.py    CLI entry point
```

## History / performance panels

Each run archives position-level analytics to `snapshots/<asof>/`. Once daily
snapshots accumulate, performance panels (P&L, exposure trends, attribution)
can be built on top of that store — they are intentionally out of scope for
v1 because a single position snapshot carries no return history.

## Requirements

Python 3.11+ with: `yfinance pandas numpy scipy matplotlib reportlab pyarrow`
