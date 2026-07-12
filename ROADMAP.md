# Roadmap — Omega Point feature parity for risk

Based on a July 2026 research sweep of OmegaPoint.ai (rebranded from ompnt.com
Sept 2025), their support docs, and their GraphQL API surface (reconstructed
from the open-source `omegapoint` Python client). 42 features catalogued;
this maps the risk-relevant ones onto what this tool can realistically build.

**Key insight from their API:** Omega Point has *no VaR endpoint*. Their core
risk number is factor-model **predicted volatility, decomposed into factor vs
idiosyncratic (specific) risk**, with per-security / per-group risk
contributors. Parity on "risk" means a factor model, not a VaR engine —
though a historical-simulation VaR is cheap for us to add and useful.

## What we already cover (v1)

* Exposures: long/short/gross/net MV and delta-adjusted, options overlay,
  beta-adjusted net (their "portfolio beta" analytics, simplified)
* Composition & concentration: sector/cap/region, top issuers, counts
  (their `composition`/`concentration` endpoints)
* Coverage/data-quality reporting (their coverage report → our footer notes)
* Snapshot store (their position-set upload, minus the platform)

## Tier 1 — the heart of Omega Point risk

1. ~~**Factor exposures + risk decomposition.**~~ **SHIPPED 2026-07-11**
   (`riskreport/factors.py`, tearsheet page 2): Ken French 5F+MOM daily
   loadings, dollar factor exposures, predicted vol factor/specific split,
   per-factor / per-sector / per-issuer risk contributions.
   **Deepened 2026-07-12**: 8 factors (added short/long-term reversal,
   selectable set), EWMA-weighted loadings + covariance, Ledoit-Wolf
   covariance shrinkage, short-history sector-median shrinkage (models spins
   instead of dropping them), and a realized-vs-predicted-vol bias test.
   Result: factor share 86%→90%, bias ratio 1.00, well-conditioned cov.
2. ~~**Stress tests / scenarios.**~~ **SHIPPED 2026-07-11**
   (`riskreport/scenarios.py`): market ±5/10/15% via per-name beta × IV
   +0/+25% grid, options fully revalued under Black-Scholes.
3. ~~**VaR.**~~ **SHIPPED 2026-07-11**: 1-day 95/99% historical-simulation
   VaR + expected shortfall, options fully repriced per scenario day.
4. ~~**What-if simulator.**~~ **SHIPPED 2026-07-11** (`run_whatif.py`,
   `riskreport/whatif.py`): apply a Symbol,Quantity trade list (or diff two
   exports) → before/after/Δ exposures, factor exposures, predicted vol,
   VaR, top issuer movers, one-page PDF.
5. ~~**Performance & factor attribution.**~~ **SHIPPED 2026-07-11**
   (`run_attribution.py`, `riskreport/attribution.py`): buy-and-hold daily
   model P&L from archived snapshots split market / style / specific, with
   issuer and sector contributors; window grows as daily snapshots accrue.
   Ken French publication lag means the newest days are market-only
   (disclosed); trading effect not measurable from quantity-only exports.

**Tier 1 complete.** Next candidates are in Tier 2 below.

## Tier 2 — feasible with free/public data, more work

* ~~**Liquidity**~~ **SHIPPED 2026-07-12**: % ADV, days-to-liquidate
  (median + 95th %ile at 20% participation), least-liquid issuer table on
  the risk page; ADV from yfinance 60d volume.
* ~~**Monitoring/alerts**~~ **SHIPPED 2026-07-12** (`riskreport/alerts.py`,
  `--alerts config.json`): threshold checks on net/gross, issuer & sector
  concentration, VaR, predicted vol, beta-adj net, days-to-liquidate.
  Breaches print to console and render as a banner atop page 1.
* ~~**Hedge-basket suggestion**~~ **SHIPPED 2026-07-12**
  (`riskreport/hedge.py`): ridge-regularized solve for the ETF basket that
  minimizes residual factor variance over a liquid menu; output is a trade
  list ready to drop into run_whatif.py.
* ~~**Crowding / squeeze**~~ **SHIPPED 2026-07-12** (`riskreport/crowding.py`):
  short % of float, days-to-cover, month-over-month short-interest trend, and
  institutional ownership from yfinance (same `.info` call as profiles, no
  extra fetch). Panel on page 1 flags your crowded shorts (squeeze risk) and
  a `max_crowded_short_pct_gross` alert. NOTE: this is a short-interest proxy;
  true 13F hedge-fund-overlap crowding (download quarterly SEC 13F datasets +
  ticker→CUSIP map + distinct-filer aggregation) is a heavier lift left for
  later — the ticker→CUSIP mapping is the free-data sticking point.
* ~~**Custom classifications/tags**~~ **SHIPPED 2026-07-12**
  (`riskreport/tags.py`, app **Themes** tab): upload a `Ticker,Theme` map →
  delta-adjusted long/short/net/gross exposure grouped by theme (names can sit
  in several themes; exposures overlap by design), with tag coverage.
* **Scheduling**: run run_report.py daily so attribution/history accrue
  automatically (Claude Code /schedule or OS task scheduler).

## Tier 2b — interactive app views (SHIPPED 2026-07-12)

Built as Streamlit tabs in `app.py`; on-screen, not PDF.

* ~~**Time-series trends**~~ (`riskreport/trends.py`): reads the snapshot
  store, plots net/gross/long/short exposure, predicted vol, VaR, factor
  share, bias ratio, issuer count over time. History accrues from daily runs.
* ~~**Benchmark-relative / active risk**~~ (`riskreport/benchmark.py`):
  active factor exposure (portfolio − benchmark), tracking error, beta to the
  benchmark, active variance factor/specific split. Benchmark = an ETF held
  at a notional (default beta-matches the book's market exposure).
* ~~**Portfolio optimizer**~~ (`riskreport/optimizer.py`, scipy SLSQP):
  min total risk / min factor risk / min tracking error over a tradable ETF
  universe, with a constraint library (max turnover, per-name size, factor
  exposure caps incl. market-neutral); returns a trade list + before/after.
* ~~**Cash / delta-adj / beta-adj basis toggle**~~ and redefined market-cap
  buckets (Nano <$50M · Micro $50–200M · Small $200M–2B · Mid $2–10B ·
  Large $10–200B · Mega >$200B).

## Tier 2c — macro, screener, AI (SHIPPED 2026-07-12)

* ~~**Macro factor overlay**~~ (`riskreport/macro.py`): book P&L regressed on
  macro-proxy ETF returns (IEF/LQD/HYG/TIP/USO/UUP/GLD), market-controlled →
  incremental $ P&L per +1% move in rates/credit/inflation/oil/USD/gold, with
  t-stats. Their Quant Insight lens, free-data version.
* ~~**Factor screener**~~ (`riskreport/screener.py`): screen the fitted-loadings
  universe (book names + hedge/macro ETFs) by factor loading, beta, R², sector,
  held/not-held; find hedges or replacements with a target profile.
* ~~**AI risk narrative + chat**~~ (`riskreport/narrative.py`): feeds the
  report's computed metrics to an LLM for a plain-English risk commentary and
  a multi-turn **chat** ("how would I lower the net short momentum exposure?").
  Uses GLM (Zhipu / z.ai) via its OpenAI-compatible API by default —
  provider-agnostic through `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL`.
  Optional. Matches their AI-native "AI Teammates" pivot.

## Tier 2d — AI-native surface & aggregation (SHIPPED 2026-07-12)

* ~~**MCP server**~~ (`riskreport/mcp_server.py`): stdio MCP server exposing the
  analytics as tools (`load_book`, `exposures`, `factor_exposures`,
  `risk_summary`, `top_issuers`, `macro_exposures`, `optimize`) so an AI
  assistant can load a book and query its risk — their other AI-native surface.
* ~~**Multi-account aggregation**~~ (`parse.merge_parse_results`): pass several
  broker CSVs (Goldman + IBKR, multiple accounts) to consolidate into one book,
  netting positions by issuer/contract and summing cash — CLI and app.

Natural next builds: forecasts/alpha into the optimizer, scheduling, and true
13F-overlap crowding.

## Tier 3 — requires licensed data; out of scope for a free-data clone

* Commercial factor models (Axioma, Wolfe QES, MSCI Barra) and model
  mix-and-match; Noonum thematic indices; Quant Insight macro analytics
* Borrow/financing rates and enhanced short interest (S3 Partners)
* Full optimizer with market-impact costs and MIP constraints
* OMS/PMS integrations (Eze, Enfusion), multi-manager aggregation, ESG sets

## Notes

* Their platform is API-first (GraphQL) with an MCP server for AI clients —
  a future step here could expose this tool the same way so an AI assistant
  can query the book.
* Full research output (42 features, 12 categories, all sources):
  archived from the research run 2026-07-11.
