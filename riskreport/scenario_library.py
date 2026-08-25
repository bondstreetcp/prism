"""Named scenario library — replay historical crises against today's book.

Two kinds of scenario:

* **Historical replay** — take an actual crisis window (e.g. the 2008 Lehman
  collapse, the 2020 COVID crash), compute each held underlying's realized
  return over that window from long price history, and apply it to *today's*
  positions with full Black-Scholes option revaluation. Implied vol is shocked
  by the VIX's actual move over the window, so a short-premium book shows its
  gamma/vega pain, not just delta. Names that didn't exist in the window fall
  back to beta × the S&P's move (disclosed); names without even a beta drop out.

* **Hypothetical shock** — a forward-looking market/vol move (e.g. "broad −20%,
  vol +100%") propagated through each name's beta. Needs no history.

Both are *instantaneous* shocks (time to expiry held fixed) — they isolate
market/vol risk, matching the main stress grid's convention. The library is
run on demand (it needs its own multi-year price fetch), separate from the
per-report pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .scenarios import _reval_position

# VIX move is applied as a uniform IV multiplier; clamp so a single bad print
# can't blow up the reval.
_VIX_CLAMP = (0.5, 4.0)
_RET_FLOOR = -0.99  # a name can lose at most ~100% (spot floored at 1%)


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    blurb: str
    kind: str  # "historical" | "hypothetical"
    start: date | None = None
    end: date | None = None
    mkt_move: float | None = None   # hypothetical: S&P move
    vol_shock: float | None = None  # hypothetical: relative IV change


# Curated crises. Windows are peak-to-trough (or the sharp leg) of each episode.
HISTORICAL: list[Scenario] = [
    Scenario("gfc2008", "GFC 2008 (Lehman)",
             "Sep–Nov 2008 credit crisis: S&P roughly halved, vol to record highs.",
             "historical", date(2008, 9, 2), date(2008, 11, 20)),
    Scenario("covid2020", "COVID crash 2020",
             "Feb 19 – Mar 23 2020: fastest bear market on record, VIX ~82.",
             "historical", date(2020, 2, 19), date(2020, 3, 23)),
    Scenario("q4_2018", "2018 Q4 selloff",
             "Oct–Dec 2018: growth scare + rate fears, S&P −19% into Christmas.",
             "historical", date(2018, 10, 1), date(2018, 12, 24)),
    Scenario("volmageddon2018", "Volmageddon (Feb 2018)",
             "Jan 26 – Feb 8 2018: short-vol blow-up, VIX doubled in days.",
             "historical", date(2018, 1, 26), date(2018, 2, 8)),
    Scenario("rates2022", "2022 rate selloff (H1)",
             "Jan–Jun 2022: fastest hiking cycle in decades hits duration/growth.",
             "historical", date(2022, 1, 3), date(2022, 6, 16)),
    Scenario("svb2023", "Regional banks / SVB (2023)",
             "Mar 6–13 2023: SVB failure, a sharp financials + vol shock.",
             "historical", date(2023, 3, 6), date(2023, 3, 13)),
    Scenario("china2015", "Aug 2015 China / VIX spike",
             "Aug 10–25 2015: yuan devaluation, an ~1100-pt Dow flash drop.",
             "historical", date(2015, 8, 10), date(2015, 8, 25)),
    Scenario("taper2013", "Taper tantrum (2013)",
             "May 22 – Jun 24 2013: rate shock as the Fed signalled tapering.",
             "historical", date(2013, 5, 22), date(2013, 6, 24)),
]

HYPOTHETICAL: list[Scenario] = [
    Scenario("hy_down10", "Broad −10%, vol +50%",
             "A routine correction with a moderate vol bid.",
             "hypothetical", mkt_move=-0.10, vol_shock=0.50),
    Scenario("hy_down20", "Broad −20%, vol +100%",
             "A severe risk-off: bear-market leg with vol doubling.",
             "hypothetical", mkt_move=-0.20, vol_shock=1.00),
    Scenario("hy_up10", "Melt-up +10%, vol −20%",
             "A sharp rally with vol compression (tests short-delta pain).",
             "hypothetical", mkt_move=0.10, vol_shock=-0.20),
]

ALL_SCENARIOS = HISTORICAL + HYPOTHETICAL


@dataclass
class ScenarioResult:
    key: str
    name: str
    blurb: str
    kind: str
    pnl: float                       # book P&L under the scenario ($)
    pnl_pct_aum: float | None        # as % of AUM, when AUM known
    spx_move: float | None           # S&P move over the window (historical)
    vix_move: float | None           # relative VIX change over the window
    contributors: pd.DataFrame       # per-underlying P&L, worst first
    coverage: float                  # share of gross exposure with a real move
    n_proxied: int                   # names moved via beta (no window history)
    n_missing: int                   # names dropped (no history, no beta)
    note: str = ""


def earliest_start() -> date:
    return min(s.start for s in HISTORICAL if s.start)


def fetch_long_history(underlyings, asof: date, cache_dir, log=None):
    """Daily closes for the book's names + SPY + ^VIX back to the oldest crisis.

    Cached per as-of date. Returns a closes DataFrame (may be missing columns
    for names that post-date a given window — the replay proxies those).
    """
    import yfinance as yf
    from pathlib import Path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"scenlib_{asof:%Y%m%d}.parquet"
    wanted = sorted(set(underlyings) | {"SPY", "^VIX"})
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        have = [c for c in wanted if c in cached.columns
                and cached[c].notna().any()]
        if len(have) >= 0.9 * len(wanted):
            return cached

    start = earliest_start() - timedelta(days=15)
    frames = []
    for i in range(0, len(wanted), 80):
        chunk = wanted[i:i + 80]
        if log:
            log(f"  fetching long history {i + 1}–{i + len(chunk)} of {len(wanted)}…")
        df = yf.download(chunk, start=start.isoformat(), auto_adjust=True,
                         progress=False, group_by="column", threads=True)
        if df is None or df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product([df.columns, chunk])
        if "Close" in df.columns.get_level_values(0):
            frames.append(df["Close"])
    if not frames:
        raise RuntimeError("Long-history download returned no data.")
    closes = pd.concat(frames, axis=1)
    closes = closes.loc[:, ~closes.columns.duplicated()]
    if getattr(closes.index, "tz", None) is not None:
        closes.index = closes.index.tz_localize(None)
    closes.to_parquet(cache_file)
    return closes


def _window_return(series: pd.Series, start: date, end: date) -> float | None:
    """Total return between the last close <= end and the first close >= start."""
    s = series.dropna()
    if s.empty:
        return None
    s = s.loc[(s.index >= pd.Timestamp(start) - pd.Timedelta(days=10)) &
              (s.index <= pd.Timestamp(end) + pd.Timedelta(days=10))]
    pre = s.loc[s.index <= pd.Timestamp(start)]
    post = s.loc[s.index >= pd.Timestamp(end)]
    p0 = pre.iloc[-1] if len(pre) else (s.iloc[0] if len(s) else None)
    p1 = post.iloc[0] if len(post) else (s.iloc[-1] if len(s) else None)
    if p0 is None or p1 is None or p0 <= 0:
        return None
    return float(p1 / p0 - 1.0)


def _reprice(positions: pd.DataFrame, ret_by_u: dict, iv_scale: float,
             betas: dict, spx_move: float | None, asof: date):
    """Reprice the book given per-name returns; proxy missing names via beta."""
    pnl_by_u: dict[str, float] = {}
    meta: dict[str, tuple] = {}
    proxied = missing = 0
    covered = gross = 0.0
    for row in positions.itertuples():
        u = row.underlying
        expo = abs(float(getattr(row, "exposure", 0.0)))
        gross += expo
        ret = ret_by_u.get(u)
        if ret is None:
            b = betas.get(u)
            if b is None or spx_move is None:
                missing += 1
                continue
            ret = b * spx_move
            proxied += 1
        else:
            covered += expo
        ret = max(ret, _RET_FLOOR)
        new_spot = max(row.spot * (1.0 + ret), row.spot * 0.01)
        pnl = _reval_position(row, new_spot, iv_scale, asof)
        pnl_by_u[u] = pnl_by_u.get(u, 0.0) + pnl
        if u not in meta:
            meta[u] = (getattr(row, "name", u), getattr(row, "sector", "Unknown"))
    rows = [{"underlying": u, "name": meta[u][0], "sector": meta[u][1],
             "pnl": v} for u, v in pnl_by_u.items()]
    contrib = (pd.DataFrame(rows).sort_values("pnl").reset_index(drop=True)
               if rows else pd.DataFrame(columns=["underlying", "name", "sector", "pnl"]))
    total = float(sum(pnl_by_u.values()))
    coverage = covered / gross if gross else 0.0
    return total, contrib, coverage, proxied, missing


def run_scenario(scenario: Scenario, positions: pd.DataFrame, closes_long,
                 betas: dict, asof: date, aum: float | None) -> ScenarioResult:
    note = ""
    if scenario.kind == "hypothetical":
        # every name moves beta × market; uniform vol shock
        ret_by_u = {}  # all proxied through beta
        iv_scale = 1.0 + (scenario.vol_shock or 0.0)
        total, contrib, coverage, proxied, missing = _reprice(
            positions, ret_by_u, iv_scale, betas, scenario.mkt_move, asof)
        spx_move, vix_move = scenario.mkt_move, scenario.vol_shock
    else:
        # historical: real per-name window returns; VIX move drives IV shock
        spx = closes_long["SPY"] if "SPY" in closes_long else None
        spx_move = _window_return(spx, scenario.start, scenario.end) if spx is not None else None
        vix_move = None
        iv_scale = 1.0
        if "^VIX" in closes_long:
            vr = _window_return(closes_long["^VIX"], scenario.start, scenario.end)
            if vr is not None:
                vix_move = vr
                iv_scale = float(np.clip(1.0 + vr, *_VIX_CLAMP))
        else:
            note = "No VIX history — implied vol held flat (understates the tail)."
        ret_by_u = {}
        for u in positions["underlying"].unique():
            if u in closes_long:
                wr = _window_return(closes_long[u], scenario.start, scenario.end)
                if wr is not None:
                    ret_by_u[u] = wr
        total, contrib, coverage, proxied, missing = _reprice(
            positions, ret_by_u, iv_scale, betas, spx_move, asof)

    return ScenarioResult(
        key=scenario.key, name=scenario.name, blurb=scenario.blurb,
        kind=scenario.kind, pnl=total,
        pnl_pct_aum=(total / aum if aum else None),
        spx_move=spx_move, vix_move=vix_move, contributors=contrib,
        coverage=coverage, n_proxied=proxied, n_missing=missing, note=note,
    )


def run_library(positions: pd.DataFrame, closes_long, betas: dict, asof: date,
                aum: float | None, log=None) -> list[ScenarioResult]:
    out = []
    for sc in ALL_SCENARIOS:
        if log:
            log(f"  scenario: {sc.name}")
        try:
            out.append(run_scenario(sc, positions, closes_long, betas, asof, aum))
        except Exception as exc:  # one bad scenario shouldn't sink the rest
            if log:
                log(f"    skipped ({exc})")
    return out
