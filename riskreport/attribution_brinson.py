"""Brinson-Fachler performance attribution vs the S&P 500.

Decomposes the book's active return (portfolio − benchmark) over a trailing
window into three sector-level effects:

  * Allocation  — from over/under-weighting sectors vs the index:
        (w_p,i − w_b,i) · (R_b,i − R_b)
  * Selection   — from picking better/worse names within a sector:
        w_b,i · (R_p,i − R_b,i)
  * Interaction — the cross term:
        (w_p,i − w_b,i) · (R_p,i − R_b,i)

They sum exactly to R_p − R_b.

Data, all self-contained from what the app already fetches:
  * Portfolio side — current holdings held over the window (buy-and-hold),
    delta-adjusted net exposure as weights, each issuer returning at its
    underlying's realized spot return.
  * Benchmark side — the 11 SPDR sector ETFs give sector returns; S&P 500
    GICS sector weights are a dated constant (approximate, disclosed).

Caveats surfaced to the user: it is a *current-holdings* attribution (not a
start-of-period reconstruction), option legs are attributed at their
underlying's return (theta/vega/gamma P&L is out of scope — the risk tabs
cover that), and net-exposure weighting stretches the classic long-only
Brinson frame for a long/short book.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

# yfinance GICS sector name -> SPDR sector ETF (benchmark sector proxy)
SECTOR_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}
SECTOR_ETFS = sorted(set(SECTOR_ETF.values()))

# Approximate S&P 500 GICS sector weights (dated — refresh occasionally).
# Normalised at use, so only the *relative* weights matter.
SP500_SECTOR_WEIGHT = {
    "Technology": 0.315,
    "Financial Services": 0.130,
    "Healthcare": 0.105,
    "Consumer Cyclical": 0.102,
    "Communication Services": 0.095,
    "Industrials": 0.083,
    "Consumer Defensive": 0.058,
    "Energy": 0.035,
    "Utilities": 0.025,
    "Real Estate": 0.022,
    "Basic Materials": 0.020,
}
SP500_WEIGHT_ASOF = "2026-06"

WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}


@dataclass
class BrinsonResult:
    table: pd.DataFrame      # per-sector weights, returns, and the 3 effects
    r_port: float            # portfolio return over the window
    r_bench: float           # benchmark return over the window
    active: float            # r_port − r_bench
    allocation: float
    selection: float
    interaction: float
    base_dollars: float      # net-exposure base the % effects scale by
    start: date
    end: date
    coverage: float          # share of gross exposure with a usable return
    issues: list[str]


def _window_return(series: pd.Series, start: pd.Timestamp,
                   end: pd.Timestamp) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    pre = s.loc[s.index <= start]
    post = s.loc[s.index <= end]
    if pre.empty or post.empty:
        return None
    p0, p1 = pre.iloc[-1], post.iloc[-1]
    if p0 <= 0:
        return None
    return float(p1 / p0 - 1.0)


def brinson_attribution(
    issuers: pd.DataFrame, closes: pd.DataFrame, asof: date,
    window: str = "3M",
) -> BrinsonResult | None:
    """Sector Brinson-Fachler attribution of current holdings vs the S&P 500."""
    issues: list[str] = []
    ndays = WINDOWS.get(window, 63)
    end = pd.Timestamp(asof)
    idx = closes.index[closes.index <= end]
    if len(idx) <= ndays:
        return None
    start = idx[-(ndays + 1)]

    # ---- per-issuer window returns (underlying spot return) ----------------
    rets = {}
    for u in issuers["underlying"].unique():
        if u in closes.columns:
            r = _window_return(closes[u], start, end)
            if r is not None:
                rets[u] = r
    iss = issuers.copy()
    iss["ret"] = iss["underlying"].map(rets)
    gross = float(iss["exposure"].abs().sum()) or 1.0
    covered = float(iss.loc[iss["ret"].notna(), "exposure"].abs().sum())
    coverage = covered / gross
    iss = iss.dropna(subset=["ret"])
    if iss.empty:
        return None

    net_total = float(iss["exposure"].sum())
    if abs(net_total) < 0.1 * gross:
        issues.append("Book is near market-neutral — net-exposure weights (and "
                      "thus the attribution) are unstable; read with care.")
    base = net_total if abs(net_total) > 1e-9 else gross

    # ---- benchmark sector returns from SPDR ETFs ---------------------------
    bench_ret = {}
    for sec, etf in SECTOR_ETF.items():
        if etf in closes.columns:
            r = _window_return(closes[etf], start, end)
            if r is not None:
                bench_ret[sec] = r
    if len(bench_ret) < 6:
        issues.append("Sector ETF history is sparse — benchmark returns are "
                      "incomplete.")
    # normalise benchmark weights over sectors we have a return for
    bw = {s: w for s, w in SP500_SECTOR_WEIGHT.items() if s in bench_ret}
    wsum = sum(bw.values()) or 1.0
    bw = {s: w / wsum for s, w in bw.items()}
    r_bench = sum(bw[s] * bench_ret[s] for s in bw)

    # ---- portfolio sector weights & returns --------------------------------
    grp = iss.groupby("sector").apply(
        lambda g: pd.Series({
            "sec_exp": g["exposure"].sum(),
            "sec_ret": (g["exposure"] * g["ret"]).sum() / g["exposure"].sum()
            if g["exposure"].sum() != 0 else 0.0,
        }), include_groups=False)

    rows = []
    for sec in SECTOR_ETF:
        if sec not in bench_ret:
            continue
        w_b = bw[sec]
        r_b = bench_ret[sec]
        w_p = float(grp.loc[sec, "sec_exp"] / base) if sec in grp.index else 0.0
        # not held → selection is zero (only the allocation effect applies)
        r_p = float(grp.loc[sec, "sec_ret"]) if sec in grp.index else r_b
        alloc = (w_p - w_b) * (r_b - r_bench)
        selec = w_b * (r_p - r_b)
        inter = (w_p - w_b) * (r_p - r_b)
        rows.append({
            "sector": sec, "w_port": w_p, "w_bench": w_b,
            "r_port": r_p, "r_bench": r_b,
            "allocation": alloc, "selection": selec, "interaction": inter,
            "total": alloc + selec + inter,
        })
    table = pd.DataFrame(rows)
    r_port = float((table["w_port"] * table["r_port"]).sum())
    allocation = float(table["allocation"].sum())
    selection = float(table["selection"].sum())
    interaction = float(table["interaction"].sum())
    table = table.sort_values("total").reset_index(drop=True)

    return BrinsonResult(
        table=table, r_port=r_port, r_bench=r_bench, active=r_port - r_bench,
        allocation=allocation, selection=selection, interaction=interaction,
        base_dollars=base, start=start.date(), end=end.date(),
        coverage=coverage, issues=issues,
    )
