"""Fixed-income (interest-rate) risk for the bond-ETF sleeve.

The equity factor model treats bond/rate ETFs as if they were stocks, which
mis-states their real risk — it is interest-rate and credit-spread driven, not
equity-factor driven. This module gives the correct lens for that sleeve:

  * DV01 / dollar duration   — $ P&L per +1bp parallel rate move
  * key-rate DV01            — the same split across 2y / 5y / 10y / 30y, so a
                               curve twist (steepener/flattener) can be priced
  * CS01                     — $ P&L per +1bp credit-spread widening (credit ETFs)
  * rate scenarios           — parallel shifts and curve twists → book P&L

Durations come from a curated, dated table of published ETF effective (and
spread) durations — the pragmatic free-data path, refreshed occasionally. Only
holdings in the table are treated as fixed income; everything else is left to
the equity engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BUCKETS = ["2y", "5y", "10y", "30y"]

# ticker -> (effective_duration, spread_duration, kind, key-rate weights)
# weights over BUCKETS sum to 1; kind in govt/credit/hy/tips/muni/mbs/em/short.
# Durations are approximate, dated ~2026-06 — refresh periodically.
def _w(**kw):
    v = {b: 0.0 for b in BUCKETS}
    v.update(kw)
    return v


BOND_ETF: dict[str, tuple] = {
    # Treasuries — short to long
    "BIL": (0.1, 0.0, "short", _w(**{"2y": 1.0})),
    "SHV": (0.3, 0.0, "short", _w(**{"2y": 1.0})),
    "SGOV": (0.1, 0.0, "short", _w(**{"2y": 1.0})),
    "SHY": (1.9, 0.0, "govt", _w(**{"2y": 1.0})),
    "VGSH": (1.9, 0.0, "govt", _w(**{"2y": 1.0})),
    "SCHO": (1.9, 0.0, "govt", _w(**{"2y": 1.0})),
    "IEI": (4.5, 0.0, "govt", _w(**{"2y": 0.2, "5y": 0.8})),
    "SCHR": (4.5, 0.0, "govt", _w(**{"2y": 0.2, "5y": 0.8})),
    "VGIT": (5.5, 0.0, "govt", _w(**{"5y": 0.6, "10y": 0.4})),
    "IEF": (7.5, 0.0, "govt", _w(**{"5y": 0.3, "10y": 0.7})),
    "GOVT": (6.0, 0.0, "govt", _w(**{"2y": 0.15, "5y": 0.35, "10y": 0.4, "30y": 0.1})),
    "TLH": (12.0, 0.0, "govt", _w(**{"10y": 0.6, "30y": 0.4})),
    "VGLT": (15.0, 0.0, "govt", _w(**{"10y": 0.3, "30y": 0.7})),
    "SPTL": (15.5, 0.0, "govt", _w(**{"10y": 0.3, "30y": 0.7})),
    "TLT": (16.5, 0.0, "govt", _w(**{"10y": 0.3, "30y": 0.7})),
    "EDV": (24.0, 0.0, "govt", _w(**{"30y": 1.0})),
    # Aggregate
    "AGG": (6.0, 1.5, "credit", _w(**{"2y": 0.15, "5y": 0.35, "10y": 0.4, "30y": 0.1})),
    "BND": (6.0, 1.5, "credit", _w(**{"2y": 0.15, "5y": 0.35, "10y": 0.4, "30y": 0.1})),
    "SPAB": (6.0, 1.5, "credit", _w(**{"2y": 0.15, "5y": 0.35, "10y": 0.4, "30y": 0.1})),
    "SCHZ": (6.0, 1.5, "credit", _w(**{"2y": 0.15, "5y": 0.35, "10y": 0.4, "30y": 0.1})),
    # Investment-grade credit
    "LQD": (8.5, 8.5, "credit", _w(**{"5y": 0.3, "10y": 0.5, "30y": 0.2})),
    "VCIT": (5.0, 5.0, "credit", _w(**{"5y": 0.6, "10y": 0.4})),
    "VCSH": (2.7, 2.7, "credit", _w(**{"2y": 0.6, "5y": 0.4})),
    "IGIB": (5.0, 5.0, "credit", _w(**{"5y": 0.6, "10y": 0.4})),
    "IGSB": (2.7, 2.7, "credit", _w(**{"2y": 0.6, "5y": 0.4})),
    "VCLT": (13.0, 13.0, "credit", _w(**{"10y": 0.4, "30y": 0.6})),
    "IGLB": (13.0, 13.0, "credit", _w(**{"10y": 0.4, "30y": 0.6})),
    "USIG": (7.0, 7.0, "credit", _w(**{"5y": 0.4, "10y": 0.5, "30y": 0.1})),
    # High yield
    "HYG": (3.5, 3.5, "hy", _w(**{"2y": 0.3, "5y": 0.5, "10y": 0.2})),
    "JNK": (3.5, 3.5, "hy", _w(**{"2y": 0.3, "5y": 0.5, "10y": 0.2})),
    "SHYG": (2.0, 2.0, "hy", _w(**{"2y": 0.6, "5y": 0.4})),
    "USHY": (3.5, 3.5, "hy", _w(**{"2y": 0.3, "5y": 0.5, "10y": 0.2})),
    "ANGL": (4.5, 4.5, "hy", _w(**{"5y": 0.7, "10y": 0.3})),
    # TIPS (real duration)
    "TIP": (7.0, 0.0, "tips", _w(**{"5y": 0.3, "10y": 0.6, "30y": 0.1})),
    "SCHP": (7.0, 0.0, "tips", _w(**{"5y": 0.3, "10y": 0.6, "30y": 0.1})),
    "VTIP": (2.5, 0.0, "tips", _w(**{"2y": 0.7, "5y": 0.3})),
    "STIP": (2.5, 0.0, "tips", _w(**{"2y": 0.7, "5y": 0.3})),
    # Municipals
    "MUB": (6.0, 4.0, "muni", _w(**{"5y": 0.4, "10y": 0.5, "30y": 0.1})),
    "VTEB": (6.0, 4.0, "muni", _w(**{"5y": 0.4, "10y": 0.5, "30y": 0.1})),
    "TFI": (6.5, 4.0, "muni", _w(**{"5y": 0.4, "10y": 0.5, "30y": 0.1})),
    # Agency MBS
    "MBB": (6.0, 1.0, "mbs", _w(**{"5y": 0.4, "10y": 0.5, "30y": 0.1})),
    "VMBS": (6.0, 1.0, "mbs", _w(**{"5y": 0.4, "10y": 0.5, "30y": 0.1})),
    # Emerging / international
    "EMB": (7.0, 7.0, "em", _w(**{"5y": 0.2, "10y": 0.6, "30y": 0.2})),
    "PCY": (8.0, 8.0, "em", _w(**{"10y": 0.6, "30y": 0.4})),
    "BNDX": (7.0, 2.0, "credit", _w(**{"5y": 0.3, "10y": 0.6, "30y": 0.1})),
    # Bank loans / floating (near-zero rate duration)
    "BKLN": (0.2, 3.0, "hy", _w(**{"2y": 1.0})),
    "SRLN": (0.2, 3.0, "hy", _w(**{"2y": 1.0})),
    "FLOT": (0.1, 0.5, "credit", _w(**{"2y": 1.0})),
}
DURATION_ASOF = "2026-06"

# rate scenarios: label -> per-bucket bp change (parallel or twist)
def _parallel(bp):
    return {b: bp for b in BUCKETS}


RATE_SCENARIOS = {
    "+100bp parallel": _parallel(100),
    "+50bp parallel": _parallel(50),
    "+25bp parallel": _parallel(25),
    "-25bp parallel": _parallel(-25),
    "-50bp parallel": _parallel(-50),
    "-100bp parallel": _parallel(-100),
    "Bear steepener": {"2y": 10, "5y": 25, "10y": 40, "30y": 50},
    "Bull flattener": {"2y": -10, "5y": -25, "10y": -40, "30y": -50},
    "Bear flattener": {"2y": 50, "5y": 40, "10y": 25, "30y": 10},
    "Bull steepener": {"2y": -50, "5y": -40, "10y": -25, "30y": -10},
}


@dataclass
class FIRisk:
    holdings: pd.DataFrame       # per FI holding: mv, duration, dv01, cs01, kind
    total_mv: float
    total_dv01: float            # $ per +1bp parallel (negative = loses when rates rise)
    dollar_duration: float       # $ per +1% (100bp) parallel = total_dv01 * 100
    total_cs01: float            # $ per +1bp credit-spread widening
    krd_dv01: dict               # per-bucket DV01
    by_kind: pd.DataFrame        # DV01 grouped by instrument kind
    scenarios: pd.DataFrame      # rate-scenario P&L
    issues: list = field(default_factory=list)


def compute_fi_risk(issuers: pd.DataFrame) -> FIRisk | None:
    """Interest-rate risk for the fixed-income ETFs in the book.

    `issuers` must carry `underlying` and `mv` (market value per name).
    Returns None when the book holds no recognised fixed-income ETFs.
    """
    df = issuers[issuers["underlying"].isin(BOND_ETF)].copy()
    if df.empty:
        return None

    rows = []
    krd = {b: 0.0 for b in BUCKETS}
    for _, r in df.iterrows():
        eff, spr, kind, w = BOND_ETF[r["underlying"]]
        mv = float(r["mv"])
        dv01 = -mv * eff * 1e-4              # $ per +1bp (long bond -> negative)
        cs01 = -mv * spr * 1e-4
        for b in BUCKETS:
            krd[b] += -mv * eff * w[b] * 1e-4
        rows.append({
            "underlying": r["underlying"], "kind": kind, "mv": mv,
            "duration": eff, "spread_dur": spr, "dv01": dv01, "cs01": cs01,
        })
    holdings = pd.DataFrame(rows).sort_values("dv01").reset_index(drop=True)
    total_mv = float(holdings["mv"].sum())
    total_dv01 = float(holdings["dv01"].sum())
    total_cs01 = float(holdings["cs01"].sum())

    by_kind = (holdings.groupby("kind")
               .agg(mv=("mv", "sum"), dv01=("dv01", "sum"), cs01=("cs01", "sum"))
               .sort_values("dv01").reset_index())

    # rate scenarios: P&L = Σ_bucket krd_dv01[bucket] × Δbp[bucket]
    scen_rows = []
    for label, shifts in RATE_SCENARIOS.items():
        pnl = sum(krd[b] * shifts.get(b, 0) for b in BUCKETS)
        scen_rows.append({"scenario": label, "pnl": pnl})
    scenarios = pd.DataFrame(scen_rows)

    return FIRisk(
        holdings=holdings, total_mv=total_mv, total_dv01=total_dv01,
        dollar_duration=total_dv01 * 100.0, total_cs01=total_cs01,
        krd_dv01=krd, by_kind=by_kind, scenarios=scenarios,
        issues=["Durations are a dated approximation (~" + DURATION_ASOF
                + "); rate scenarios use a duration (linear) approximation, no "
                "convexity. These ETFs also appear in the equity exposures, "
                "where their rate risk is not modeled."],
    )
