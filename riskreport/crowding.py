"""Crowding and short-squeeze analytics from free short-interest data.

Uses the short-interest and institutional-ownership fields already pulled with
each name's profile (yfinance `.info`): short % of float, days-to-cover
(short ratio), the month-over-month change in shares short, and institutional
ownership. No paid securities-finance feed, so this is a proxy for the
Omega Point "crowding / squeeze" lens rather than a hedge-fund 13F overlap.

The risk that matters for this book:
  * Crowded SHORTS you also hold short  -> squeeze risk (a rally forces
    covering; hard/expensive to buy back). Flagged when you are net short a
    name with high short % of float and high days-to-cover.
  * Crowded LONGS in heavily-shorted names -> you are on the popular long
    side of a contested name; informational, lower direct risk.

Exposure weighting uses issuer-level net delta-adjusted dollars, so the
summary reflects where the book's money actually sits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# thresholds for "crowded" (short interest as a fraction of float) and
# "hard to cover" (days of average volume to buy back the short interest)
CROWDED_SI_FLOAT = 0.10
HARD_TO_COVER_DAYS = 5.0


@dataclass
class CrowdingResult:
    coverage: float  # share of gross exposure with short-interest data
    wavg_si_float_long: float | None  # exposure-weighted short%float, long book
    wavg_si_float_short: float | None
    wavg_inst_long: float | None  # exposure-weighted institutional ownership
    squeeze_names: pd.DataFrame  # your shorts in crowded, hard-to-cover names
    crowded_longs: pd.DataFrame  # your longs in heavily-shorted names
    n_crowded_short_exposure: float  # $ of short exposure in crowded names
    issues: list[str] = field(default_factory=list)


def compute_crowding(issuers: pd.DataFrame, positions: pd.DataFrame) -> CrowdingResult:
    """Portfolio crowding/squeeze summary from issuer-level exposure + short data.

    issuers: analytics issuer frame (underlying, name, sector, exposure).
    positions: analytics position frame carrying the per-name short fields.
    """
    issues: list[str] = []
    # one short-data row per underlying (fields are per-issuer, not per-leg)
    si = (
        positions.groupby("underlying")
        .agg(
            short_pct_float=("short_pct_float", "first"),
            short_ratio=("short_ratio", "first"),
            shares_short=("shares_short", "first"),
            shares_short_prior=("shares_short_prior", "first"),
            held_pct_inst=("held_pct_inst", "first"),
        )
    )
    df = issuers.merge(si, on="underlying", how="left")

    gross = float(df["exposure"].abs().sum()) or 1.0
    covered_mask = df["short_pct_float"].notna()
    coverage = float(df.loc[covered_mask, "exposure"].abs().sum()) / gross
    if coverage < 0.80:
        issues.append(
            f"Short-interest data covers {coverage:.0%} of gross exposure "
            "(ETFs and new listings usually lack it)."
        )

    def wavg(mask, col):
        sub = df.loc[mask & df[col].notna()]
        w = sub["exposure"].abs()
        return float((sub[col] * w).sum() / w.sum()) if w.sum() > 0 else None

    long_mask = df["exposure"] > 0
    short_mask = df["exposure"] < 0

    # month-over-month short-interest trend (rising = building pressure)
    df["si_change"] = df["shares_short"] - df["shares_short_prior"]

    # squeeze risk: names you are SHORT that are crowded and hard to cover
    squeeze = df[
        short_mask
        & (df["short_pct_float"] >= CROWDED_SI_FLOAT)
        & (df["short_ratio"] >= HARD_TO_COVER_DAYS)
    ].copy()
    # lead with the largest exposure (money at risk), not the most crowded
    # tiny name — this is a risk report, so material positions come first
    squeeze["_abs_exp"] = squeeze["exposure"].abs()
    squeeze = squeeze.sort_values(
        ["_abs_exp", "short_pct_float"], ascending=False
    )

    crowded_longs = df[
        long_mask & (df["short_pct_float"] >= CROWDED_SI_FLOAT)
    ].copy()
    crowded_longs["_abs_exp"] = crowded_longs["exposure"].abs()
    crowded_longs = crowded_longs.sort_values(
        ["_abs_exp", "short_pct_float"], ascending=False
    )

    crowded_short_exp = float(
        df.loc[
            short_mask & (df["short_pct_float"] >= CROWDED_SI_FLOAT),
            "exposure",
        ].abs().sum()
    )

    return CrowdingResult(
        coverage=coverage,
        wavg_si_float_long=wavg(long_mask, "short_pct_float"),
        wavg_si_float_short=wavg(short_mask, "short_pct_float"),
        wavg_inst_long=wavg(long_mask, "held_pct_inst"),
        squeeze_names=squeeze[
            ["underlying", "name", "sector", "exposure",
             "short_pct_float", "short_ratio", "si_change"]
        ],
        crowded_longs=crowded_longs[
            ["underlying", "name", "sector", "exposure",
             "short_pct_float", "short_ratio", "si_change"]
        ],
        n_crowded_short_exposure=crowded_short_exp,
        issues=issues,
    )
