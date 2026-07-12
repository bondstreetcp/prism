"""Custom thematic tags — Omega Point's thematic lens (user-supplied version).

Upload a simple CSV mapping tickers to themes ("Ticker,Theme", one row per
tag; a ticker can appear on several rows), and the book's delta-adjusted
exposure is grouped by theme. A name can belong to several themes, so theme
exposures overlap by design (a name can be both "AI" and "Semis").
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from .parse import normalize_ticker


def parse_tags(path: str | Path) -> dict[str, list[str]]:
    """Parse a Ticker,Theme[,Theme...] CSV into {normalized_ticker: [themes]}.

    Extra theme cells and ';'/','/'|'-separated lists are all supported."""
    tags: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row or not row[0].strip():
                continue
            first = row[0].strip()
            if i == 0 and first.lower() in ("ticker", "symbol"):
                continue  # header
            themes: list[str] = []
            for cell in row[1:]:
                for t in re.split(r"[;,|]", cell):
                    t = t.strip()
                    if t:
                        themes.append(t)
            if themes:
                tags.setdefault(normalize_ticker(first), []).extend(themes)
    return {k: sorted(set(v)) for k, v in tags.items()}


def theme_exposure(
    issuers: pd.DataFrame, tags: dict[str, list[str]], basis_col: str = "exposure"
) -> tuple[pd.DataFrame, float]:
    """Long/short/net/gross of `basis_col` grouped by theme, plus tag coverage.

    Returns (table sorted by gross, coverage = share of gross exposure that
    carries at least one tag)."""
    s = issuers.assign(_b=issuers[basis_col].fillna(0.0))
    gross_total = float(s["_b"].abs().sum()) or 1.0

    acc: dict[str, dict] = {}
    tagged_gross = 0.0
    for _, r in s.iterrows():
        themes = tags.get(r["underlying"], [])
        v = float(r["_b"])
        if themes:
            tagged_gross += abs(v)
        for th in themes:
            a = acc.setdefault(th, {"long": 0.0, "short": 0.0, "gross": 0.0, "n_issuers": 0})
            a["long"] += max(v, 0.0)
            a["short"] += min(v, 0.0)
            a["gross"] += abs(v)
            a["n_issuers"] += 1

    rows = []
    for th, a in acc.items():
        rows.append({
            "theme": th, "long": a["long"], "short": a["short"],
            "net": a["long"] + a["short"], "gross": a["gross"],
            "n_issuers": a["n_issuers"],
            "pct_gross": a["gross"] / gross_total,
        })
    table = pd.DataFrame(rows).sort_values("gross", ascending=False) if rows \
        else pd.DataFrame(columns=["theme", "long", "short", "net", "gross",
                                   "n_issuers", "pct_gross"])
    coverage = tagged_gross / gross_total
    return table.reset_index(drop=True), coverage
