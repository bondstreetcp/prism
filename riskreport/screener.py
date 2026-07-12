"""Factor screener — Omega Point's security search / screener.

Screens the fitted-loadings universe (the book's own names plus the hedge-ETF
and macro-ETF menu) by factor loading, beta, and model fit, to find hedge
candidates or replacement ideas with a target factor profile. Marks which
names the book already holds so you can tell a new idea from an existing one.
"""

from __future__ import annotations

import pandas as pd

from .factors import FactorModel


def build_screen_frame(
    model: FactorModel, stats: dict, profiles: dict, positions: pd.DataFrame
) -> pd.DataFrame:
    """One row per fitted name: loadings + beta + R² + sector + held flag."""
    held = set(positions["underlying"].unique())
    exp_by_u = positions.groupby("underlying")["exposure"].sum()

    rows = []
    for ticker, loadings in model.loadings.iterrows():
        st = stats.get(ticker)
        prof = profiles.get(ticker, {})
        row = {
            "ticker": ticker,
            "name": (prof.get("name") or ticker)[:30],
            "sector": prof.get("sector") or ("ETF/Index" if prof.get("quote_type") == "ETF" else "—"),
            "beta": None if st is None else st.beta,
            "r2": float(model.r2.get(ticker)) if ticker in model.r2.index and pd.notna(model.r2.get(ticker)) else None,
            "held": ticker in held,
            "exposure": float(exp_by_u.get(ticker, 0.0)),
        }
        for f in model.factor_names:
            row[f] = float(loadings[f])
        rows.append(row)
    return pd.DataFrame(rows)


def screen(
    frame: pd.DataFrame, model: FactorModel, *,
    factor_ranges: dict[str, tuple[float, float]] | None = None,
    beta_range: tuple[float, float] | None = None,
    min_r2: float | None = None,
    sectors: list[str] | None = None,
    held: str = "all",              # "all" | "held" | "not_held"
    sort_by: str | None = None,
    ascending: bool = False,
    limit: int = 40,
) -> pd.DataFrame:
    """Filter and sort the screen frame."""
    df = frame.copy()
    for f, (lo, hi) in (factor_ranges or {}).items():
        if f in df.columns:
            df = df[(df[f] >= lo) & (df[f] <= hi)]
    if beta_range is not None:
        lo, hi = beta_range
        df = df[df["beta"].notna() & (df["beta"] >= lo) & (df["beta"] <= hi)]
    if min_r2 is not None:
        df = df[df["r2"].notna() & (df["r2"] >= min_r2)]
    if sectors:
        df = df[df["sector"].isin(sectors)]
    if held == "held":
        df = df[df["held"]]
    elif held == "not_held":
        df = df[~df["held"]]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending,
                            key=lambda s: s.abs() if sort_by in model.factor_names else s)
    return df.head(limit)
