"""Performance attribution from archived snapshots.

For each trading day, the previous snapshot's book is held (buy-and-hold
between snapshots) and its model P&L is decomposed:

    total    = full-revaluation P&L (equities linear; options repriced with
               Black-Scholes at the day's spot and shrunken expiry, IV held
               at the snapshot value — so theta and gamma are captured)
    market   = exposure_{t-1} x beta_mkt x MktRF_t
    style    = exposure_{t-1} x (SMB,HML,RMW,CMA,MOM loadings . factor returns)
    specific = total - market - style

Ken French factor returns publish with a ~5-week lag; days beyond their
coverage fall back to a market-only decomposition (SPY return as the market
factor), leaving style at zero and letting specific absorb it — disclosed in
the result notes.

Because broker exports carry quantities only (no trade prices), trades
between snapshots are assumed executed at the prior close; a true trading
effect is not measurable and is disclosed as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import FACTORS, FactorModel
from .scenarios import _bs_price_vec

STYLE_FACTORS = [f for f in FACTORS if f != "Mkt-RF"]  # default; model overrides


def load_snapshots(base_dir: str | Path = "snapshots") -> list[tuple[date, pd.DataFrame]]:
    """All archived snapshots, oldest first."""
    out = []
    base = Path(base_dir)
    if not base.exists():
        return out
    for d in sorted(base.iterdir()):
        f = d / "positions.csv"
        if not d.is_dir() or not f.exists():
            continue
        try:
            snap_date = date.fromisoformat(d.name)
        except ValueError:
            continue
        df = pd.read_csv(f)
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
        out.append((snap_date, df))
    return out


@dataclass
class AttributionResult:
    start: date
    end: date
    daily: pd.DataFrame  # index=date, cols: total, market, style, specific + per-factor
    issuer_specific: pd.Series  # cumulative specific P&L per issuer
    sector_total: pd.Series  # cumulative total P&L per sector
    n_days: int
    n_proxy_days: int  # days attributed market-only (KF not yet published)
    factor_data_end: date | None
    notes: list[str] = field(default_factory=list)


def _day_pnl(book: pd.DataFrame, spot_prev: pd.Series, spot_now: pd.Series,
             prev_day: date, day: date) -> pd.Series:
    """Full-revaluation P&L per position row for one trading day.

    t0 is measured from the PREVIOUS trading day, not day-1, so weekend and
    holiday theta is booked on the following trading day instead of vanishing.
    """
    pnl = np.zeros(len(book))
    for i, row in enumerate(book.itertuples()):
        s0 = spot_prev.get(row.underlying)
        s1 = spot_now.get(row.underlying)
        if s0 is None or s1 is None or not np.isfinite(s0) or not np.isfinite(s1):
            continue
        if row.kind == "equity":
            pnl[i] = row.qty * (s1 - s0)
        else:
            iv = row.iv if row.iv and np.isfinite(row.iv) else 0.35
            t0 = max((row.expiry - prev_day).days, 0) / 365.0
            t1 = max((row.expiry - day).days, 0) / 365.0
            p0 = _bs_price_vec(np.array([s0]), row.strike, t0, iv, row.cp)[0]
            p1 = _bs_price_vec(np.array([s1]), row.strike, t1, iv, row.cp)[0]
            pnl[i] = row.qty * 100.0 * (p1 - p0)
    return pd.Series(pnl, index=book.index)


def compute_attribution(
    snapshots: list[tuple[date, pd.DataFrame]],
    closes: pd.DataFrame,
    factor_returns: pd.DataFrame | None,
    model: FactorModel | None,
    end: date | None = None,
) -> AttributionResult:
    if not snapshots:
        raise ValueError("No snapshots found — run run_report.py first.")

    closes = closes.copy()
    closes.index = pd.to_datetime(closes.index)
    trade_days = [d.date() for d in closes.index]
    start = snapshots[0][0]
    end = end or trade_days[-1]
    days = [d for d in trade_days if start < d <= end]
    if not days:
        raise ValueError(
            f"No trading days between first snapshot ({start}) and {end}."
        )

    fac_end = factor_returns.index.max().date() if factor_returns is not None else None
    spy_ret = closes["SPY"].pct_change() if "SPY" in closes.columns else None

    # factor set is whatever the model was fit on (ordered, Mkt-RF included)
    fnames = list(model.factor_names) if model is not None else list(FACTORS)
    style_factors = [f for f in fnames if f != "Mkt-RF"]
    mkt_idx = fnames.index("Mkt-RF") if "Mkt-RF" in fnames else 0

    daily_rows = []
    issuer_specific: dict[str, float] = {}
    sector_total: dict[str, float] = {}
    n_proxy = 0

    for day in days:
        # book = latest snapshot strictly before this day
        book = None
        for snap_date, df in snapshots:
            if snap_date < day:
                book = df
            else:
                break
        if book is None:
            continue
        day_ts = pd.Timestamp(day)
        prev_idx = closes.index[closes.index < day_ts]
        if not len(prev_idx):
            continue
        spot_prev = closes.loc[prev_idx[-1]]
        spot_now = closes.loc[day_ts]

        pnl = _day_pnl(book, spot_prev, spot_now, prev_idx[-1].date(), day)
        total = float(pnl.sum())

        # prior-day exposure per position (delta-adjusted, approximated by
        # scaling the snapshot exposure to the prior-day spot)
        spots_prev = book["underlying"].map(spot_prev)
        scale = (spots_prev / book["spot"]).fillna(1.0)
        exp_prev = book["exposure"] * scale

        market = style = 0.0
        style_parts = dict.fromkeys(style_factors, 0.0)
        if model is not None:
            B = model.loadings.reindex(index=book["underlying"], columns=fnames).to_numpy()
            covered = ~np.isnan(B).any(axis=1)
            x = exp_prev.to_numpy()
            if fac_end is not None and day <= fac_end and pd.Timestamp(day) in factor_returns.index:
                f = factor_returns.loc[pd.Timestamp(day)]
                market = float(np.nansum(
                    np.where(covered, x * B[:, mkt_idx] * f["Mkt-RF"], 0.0)
                ))
                for sf in style_factors:
                    j = fnames.index(sf)
                    part = float(np.nansum(
                        np.where(covered, x * B[:, j] * f[sf], 0.0)
                    ))
                    style_parts[sf] = part
                    style += part
            elif spy_ret is not None and day_ts in spy_ret.index:
                n_proxy += 1
                r_m = float(spy_ret.loc[day_ts])
                market = float(np.nansum(
                    np.where(covered, x * B[:, mkt_idx] * r_m, 0.0)
                ))

        specific = total - market - style
        daily_rows.append(
            {"date": day, "total": total, "market": market,
             "style": style, "specific": specific, **style_parts}
        )

        # cumulative contributions (specific by issuer, total by sector)
        pos_factor = pd.Series(0.0, index=book.index)
        if model is not None:
            f_vec = None
            if fac_end is not None and day <= fac_end and pd.Timestamp(day) in factor_returns.index:
                f_vec = factor_returns.loc[pd.Timestamp(day), fnames].to_numpy()
                B_all = model.loadings.reindex(index=book["underlying"], columns=fnames).to_numpy()
                pos_factor = pd.Series(
                    np.where(~np.isnan(B_all).any(axis=1),
                             exp_prev.to_numpy() * (B_all @ f_vec), 0.0),
                    index=book.index,
                )
            elif spy_ret is not None and day_ts in spy_ret.index:
                B_all = model.loadings.reindex(index=book["underlying"], columns=fnames).to_numpy()
                pos_factor = pd.Series(
                    np.where(~np.isnan(B_all).any(axis=1),
                             exp_prev.to_numpy() * B_all[:, mkt_idx] * float(spy_ret.loc[day_ts]), 0.0),
                    index=book.index,
                )
        pos_specific = pnl - pos_factor
        for u, v in pos_specific.groupby(book["underlying"]).sum().items():
            issuer_specific[u] = issuer_specific.get(u, 0.0) + float(v)
        for sec, v in pnl.groupby(book["sector"]).sum().items():
            sector_total[sec] = sector_total.get(sec, 0.0) + float(v)

    daily = pd.DataFrame(daily_rows).set_index("date")
    notes = []
    if n_proxy:
        notes.append(
            f"{n_proxy} of {len(daily)} day(s) fall after the Ken French "
            f"data end ({fac_end}) and use a market-only decomposition "
            "(SPY as market factor); style P&L for those days sits in "
            "'specific'."
        )
    notes.append(
        "Broker exports carry no trade prices; trades between snapshots are "
        "assumed executed at the prior close, so no separate trading effect "
        "is shown."
    )

    return AttributionResult(
        start=start, end=days[-1], daily=daily,
        issuer_specific=pd.Series(issuer_specific).sort_values(),
        sector_total=pd.Series(sector_total).sort_values(ascending=False),
        n_days=len(daily), n_proxy_days=n_proxy,
        factor_data_end=fac_end, notes=notes,
    )
