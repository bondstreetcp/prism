"""Time-series trends from the archived snapshot store.

Every report run writes snapshots/<asof>/summary.json (exposures, risk, VaR,
liquidity). This reads them all into one tidy frame so the app can plot how
the book's exposure and risk evolved over time — Omega Point's monitoring
time-series view, built from data the tool already persists.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


def load_trend_series(base_dir: str | Path = "snapshots") -> pd.DataFrame:
    """One row per snapshot date with the headline exposure/risk metrics.

    Missing metrics (e.g. VaR when a run used --no-factors) come back as NaN.
    Returns an empty frame if there are no snapshots yet.
    """
    base = Path(base_dir)
    rows = []
    if not base.exists():
        return pd.DataFrame()

    for d in sorted(base.iterdir()):
        f = d / "summary.json"
        if not d.is_dir() or not f.exists():
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            snap_date = date.fromisoformat(payload.get("asof") or d.name)
        except (json.JSONDecodeError, ValueError, OSError):
            continue
        s = payload.get("summary", {})
        risk = s.get("risk") or {}
        liq = s.get("liquidity") or {}
        rows.append({
            "date": snap_date,
            "exp_long": s.get("exp_long"),
            "exp_short": s.get("exp_short"),
            "exp_gross": s.get("exp_gross"),
            "exp_net": s.get("exp_net"),
            "mv_gross": s.get("mv_gross"),
            "mv_net": s.get("mv_net"),
            "beta_net": s.get("beta_net"),
            "n_instruments": s.get("n_instruments"),
            "n_issuers": s.get("n_issuers"),
            "opt_exp_net": s.get("opt_exp_net"),
            "vol_total": risk.get("vol_total"),
            "vol_factor": risk.get("vol_factor"),
            "vol_specific": risk.get("vol_specific"),
            "factor_var_share": risk.get("factor_var_share"),
            "var_95": risk.get("var_95"),
            "var_99": risk.get("var_99"),
            "es_95": risk.get("es_95"),
            "var_95_spot": risk.get("var_95_spot"),
            "net_gamma_1pct": risk.get("net_gamma_1pct"),
            "net_vega_1pt": risk.get("net_vega_1pt"),
            "net_theta_day": risk.get("net_theta_day"),
            "bias_ratio": risk.get("bias_ratio"),
            "days_to_liq_p95": liq.get("days_to_liq_p95"),
        })
        # per-factor net exposure, if present, as factor__<name> columns
        for fac, val in (risk.get("factor_exposures_net") or {}).items():
            rows[-1][f"factor__{fac}"] = val

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


FACTOR_COLS_PREFIX = "factor__"

TREND_METRICS = {
    # label -> (column, scale, unit) ; scale divides the raw value
    "Net exposure": ("exp_net", 1e6, "$M"),
    "Gross exposure": ("exp_gross", 1e6, "$M"),
    "Long exposure": ("exp_long", 1e6, "$M"),
    "Short exposure": ("exp_short", 1e6, "$M"),
    "Beta-adj net": ("beta_net", 1e6, "$M"),
    "Predicted vol (ann.)": ("vol_total", 1e6, "$M"),
    "1-day 95% VaR": ("var_95", 1e6, "$M"),
    "1-day 99% VaR": ("var_99", 1e6, "$M"),
    "Net vega (per +1 vol pt)": ("net_vega_1pt", 1e3, "$k"),
    "Net theta (per day)": ("net_theta_day", 1e3, "$k"),
    "Net gamma (P&L per ±1%)": ("net_gamma_1pct", 1e3, "$k"),
    "Factor share of variance": ("factor_var_share", 1.0, "fraction"),
    "OOS bias ratio": ("bias_ratio", 1.0, "ratio"),
    "# issuers": ("n_issuers", 1.0, "count"),
}
