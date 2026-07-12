"""Threshold monitoring: check the book against configured risk limits.

Config is a JSON file (see alerts_example.json). Any limit set to null is
skipped. Ratios are fractions of gross delta-adjusted exposure unless the
name says otherwise. Violations surface on the console and in a highlighted
block on the tearsheet.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT_CONFIG = {
    "max_net_pct_gross": None,        # e.g. 0.70 -> net must stay <= 70% of gross
    "max_gross_usd": None,            # e.g. 40e6
    "max_issuer_pct_gross": None,     # e.g. 0.10 per single issuer
    "max_sector_pct_gross": None,     # e.g. 0.35 per sector
    "max_var95_pct_gross": None,      # e.g. 0.015
    "max_predicted_vol_pct_gross": None,  # e.g. 0.12 annualized
    "max_beta_adj_net_usd": None,
    "max_days_to_liq_p95": None,      # e.g. 5.0 trading days
    "max_crowded_short_pct_gross": None,  # short exposure in crowded names
}


def load_config(path: str | Path | None) -> dict | None:
    """Load an alert config; None when no config file is available."""
    if path is None:
        default = Path("alerts.json")
        if not default.exists():
            return None
        path = default
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = dict(DEFAULT_CONFIG)
    unknown = set(raw) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown alert config keys: {sorted(unknown)}")
    cfg.update(raw)
    return cfg


def check_alerts(analytics, factor_risk=None, scenarios=None,
                 config: dict | None = None, crowding=None) -> list[str]:
    """Return a list of human-readable limit violations (empty = all clear)."""
    if not config:
        return []
    s = analytics.summary
    gross = s["exp_gross"] or 1.0
    hits: list[str] = []

    def check(limit_key, value, label, fmt=lambda v: f"{v:.1%}"):
        limit = config.get(limit_key)
        if limit is None or value is None:
            return
        if isinstance(value, float) and math.isnan(value):
            return
        if value > limit:
            hits.append(f"{label}: {fmt(value)} exceeds limit {fmt(limit)}")

    check("max_net_pct_gross", abs(s["exp_net"]) / gross, "Net/gross ratio")
    check("max_gross_usd", s["exp_gross"], "Gross exposure",
          fmt=lambda v: f"${v/1e6:,.1f}M")
    check("max_beta_adj_net_usd", abs(s["beta_net"]), "Beta-adjusted net",
          fmt=lambda v: f"${v/1e6:,.1f}M")

    top_issuer = analytics.issuers.loc[
        analytics.issuers["exposure"].abs().idxmax()
    ]
    check("max_issuer_pct_gross", abs(top_issuer["exposure"]) / gross,
          f"Largest issuer ({top_issuer['underlying']})")

    if analytics.sector_table is not None and len(analytics.sector_table):
        top_sec = analytics.sector_table.iloc[0]
        check("max_sector_pct_gross", float(top_sec["gross"]) / gross,
              f"Largest sector ({top_sec['sector']})")

    if scenarios is not None:
        check("max_var95_pct_gross",
              scenarios.var_95 / gross if scenarios.var_95 == scenarios.var_95 else None,
              "1-day 95% VaR / gross")
    if factor_risk is not None:
        check("max_predicted_vol_pct_gross", factor_risk.vol_total / gross,
              "Predicted vol / gross")

    liq = s.get("liquidity") or {}
    check("max_days_to_liq_p95", liq.get("days_to_liq_p95"),
          "Days-to-liquidate 95th %ile", fmt=lambda v: f"{v:,.1f}d")

    if crowding is not None:
        check("max_crowded_short_pct_gross",
              crowding.n_crowded_short_exposure / gross,
              "Short exposure in crowded names")

    return hits
