"""Archive each run so performance/history panels can be built later.

Every report run writes:
  snapshots/<asof>/positions.csv   — position-level analytics
  snapshots/<asof>/summary.json    — portfolio summary block

Once several dated snapshots accumulate, future versions can compute
day-over-day P&L, exposure trends, and return attribution.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .analytics import PortfolioAnalytics


def save_snapshot(
    analytics: PortfolioAnalytics, base_dir: str | Path = "snapshots"
) -> Path:
    out_dir = Path(base_dir) / analytics.asof.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    analytics.positions.to_csv(out_dir / "positions.csv", index=False)

    payload = {
        "asof": analytics.asof.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": analytics.summary,
        "issues": analytics.issues,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return out_dir
