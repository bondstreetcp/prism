"""Correlation-based risk clustering — the book's 'real bets'.

Concentration says the book behaves like ~N independent risk bets; this shows
*what those bets are*. It builds the name-name return correlation from the
factor model (Σ = B·F·Bᵀ + diag(specific)), clusters the largest positions by
correlation (hierarchical, average linkage on 1−corr distance), and reports
each cluster's members, dominant sector, net exposure, share of portfolio risk,
and average internal correlation.

A cluster is a group of holdings that tend to move together — an implicit
thematic bet the position-by-position view doesn't reveal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ClusterResult:
    table: pd.DataFrame          # one row per cluster
    n_names: int                 # names clustered (largest positions)
    n_clusters: int
    coverage: float              # share of gross exposure clustered
    issues: list = field(default_factory=list)


def risk_clusters(
    factor_risk, model, n_names: int = 40, n_clusters: int = 6,
) -> ClusterResult | None:
    pr = getattr(factor_risk, "position_risk", None)
    if pr is None or len(pr) < 4:
        return None
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        return None

    g = pr.groupby("underlying").agg(
        exposure=("exposure", "sum"), sector=("sector", "first"),
        rc=("risk_contrib", "sum"))
    gross_all = float(pr.groupby("underlying")["exposure"].sum().abs().sum()) or 1.0
    # cluster the largest positions (keeps the correlation matrix tractable)
    top = g.reindex(g["exposure"].abs().sort_values(ascending=False).index)
    top = top.head(min(n_names, len(top)))
    names = list(top.index)
    if len(names) < 4:
        return None

    # cluster on co-movement EXCLUDING the market factor — otherwise every
    # long equity correlates via the market and collapses into one cluster.
    # Stripping market beta reveals the style/thematic (active) groupings.
    fn = [f for f in model.factor_names if f not in ("Mkt-RF", "Mkt")]
    if len(fn) < 2:
        fn = model.factor_names
    F = model.fcov.loc[fn, fn].to_numpy()
    B = model.loadings.loc[names, fn].to_numpy()
    s = model.resid_vol.reindex(names).fillna(0.0).to_numpy()
    cov = B @ F @ B.T + np.diag(s ** 2)
    d = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    corr = np.clip(cov / np.outer(d, d), -1.0, 1.0)

    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0                 # enforce symmetry for squareform
    Z = linkage(squareform(dist, checks=False), method="average")
    k = int(min(max(2, n_clusters), len(names)))
    labels = fcluster(Z, t=k, criterion="maxclust")

    exps = top["exposure"].to_numpy()
    rcs = top["rc"].to_numpy()
    secs = top["sector"].astype(str).to_numpy()
    rows = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        members = [names[i] for i in idx]
        # average pairwise internal correlation
        if len(idx) > 1:
            sub = corr[np.ix_(idx, idx)]
            # off-diagonal mean; subtract the actual trace (not len) so a
            # degenerate zero-variance name with a non-unit diagonal is exact
            avg_corr = float((sub.sum() - np.trace(sub))
                             / (len(idx) * (len(idx) - 1)))
        else:
            avg_corr = float("nan")
        # dominant sector by |exposure|
        sec_w = {}
        for i in idx:
            sec_w[secs[i]] = sec_w.get(secs[i], 0.0) + abs(exps[i])
        dom_sector = max(sec_w, key=sec_w.get) if sec_w else "—"
        # top members by |exposure|
        order = idx[np.argsort(-np.abs(exps[idx]))]
        top_members = ", ".join(names[i] for i in order[:5])
        rows.append({
            "n": len(idx),
            "dominant_sector": dom_sector,
            "top_members": top_members,
            "net_exposure": float(exps[idx].sum()),
            "risk_share": float(rcs[idx].sum()),
            "avg_corr": avg_corr,
        })
    table = (pd.DataFrame(rows).sort_values("risk_share", ascending=False)
             .reset_index(drop=True))
    coverage = float(np.abs(exps).sum()) / gross_all
    return ClusterResult(table=table, n_names=len(names), n_clusters=k,
                         coverage=coverage)
