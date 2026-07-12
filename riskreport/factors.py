"""Fama-French factor model: loadings, exposures, risk decomposition.

Data: Ken French daily library. Default factor set is the FF5 + momentum +
short/long-term reversal (8 factors); the set is selectable via FACTOR_SETS.

Beyond plain OLS, this estimator adds the pieces a production risk model uses:

  * EWMA weighting (half-life ~90d) of both the loadings regression and the
    factor covariance, so the model tracks the current regime rather than
    weighting a two-year-old day equally with yesterday.
  * Ledoit-Wolf shrinkage of the factor covariance toward its diagonal, so
    the matrix stays well-conditioned as the factor count grows.
  * Short-history shrinkage: a name with 20-60 days of history (recent spins
    and IPOs) is fit and then shrunk toward its sector's median loadings
    instead of being dropped, so coverage does not crater on new listings.
  * A portfolio-level bias test (realized vs predicted vol over a trailing
    window) to validate that the model is calibrated, not just fitted.

The risk math is the standard fundamental-factor form:

    Cov(r_i, r_j) = b_i' F b_j + 1{i=j} * s_i^2
    sigma_p^2     = x' B F B' x + sum_u x_u^2 s_u^2   (specific netted per name)
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

KF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
KF_FILES = {
    "5f": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
    "st_rev": "F-F_ST_Reversal_Factor_daily_CSV.zip",
    "lt_rev": "F-F_LT_Reversal_Factor_daily_CSV.zip",
}

FACTOR_SETS = {
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "ff5mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
    "ff5mom_rev": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "ST_Rev", "LT_Rev"],
}
DEFAULT_FACTOR_SET = "ff5mom_rev"
FACTORS = FACTOR_SETS[DEFAULT_FACTOR_SET]  # canonical default for consumers
ALL_FACTORS = FACTOR_SETS["ff5mom_rev"]

MIN_OBS = 60          # full-confidence fit
SHRINK_OBS = 20       # 20 <= n < 60 -> fit then shrink toward sector median
MAX_OBS = 504         # ~2y window (EWMA down-weights the far tail)
HALF_LIFE = 90        # EWMA half-life, trading days
TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Ken French data
# ----------------------------------------------------------------------
def _parse_kf_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV: header row then YYYYMMDD rows, values in
    percent; stops at the first non-date row (annual blocks / copyright)."""
    lines = text.splitlines()
    start = next(
        i for i, ln in enumerate(lines) if re.match(r"^\d{8},", ln.strip())
    )
    header = [h.strip() for h in lines[start - 1].split(",")]
    header[0] = "date"
    rows = []
    for ln in lines[start:]:
        if not re.match(r"^\d{8},", ln.strip()):
            break
        rows.append([p.strip() for p in ln.split(",")])
    df = pd.DataFrame(rows)
    df = df.iloc[:, : len(header)]
    df.columns = header[: df.shape[1]]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    return df.dropna(how="all", axis=1)


def _normalize_factor_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map Ken French column spellings to our canonical factor names."""
    ren = {}
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "").replace("_", "")
        if cl == "mom":
            ren[c] = "MOM"
        elif cl == "strev":
            ren[c] = "ST_Rev"
        elif cl == "ltrev":
            ren[c] = "LT_Rev"
    return df.rename(columns=ren)


def fetch_factor_returns(cache_dir: str | Path = "cache") -> pd.DataFrame:
    """Daily factor returns (decimal): all available factors + RF, cached ~3d."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "ff_factors_daily.parquet"
    if cache_file.exists():
        age = date.today() - date.fromtimestamp(cache_file.stat().st_mtime)
        if age <= timedelta(days=3):
            cached = pd.read_parquet(cache_file)
            if set(ALL_FACTORS) | {"RF"} <= set(cached.columns):
                return cached

    merged = None
    for name in KF_FILES.values():
        req = urllib.request.Request(
            KF_BASE + name, headers={"User-Agent": "Mozilla/5.0"}
        )
        raw = urllib.request.urlopen(req, timeout=120).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        part = _normalize_factor_cols(
            _parse_kf_csv(z.read(z.namelist()[0]).decode("utf-8", "replace"))
        )
        merged = part if merged is None else merged.join(part, how="inner")

    keep = [c for c in ALL_FACTORS + ["RF"] if c in merged.columns]
    missing = set(FACTORS + ["RF"]) - set(keep)  # only the default set is required
    if missing:
        raise RuntimeError(f"Factor download missing required columns: {missing}")
    merged = merged[keep]
    merged.to_parquet(cache_file)
    return merged


# ----------------------------------------------------------------------
# Estimation helpers
# ----------------------------------------------------------------------
def _ewma_weights(n: int, half_life: float) -> np.ndarray:
    """Normalized EWMA weights, most-recent observation last (largest weight)."""
    lam = 0.5 ** (1.0 / half_life)
    w = lam ** np.arange(n - 1, -1, -1)
    return w / w.sum()


def _ledoit_wolf_delta(X: np.ndarray) -> float:
    """Ledoit-Wolf linear shrinkage intensity toward the diagonal target,
    computed with equal weights (a robust intensity estimate).

    X: T x k, already demeaned. Returns delta in [0, 1]."""
    T = X.shape[0]
    if T < 3:
        return 1.0
    S = (X.T @ X) / T
    # pi: sum over i,j of Var of the entrywise cross-products
    cross = np.einsum("ti,tj->tij", X, X)  # T x k x k
    pi_mat = ((cross - S) ** 2).mean(axis=0)
    pi = pi_mat.sum()
    rho = np.trace(pi_mat)  # diagonal-target: only diagonal error terms count
    off = S - np.diag(np.diag(S))
    gamma = float((off ** 2).sum())
    if gamma <= 0:
        return 0.0
    delta = (pi - rho) / gamma / T
    return float(min(1.0, max(0.0, delta)))


def _factor_covariance(fac_window: pd.DataFrame, factor_names: list[str]) -> tuple[pd.DataFrame, float]:
    """EWMA-weighted factor covariance shrunk (Ledoit-Wolf) toward its diagonal.

    Returns (annualizable daily covariance, shrinkage delta)."""
    F = fac_window[factor_names].to_numpy()
    n = len(F)
    w = _ewma_weights(n, HALF_LIFE)
    mean = w @ F
    Xc = F - mean
    S = (Xc * w[:, None]).T @ Xc  # EWMA-weighted covariance
    delta = _ledoit_wolf_delta(F - F.mean(axis=0))
    D = np.diag(np.diag(S))
    shrunk = (1.0 - delta) * S + delta * D
    return pd.DataFrame(shrunk, index=factor_names, columns=factor_names), delta


def _wls_fit(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """EWMA-weighted least squares with intercept. Returns (betas, resid, r2)."""
    A = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    resid = y - A @ coef
    wmean = w @ y
    ss_tot = float(w @ (y - wmean) ** 2)
    ss_res = float(w @ resid ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coef[1:], resid, r2


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
@dataclass
class FactorModel:
    loadings: pd.DataFrame           # index=ticker, columns=factor_names
    resid_vol: pd.Series             # daily residual vol per ticker
    fcov: pd.DataFrame               # daily factor covariance
    r2: pd.Series
    nobs: pd.Series
    factor_names: list[str]
    data_end: date
    # diagnostics
    avg_r2: float = 0.0
    cond_number: float = 0.0         # condition number of fcov
    cov_shrinkage: float = 0.0       # Ledoit-Wolf delta
    n_full: int = 0                  # names fit with full history
    n_shrunk: int = 0                # names shrunk toward sector median
    half_life: float = HALF_LIFE


def fit_loadings(
    closes: pd.DataFrame,
    factors: pd.DataFrame,
    asof: date,
    factor_names: list[str] | None = None,
    sectors: dict[str, str] | None = None,
) -> FactorModel:
    """EWMA-weighted loadings on the factor set, with short-history names
    shrunk toward their sector's median loadings."""
    factor_names = factor_names or FACTORS
    missing = set(factor_names) - set(factors.columns)
    if missing:
        raise RuntimeError(f"Factor returns missing {missing} for the chosen set.")
    sectors = sectors or {}

    rets = closes.pct_change()
    rets = rets.loc[rets.index <= pd.Timestamp(asof)]
    joint = rets.index.intersection(factors.index)
    rets = rets.loc[joint]
    fac = factors.loc[joint]

    fcov, delta = _factor_covariance(fac.tail(MAX_OBS), factor_names)

    X_all = fac[factor_names]
    rf = fac["RF"]

    full_load, full_resid, r2s, nobs = {}, {}, {}, {}
    short_raw: dict[str, tuple[np.ndarray, float, int]] = {}

    for ticker in rets.columns:
        y_full = (rets[ticker] - rf).dropna().tail(MAX_OBS)
        n = len(y_full)
        if n < SHRINK_OBS:
            continue
        X = X_all.loc[y_full.index].to_numpy()
        y = y_full.to_numpy()
        w = _ewma_weights(n, HALF_LIFE)
        betas, resid, r2 = _wls_fit(y, X, w)
        rvol = float(np.sqrt(w @ resid ** 2) * np.sqrt(n / max(n - len(betas) - 1, 1)))
        if n >= MIN_OBS:
            full_load[ticker] = betas
            full_resid[ticker] = rvol
            r2s[ticker] = r2
            nobs[ticker] = n
        else:
            short_raw[ticker] = (betas, rvol, n)

    load_df = pd.DataFrame.from_dict(
        full_load, orient="index", columns=factor_names
    )

    # sector medians from the fully-fit names, for shrinking short-history names
    n_shrunk = 0
    if short_raw and sectors:
        sec_series = pd.Series({t: sectors.get(t) for t in load_df.index})
        sec_load_median = load_df.groupby(sec_series).median()
        sec_resid_median = pd.Series(full_resid).groupby(sec_series).median()
        overall_resid_median = float(np.median(list(full_resid.values()))) if full_resid else 0.02
        for ticker, (betas, rvol, n) in short_raw.items():
            sec = sectors.get(ticker)
            if sec is None or sec not in sec_load_median.index:
                continue  # no peer group -> leave uncovered
            alpha = n / MIN_OBS  # more history -> trust the raw fit more
            prior = sec_load_median.loc[sec].to_numpy()
            shrunk = alpha * betas + (1.0 - alpha) * prior
            full_load[ticker] = shrunk
            # be conservative on residual vol for thin-history names
            full_resid[ticker] = max(
                rvol, float(sec_resid_median.get(sec, overall_resid_median))
            )
            r2s[ticker] = np.nan
            nobs[ticker] = n
            n_shrunk += 1

    loadings = pd.DataFrame.from_dict(
        full_load, orient="index", columns=factor_names
    )
    resid_vol = pd.Series(full_resid)
    r2_series = pd.Series(r2s)
    cond = float(np.linalg.cond(fcov.to_numpy())) if len(fcov) else 0.0

    return FactorModel(
        loadings=loadings,
        resid_vol=resid_vol,
        fcov=fcov,
        r2=r2_series,
        nobs=pd.Series(nobs),
        factor_names=factor_names,
        data_end=fac.index.max().date(),
        avg_r2=float(r2_series.dropna().mean()) if r2_series.notna().any() else 0.0,
        cond_number=cond,
        cov_shrinkage=delta,
        n_full=len(full_resid) - n_shrunk,
        n_shrunk=n_shrunk,
    )


@dataclass
class FactorRisk:
    exposures: pd.DataFrame
    vol_total: float
    vol_factor: float
    vol_specific: float
    factor_var_share: float
    factor_risk_contrib: pd.Series
    position_risk: pd.DataFrame
    coverage: float
    data_end: date
    factor_names: list[str] = field(default_factory=lambda: list(FACTORS))
    issues: list[str] = field(default_factory=list)


def compute_factor_risk(
    positions: pd.DataFrame, model: FactorModel
) -> FactorRisk:
    """Portfolio factor exposures and predicted-vol decomposition."""
    fnames = model.factor_names
    issues: list[str] = []
    df = positions[["symbol", "name", "underlying", "sector", "exposure"]].copy()
    df["has_model"] = df["underlying"].isin(model.loadings.index)
    gross = float(df["exposure"].abs().sum())
    covered = float(df.loc[df["has_model"], "exposure"].abs().sum())
    coverage = covered / gross if gross else 0.0
    if coverage < 0.95:
        issues.append(
            f"Factor loadings cover {coverage:.0%} of gross exposure; "
            "uncovered positions contribute no modeled risk."
        )

    m = df[df["has_model"]]
    B = model.loadings.loc[m["underlying"], fnames].to_numpy()
    x = m["exposure"].to_numpy()
    s = model.resid_vol.reindex(m["underlying"]).fillna(0.0).to_numpy()
    F = model.fcov.loc[fnames, fnames].to_numpy()

    k = len(fnames)
    expo = np.zeros((3, k))
    for j, mask in enumerate([x > 0, x < 0]):
        expo[j] = (x[mask, None] * B[mask]).sum(axis=0)
    expo[2] = expo[0] + expo[1]
    exposures = pd.DataFrame(expo.T, index=fnames, columns=["long", "short", "net"])

    xf = expo[2]
    factor_var = float(xf @ F @ xf) * TRADING_DAYS

    x_by_u = m.groupby("underlying")["exposure"].sum()
    s_by_u = model.resid_vol.reindex(x_by_u.index).fillna(0.0)
    specific_var = float(((x_by_u * s_by_u) ** 2).sum()) * TRADING_DAYS
    total_var = factor_var + specific_var
    vol_total = float(np.sqrt(total_var)) if total_var > 0 else 0.0

    contrib = pd.Series(xf * (F @ xf) * TRADING_DAYS, index=fnames)
    factor_risk_contrib = contrib / total_var if total_var > 0 else contrib * 0.0

    x_net = m["underlying"].map(x_by_u).to_numpy()
    cov_x = (B @ (F @ (B.T @ x)) + (s**2) * x_net) * TRADING_DAYS
    pos_contrib = x * cov_x
    position_risk = m.assign(
        risk_contrib=pos_contrib / total_var if total_var > 0 else 0.0
    )

    return FactorRisk(
        exposures=exposures,
        vol_total=vol_total,
        vol_factor=float(np.sqrt(factor_var)),
        vol_specific=float(np.sqrt(specific_var)),
        factor_var_share=factor_var / total_var if total_var > 0 else 0.0,
        factor_risk_contrib=factor_risk_contrib,
        position_risk=position_risk,
        coverage=coverage,
        data_end=model.data_end,
        factor_names=fnames,
        issues=issues,
    )


@dataclass
class BiasTest:
    realized_vol: float          # annualized $ vol over the holdout
    predicted_vol: float         # out-of-sample model's predicted $ vol
    ratio: float                 # realized / predicted (target ~1.0)
    window: int                  # holdout trading days
    coverage: float              # share of gross exposure tested


def _book_predicted_vol(x_by_u: pd.Series, model: FactorModel, names: list[str]) -> float:
    """Annualized predicted $ vol of net-per-underlying exposures under a model,
    restricted to `names` (net specific per underlying, same math as the risk
    page) — used so the bias test predicts on exactly the tested universe."""
    fn = model.factor_names
    x = x_by_u.reindex(names).to_numpy()
    B = model.loadings.loc[names, fn].to_numpy()
    F = model.fcov.loc[fn, fn].to_numpy()
    s = model.resid_vol.reindex(names).fillna(0.0).to_numpy()
    xf = x @ B
    var = (float(xf @ F @ xf) + float((x**2 * s**2).sum())) * TRADING_DAYS
    return float(np.sqrt(max(var, 0.0)))


def factor_bias_test(
    positions: pd.DataFrame, closes: pd.DataFrame, factors: pd.DataFrame,
    asof: date, sectors: dict[str, str] | None = None,
    factor_names: list[str] | None = None, window: int = 120,
) -> BiasTest | None:
    """Out-of-sample calibration check (Barra-style predict-then-observe).

    Fit the model on data ending `window` trading days BEFORE the as-of date,
    then compare its predicted vol for the current book to the vol that book
    actually realized over the held-out window. Ratio ≈ 1.0 means the
    estimation methodology (EWMA + shrinkage + specific) is well-scaled
    out-of-sample; a model built the same way today is then trustworthy.
    """
    rets = closes.pct_change()
    idx = rets.index[rets.index <= pd.Timestamp(asof)]
    if len(idx) < window + MIN_OBS + 5:
        return None  # not enough pre-holdout history to fit an OOS model
    holdout = idx[-window:]
    est_end = idx[-window - 1].date()

    model_oos = fit_loadings(closes, factors, est_end, factor_names, sectors)

    x_by_u = positions.groupby("underlying")["exposure"].sum()
    # tested universe: names the OOS model covers AND that have holdout returns
    covered = [u for u in x_by_u.index
               if u in model_oos.loadings.index and u in rets.columns]
    if not covered:
        return None
    R = rets.loc[holdout, covered]
    # drop names too sparse over the holdout instead of zero-filling them
    name_cov = R.notna().mean()
    keep = list(name_cov[name_cov >= 0.9].index)
    if len(keep) < 2:
        return None
    R = R[keep]
    R = R.loc[R.notna().mean(axis=1) >= 0.8].fillna(0.0)
    if len(R) < 40:
        return None

    x = x_by_u.reindex(keep).to_numpy()
    realized = float(np.std(R.to_numpy() @ x) * np.sqrt(TRADING_DAYS))
    predicted = _book_predicted_vol(x_by_u, model_oos, keep)
    if predicted <= 0:
        return None
    gross = float(x_by_u.abs().sum()) or 1.0
    coverage = float(x_by_u.reindex(keep).abs().sum()) / gross
    return BiasTest(
        realized_vol=realized,
        predicted_vol=predicted,
        ratio=realized / predicted,
        window=len(R),
        coverage=coverage,
    )
