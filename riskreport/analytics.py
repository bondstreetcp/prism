"""Portfolio analytics: pricing, delta-adjusted exposures, and aggregations.

Conventions
-----------
* "Raw" market value: what the positions are worth (equity qty*spot; option
  qty*100*premium). Long/short split is by sign of each position's MV.
* "Delta-adjusted" exposure: equity MV for stocks; qty*100*delta*spot for
  options — the equivalent underlying dollars at risk.
* Issuer-level tables aggregate delta-adjusted exposure per underlying, so a
  short put and a long stock position on the same name net together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from scipy.stats import norm

from .marketdata import TickerStats
from .parse import Position

RISK_FREE_RATE = 0.04
DEFAULT_IV = 0.35
IV_SANITY = (0.01, 5.0)

CAP_BUCKETS = [
    ("Mega (>$200B)", 200e9, math.inf),
    ("Large ($10-200B)", 10e9, 200e9),
    ("Mid ($2-10B)", 2e9, 10e9),
    ("Small ($200M-2B)", 200e6, 2e9),
    ("Micro ($50-200M)", 50e6, 200e6),
    ("Nano (<$50M)", 0, 50e6),
]

NORTH_AMERICA = {"United States", "Canada", "Mexico"}
EUROPE = {
    "United Kingdom", "Germany", "France", "Switzerland", "Netherlands",
    "Ireland", "Spain", "Italy", "Sweden", "Denmark", "Norway", "Finland",
    "Belgium", "Austria", "Portugal", "Luxembourg", "Greece", "Poland",
    "Jersey", "Guernsey", "Isle of Man", "Monaco", "Liechtenstein",
}


def bs_price_delta(
    spot: float, strike: float, t_years: float, iv: float, cp: str,
    rate: float = RISK_FREE_RATE,
) -> tuple[float, float]:
    """Black-Scholes European price and delta (q=0)."""
    if t_years <= 0:
        intrinsic = max(spot - strike, 0.0) if cp == "C" else max(strike - spot, 0.0)
        if cp == "C":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return intrinsic, delta
    sq = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t_years) / sq
    d2 = d1 - sq
    if cp == "C":
        price = spot * norm.cdf(d1) - strike * math.exp(-rate * t_years) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = strike * math.exp(-rate * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
    return price, delta


def bs_greeks(
    spot: float, strike: float, t_years: float, iv: float, cp: str,
    rate: float = RISK_FREE_RATE,
) -> tuple[float, float, float, float, float]:
    """Black-Scholes price and per-share greeks (q=0).

    Returns (price, delta, gamma, vega, theta) where:
      * gamma  = d(delta)/d(spot)          — per $1 of spot
      * vega   = d(price)/d(sigma)         — per 1.00 of vol (×0.01 = per vol pt)
      * theta  = d(price)/d(time)          — per YEAR (÷365 = per calendar day)
    At/after expiry (or degenerate inputs) gamma/vega/theta are zero and price/
    delta fall back to intrinsic.
    """
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        price, delta = bs_price_delta(spot, strike, t_years, iv, cp, rate)
        return price, delta, 0.0, 0.0, 0.0
    sqt = math.sqrt(t_years)
    sq = iv * sqt
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t_years) / sq
    d2 = d1 - sq
    pdf = norm.pdf(d1)
    disc = math.exp(-rate * t_years)
    gamma = pdf / (spot * sq)
    vega = spot * pdf * sqt
    if cp == "C":
        price = spot * norm.cdf(d1) - strike * disc * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -(spot * pdf * iv) / (2.0 * sqt) - rate * strike * disc * norm.cdf(d2)
    else:
        price = strike * disc * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = -(spot * pdf * iv) / (2.0 * sqt) + rate * strike * disc * norm.cdf(-d2)
    return price, delta, gamma, vega, theta


@dataclass
class PortfolioAnalytics:
    asof: date
    positions: pd.DataFrame  # one row per priced position
    issuers: pd.DataFrame  # aggregated per underlying
    summary: dict = field(default_factory=dict)
    sector_table: pd.DataFrame | None = None
    cap_table: pd.DataFrame | None = None
    region_table: pd.DataFrame | None = None
    issues: list[str] = field(default_factory=list)


def _cap_bucket(market_cap) -> str:
    if market_cap is None:
        return "Unknown"
    try:
        mc = float(market_cap)
    except (TypeError, ValueError):
        return "Unknown"
    if not math.isfinite(mc) or mc < 0:
        return "Unknown"
    for label, lo, hi in CAP_BUCKETS:
        if lo <= mc < hi:
            return label
    return "Unknown"


def _region(country: str | None, quote_type: str | None) -> str:
    if quote_type == "ETF":
        return "ETF/Index"
    if not country:
        return "Unknown"
    if country in NORTH_AMERICA:
        return "North America"
    if country in EUROPE:
        return "Europe"
    return "Other"


def _sector(profile: dict) -> str:
    if profile.get("quote_type") == "ETF":
        return "ETF/Index"
    return profile.get("sector") or "Unknown"


# ----------------------------------------------------------------------
# Exposure-basis helpers (Cash / Delta-adjusted / Beta-adjusted)
# Each maps to a column already computed on the issuer / position frames:
#   Cash (MV)      -> market value (equity qty*px; option qty*100*premium)
#   Delta-adjusted -> equity MV + option delta notional
#   Beta-adjusted  -> delta-adjusted * beta vs SPY
# ----------------------------------------------------------------------
BASIS_COLUMNS = {
    "Cash (MV)": "mv",
    "Delta-adjusted": "exposure",
    "Beta-adjusted": "beta_exposure",
}

CAP_ORDER = [label for label, _, _ in CAP_BUCKETS] + ["Unknown"]


def basis_category_table(
    issuers: pd.DataFrame, category_col: str, basis_col: str,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Long/short/net/gross of `basis_col` grouped by a category, per issuer.

    NaNs (e.g. beta_exposure for names without a beta) count as zero, so the
    caller should surface coverage separately for the beta-adjusted basis."""
    s = issuers.assign(_b=issuers[basis_col].fillna(0.0))
    g = s.groupby(category_col).agg(
        long=("_b", lambda x: x[x > 0].sum()),
        short=("_b", lambda x: x[x < 0].sum()),
        net=("_b", "sum"),
        gross=("_b", lambda x: x.abs().sum()),
        n_issuers=("underlying", "count"),
    )
    gross_total = float(g["gross"].sum())
    g["pct_gross"] = g["gross"] / gross_total if gross_total else 0.0
    g = g.sort_values("gross", ascending=False)
    if order:
        present = [x for x in order if x in g.index]
        rest = [x for x in g.index if x not in order]
        g = g.reindex(present + rest)
    return g.reset_index()


def basis_summary(issuers: pd.DataFrame, basis_col: str) -> dict:
    """Long/short/gross/net totals of a basis, plus its coverage of gross."""
    raw = issuers[basis_col]
    s = raw.fillna(0.0)
    long = float(s[s > 0].sum())
    short = float(s[s < 0].sum())
    # coverage = share of |delta-adj| exposure that has a value on this basis
    delta_gross = float(issuers["exposure"].abs().sum()) or 1.0
    covered = float(issuers.loc[raw.notna(), "exposure"].abs().sum())
    return {
        "long": long, "short": short,
        "gross": long - short, "net": long + short,
        "coverage": covered / delta_gross,
    }


def basis_top_issuers(
    issuers: pd.DataFrame, basis_col: str, side: str, n: int = 10
) -> tuple[pd.DataFrame, float]:
    """Top `n` issuers on a basis for one side; returns (rows, side_total)."""
    s = issuers.assign(_b=issuers[basis_col].fillna(0.0))
    if side == "long":
        sel = s[s["_b"] > 0].nlargest(n, "_b")
        total = float(s.loc[s["_b"] > 0, "_b"].sum())
    else:
        sel = s[s["_b"] < 0].nsmallest(n, "_b")
        total = abs(float(s.loc[s["_b"] < 0, "_b"].sum()))
    return sel, total


def build_analytics(
    positions: list[Position],
    stats: dict[str, TickerStats],
    profiles: dict[str, dict],
    option_quotes: dict[str, dict],
    asof: date,
    aum: float | None = None,
    cash: float | None = None,
    issues: list[str] | None = None,
) -> PortfolioAnalytics:
    issues = list(issues or [])
    rows = []
    unpriced: list[str] = []
    iv_fallbacks = 0
    theo_priced = 0

    for p in positions:
        st = stats.get(p.underlying)
        if st is None:
            unpriced.append(p.raw_symbol)
            continue
        profile = profiles.get(p.underlying, {})
        beta = st.beta

        # dollar greeks (option positions only; equities carry delta only)
        gamma_pnl_1pct = 0.0   # convexity P&L from a ±1% underlying move
        vega_dollar = 0.0      # P&L per +1 vol point (0.01 change in IV)
        theta_dollar = 0.0     # P&L per calendar day

        if p.kind == "equity":
            mv = p.qty * st.spot
            exposure = mv
            delta = 1.0
            price_src = "close"
            iv_used = None
            delta_shares = p.qty
        else:
            t_years = (p.expiry - asof).days / 365.0
            quote = option_quotes.get(p.contract_key, {})
            iv = quote.get("iv")
            if iv is None or not (IV_SANITY[0] <= iv <= IV_SANITY[1]):
                iv = st.realized_vol if st.realized_vol else DEFAULT_IV
                iv_fallbacks += 1
            theo, delta, gamma, vega, theta = bs_greeks(
                st.spot, p.strike, t_years, iv, p.cp)
            signed_shares = p.qty * p.multiplier   # signed contract → share count
            gamma_pnl_1pct = 0.5 * (signed_shares * gamma) * (0.01 * st.spot) ** 2
            vega_dollar = signed_shares * vega * 0.01
            theta_dollar = signed_shares * (theta / 365.0)

            bid, ask = quote.get("bid"), quote.get("ask")
            if bid is not None and ask is not None and ask > 0 and ask >= bid:
                premium = (bid + ask) / 2
                price_src = "chain_mid"
            elif quote.get("last"):
                premium = quote["last"]
                price_src = "chain_last"
            else:
                premium = theo
                price_src = "bs_theoretical"
                theo_priced += 1

            mv = p.qty * p.multiplier * premium
            exposure = p.qty * p.multiplier * delta * st.spot
            iv_used = iv
            delta_shares = p.qty * p.multiplier * delta

        rows.append(
            {
                "account": p.account,
                "symbol": p.raw_symbol,
                "underlying": p.underlying,
                "kind": p.kind,
                "qty": p.qty,
                "spot": st.spot,
                "expiry": p.expiry,
                "strike": p.strike,
                "cp": p.cp,
                "adjusted": p.adjusted,
                "iv": iv_used,
                "delta": delta,
                "price_source": price_src,
                "mv": mv,
                "exposure": exposure,
                "delta_shares": delta_shares,
                "gamma_pnl_1pct": gamma_pnl_1pct,
                "vega_dollar": vega_dollar,
                "theta_dollar": theta_dollar,
                "adv_shares": st.adv_shares,
                "spot_date": st.spot_date,
                "beta": beta,
                "beta_exposure": exposure * beta if beta is not None else None,
                "name": profile.get("name") or p.underlying,
                "sector": _sector(profile),
                "cap_bucket": _cap_bucket(profile.get("market_cap")),
                "region": _region(profile.get("country"), profile.get("quote_type")),
                "short_pct_float": profile.get("short_pct_float"),
                "short_ratio": profile.get("short_ratio"),
                "shares_short": profile.get("shares_short"),
                "shares_short_prior": profile.get("shares_short_prior"),
                "held_pct_inst": profile.get("held_pct_inst"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No positions could be priced — cannot build a report.")

    if unpriced:
        issues.append(
            f"{len(unpriced)} position(s) had no price data and were excluded: "
            + ", ".join(unpriced[:8])
            + ("…" if len(unpriced) > 8 else "")
        )
    post_asof = df.loc[
        df["spot_date"].notna() & (df["spot_date"] > asof), "underlying"
    ].unique()
    if len(post_asof):
        issues.append(
            "New listings priced with their first post-as-of close (no "
            f"market close existed on {asof}): " + ", ".join(sorted(post_asof))
        )
    if iv_fallbacks:
        issues.append(
            f"{iv_fallbacks} option(s) used realized-vol fallback for delta "
            "(no usable chain IV)."
        )
    if theo_priced:
        issues.append(
            f"{theo_priced} option(s) priced with Black-Scholes theoretical "
            "value (no usable chain quote)."
        )
    n_adj = int(df["adjusted"].sum())
    if n_adj:
        issues.append(
            f"{n_adj} adjusted option contract(s) treated as standard terms; "
            "their non-standard deliverables make both delta and premium "
            "approximate, not just the multiplier."
        )
    run_date = date.today()
    if run_date != asof and len(df.loc[df["kind"] == "option"]):
        issues.append(
            f"Option quotes (bid/ask/IV) are live as of {run_date}; equity "
            f"spots and deltas use {asof} closes — a market move between the "
            "two dates makes the option book's MV inconsistent with the "
            "equity book."
        )

    # ------------------------------------------------------------------
    # Issuer-level aggregation (delta-adjusted)
    # ------------------------------------------------------------------
    issuers = (
        df.groupby("underlying")
        .agg(
            name=("name", "first"),
            sector=("sector", "first"),
            cap_bucket=("cap_bucket", "first"),
            region=("region", "first"),
            beta=("beta", "first"),
            exposure=("exposure", "sum"),
            mv=("mv", "sum"),
            net_shares=("delta_shares", "sum"),
            adv_shares=("adv_shares", "first"),
            n_positions=("symbol", "count"),
        )
        .reset_index()
    )
    issuers["beta_exposure"] = issuers["exposure"] * issuers["beta"]
    issuers["pct_adv"] = (
        issuers["net_shares"].abs() / issuers["adv_shares"]
    ).where(issuers["adv_shares"] > 0)
    # days to liquidate the net delta-equivalent position at 20% of ADV
    issuers["days_to_liq"] = issuers["pct_adv"] / 0.20

    # ------------------------------------------------------------------
    # Summary block
    # ------------------------------------------------------------------
    mv_long = float(df.loc[df["mv"] > 0, "mv"].sum())
    mv_short = float(df.loc[df["mv"] < 0, "mv"].sum())
    exp_long = float(df.loc[df["exposure"] > 0, "exposure"].sum())
    exp_short = float(df.loc[df["exposure"] < 0, "exposure"].sum())
    iss_long = issuers.loc[issuers["exposure"] > 0]
    iss_short = issuers.loc[issuers["exposure"] < 0]

    opts = df.loc[df["kind"] == "option"]
    eq = df.loc[df["kind"] == "equity"]

    beta_known = df.dropna(subset=["beta_exposure"])
    beta_net = float(beta_known["beta_exposure"].sum())
    beta_coverage = (
        float(beta_known["exposure"].abs().sum() / df["exposure"].abs().sum())
        if float(df["exposure"].abs().sum()) > 0
        else 0.0
    )
    if beta_coverage < 0.9:
        issues.append(
            f"Betas available for only {beta_coverage:.0%} of gross exposure "
            "— beta-adjusted net is understated."
        )

    # AUM = net market value of the book plus cash. Cash comes from the broker
    # file (IBKR) or is supplied by the user (Goldman has no cash line). An
    # explicit `aum` override wins; otherwise derive it when cash is known.
    mv_net = mv_long + mv_short
    if aum is None and cash is not None:
        aum = mv_net + cash

    summary = {
        "aum": aum,
        "cash": cash,
        "mv_long": mv_long,
        "mv_short": mv_short,
        "mv_gross": mv_long - mv_short,
        "mv_net": mv_net,
        "exp_long": exp_long,
        "exp_short": exp_short,
        "exp_gross": exp_long - exp_short,
        "exp_net": exp_long + exp_short,
        "beta_net": beta_net,
        "beta_coverage": beta_coverage,
        "n_instruments": int(len(df)),
        "n_options": int(len(opts)),
        "n_equities": int(len(eq)),
        "n_issuers": int(issuers["underlying"].nunique()),
        "n_issuers_long": int(len(iss_long)),
        "n_issuers_short": int(len(iss_short)),
        # issuer-level (netted) side totals — denominators for top-issuer %s
        "iss_exp_long": float(iss_long["exposure"].sum()),
        "iss_exp_short": float(iss_short["exposure"].sum()),
        "opt_mv_gross": float(opts["mv"].abs().sum()),
        "opt_mv_net": float(opts["mv"].sum()),
        "opt_exp_gross": float(opts["exposure"].abs().sum()),
        "opt_exp_net": float(opts["exposure"].sum()),
        "eq_exp_gross": float(eq["exposure"].abs().sum()),
        "eq_exp_net": float(eq["exposure"].sum()),
    }

    # ------------------------------------------------------------------
    # Portfolio greeks (dollar terms). Delta = delta-adjusted net exposure;
    # gamma/vega/theta come from the option book. A short-premium book reads
    # as short gamma (neg), short vega (neg), long theta (pos).
    # ------------------------------------------------------------------
    summary["greeks"] = {
        "net_delta": summary["exp_net"],
        "net_gamma_1pct": float(df["gamma_pnl_1pct"].sum()),
        "net_vega_1pt": float(df["vega_dollar"].sum()),
        "net_theta_day": float(df["theta_dollar"].sum()),
        "gross_vega_1pt": float(df["vega_dollar"].abs().sum()),
        "opt_delta_net": float(opts["exposure"].sum()),
    }

    # ------------------------------------------------------------------
    # Liquidity (issuer-level, net delta-equivalent shares vs 60d ADV)
    # ------------------------------------------------------------------
    liq = issuers.dropna(subset=["pct_adv"]).copy()
    total_gross = float(issuers["exposure"].abs().sum()) or 1.0
    liq_gross = float(liq["exposure"].abs().sum())
    liq_cov = liq_gross / total_gross

    # bucket shares are % of TOTAL gross (uncovered names contribute 0 to the
    # numerator), so the "% gross" label is honest; coverage is reported
    # separately so a low-coverage book is not read as low-liquidity-risk
    def bucket_share(threshold: float) -> float:
        heavy = liq.loc[liq["pct_adv"] > threshold, "exposure"].abs().sum()
        return float(heavy) / total_gross

    sorted_liq = liq.sort_values("days_to_liq")
    cum_w = sorted_liq["exposure"].abs().cumsum() / (liq_gross or 1.0)

    def weighted_pctile(q: float) -> float | None:
        hit = sorted_liq.loc[cum_w >= q, "days_to_liq"]
        return float(hit.iloc[0]) if len(hit) else None

    summary["liquidity"] = {
        "adv_coverage": liq_cov,
        "pct_gross_over_25adv": bucket_share(0.25),
        "pct_gross_over_50adv": bucket_share(0.50),
        "pct_gross_over_100adv": bucket_share(1.00),
        "days_to_liq_p50": weighted_pctile(0.50),
        "days_to_liq_p95": weighted_pctile(0.95),
    }

    # ------------------------------------------------------------------
    # Category tables (issuer-level, delta-adjusted by default; the app can
    # re-derive any of these on the cash / beta basis via basis_category_table)
    # ------------------------------------------------------------------
    return PortfolioAnalytics(
        asof=asof,
        positions=df,
        issuers=issuers,
        summary=summary,
        sector_table=basis_category_table(issuers, "sector", "exposure"),
        cap_table=basis_category_table(issuers, "cap_bucket", "exposure",
                                       order=CAP_ORDER),
        region_table=basis_category_table(issuers, "region", "exposure"),
        issues=issues,
    )
