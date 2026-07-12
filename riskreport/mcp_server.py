"""MCP server — exposes the portfolio risk analytics to AI assistants.

Omega Point's other AI-native surface: an assistant (Claude Desktop, etc.)
connects to this stdio MCP server, loads a book, and queries its risk.

Run:
    pip install mcp
    python -m riskreport.mcp_server            # stdio

MCP client config (e.g. Claude Desktop) — a stdio server:
    { "command": "python", "args": ["-m", "riskreport.mcp_server"],
      "cwd": "<this project>", "env": {"RISK_CACHE_DIR": "cache"} }

The analytics query functions below take a ReportResult and are plain,
importable, and testable; the MCP wiring in build_server() lazily imports the
`mcp` package so this module imports fine without it.
"""

from __future__ import annotations


# ----------------------------------------------------------------------
# Query functions (no MCP dependency — pure, testable)
# ----------------------------------------------------------------------
def q_exposures(res) -> dict:
    s = res.summary
    return {
        "as_of": str(res.asof), "name": res.name,
        "aum_$": s.get("aum"), "cash_$": s.get("cash"),
        "long_$": s["exp_long"], "short_$": s["exp_short"],
        "gross_$": s["exp_gross"], "net_$": s["exp_net"],
        "beta_adj_net_$": s["beta_net"],
        "n_instruments": s["n_instruments"], "n_issuers": s["n_issuers"],
    }


def q_factor_exposures(res) -> dict:
    fr = res.factor_risk
    if fr is None:
        return {"error": "factor model unavailable for this book"}
    return {
        "predicted_vol_ann_$": fr.vol_total,
        "factor_share_of_variance": fr.factor_var_share,
        "net_factor_exposures_$": {f: float(v) for f, v in fr.exposures["net"].items()},
        "coverage": fr.coverage,
    }


def q_risk_summary(res) -> dict:
    s = res.summary
    risk = s.get("risk") or {}
    out = {k: risk.get(k) for k in ("var_95", "var_99", "es_95", "vol_total",
                                    "factor_var_share", "bias_ratio")}
    if res.crowding is not None:
        cr = res.crowding
        out["crowding"] = {
            "short_book_avg_short_pct_float": cr.wavg_si_float_short,
            "num_squeeze_risk_names": int(len(cr.squeeze_names)),
        }
    out["liquidity"] = s.get("liquidity")
    out["limit_breaches"] = res.alert_hits
    return out


def q_top_issuers(res, side: str = "long", n: int = 10) -> list:
    iss = res.analytics.issuers
    sel = (iss[iss["exposure"] > 0].nlargest(n, "exposure") if side == "long"
           else iss[iss["exposure"] < 0].nsmallest(n, "exposure"))
    aum = res.summary.get("aum")
    return [{
        "ticker": r["underlying"], "name": str(r["name"] or r["underlying"]),
        "sector": str(r["sector"]), "exposure_$": float(r["exposure"]),
        "pct_aum": (float(r["exposure"]) / aum) if aum else None,
    } for _, r in sel.iterrows()]


def q_macro(res) -> dict:
    if res.factor_risk is None or res.closes is None:
        return {"error": "macro overlay needs the factor model and price history"}
    from .macro import compute_macro
    mac = compute_macro(res.analytics.positions, res.closes, res.asof)
    return {
        "market_beta_$_per_1pct": mac.market_beta, "r2": mac.r2,
        "macro_betas_$_per_1pct": {r["factor"]: r["beta_per_1pct"]
                                   for _, r in mac.betas.iterrows()},
    }


def q_optimize(res, objective: str = "Minimize total risk",
               turnover_max_usd: float | None = None,
               market_cap_usd: float | None = None) -> dict:
    if res.factor_risk is None or res.model is None:
        return {"error": "optimizer needs the factor model"}
    from .optimizer import optimize_overlay
    caps = {"Mkt-RF": market_cap_usd} if market_cap_usd else None
    opt = optimize_overlay(res.factor_risk, res.model, res.stats,
                           objective=objective, turnover_max=turnover_max_usd,
                           factor_caps=caps)
    return {
        "objective": objective, "success": opt.success,
        "vol_before_$": opt.vol_before, "vol_after_$": opt.vol_after,
        "turnover_$": opt.turnover, "binding": opt.binding,
        "trades": [{"etf": r["etf"], "notional_$": float(r["notional"]),
                    "shares": r["shares"]} for _, r in opt.trades.iterrows()],
    }


# ----------------------------------------------------------------------
# MCP wiring (lazy import so this module loads without the `mcp` package)
# ----------------------------------------------------------------------
def build_server():
    from mcp.server.fastmcp import FastMCP

    from .pipeline import generate_report

    mcp = FastMCP("portfolio-risk")
    state: dict = {"result": None}

    def _require():
        if state["result"] is None:
            raise ValueError("No book loaded. Call load_book(csv_path) first.")
        return state["result"]

    @mcp.tool()
    def load_book(csv_path: str, cash: float | None = None) -> dict:
        """Run the risk pipeline on a broker position CSV (Goldman or IBKR)
        and cache it for subsequent queries. Returns headline exposure/risk."""
        state["result"] = generate_report(csv_path, cash=cash)
        return q_exposures(state["result"])

    @mcp.tool()
    def exposures() -> dict:
        """Long/short/gross/net delta-adjusted exposure and AUM of the book."""
        return q_exposures(_require())

    @mcp.tool()
    def factor_exposures() -> dict:
        """Net factor exposures (Fama-French + reversal) and predicted vol."""
        return q_factor_exposures(_require())

    @mcp.tool()
    def risk_summary() -> dict:
        """VaR, predicted vol, crowding, liquidity, and any limit breaches."""
        return q_risk_summary(_require())

    @mcp.tool()
    def top_issuers(side: str = "long", n: int = 10) -> list:
        """Top issuers by delta-adjusted exposure. side = 'long' or 'short'."""
        return q_top_issuers(_require(), side, n)

    @mcp.tool()
    def macro_exposures() -> dict:
        """Book sensitivity to macro drivers ($ P&L per +1% move, net of market)."""
        return q_macro(_require())

    @mcp.tool()
    def optimize(objective: str = "Minimize total risk",
                 turnover_max_usd: float | None = None,
                 market_cap_usd: float | None = None) -> dict:
        """Solve for a hedge overlay over a liquid ETF universe. objective:
        'Minimize total risk' | 'Minimize factor risk'. Optional dollar caps."""
        return q_optimize(_require(), objective, turnover_max_usd, market_cap_usd)

    return mcp


def main():
    build_server().run()


if __name__ == "__main__":
    main()
