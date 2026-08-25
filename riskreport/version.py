"""App version / build metadata and the 'What's new' changelog.

Bump ``VERSION`` / ``REVISION`` / ``BUILD_DATE`` and prepend to ``WHATS_NEW``
whenever a change ships, so the splash shows users what changed. REVISION is
the git commit count at release; keep it roughly in step (it is informational).
"""

from __future__ import annotations

VERSION = "2.5"
REVISION = 43          # git commit count at release (informational)
BUILD_DATE = "2026-08-25"


def build_label() -> str:
    """Short build string for the header/footer, e.g. 'v2.0 · rev 33 · 2026-08-25'.

    Appends the short git SHA when it can be read at runtime (best-effort).
    """
    label = f"v{VERSION} · rev {REVISION} · {BUILD_DATE}"
    try:  # best-effort — Streamlit Cloud clones the repo, so .git usually exists
        import subprocess
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        if sha:
            label += f" · {sha}"
    except Exception:
        pass
    return label


# Newest release first. Each entry: (version, date, [bullet, ...]).
# Only the top entry is highlighted in the splash; older ones are collapsed.
WHATS_NEW = [
    ("2.5", "2026-08-25", [
        "**Options skew view** (Options tab) — vega and premium by strike "
        "moneyness (deep-OTM → ITM), so you can see where your *tail* vol "
        "exposure sits. Isolates deep-OTM short-put vega (the crash exposure).",
    ]),
    ("2.4", "2026-08-25", [
        "**New 🛡 Risk tab** — the detailed risk analytics (VaR & greeks, "
        "predicted-vol drivers, concentration, risk clusters, liquidity, "
        "realized drawdown, Monte Carlo VaR) moved off the Report tab into "
        "their own tab, so the Report stays a clean exposure dashboard.",
    ]),
    ("2.3", "2026-08-25", [
        "**Risk clusters** (Report tab) — groups your largest positions into "
        "correlated clusters (implicit thematic bets) from the factor-model "
        "correlation, ranked by share of risk. Complements the effective-bets "
        "count with *what* the bets are.",
    ]),
    ("2.2", "2026-08-25", [
        "**Realized risk & drawdown** (Report tab) — a trailing-1y backtest of "
        "the current book: realized vol vs predicted, max drawdown, and an "
        "empirical VaR that backtests the model VaR, with return & drawdown "
        "charts.",
    ]),
    ("2.1", "2026-08-25", [
        "**Liquidity & liquidation cost** (Report tab) — estimated cost to "
        "unwind the book via a market-impact model, and a liquidity-adjusted "
        "VaR (VaR plus that cost), with the hardest-to-exit names ranked.",
    ]),
    ("2.0", "2026-08-25", [
        "**Options tab** — expiry / theta ladder: greeks and premium by time to "
        "expiry, so you see where theta is earned and the short gamma/vega sits.",
        "**Concentration & diversification** (Report tab) — effective number of "
        "risk bets vs names held, diversification ratio, top-5 share of risk.",
        "**Monte Carlo VaR** (Report tab) — a parametric VaR that cross-checks "
        "the historical-sim VaR, with a simulated P&L distribution.",
        "**Fixed Income tab** — interest-rate risk for bond ETFs: DV01, key-rate "
        "DV01, credit-spread (CS01), and rate scenarios.",
        "**Pre-trade tab** — enter proposed trades and see the risk / limit "
        "impact (would-breach / would-cure) before executing.",
        "**Attribution** (Benchmark tab) — Brinson (sector allocation vs "
        "selection) and factor-based (Barra) return attribution.",
        "**Active-risk decomposition** — which factors and which names drive "
        "tracking error (position-level MCTE).",
        "**Full option greeks** (gamma/vega/theta), **vol-aware VaR** (IV shocks "
        "with the market), and **component/marginal VaR** (which names drive the "
        "tail).",
        "**Crisis-scenario library** — replay 2008 / COVID / rate shocks against "
        "today's book.",
        "**Sharper AI** — the commentary and chat now reason over all of the "
        "above.",
    ]),
]
