"""Market data layer over yfinance with an on-disk cache.

Three fetch surfaces:
  * price history  — batch daily closes/volumes, ~15 months, cached per as-of date
  * profiles       — sector/industry/market cap/country per ticker, cached with TTL
  * option quotes  — bid/ask/IV per contract from live chains, cached per calendar day

All caches live under ``cache/`` next to the project so repeat runs are fast
and offline-capable.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BENCHMARK = "SPY"
PROFILE_TTL_DAYS = 7
PROFILE_SCHEMA = 2  # bump when the profile field set changes -> forces refetch
HISTORY_LOOKBACK_DAYS = 460  # ~15 months of calendar days
DOWNLOAD_CHUNK = 80
CHAIN_WORKERS = 8
PROFILE_WORKERS = 8


@dataclass
class TickerStats:
    spot: float
    realized_vol: float | None  # 60d annualized, as of the snapshot date
    beta: float | None  # vs SPY, up to 250d
    adv_shares: float | None  # 60d average daily volume
    spot_date: date | None = None  # date of the close used as spot


class MarketData:
    def __init__(self, cache_dir: str | Path = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------
    def fetch_history(
        self, tickers: list[str], asof: date
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Daily closes and volumes for tickers + benchmark, ending >= asof.

        Returns (closes, volumes) DataFrames indexed by naive date.
        """
        wanted = sorted(set(tickers) | {BENCHMARK})
        cache_file = self.cache_dir / f"history_{asof:%Y%m%d}.parquet"
        closes = vols = None
        if cache_file.exists():
            cached = pd.read_parquet(cache_file)
            # a column only counts as cached if it actually has data —
            # failed downloads leave all-NaN columns that must retry
            close_block = cached["Close"]
            have = set(close_block.columns[close_block.notna().any()])
            if set(wanted) <= have:
                closes = cached["Close"][wanted]
                vols = cached["Volume"][wanted]

        if closes is None:
            start = asof - timedelta(days=HISTORY_LOOKBACK_DAYS)
            frames = []
            for i in range(0, len(wanted), DOWNLOAD_CHUNK):
                chunk = wanted[i : i + DOWNLOAD_CHUNK]
                df = yf.download(
                    chunk,
                    start=start.isoformat(),
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    threads=True,
                )
                if df is not None and not df.empty:
                    if not isinstance(df.columns, pd.MultiIndex):
                        df.columns = pd.MultiIndex.from_product(
                            [df.columns, chunk]
                        )
                    frames.append(df[["Close", "Volume"]])
            if not frames:
                raise RuntimeError(
                    "Price download returned no data — check connectivity."
                )
            merged = pd.concat(frames, axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated()]
            if getattr(merged.index, "tz", None) is not None:
                merged.index = merged.index.tz_localize(None)
            # an intraday run returns today's LIVE price as today's bar; drop
            # it before the US close so a partial print is never cached (or
            # used) as a final close
            now_et = _now_eastern()
            if now_et.hour < 16 or (now_et.hour == 16 and now_et.minute < 15):
                merged = merged.loc[
                    merged.index != pd.Timestamp(now_et.date())
                ]
            # failed tickers come back as all-NaN columns; drop them so the
            # cache never claims to have them and the next run retries
            dead = [
                t for t in merged["Close"].columns
                if not merged["Close"][t].notna().any()
            ]
            if dead:
                merged = merged.drop(
                    columns=[
                        (field, t)
                        for field in ("Close", "Volume")
                        for t in dead
                        if (field, t) in merged.columns
                    ]
                )
            merged.to_parquet(cache_file)
            closes = merged["Close"].reindex(columns=wanted)
            vols = merged["Volume"].reindex(columns=wanted)

        return closes, vols

    def compute_stats(
        self, closes: pd.DataFrame, vols: pd.DataFrame, asof: date
    ) -> dict[str, TickerStats]:
        """Per-ticker spot / realized vol / beta / ADV as of the snapshot date."""
        asof_ts = pd.Timestamp(asof)
        closes_all = closes
        closes = closes.loc[closes.index <= asof_ts]
        vols = vols.loc[vols.index <= asof_ts]
        rets = np.log(closes / closes.shift(1))
        bench = rets.get(BENCHMARK)

        out: dict[str, TickerStats] = {}
        for ticker in closes.columns:
            series = closes[ticker].dropna()
            if series.empty:
                # brand-new listing (e.g. a spin-off that started trading
                # after the snapshot date): fall back to the first close
                # within 10 calendar days AFTER asof, disclosed via spot_date
                later = closes_all[ticker].dropna()
                later = later.loc[
                    (later.index > asof_ts)
                    & (later.index <= asof_ts + pd.Timedelta(days=10))
                ]
                if later.empty:
                    continue
                spot = float(later.iloc[0])
                if not math.isfinite(spot) or spot <= 0:
                    continue
                out[ticker] = TickerStats(
                    spot=spot, realized_vol=None, beta=None,
                    adv_shares=None, spot_date=later.index[0].date(),
                )
                continue
            spot = float(series.iloc[-1])
            if not math.isfinite(spot) or spot <= 0:
                continue

            r = rets[ticker].dropna().tail(60)
            realized = (
                float(r.std() * math.sqrt(252)) if len(r) >= 20 else None
            )

            beta = None
            if bench is not None and ticker != BENCHMARK:
                joint = pd.concat(
                    [rets[ticker], bench], axis=1, keys=["t", "b"]
                ).dropna().tail(250)
                if len(joint) >= 60:
                    var_b = joint["b"].var()
                    if var_b and var_b > 0:
                        beta = float(joint["t"].cov(joint["b"]) / var_b)
            elif ticker == BENCHMARK:
                beta = 1.0

            v = vols[ticker].dropna().tail(60)
            adv = float(v.mean()) if len(v) >= 20 else None

            out[ticker] = TickerStats(
                spot=spot, realized_vol=realized, beta=beta, adv_shares=adv,
                spot_date=series.index[-1].date(),
            )
        return out

    # ------------------------------------------------------------------
    # Profiles (sector / industry / market cap / country)
    # ------------------------------------------------------------------
    def fetch_profiles(self, tickers: list[str]) -> dict[str, dict]:
        cache_file = self.cache_dir / "profiles.json"
        cache: dict[str, dict] = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cache = {}

        cutoff = (datetime.now() - timedelta(days=PROFILE_TTL_DAYS)).isoformat()
        todo = [
            t
            for t in sorted(set(tickers))
            if t not in cache
            or "error" in cache[t]  # transient failures always retry
            or cache[t].get("schema") != PROFILE_SCHEMA  # fields added -> refetch
            or cache[t].get("fetched_at", "") < cutoff
        ]

        def grab(ticker: str, attempts: int = 2) -> tuple[str, dict]:
            last_exc = None
            for _ in range(attempts):
                try:
                    info = yf.Ticker(ticker).info or {}
                    break
                except Exception as exc:  # network / delisted / throttled
                    last_exc = exc
                    time.sleep(1.0)
            else:
                return ticker, {"error": str(last_exc)[:200]}
            return ticker, {
                "name": info.get("shortName") or info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "country": info.get("country"),
                "quote_type": info.get("quoteType"),
                "category": info.get("category"),
                # crowding / squeeze signals (same .info call, no extra cost)
                "short_pct_float": info.get("shortPercentOfFloat"),
                "short_ratio": info.get("shortRatio"),  # days to cover
                "shares_short": info.get("sharesShort"),
                "shares_short_prior": info.get("sharesShortPriorMonth"),
                "held_pct_inst": info.get("heldPercentInstitutions"),
            }

        if todo:
            with ThreadPoolExecutor(max_workers=PROFILE_WORKERS) as pool:
                futures = {pool.submit(grab, t): t for t in todo}
                for fut in as_completed(futures):
                    ticker, data = fut.result()
                    data["fetched_at"] = datetime.now().isoformat()
                    if "error" not in data:
                        data["schema"] = PROFILE_SCHEMA
                    cache[ticker] = data
            cache_file.write_text(
                json.dumps(cache, indent=1), encoding="utf-8"
            )

        return {t: cache.get(t, {}) for t in tickers}

    # ------------------------------------------------------------------
    # Option chains
    # ------------------------------------------------------------------
    def fetch_option_quotes(
        self, contracts: list[dict]
    ) -> dict[str, dict]:
        """Live chain quotes for the given contracts.

        Each contract dict needs: key, underlying, root, expiry (date),
        strike, cp. Returns {key: {bid, ask, last, iv}} for contracts found.
        Chains are live-only on Yahoo, so results are cached per calendar day.
        """
        cache_file = self.cache_dir / f"chains_{date.today():%Y%m%d}.json"
        cache: dict[str, dict] = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cache = {}

        def needs_fetch(key: str) -> bool:
            entry = cache.get(key)
            if entry is None:
                return True
            # transient chain errors retry; structural misses (expiry/contract
            # not listed) stay cached for the day
            return str(entry.get("missing", "")).startswith("chain_error")

        todo = []
        for c in contracts:
            if not needs_fetch(c["key"]):
                continue
            if c["expiry"] < date.today():
                # no live chain can exist; structural miss, cached for the day
                cache[c["key"]] = {"missing": "expired_before_run"}
                continue
            todo.append(c)
        groups: dict[tuple[str, date], list[dict]] = {}
        for c in todo:
            groups.setdefault((c["underlying"], c["expiry"]), []).append(c)

        def grab(underlying: str, expiry: date, members: list[dict]) -> dict:
            found: dict[str, dict] = {}
            try:
                tk = yf.Ticker(underlying)
                expiries = set(tk.options)
                exp_str = expiry.isoformat()
                if exp_str not in expiries:
                    return {m["key"]: {"missing": "expiry_not_listed"} for m in members}
                chain = tk.option_chain(exp_str)
            except Exception as exc:
                return {m["key"]: {"missing": f"chain_error: {str(exc)[:120]}"} for m in members}

            for m in members:
                try:
                    df = chain.calls if m["cp"] == "C" else chain.puts
                    if df is None or df.empty or "strike" not in df.columns:
                        # one-sided chains come back as column-less frames
                        found[m["key"]] = {"missing": "empty_chain_side"}
                        continue
                    strike_milli = int(round(m["strike"] * 1000))
                    hit = df[
                        (df["strike"].round(4) == round(m["strike"], 4))
                        & df["contractSymbol"].str.startswith(
                            f"{m['root']}{m['expiry']:%y%m%d}"
                        )
                    ]
                    if hit.empty:
                        # fall back to strike-only match (root prefix quirks)
                        hit = df[
                            df["contractSymbol"].str.endswith(
                                f"{m['cp']}{strike_milli:08d}"
                            )
                            & df["contractSymbol"].str.startswith(m["root"])
                        ]
                    if hit.empty:
                        found[m["key"]] = {"missing": "contract_not_in_chain"}
                        continue
                    row = hit.iloc[0]
                    found[m["key"]] = {
                        "bid": _clean(row.get("bid")),
                        "ask": _clean(row.get("ask")),
                        "last": _clean(row.get("lastPrice")),
                        "iv": _clean(row.get("impliedVolatility")),
                    }
                except Exception as exc:
                    found[m["key"]] = {
                        "missing": f"match_error: {str(exc)[:120]}"
                    }
            return found

        if groups:
            with ThreadPoolExecutor(max_workers=CHAIN_WORKERS) as pool:
                futures = {
                    pool.submit(grab, u, e, members): members
                    for (u, e), members in groups.items()
                }
                for fut in as_completed(futures):
                    try:
                        cache.update(fut.result())
                    except Exception as exc:
                        # never let one bad group discard the rest
                        for m in futures[fut]:
                            cache[m["key"]] = {
                                "missing": f"chain_error: {str(exc)[:120]}"
                            }
        # write even when only structural misses were added (expired etc.)
        cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")

        return {
            c["key"]: cache[c["key"]]
            for c in contracts
            if c["key"] in cache and "missing" not in cache[c["key"]]
        }


def _now_eastern() -> datetime:
    """Current time in US Eastern, robust to a missing IANA tz database.

    Windows has no system tz DB; ZoneInfo needs the `tzdata` package. If it
    is absent, fall back to a fixed UTC offset (a rough DST guess). The only
    consumer is the 16:15 intraday-close guard, so an hour of DST slop just
    shifts that cutoff slightly — it never corrupts data.
    """
    from datetime import timezone

    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # crude DST: EDT (UTC-4) roughly Mar–Nov, else EST (UTC-5)
        utc_now = datetime.now(timezone.utc)
        offset = -4 if 3 <= utc_now.month <= 11 else -5
        return utc_now + timedelta(hours=offset)


def _clean(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0:
        return None
    return v
