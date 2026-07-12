"""Parse broker position CSV exports into typed Position records.

Handles the "Intraday Position" export format:
  Account Number, Symbol, Beginning Position, Today Buy, Today Sell,
  Non trade Activity In, Non trade Activity Out, Net Intraday Activity,
  Current Position

Equity symbols are plain tickers ("MDT"). Listed options are encoded as
"ROOT   MON DD YYYY   STRIKE.000 C|P", e.g. "ALLE   SEP 18 2026   160.000 P".
Adjusted option roots carry a trailing digit ("APTV1") from corporate actions.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_OPTION_RE = re.compile(
    r"^(?P<root>[A-Z]+\d?)\s+"
    r"(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+"
    r"(?P<day>\d{1,2})\s+(?P<year>\d{4})\s+"
    r"(?P<strike>\d+(?:\.\d+)?)\s+(?P<cp>[CP])$"
)


@dataclass
class Position:
    account: str
    raw_symbol: str
    qty: float
    kind: str  # "equity" | "option"
    underlying: str  # normalized ticker of the issuer/underlier
    root: str = ""  # option root as printed (may be adjusted, e.g. APTV1)
    expiry: date | None = None
    strike: float | None = None
    cp: str | None = None  # "C" | "P"
    adjusted: bool = False  # corporate-action-adjusted contract
    multiplier: int = 100

    @property
    def contract_key(self) -> str:
        """Stable key for an option contract (OCC-style)."""
        if self.kind != "option":
            return self.underlying
        return (
            f"{self.root}{self.expiry:%y%m%d}{self.cp}"
            f"{int(round(self.strike * 1000)):08d}"
        )


@dataclass
class ParseResult:
    positions: list[Position] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    accounts: list[str] = field(default_factory=list)
    asof: date | None = None


# Brokers print class shares without a separator; Yahoo wants a dash.
# Explicit map only — many normal tickers end in A/B (TSLA), so no regex.
TICKER_ALIASES = {
    "BRKA": "BRK-A",
    "BRKB": "BRK-B",
    "BFA": "BF-A",
    "BFB": "BF-B",
    "LGFA": "LGF-A",
    "LGFB": "LGF-B",
    "HEIA": "HEI-A",
    "MOGA": "MOG-A",
    "MOGB": "MOG-B",
    "CWENA": "CWEN-A",
    "GEFB": "GEF-B",
    "UHALB": "UHAL-B",
}


def normalize_ticker(sym: str) -> str:
    """Map broker ticker punctuation to Yahoo Finance style (BRK.B -> BRK-B)."""
    t = sym.strip().upper().replace(".", "-").replace(" ", "-").replace("/", "-")
    return TICKER_ALIASES.get(t, t)


def _strip_adjustment(root: str) -> tuple[str, bool]:
    base = root.rstrip("0123456789")
    return (base, base != root)


def asof_from_filename(path: str | Path) -> date | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", Path(path).name)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    return None


def build_position(
    account: str, raw_symbol: str, qty: float
) -> tuple[Position | None, str | None]:
    """Build a Position from a broker-format symbol; (None, error) on failure."""
    collapsed = re.sub(r"\s+", " ", raw_symbol.strip()).upper()
    if not collapsed:
        return None, "blank symbol"
    m = _OPTION_RE.match(collapsed)
    if m is None and " " in collapsed:
        # spaces mean option intent; never let a near-miss option symbol
        # fall through as a garbage "equity" that silently fails to price
        return None, (
            f"{raw_symbol!r} looks like an option symbol but does not match "
            "the 'ROOT MON DD YYYY STRIKE C|P' format"
        )
    if m:
        try:
            expiry = date(int(m["year"]), MONTHS[m["mon"]], int(m["day"]))
        except ValueError:
            return None, f"invalid option expiry in {raw_symbol!r}"
        root = m["root"]
        base, adjusted = _strip_adjustment(root)
        return Position(
            account=account, raw_symbol=raw_symbol.strip(), qty=qty,
            kind="option", underlying=normalize_ticker(base), root=root,
            expiry=expiry, strike=float(m["strike"]), cp=m["cp"],
            adjusted=adjusted,
        ), None
    return Position(
        account=account, raw_symbol=raw_symbol.strip(), qty=qty,
        kind="equity", underlying=normalize_ticker(raw_symbol),
    ), None


def parse_positions_csv(path: str | Path) -> ParseResult:
    result = ParseResult(asof=asof_from_filename(path))
    accounts: set[str] = set()

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"Account Number", "Symbol", "Current Position"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing expected columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for lineno, row in enumerate(reader, start=2):
            account = (row.get("Account Number") or "").strip()
            raw_symbol = (row.get("Symbol") or "").strip()
            qty_text = (row.get("Current Position") or "").strip()

            try:
                qty = float(qty_text.replace(",", ""))
            except ValueError:
                result.issues.append(
                    f"line {lineno}: unreadable quantity {qty_text!r} "
                    f"for symbol {raw_symbol!r} — row skipped"
                )
                continue

            if account:
                accounts.add(account)

            if not raw_symbol:
                result.issues.append(
                    f"line {lineno}: blank symbol with position of "
                    f"{qty:,.0f} shares — cannot price, excluded"
                )
                continue

            if qty == 0:
                continue  # closed intraday; nothing to report

            pos, err = build_position(account, raw_symbol, qty)
            if err:
                result.issues.append(f"line {lineno}: {err} — row skipped")
                continue
            result.positions.append(pos)

    result.accounts = sorted(accounts)
    return result
