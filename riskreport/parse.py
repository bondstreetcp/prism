"""Parse broker position CSV exports into typed Position records.

Two formats are supported and auto-detected:

* Goldman "Intraday Position": Account Number, Symbol, ... Current Position.
  Equity symbols are plain tickers ("MDT"); options are
  "ROOT   MON DD YYYY   STRIKE.000 C|P" (e.g. "ALLE   SEP 18 2026   160.000 P").
  Adjusted roots carry a trailing digit ("APTV1") from corporate actions.

* Interactive Brokers Activity Statement: a multi-section CSV. Positions come
  from the "Open Positions" section; options are "ROOT DDMMMYY STRIKE C|P"
  (e.g. "AKAM 20NOV26 85 P"). Cash and total NAV come from the "Net Asset
  Value" section, so an IBKR file carries its own AUM (no manual cash input).
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
    cash: float | None = None   # from the broker file (IBKR); None if absent
    nav: float | None = None    # broker-reported total NAV, if available
    source: str = "goldman"     # "goldman" | "ibkr"


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
    result.source = "goldman"
    return result


# ----------------------------------------------------------------------
# Interactive Brokers Activity Statement
# ----------------------------------------------------------------------
_IBKR_OPT_RE = re.compile(
    r"^(?P<root>\S+)\s+"
    r"(?P<day>\d{1,2})(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
    r"(?P<year>\d{2})\s+"
    r"(?P<strike>\d+(?:\.\d+)?)\s+(?P<cp>[CP])$"
)


def _build_ibkr_option(account: str, raw: str, qty: float, multiplier: int):
    """Build an option Position from an IBKR symbol 'ROOT DDMMMYY STRIKE C|P'."""
    m = _IBKR_OPT_RE.match(re.sub(r"\s+", " ", raw.strip()))
    if not m:
        return None, f"unrecognized IBKR option symbol {raw!r}"
    try:
        expiry = date(2000 + int(m["year"]), MONTHS[m["mon"]], int(m["day"]))
    except ValueError:
        return None, f"invalid expiry in IBKR option {raw!r}"
    root = m["root"]
    base, adjusted = _strip_adjustment(root)
    return Position(
        account=account, raw_symbol=raw.strip(), qty=qty, kind="option",
        underlying=normalize_ticker(base), root=root, expiry=expiry,
        strike=float(m["strike"]), cp=m["cp"], adjusted=adjusted,
        multiplier=multiplier or 100,
    ), None


def _ibkr_sections(path: str | Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def is_ibkr(path: str | Path) -> bool:
    """Sniff whether a CSV is an IBKR Activity Statement."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            head = f.read(4000)
    except OSError:
        return False
    return ("Interactive Brokers" in head
            or head.startswith("Statement,Header,Field Name")
            or "\nOpen Positions,Header" in head
            or "Open Positions,Header" in head[:2000])


def parse_ibkr_csv(path: str | Path) -> ParseResult:
    rows = _ibkr_sections(path)
    result = ParseResult(source="ibkr", asof=asof_from_filename(path))
    accounts: set[str] = set()

    # ---- account, as-of date, cash, and NAV --------------------------
    for r in rows:
        if len(r) < 3:
            continue
        if r[0] == "Account Information" and r[1] == "Data" and r[2].strip() == "Account":
            accounts.add(r[3].strip())
        elif r[0] == "Statement" and r[1] == "Data" and r[2].strip() == "Period":
            # A ranged Period ("January 1, 2026 - July 10, 2026") is common on
            # monthly/YTD statements; positions are as-of the period END, so
            # take the LAST date, not the first.
            dates = re.findall(r"[A-Z][a-z]+ \d{1,2}, \d{4}", r[3])
            if dates:
                try:
                    result.asof = datetime.strptime(dates[-1], "%B %d, %Y").date()
                except ValueError:
                    pass
        elif r[0] == "Net Asset Value" and r[1] == "Data" and len(r) > 6:
            label = r[2].strip()
            try:
                current_total = float(r[6])
            except (ValueError, IndexError):
                continue
            if label == "Cash":
                result.cash = current_total
            elif label == "Total":
                result.nav = current_total

    # ---- open positions ---------------------------------------------
    op_header = None
    for r in rows:
        if r and r[0] == "Open Positions" and len(r) > 1 and r[1] == "Header":
            op_header = [c.strip() for c in r[2:]]
            break
    if op_header is None:
        raise ValueError("IBKR file has no 'Open Positions' section.")
    idx = {name: i for i, name in enumerate(op_header)}
    need = {"Asset Category", "Symbol", "Quantity", "Mult"}
    if not need <= set(idx):
        raise ValueError(f"IBKR Open Positions missing columns: {need - set(idx)}")

    acct = sorted(accounts)[0] if accounts else "IBKR"
    for r in rows:
        if not (r and r[0] == "Open Positions" and len(r) > 1 and r[1] == "Data"):
            continue
        cells = r[2:]
        if len(cells) <= max(idx.values()):
            continue
        # only the per-instrument 'Summary' rows (skip lot/subtotal breakdowns)
        if idx.get("DataDiscriminator") is not None and \
                cells[idx["DataDiscriminator"]].strip() != "Summary":
            continue
        cat = cells[idx["Asset Category"]].strip()
        symbol = cells[idx["Symbol"]].strip()
        qty_text = cells[idx["Quantity"]].strip().replace(",", "")
        mult_text = cells[idx["Mult"]].strip().replace(",", "")
        if not symbol:
            continue
        try:
            qty = float(qty_text)
        except ValueError:
            result.issues.append(f"IBKR: unreadable quantity {qty_text!r} for "
                                 f"{symbol!r} — skipped")
            continue
        if qty == 0:
            continue
        try:
            mult = int(float(mult_text)) if mult_text else 100
        except ValueError:
            mult = 100

        if "Option" in cat:
            pos, err = _build_ibkr_option(acct, symbol, qty, mult)
            if err:
                result.issues.append(f"IBKR: {err} — skipped")
                continue
            result.positions.append(pos)
        elif cat in ("Stocks", "Equity", "ETFs", "Funds"):
            result.positions.append(Position(
                account=acct, raw_symbol=symbol, qty=qty, kind="equity",
                underlying=normalize_ticker(symbol),
            ))
        else:
            result.issues.append(f"IBKR: unsupported asset category {cat!r} "
                                 f"for {symbol!r} — excluded")

    result.accounts = sorted(accounts)
    return result


def parse_positions(path: str | Path) -> ParseResult:
    """Auto-detect the broker format (Goldman vs IBKR) and parse."""
    if is_ibkr(path):
        return parse_ibkr_csv(path)
    return parse_positions_csv(path)


def _pos_key(p: Position) -> str:
    return f"OPT:{p.contract_key}" if p.kind == "option" else f"EQ:{p.underlying}"


def merge_parse_results(results: list[ParseResult]) -> ParseResult:
    """Consolidate several parsed books into one — for firmwide/multi-account
    aggregation. Positions on the same underlying/contract sum across files;
    cash and NAV sum where reported. Positions that net to zero drop out."""
    merged = ParseResult(source="+".join(sorted({r.source for r in results})) or "merged")
    book: dict[str, Position] = {}
    accounts: list[str] = []
    for r in results:
        for p in r.positions:
            k = _pos_key(p)
            if k in book:
                book[k].qty += p.qty
            else:
                book[k] = Position(**vars(p))
        merged.issues.extend(r.issues)
        accounts.extend(r.accounts)
    merged.positions = [p for p in book.values() if p.qty != 0]
    merged.accounts = sorted(set(accounts))

    cashes = [r.cash for r in results if r.cash is not None]
    navs = [r.nav for r in results if r.nav is not None]
    merged.cash = sum(cashes) if cashes else None
    merged.nav = sum(navs) if navs else None
    if navs and len(navs) < len(results):
        merged.nav = None  # incomplete NAV coverage — don't report a partial sum
    asofs = [r.asof for r in results if r.asof]
    merged.asof = max(asofs) if asofs else None
    if len({r.asof for r in results if r.asof}) > 1:
        merged.issues.append(
            "Aggregated files have different as-of dates; using the latest "
            f"({merged.asof}). Positions are combined as-is."
        )
    return merged
