"""What-if simulator: apply a trade list to a book, or diff two books.

Trade file format (CSV, header optional, `#` comments allowed):

    Symbol,Quantity
    SPY,-2000
    IWM    DEC 18 2026   200.000 P,-50
    MDT,1000

Quantities are signed DELTAS: positive buys, negative sells/shorts. Symbols
use the same broker format as the position export, so options trade too.
Alternatively pass a second full position export and the simulator diffs the
two books directly.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .parse import Position, build_position, parse_positions_csv


@dataclass
class TradeList:
    trades: list[Position] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def parse_trades_csv(path: str | Path) -> TradeList:
    """Parse a simple Symbol,Quantity trade list (broker symbol format)."""
    result = TradeList()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for lineno, row in enumerate(csv.reader(f), start=1):
            if not row or not "".join(row).strip():
                continue
            first = row[0].strip()
            if first.startswith("#"):
                continue
            if first.lower() in ("symbol", "ticker"):
                continue  # header row (may follow comment lines)
            if len(row) < 2:
                result.issues.append(
                    f"trades line {lineno}: expected 'Symbol,Quantity', "
                    f"got {row!r} — skipped"
                )
                continue
            symbol = first
            # an unquoted '1,000' splits across fields — rejoin the tail so
            # it parses as 1000 rather than silently as 1
            qty_text = "".join(p.strip() for p in row[1:])
            try:
                qty = float(qty_text.replace(",", ""))
            except ValueError:
                result.issues.append(
                    f"trades line {lineno}: unreadable quantity {qty_text!r} "
                    "— skipped"
                )
                continue
            if qty == 0:
                continue
            pos, err = build_position("WHATIF", symbol, qty)
            if err:
                result.issues.append(f"trades line {lineno}: {err} — skipped")
                continue
            result.trades.append(pos)
    return result


def _merge_key(p: Position) -> str:
    # namespaced so an equity whose ticker looks like an OCC contract string
    # can never merge into (or flatten) a real option position
    if p.kind == "option":
        return f"OPT:{p.contract_key}"
    return f"EQ:{p.underlying}"


def apply_trades(
    base: list[Position], trades: list[Position]
) -> list[Position]:
    """Return a new position list with signed trade quantities merged in."""
    book: dict[str, Position] = {}
    for p in base:
        key = _merge_key(p)
        if key in book:
            book[key].qty += p.qty
        else:
            book[key] = Position(**vars(p))
    for t in trades:
        key = _merge_key(t)
        if key in book:
            book[key].qty += t.qty
        else:
            book[key] = Position(**vars(t))
    return [p for p in book.values() if p.qty != 0]


def load_proposed_book(
    base_csv: str | Path,
    trades_csv: str | Path | None,
    proposed_csv: str | Path | None,
) -> tuple[list[Position], list[Position], list[Position], list[str]]:
    """Returns (base_positions, proposed_positions, trades, issues).

    Exactly one of trades_csv / proposed_csv must be given. When a full
    proposed export is given, the implied trade list is the position diff.
    """
    parsed = parse_positions_csv(base_csv)
    issues = list(parsed.issues)
    base = parsed.positions

    if (trades_csv is None) == (proposed_csv is None):
        raise ValueError("Provide exactly one of --trades or --proposed.")

    if trades_csv is not None:
        tl = parse_trades_csv(trades_csv)
        issues += tl.issues
        proposed = apply_trades(base, tl.trades)
        return base, proposed, tl.trades, issues

    parsed2 = parse_positions_csv(proposed_csv)
    issues += [f"(proposed) {i}" for i in parsed2.issues]
    proposed = parsed2.positions
    # implied trades = proposed minus base
    base_q = {}
    for p in base:
        base_q[_merge_key(p)] = base_q.get(_merge_key(p), 0.0) + p.qty
    trades = []
    seen = set()
    for p in proposed:
        key = _merge_key(p)
        if key in seen:
            continue
        seen.add(key)
        total = sum(q.qty for q in proposed if _merge_key(q) == key)
        delta = total - base_q.get(key, 0.0)
        if delta != 0:
            t = Position(**vars(p))
            t.qty = delta
            trades.append(t)
    for p in base:
        key = _merge_key(p)
        if key not in seen and p.qty != 0:
            t = Position(**vars(p))
            t.qty = -base_q[key]
            trades.append(t)
            seen.add(key)
    return base, proposed, trades, issues
