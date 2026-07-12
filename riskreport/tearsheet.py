"""Render a one-page PDF risk tearsheet from PortfolioAnalytics.

Layout (US Letter portrait):
  header    — portfolio name, as-of date
  row 1     — exposure summary / options overlay / issuer counts tables
  row 2     — diverging long/short bar charts: sector, market cap, region
  row 3     — top 10 long and short issuer tables
  footer    — methodology and data-quality notes
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Table, TableStyle

from .analytics import PortfolioAnalytics

# dataviz reference palette (light mode)
BLUE = "#2a78d6"  # diverging pole: long
RED = "#e34948"  # diverging pole: short
CRITICAL = "#d03b3b"  # status: limit breach
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

PAGE_W, PAGE_H = letter
MARGIN = 36
CONTENT_W = PAGE_W - 2 * MARGIN

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
    }
)


def _m(dollars: float) -> str:
    """Format dollars in $M, negatives in parentheses."""
    v = dollars / 1e6
    return f"({abs(v):,.1f})" if v < 0 else f"{v:,.1f}"


def _pct(x: float) -> str:
    return f"({abs(x):.1%})" if x < 0 else f"{x:.1%}"


def _table(
    data: list[list[str]],
    col_widths: list[float],
    header_rows: int = 1,
    align_left_cols: int = 1,
    font_size: float = 6.8,
) -> Table:
    t = Table(data, colWidths=col_widths)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
        ("FONT", (0, 0), (-1, header_rows - 1), "Helvetica-Bold", font_size),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), HexColor(INK_2)),
        ("TEXTCOLOR", (0, header_rows), (-1, -1), HexColor(INK)),
        ("ALIGN", (align_left_cols, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, HexColor(BASELINE)),
        ("LINEBELOW", (0, header_rows), (-1, -2), 0.4, HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]
    t.setStyle(TableStyle(style))
    return t


def _draw_table(c, table: Table, x: float, y_top: float) -> float:
    """Draw a platypus table with its top-left corner at (x, y_top)."""
    w, h = table.wrapOn(c, CONTENT_W, PAGE_H)
    table.drawOn(c, x, y_top - h)
    return h


def _diverging_chart(
    cat: pd.DataFrame,
    label_col: str,
    title: str,
    width_in: float,
    height_in: float,
    max_rows: int = 12,
    label_extremes: bool = True,
) -> io.BytesIO:
    """Horizontal diverging bars: long (blue, right), short (red, left),
    net as an ink diamond."""
    if len(cat) > max_rows:
        # roll everything past the cap into "Other" so every dollar plots
        head, rest = cat.head(max_rows - 1), cat.iloc[max_rows - 1 :]
        other = {
            label_col: "Other",
            "long": rest["long"].sum(),
            "short": rest["short"].sum(),
            "net": rest["net"].sum(),
            "gross": rest["gross"].sum(),
        }
        cat = pd.concat([head, pd.DataFrame([other])], ignore_index=True)
    df = cat.head(max_rows).iloc[::-1]  # largest gross at top
    labels = [str(x)[:22] for x in df[label_col]]
    longs = df["long"] / 1e6
    shorts = df["short"] / 1e6
    nets = df["net"] / 1e6
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(y, longs, color=BLUE, height=0.58, zorder=3)
    ax.barh(y, shorts, color=RED, height=0.58, zorder=3)
    ax.scatter(nets, y, marker="D", s=9, color=INK, zorder=4, linewidths=0)

    ax.axvline(0, color=BASELINE, lw=0.8, zorder=2)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis="y", labelsize=6.4, length=0)
    ax.tick_params(axis="x", labelsize=6.0, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if label_extremes and len(df):
        span = max(longs.max(), 0) - min(shorts.min(), 0)
        pad = span * 0.02 if span else 0.1
        i_long = int(longs.reset_index(drop=True).idxmax())
        if longs.iloc[i_long] > 0:
            ax.text(longs.iloc[i_long] + pad, i_long, f"{longs.iloc[i_long]:,.0f}",
                    va="center", ha="left", fontsize=6, color=INK_2)
        i_short = int(shorts.reset_index(drop=True).idxmin())
        if shorts.iloc[i_short] < 0:
            ax.text(shorts.iloc[i_short] - pad, i_short, f"({abs(shorts.iloc[i_short]):,.0f})",
                    va="center", ha="right", fontsize=6, color=INK_2)
        ax.margins(x=0.12)

    ax.set_title(title, fontsize=7.5, fontweight="bold", color=INK, loc="left", pad=4)
    ax.set_xlabel("Delta-adjusted exposure, $M", fontsize=6, color=MUTED, labelpad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _draw_legend(c, x: float, y: float) -> None:
    """Long/Short/Net legend drawn natively in reportlab at (x, y)."""
    c.setFont("Helvetica", 7)
    for label, color, shape in [("Long", BLUE, "rect"), ("Short", RED, "rect"),
                                ("Net", INK, "diamond")]:
        c.setFillColor(HexColor(color))
        if shape == "rect":
            c.rect(x, y - 1.5, 7, 7, stroke=0, fill=1)
        else:
            p = c.beginPath()
            p.moveTo(x + 3.5, y - 2)
            p.lineTo(x + 7, y + 1.5)
            p.lineTo(x + 3.5, y + 5)
            p.lineTo(x, y + 1.5)
            p.close()
            c.drawPath(p, stroke=0, fill=1)
        c.setFillColor(HexColor(INK_2))
        c.drawString(x + 10, y, label)
        x += 12 + c.stringWidth(label, "Helvetica", 7) + 14


FACTOR_LABELS = {
    "Mkt-RF": "Market",
    "SMB": "Size (SMB)",
    "HML": "Value (HML)",
    "RMW": "Profitability (RMW)",
    "CMA": "Investment (CMA)",
    "MOM": "Momentum (MOM)",
    "ST_Rev": "Short-term reversal",
    "LT_Rev": "Long-term reversal",
}


def render_tearsheet(
    analytics: PortfolioAnalytics,
    name: str,
    out_path: str | Path,
    factor_risk=None,
    scenarios=None,
    hedge=None,
    alert_hits=None,
    crowding=None,
    model=None,
    bias=None,
) -> Path:
    a = analytics
    s = a.summary
    out_path = Path(out_path)
    c = rl_canvas.Canvas(str(out_path), pagesize=letter)
    c.setTitle(f"Risk Report {name} {a.asof}")

    # ------------------------------------------------------------- header
    y = PAGE_H - MARGIN
    c.setFillColor(HexColor(INK))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y - 14, f"Risk Report — {name}")
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor(INK_2))
    c.drawRightString(PAGE_W - MARGIN, y - 8, f"As of {a.asof:%B %d, %Y}")
    c.drawRightString(PAGE_W - MARGIN, y - 18, "Internal use only")
    c.setStrokeColor(HexColor(BASELINE))
    c.setLineWidth(0.8)
    c.line(MARGIN, y - 24, PAGE_W - MARGIN, y - 24)
    y -= 34

    # ---------------------------------------------------- alert banner
    if alert_hits:
        band_h = 12 + 8.5 * len(alert_hits)
        c.setFillColor(HexColor("#fbe9e7"))  # light critical wash
        c.rect(MARGIN, y - band_h, CONTENT_W, band_h, stroke=0, fill=1)
        c.setFillColor(HexColor(CRITICAL))
        c.setFont("Helvetica-Bold", 8)
        c.drawString(MARGIN + 6, y - 11,
                     f"⚠  {len(alert_hits)} risk limit breach(es)")
        c.setFont("Helvetica", 7)
        c.setFillColor(HexColor(INK))
        yy = y - 21
        for hit in alert_hits[:6]:
            c.drawString(MARGIN + 12, yy, f"• {hit}")
            yy -= 8.5
        y -= band_h + 8

    # ------------------------------------------------------- row 1: tables
    denom = s["aum"] if s["aum"] else s["exp_gross"]
    denom_label = "% AUM" if s["aum"] else "% Gross"

    exposure_rows = [
        ["Exposure", "MV $M", "Δ-adj $M", denom_label],
        ["Long", _m(s["mv_long"]), _m(s["exp_long"]), _pct(s["exp_long"] / denom)],
        ["Short", _m(s["mv_short"]), _m(s["exp_short"]), _pct(s["exp_short"] / denom)],
        ["Gross", _m(s["mv_gross"]), _m(s["exp_gross"]), _pct(s["exp_gross"] / denom)],
        ["Net", _m(s["mv_net"]), _m(s["exp_net"]), _pct(s["exp_net"] / denom)],
        ["Beta-adj net", "", _m(s["beta_net"]), _pct(s["beta_net"] / denom)],
    ]

    options_rows = [
        ["Options overlay", "$M", "% of book"],
        ["Options MV, gross", _m(s["opt_mv_gross"]),
         _pct(s["opt_mv_gross"] / s["mv_gross"] if s["mv_gross"] else 0)],
        ["Options MV, net", _m(s["opt_mv_net"]), ""],
        ["Options Δ-adj, gross", _m(s["opt_exp_gross"]),
         _pct(s["opt_exp_gross"] / s["exp_gross"] if s["exp_gross"] else 0)],
        ["Options Δ-adj, net", _m(s["opt_exp_net"]), ""],
        ["Equities Δ-adj, net", _m(s["eq_exp_net"]), ""],
    ]

    counts_rows = [
        ["Portfolio", "Count"],
        ["Instruments", f"{s['n_instruments']:,}"],
        ["  equities / ETFs", f"{s['n_equities']:,}"],
        ["  options", f"{s['n_options']:,}"],
        ["Issuers", f"{s['n_issuers']:,}"],
        ["  net long", f"{s['n_issuers_long']:,}"],
        ["  net short", f"{s['n_issuers_short']:,}"],
    ]

    gap = 12
    w1, w2, w3 = 200, 190, 126
    t1 = _table(exposure_rows, [52, 50, 50, 48])
    t2 = _table(options_rows, [95, 50, 45])
    t3 = _table(counts_rows, [78, 48])
    h1 = _draw_table(c, t1, MARGIN, y)
    h2 = _draw_table(c, t2, MARGIN + w1 + gap, y)
    h3 = _draw_table(c, t3, MARGIN + w1 + w2 + 2 * gap, y)
    y -= max(h1, h2, h3) + 14

    # ------------------------------------------------------- row 2: charts
    _draw_legend(c, MARGIN, y - 7)
    y -= 14

    chart_h_pt = 210
    sector_w_pt = CONTENT_W * 0.52
    side_w_pt = CONTENT_W * 0.44

    sector_img = ImageReader(
        _diverging_chart(a.sector_table, "sector", "Exposure by sector",
                         sector_w_pt / 72, chart_h_pt / 72)
    )
    cap_img = ImageReader(
        _diverging_chart(a.cap_table, "cap_bucket", "Exposure by market cap",
                         side_w_pt / 72, (chart_h_pt / 2 - 6) / 72,
                         max_rows=6, label_extremes=False)
    )
    region_img = ImageReader(
        _diverging_chart(a.region_table, "region", "Exposure by region",
                         side_w_pt / 72, (chart_h_pt / 2 - 6) / 72,
                         max_rows=5, label_extremes=False)
    )

    def _draw_img(img: ImageReader, x: float, y_top: float, box_w: float, box_h: float):
        iw, ih = img.getSize()
        k = min(box_w / iw, box_h / ih)
        c.drawImage(img, x, y_top - ih * k, width=iw * k, height=ih * k, mask="auto")

    _draw_img(sector_img, MARGIN, y, sector_w_pt, chart_h_pt)
    right_x = MARGIN + sector_w_pt + 14
    _draw_img(cap_img, right_x, y, side_w_pt, chart_h_pt / 2 - 6)
    _draw_img(region_img, right_x, y - chart_h_pt / 2 - 6, side_w_pt, chart_h_pt / 2 - 6)
    y -= chart_h_pt + 18

    # ------------------------------------------- row 3: top issuer tables
    def top_table(side: str) -> list[list[str]]:
        iss = a.issuers
        # denominators are issuer-level side totals — same aggregation level
        # as the netted issuer exposures in the rows
        if side == "long":
            sel = iss[iss["exposure"] > 0].nlargest(10, "exposure")
            total = s["iss_exp_long"]
            head = ["Top long issuers", "Sector", "$M", "% Long"]
        else:
            sel = iss[iss["exposure"] < 0].nsmallest(10, "exposure")
            total = abs(s["iss_exp_short"])
            head = ["Top short issuers", "Sector", "$M", "% Short"]
        rows = [head]
        for _, r in sel.iterrows():
            nm = str(r["name"] or r["underlying"])[:26]
            rows.append([
                nm,
                str(r["sector"])[:18],
                _m(r["exposure"]),
                _pct(abs(r["exposure"]) / total if total else 0),
            ])
        return rows

    half = (CONTENT_W - 14) / 2
    tl = _table(top_table("long"), [half * 0.42, half * 0.30, half * 0.14, half * 0.14],
                align_left_cols=2)
    tr = _table(top_table("short"), [half * 0.42, half * 0.30, half * 0.14, half * 0.14],
                align_left_cols=2)
    hl = _draw_table(c, tl, MARGIN, y)
    hr = _draw_table(c, tr, MARGIN + half + 14, y)
    y -= max(hl, hr) + 14

    # ------------------------------------------- crowding & squeeze risk
    if crowding is not None:
        cr = crowding

        def _opt_pct(v):
            return "n/a" if v is None else _pct(v)

        summ_rows = [
            ["Crowding & squeeze (short interest)", ""],
            ["Wavg short % float — long book", _opt_pct(cr.wavg_si_float_long)],
            ["Wavg short % float — short book", _opt_pct(cr.wavg_si_float_short)],
            ["Wavg institutional own. — long book", _opt_pct(cr.wavg_inst_long)],
            ["Short exposure in crowded names", _m(cr.n_crowded_short_exposure)],
            ["Short-interest data coverage", _pct(cr.coverage)],
        ]
        sq = cr.squeeze_names.head(6)
        if len(sq):
            sq_rows = [["Squeeze risk (your crowded shorts)", "SI%Flt", "DTC", "Exp $M"]]
            for _, r in sq.iterrows():
                sq_rows.append([
                    str(r["name"] or r["underlying"])[:24],
                    f"{r['short_pct_float']:.0%}",
                    f"{r['short_ratio']:.1f}",
                    _m(r["exposure"]),
                ])
        else:
            sq_rows = [["Squeeze risk (your crowded shorts)", ""],
                       ["No shorts above crowding thresholds", ""]]

        cw1 = _table(summ_rows, [half * 0.74, half * 0.26])
        cw2 = _table(sq_rows,
                     [half * 0.46, half * 0.18, half * 0.14, half * 0.22]
                     if len(sq) else [half * 0.7, half * 0.3],
                     align_left_cols=1)
        hc1 = _draw_table(c, cw1, MARGIN, y)
        hc2 = _draw_table(c, cw2, MARGIN + half + 14, y)
        y -= max(hc1, hc2) + 14

    # ------------------------------------------------------------- footer
    c.setStrokeColor(HexColor(GRID))
    c.setLineWidth(0.5)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 9

    c.setFont("Helvetica", 5.8)
    c.setFillColor(HexColor(MUTED))
    notes = [
        "Methodology: equity prices are Yahoo Finance closes as of the report date; option quotes (bid/ask mid, else last trade, else "
        "Black-Scholes theoretical) and chain implied vols are live as of the run date. Option deltas are Black-Scholes (realized-vol "
        "fallback). Betas vs SPY (250d daily). Delta-adjusted exposure = equity MV + option delta notional, aggregated per issuer. "
        "Crowding: SI%Flt = short interest / float, DTC = days-to-cover (short ratio); crowded = SI ≥ 10% of float and DTC ≥ 5.",
    ]
    for issue in a.issues[:6]:
        notes.append(f"• {issue}")
    if len(a.issues) > 6:
        notes.append(f"• …and {len(a.issues) - 6} more (see snapshot summary.json)")
    import textwrap

    for line in notes:
        for chunk in textwrap.wrap(line, width=170, subsequent_indent="   "):
            c.drawString(MARGIN, y, chunk)
            y -= 7.2

    c.showPage()

    # scenarios can succeed even when the factor model fails — render the
    # risk page whenever either exists rather than dropping computed numbers
    if factor_risk is not None or scenarios is not None:
        _render_risk_page(c, analytics, name, factor_risk, scenarios, hedge,
                          model, bias)
        c.showPage()

    c.save()
    return out_path


def render_whatif(
    before, after, name: str, out_path: str | Path,
    trades: list, fr_before=None, fr_after=None,
    sc_before=None, sc_after=None, issues: list[str] | None = None,
) -> Path:
    """One-page before/after/delta what-if report.

    before/after are PortfolioAnalytics; fr_*/sc_* the optional factor-risk
    and scenario results for each book.
    """
    import math
    import textwrap

    out_path = Path(out_path)
    c = rl_canvas.Canvas(str(out_path), pagesize=letter)
    c.setTitle(f"What-If {name} {before.asof}")

    y = PAGE_H - MARGIN
    c.setFillColor(HexColor(INK))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y - 14, f"What-If Analysis — {name}")
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor(INK_2))
    c.drawRightString(PAGE_W - MARGIN, y - 8, f"As of {before.asof:%B %d, %Y}")
    c.drawRightString(PAGE_W - MARGIN, y - 18,
                      f"{len(trades)} trade(s) applied · Internal use only")
    c.setStrokeColor(HexColor(BASELINE))
    c.setLineWidth(0.8)
    c.line(MARGIN, y - 24, PAGE_W - MARGIN, y - 24)
    y -= 34

    # ------------------------------------------------------- trades table
    trade_rows = [["Trade list", "Qty"]]
    for t in trades[:14]:
        trade_rows.append([t.raw_symbol[:38], f"{t.qty:+,.0f}"])
    if len(trades) > 14:
        trade_rows.append([f"…and {len(trades) - 14} more", ""])

    # ------------------------------------------- before/after/delta tables
    sb, sa = before.summary, after.summary

    def bad_row(label, key, is_pct=False, denom=1.0):
        b, a = sb.get(key), sa.get(key)
        if b is None or a is None:
            return [label, "", "", ""]
        return [label, _m(b), _m(a), _m(a - b)]

    expo_rows = [["Exposure ($M, Δ-adj)", "Before", "After", "Δ"]]
    for label, key in [("Long", "exp_long"), ("Short", "exp_short"),
                       ("Gross", "exp_gross"), ("Net", "exp_net"),
                       ("Beta-adj net", "beta_net")]:
        expo_rows.append(bad_row(label, key))

    risk_rows = [["Risk", "Before", "After", "Δ"]]

    def _num(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else x

    pairs = []
    if fr_before is not None and fr_after is not None:
        pairs += [
            ("Predicted vol (ann.)", fr_before.vol_total, fr_after.vol_total),
            ("  from factors", fr_before.vol_factor, fr_after.vol_factor),
            ("  stock-specific", fr_before.vol_specific, fr_after.vol_specific),
        ]
    if sc_before is not None and sc_after is not None:
        pairs += [
            ("VaR 95% (1-day)", _num(sc_before.var_95), _num(sc_after.var_95)),
            ("VaR 99% (1-day)", _num(sc_before.var_99), _num(sc_after.var_99)),
            ("Expected shortfall 95%", _num(sc_before.es_95), _num(sc_after.es_95)),
        ]
    for label, b, a in pairs:
        if b is None or a is None:
            risk_rows.append([label, "n/a", "n/a", ""])
        else:
            risk_rows.append([label, _m(b), _m(a), _m(a - b)])

    t1 = _table(trade_rows, [150, 46])
    t2 = _table(expo_rows, [80, 44, 44, 44])
    h1 = _draw_table(c, t1, MARGIN, y)
    h2 = _draw_table(c, t2, MARGIN + 214, y)
    y_col2 = y - h2 - 10
    t3 = _table(risk_rows, [96, 44, 44, 44])
    h3 = _draw_table(c, t3, MARGIN + 214, y_col2)
    y -= max(h1, h2 + h3 + 10) + 14

    # ------------------------------------------------ factor exposure delta
    if fr_before is not None and fr_after is not None:
        fac_rows = [["Factor exposure (net $M)", "Before", "After", "Δ"]]
        for f in fr_before.exposures.index:
            b = float(fr_before.exposures.loc[f, "net"])
            a = float(fr_after.exposures.loc[f, "net"])
            fac_rows.append([FACTOR_LABELS.get(f, f), _m(b), _m(a), _m(a - b)])
        t4 = _table(fac_rows, [96, 46, 46, 46])
        h4 = _draw_table(c, t4, MARGIN, y)
    else:
        h4 = 0

    # --------------------------------------------------- top issuer movers
    ib = before.issuers.set_index("underlying")["exposure"]
    ia = after.issuers.set_index("underlying")["exposure"]
    meta = pd.concat([
        before.issuers.set_index("underlying")[["name", "sector"]],
        after.issuers.set_index("underlying")[["name", "sector"]],
    ])
    meta = meta[~meta.index.duplicated()]
    delta = (ia.reindex(ib.index.union(ia.index)).fillna(0.0)
             - ib.reindex(ib.index.union(ia.index)).fillna(0.0))
    movers = delta.abs().nlargest(10).index
    mover_rows = [["Largest exposure changes", "Sector", "Before", "After", "Δ"]]
    for u in movers:
        if abs(delta[u]) < 1.0:
            continue
        mover_rows.append([
            str(meta.loc[u, "name"])[:24],
            str(meta.loc[u, "sector"])[:16],
            _m(float(ib.get(u, 0.0))),
            _m(float(ia.get(u, 0.0))),
            _m(float(delta[u])),
        ])
    t5 = _table(mover_rows, [110, 74, 42, 42, 42], align_left_cols=2)
    h5 = _draw_table(c, t5, MARGIN + 250, y)
    y -= max(h4, h5) + 14

    # ------------------------------------------------------------- footer
    c.setStrokeColor(HexColor(GRID))
    c.setLineWidth(0.5)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 9
    c.setFont("Helvetica", 5.8)
    c.setFillColor(HexColor(MUTED))
    notes = [
        "Methodology identical to the main risk report: delta-adjusted exposures, Black-Scholes option pricing, "
        "Ken French factor loadings, full-revaluation historical-simulation VaR. New positions introduced by trades are "
        "priced with the same market data as the base book.",
    ]
    for issue in (issues or [])[:6]:
        notes.append(f"• {issue}")
    if issues and len(issues) > 6:
        notes.append(f"• …and {len(issues) - 6} more issue(s)")
    for line in notes:
        for chunk in textwrap.wrap(line, width=170, subsequent_indent="   "):
            c.drawString(MARGIN, y, chunk)
            y -= 7.2

    c.showPage()
    c.save()
    return out_path


AQUA = "#1baf7a"  # categorical slot 2
YELLOW = "#eda100"  # categorical slot 3
VIOLET = "#4a3aa7"  # categorical slot 5


def render_attribution(result, name: str, out_path: str | Path) -> Path:
    """One-page performance attribution report from an AttributionResult."""
    import textwrap

    r = result
    out_path = Path(out_path)
    c = rl_canvas.Canvas(str(out_path), pagesize=letter)
    c.setTitle(f"Attribution {name} {r.start} to {r.end}")

    y = PAGE_H - MARGIN
    c.setFillColor(HexColor(INK))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y - 14, f"Performance Attribution — {name}")
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor(INK_2))
    c.drawRightString(PAGE_W - MARGIN, y - 8,
                      f"{r.start:%b %d, %Y} → {r.end:%b %d, %Y} · {r.n_days} trading day(s)")
    c.drawRightString(PAGE_W - MARGIN, y - 18, "Model P&L · Internal use only")
    c.setStrokeColor(HexColor(BASELINE))
    c.setLineWidth(0.8)
    c.line(MARGIN, y - 24, PAGE_W - MARGIN, y - 24)
    y -= 34

    # ------------------------------------------------------ summary tables
    def _k(v: float) -> str:
        v = v / 1e3
        return f"({abs(v):,.0f})" if v < 0 else f"{v:,.0f}"

    cum = r.daily[["total", "market", "style", "specific"]].sum()
    sum_rows = [["Cumulative model P&L", "$K"]]
    for label, key in [("Total", "total"), ("Market (beta)", "market"),
                       ("Style factors", "style"), ("Stock-specific", "specific")]:
        sum_rows.append([label, _k(cum[key])])

    style_cols = [f for f in r.daily.columns
                  if f not in ("total", "market", "style", "specific")]
    fac_rows = [["Style factor P&L", "$K"]]
    for f in style_cols:
        fac_rows.append([FACTOR_LABELS.get(f, f), _k(float(r.daily[f].sum()))])

    t1 = _table(sum_rows, [104, 44])
    t2 = _table(fac_rows, [104, 44])
    h1 = _draw_table(c, t1, MARGIN, y)
    h2 = _draw_table(c, t2, MARGIN + 172, y)

    # -------------------------------------------------- cumulative chart
    fig, ax = plt.subplots(figsize=(3.1, 1.7), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cumd = r.daily[["total", "market", "specific"]].cumsum() / 1e3
    x = range(len(cumd))
    for col, color in [("total", BLUE), ("market", AQUA), ("specific", VIOLET)]:
        ax.plot(x, cumd[col], color=color, lw=1.6,
                solid_joinstyle="round", solid_capstyle="round")
    ax.axhline(0, color=BASELINE, lw=0.7)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    labels = [d.strftime("%m/%d") for d in cumd.index]
    step = max(1, len(labels) // 6)
    ax.set_xticks(list(x)[::step], labels[::step])
    ax.tick_params(labelsize=6, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Cumulative P&L, $K", fontsize=7.5, fontweight="bold",
                 color=INK, loc="left", pad=4)
    ax.legend(["Total", "Market", "Specific"], fontsize=6, frameon=False,
              loc="best", handlelength=1.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = ImageReader(buf)
    iw, ih = img.getSize()
    k = min(185 / iw, 125 / ih)
    c.drawImage(img, PAGE_W - MARGIN - iw * k, y - ih * k,
                width=iw * k, height=ih * k, mask="auto")
    y -= max(h1, h2, ih * k) + 16

    # ------------------------------------------ contributors and sectors
    best = r.issuer_specific.tail(5).iloc[::-1]
    worst = r.issuer_specific.head(5)
    contrib_rows = [["Stock-specific P&L", "$K"]]
    for u, v in best.items():
        contrib_rows.append([f"+ {u}"[:26], _k(float(v))])
    for u, v in worst.items():
        contrib_rows.append([f"− {u}"[:26], _k(float(v))])

    sec_rows = [["Total P&L by sector", "$K"]]
    for sec, v in r.sector_total.head(12).items():
        sec_rows.append([str(sec)[:24], _k(float(v))])

    t3 = _table(contrib_rows, [120, 50])
    t4 = _table(sec_rows, [120, 50])
    h3 = _draw_table(c, t3, MARGIN, y)
    h4 = _draw_table(c, t4, MARGIN + 210, y)
    y -= max(h3, h4) + 12

    # ------------------------------------------------------------- footer
    c.setStrokeColor(HexColor(GRID))
    c.setLineWidth(0.5)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 9
    c.setFont("Helvetica", 5.8)
    c.setFillColor(HexColor(MUTED))
    notes = [
        "Methodology: buy-and-hold model P&L per day from the latest prior snapshot; options repriced daily (Black-Scholes, snapshot IV, "
        "shrinking expiry). Market = exposure x market beta x market factor return; style = Ken French SMB/HML/RMW/CMA/MOM; "
        "specific = residual. Issuer rows show the five best and five worst stock-specific contributors.",
    ] + [f"• {n}" for n in r.notes[:4]]
    for line in notes:
        for chunk in textwrap.wrap(line, width=170, subsequent_indent="   "):
            c.drawString(MARGIN, y, chunk)
            y -= 7.2

    c.showPage()
    c.save()
    return out_path


def _render_risk_page(c, analytics, name, fr, sc, hedge=None, model=None,
                      bias=None) -> None:
    """Page 2: factor model, predicted vol, stress grid, VaR."""
    import math

    a = analytics
    s = a.summary
    gross = s["exp_gross"] or 1.0

    y = PAGE_H - MARGIN
    c.setFillColor(HexColor(INK))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y - 14, f"Factor Model & Scenarios — {name}")
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor(INK_2))
    c.drawRightString(PAGE_W - MARGIN, y - 8, f"As of {a.asof:%B %d, %Y}")
    c.drawRightString(PAGE_W - MARGIN, y - 18, "Internal use only")
    c.setStrokeColor(HexColor(BASELINE))
    c.setLineWidth(0.8)
    c.line(MARGIN, y - 24, PAGE_W - MARGIN, y - 24)
    y -= 34

    # ------------------------------------------- row 1: vol + VaR panels
    if fr is not None:
        vol_rows = [
            ["Predicted volatility (ann.)", "$M", "% Gross"],
            ["Total", _m(fr.vol_total), _pct(fr.vol_total / gross)],
            ["  from factors", _m(fr.vol_factor), _pct(fr.vol_factor / gross)],
            ["  stock-specific", _m(fr.vol_specific), _pct(fr.vol_specific / gross)],
            ["Factor share of variance", "", _pct(fr.factor_var_share)],
            ["Model coverage of gross", "", _pct(fr.coverage)],
        ]
    else:
        vol_rows = [
            ["Predicted volatility (ann.)", "$M", "% Gross"],
            ["Factor model unavailable this run", "n/a", ""],
        ]

    def _v(x):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else _m(x)

    def _vp(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        return _pct(x / gross)

    if sc is not None:
        var_rows = [
            ["Value at Risk (1-day, hist-sim)", "$M", "% Gross"],
            ["VaR 95%", _v(sc.var_95), _vp(sc.var_95)],
            ["VaR 99%", _v(sc.var_99), _vp(sc.var_99)],
            ["Expected shortfall 95%", _v(sc.es_95), _vp(sc.es_95)],
            [f"Worst day in window ({sc.worst_date})", _v(-sc.pnl_worst if sc.pnl_worst == sc.pnl_worst else float('nan')), ""],
            [f"Scenario days used", f"{sc.var_obs}", ""],
        ]
    else:
        var_rows = [
            ["Value at Risk (1-day, hist-sim)", "$M", "% Gross"],
            ["Scenarios unavailable this run", "n/a", ""],
        ]

    t1 = _table(vol_rows, [118, 46, 46])
    t2 = _table(var_rows, [130, 46, 46])
    h1 = _draw_table(c, t1, MARGIN, y)
    h2 = _draw_table(c, t2, MARGIN + 240, y)
    y -= max(h1, h2) + 10

    # ------------------------------ model diagnostics + bias-test line
    if model is not None:
        parts = [
            f"Model: {len(model.factor_names)} factors (FF5+mom+rev)",
            f"EWMA hl {int(model.half_life)}d",
            f"avg R²={model.avg_r2:.2f}",
            f"cov δ={model.cov_shrinkage:.2f}",
            f"cond={model.cond_number:.0f}",
        ]
        if model.n_shrunk:
            parts.append(f"{model.n_shrunk} sector-shrunk")
        if bias is not None:
            parts.append(f"OOS bias {bias.ratio:.2f} ({bias.window}d holdout)")
        c.setFont("Helvetica", 6.4)
        c.setFillColor(HexColor(INK_2))
        c.drawString(MARGIN, y, "  ·  ".join(parts))
        y -= 14

    # ----------------------------------- row 2: factor exposures + table
    if fr is not None:
        expo = fr.exposures.copy()
        cat = expo.reset_index().rename(columns={"index": "factor"})
        cat["factor"] = cat["factor"].map(lambda f: FACTOR_LABELS.get(f, f))
        cat["long"] = cat["net"].clip(lower=0.0)
        cat["short"] = cat["net"].clip(upper=0.0)
        cat["gross"] = cat["net"].abs()

        chart = ImageReader(
            _diverging_chart(
                cat, "factor", "Net factor exposure (delta-adjusted dollars)",
                3.4, 1.9, max_rows=len(cat), label_extremes=False,
            )
        )
        iw, ih = chart.getSize()
        k = min(250 / iw, 150 / ih)
        c.drawImage(chart, MARGIN, y - ih * k, width=iw * k, height=ih * k,
                    mask="auto")

        fac_rows = [["Factor", "Net $M", "% Gross", "% of variance"]]
        for f in fr.exposures.index:
            fac_rows.append([
                FACTOR_LABELS.get(f, f),
                _m(fr.exposures.loc[f, "net"]),
                _pct(fr.exposures.loc[f, "net"] / gross),
                _pct(float(fr.factor_risk_contrib.get(f, 0.0))),
            ])
        fac_rows.append(["Stock-specific", "", "", _pct(1.0 - fr.factor_var_share)])
        t3 = _table(fac_rows, [86, 44, 44, 60])
        _draw_table(c, t3, MARGIN + 280, y)
        y -= 160

    # --------------------------------------------- row 3: stress grid
    if sc is not None:
        grid = sc.stress_grid
        stress_rows = [["Stress P&L, $M (full reval)"] + list(grid.columns)]
        for label, row in grid.iterrows():
            stress_rows.append([label] + [_m(v) for v in row])
        t4 = _table(
            stress_rows,
            [92] + [round((CONTENT_W - 92) / len(grid.columns))] * len(grid.columns),
        )
        h4 = _draw_table(c, t4, MARGIN, y)
        y -= h4 + 6
        c.setFont("Helvetica", 6)
        c.setFillColor(HexColor(MUTED))
        c.drawString(
            MARGIN, y,
            "Market moves propagate to each name via its 250d beta; options fully "
            "repriced (Black-Scholes) at shocked spot and IV. Instantaneous P&L.",
        )
        y -= 16

    # --------------------------- row 4: risk contributors + risk by sector
    if fr is not None:
        pr = fr.position_risk
        by_issuer = (
            pr.groupby("underlying")
            .agg(name=("name", "first"), sector=("sector", "first"),
                 exposure=("exposure", "sum"), risk_contrib=("risk_contrib", "sum"))
            .reset_index()
            .sort_values("risk_contrib", ascending=False)
            .head(10)
        )
        contrib_rows = [["Top risk contributors", "Sector", "$M", "% var"]]
        for _, r in by_issuer.iterrows():
            contrib_rows.append([
                str(r["name"])[:26], str(r["sector"])[:18],
                _m(r["exposure"]), _pct(float(r["risk_contrib"])),
            ])

        by_sector = (
            pr.groupby("sector")["risk_contrib"].sum()
            .sort_values(ascending=False).head(12)
        )
        sector_rows = [["Risk by sector", "% of variance"]]
        for sec, v in by_sector.items():
            sector_rows.append([str(sec)[:24], _pct(float(v))])

        half = (CONTENT_W - 14) / 2
        t5 = _table(contrib_rows, [half * 0.40, half * 0.28, half * 0.15, half * 0.17],
                    align_left_cols=2)
        t6 = _table(sector_rows, [half * 0.6, half * 0.4])
        h5 = _draw_table(c, t5, MARGIN, y)
        h6 = _draw_table(c, t6, MARGIN + half + 14, y)
        y -= max(h5, h6) + 12

    # -------------------------------------------------- row 5: liquidity
    liq = s.get("liquidity")
    if liq:
        def _days(v):
            return "n/a" if v is None else f"{v:,.1f}"

        liq_rows = [
            ["Liquidity (net Δ-shares vs 60d ADV)", ""],
            ["% gross in names >25% ADV", _pct(liq["pct_gross_over_25adv"])],
            ["% gross in names >50% ADV", _pct(liq["pct_gross_over_50adv"])],
            ["% gross in names >100% ADV", _pct(liq["pct_gross_over_100adv"])],
            ["Days to liquidate, median (20% part.)", _days(liq["days_to_liq_p50"])],
            ["Days to liquidate, 95th %ile", _days(liq["days_to_liq_p95"])],
            ["ADV data coverage of gross", _pct(liq["adv_coverage"])],
        ]
        worst_liq = (
            a.issuers.dropna(subset=["pct_adv"])
            .nlargest(6, "pct_adv")
        )
        least_rows = [["Least liquid issuers", "% ADV", "Days", "Exp $M"]]
        for _, r in worst_liq.iterrows():
            least_rows.append([
                str(r["name"])[:24],
                f"{r['pct_adv']:.0%}",
                f"{r['days_to_liq']:,.1f}",
                _m(r["exposure"]),
            ])
        half = (CONTENT_W - 14) / 2
        t7 = _table(liq_rows, [half * 0.72, half * 0.28])
        t8 = _table(least_rows, [half * 0.46, half * 0.18, half * 0.16, half * 0.20],
                    align_left_cols=1)
        h7 = _draw_table(c, t7, MARGIN, y)
        h8 = _draw_table(c, t8, MARGIN + half + 14, y)
        y -= max(h7, h8) + 12

    # ------------------------------------------- row 6: hedge suggestion
    if hedge is not None and len(hedge.trades):
        red = hedge.vol_before - hedge.vol_after
        red_pct = red / hedge.vol_before if hedge.vol_before else 0.0
        hedge_rows = [["Suggested factor hedge", "$M notional", "~Shares"]]
        for _, r in hedge.trades.iterrows():
            hedge_rows.append([
                r["etf"], _m(r["notional"]),
                "" if r["shares"] is None else f"{r['shares']:+,}",
            ])
        hedge_rows.append([
            f"Predicted vol {_m(hedge.vol_before)} → {_m(hedge.vol_after)} $M",
            f"−{red_pct:.0%}", "",
        ])
        half = (CONTENT_W - 14) / 2
        t9 = _table(hedge_rows, [half * 0.5, half * 0.28, half * 0.22],
                    align_left_cols=1)
        h9 = _draw_table(c, t9, MARGIN, y)
        c.setFont("Helvetica", 6)
        c.setFillColor(HexColor(MUTED))
        c.drawString(
            MARGIN + half + 14, y - 8,
            "Basket minimizing residual factor variance over a liquid ETF",
        )
        c.drawString(MARGIN + half + 14, y - 16,
                     "menu. Drop into run_whatif.py to see the full impact.")
        y -= h9 + 12

    # ------------------------------------------------------------- footer
    c.setStrokeColor(HexColor(GRID))
    c.setLineWidth(0.5)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 9
    import textwrap

    c.setFont("Helvetica", 5.8)
    c.setFillColor(HexColor(MUTED))
    notes = []
    if fr is not None:
        notes.append(
            "Methodology: EWMA-weighted (half-life 90d) loadings of each name's daily excess returns on the Fama-French 5 factors "
            f"+ momentum + short/long-term reversal (Ken French daily library, through {fr.data_end}), ~2y window, min 60 days "
            "(20-60 shrunk toward the sector median). Factor covariance is EWMA-weighted with Ledoit-Wolf shrinkage to its diagonal. "
            "Predicted vol combines it with per-name residual vol (netted per underlying); contributions are x_i*(Cov x)_i / variance. "
            "OOS bias test: fit the model on data ending one holdout-window before the as-of date, then compare its predicted vol "
            "for the current book to the vol that book realized over the held-out window (≈1.0 = the methodology is well-scaled out-of-sample)."
        )
    if sc is not None:
        notes.append(
            f"VaR applies each of the last {sc.var_obs} daily joint return "
            "vectors to the current book with options fully repriced (IV held constant)."
        )
    all_issues = (fr.issues if fr is not None else []) + (sc.issues if sc is not None else [])
    for issue in all_issues[:4]:
        notes.append(f"• {issue}")
    for line in notes:
        for chunk in textwrap.wrap(line, width=170, subsequent_indent="   "):
            c.drawString(MARGIN, y, chunk)
            y -= 7.2
