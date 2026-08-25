"""Tape-matched visual theme for the Streamlit app.

Prism shares a design language with the Tape research app
(tape.truporchhomesvm.com): a dark terminal palette, subtle card borders and
shadows, blue accent, green/red for gains/losses, and system sans-serif.

The base palette is set in .streamlit/config.toml; this module injects the
finer styling (cards, metrics, tabs, tables, buttons, sidebar, alerts) that
Streamlit's theme config can't reach. Call ``inject()`` once, right after
``st.set_page_config``. Selectors target stable ``data-testid`` hooks so a
Streamlit point-release is unlikely to break them; if one drifts, the app
still works — it just loses that bit of polish.
"""

from __future__ import annotations

# Tape's design tokens (captured from the live app).
TOKENS = {
    "bg": "#0b0e14",
    "surface": "#131722",
    "surface_2": "#0d1117",
    "surface_3": "#10182a",
    "surface_hover": "#1a1f2e",
    "border": "#2a2e39",
    "border_strong": "#3a4256",
    "divider": "#1f2430",
    "text": "#e6e9f0",
    "text_2": "#aab2c5",
    "text_3": "#8b93a7",
    "accent": "#60a5fa",
    "accent_strong": "#3b82f6",
    "accent_soft": "#60a5fa26",
    "pos": "#22c55e",
    "neg": "#ef4444",
    "warn": "#f59e0b",
    "shadow_sm": "0 1px 2px #0006",
    "shadow_md": "0 6px 20px -4px #0000008c",
    "radius": "12px",
    "font": ('ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, '
             '"Helvetica Neue", Arial, sans-serif'),
}

_CSS = """
<style>
:root {{
  --bg: {bg}; --surface: {surface}; --surface-2: {surface_2};
  --surface-3: {surface_3}; --surface-hover: {surface_hover};
  --border: {border}; --border-strong: {border_strong}; --divider: {divider};
  --text: {text}; --text-2: {text_2}; --text-3: {text_3};
  --accent: {accent}; --accent-strong: {accent_strong};
  --accent-soft: {accent_soft};
  --pos: {pos}; --neg: {neg}; --warn: {warn};
  --shadow-sm: {shadow_sm}; --shadow-md: {shadow_md}; --radius: {radius};
}}

/* --- base ------------------------------------------------------------- */
html, body, [class*="css"], .stApp {{
  font-family: {font};
  background: var(--bg);
  color: var(--text);
}}
.stApp {{
  background:
    radial-gradient(1200px 600px at 80% -10%, #10182a55, transparent 60%),
    var(--bg);
}}
/* tighten the default top gap and cap width like a real app shell */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* --- headings --------------------------------------------------------- */
h1, h2, h3, h4 {{ color: var(--text); letter-spacing: -0.01em; font-weight: 650; }}
h1 {{ font-size: 1.8rem; }}
h2 {{ font-size: 1.25rem; }}
h3 {{ font-size: 1.05rem; color: var(--text-2); }}
[data-testid="stCaptionContainer"], .stCaption, small {{ color: var(--text-3) !important; }}

/* --- sidebar ---------------------------------------------------------- */
[data-testid="stSidebar"] {{
  background: var(--surface-2);
  border-right: 1px solid var(--divider);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

/* --- metric "cards" --------------------------------------------------- */
[data-testid="stMetric"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.9rem 1rem;
  box-shadow: var(--shadow-sm);
}}
[data-testid="stMetric"]:hover {{ border-color: var(--border-strong); }}
[data-testid="stMetricLabel"] {{
  color: var(--text-3);
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] {{ color: var(--text); font-weight: 640; }}

/* --- tabs: a segmented nav bar like Tape ------------------------------ */
[data-testid="stTabs"] [role="tablist"] {{
  gap: 2px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 4px;
}}
[data-testid="stTabs"] [role="tablist"] {{ border-bottom: 1px solid var(--border); }}
[data-testid="stTabs"] button[role="tab"] {{
  color: var(--text-3);
  border-radius: 7px;
  padding: 0.35rem 0.8rem;
}}
[data-testid="stTabs"] button[role="tab"]:hover {{
  color: var(--text); background: var(--surface-hover);
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
  color: var(--text);
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 1px var(--accent);
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ background: transparent; }}

/* --- buttons ---------------------------------------------------------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius: 9px;
  border: 1px solid var(--border-strong);
  background: var(--surface-3);
  color: var(--text);
  font-weight: 560;
  transition: all .12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--accent);
  color: var(--text);
  background: var(--surface-hover);
}}
/* primary action (Generate report) */
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {{
  background: var(--accent-strong);
  border-color: var(--accent-strong);
  color: #fff;
}}
[data-testid="stBaseButton-primary"]:hover {{ background: var(--accent); border-color: var(--accent); }}

/* --- inputs ----------------------------------------------------------- */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {{
  background: var(--surface-2) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}}

/* --- dataframes ------------------------------------------------------- */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}}

/* --- alerts: map to Tape's semantic colors ---------------------------- */
[data-testid="stAlert"] {{ border-radius: 10px; border: 1px solid var(--border); }}
[data-testid="stAlert"][data-baseweb="notification"] {{ background: var(--surface); }}

/* --- expander / run-status block -------------------------------------- */
[data-testid="stExpander"], [data-testid="stExpander"] details {{
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}}
[data-testid="stCodeBlock"], pre {{
  background: var(--surface-2) !important;
  border: 1px solid var(--border);
  border-radius: 10px;
}}

/* --- radio as a segmented control ------------------------------------- */
[data-testid="stRadio"] [role="radiogroup"] {{ gap: 4px; }}

/* --- scrollbars ------------------------------------------------------- */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--border-strong); }}
::-webkit-scrollbar-track {{ background: transparent; }}
</style>
"""


def inject() -> None:
    """Inject the Tape-matched stylesheet. Call once after set_page_config."""
    import streamlit as st

    st.markdown(_CSS.format(**TOKENS), unsafe_allow_html=True)


def brand_header(title: str = "Prism", subtitle: str = "Portfolio Risk") -> None:
    """Render a compact branded header row echoing Tape's wordmark."""
    import streamlit as st

    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:.6rem;
                    padding:.2rem 0 1rem;border-bottom:1px solid var(--divider);
                    margin-bottom:1.2rem;">
          <span style="font-size:1.5rem;font-weight:720;letter-spacing:-.02em;
                       color:var(--text);">{title}</span>
          <span style="font-size:1.1rem;color:var(--accent);">◆</span>
          <span style="font-size:1rem;color:var(--text-3);font-weight:500;">
            {subtitle}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
