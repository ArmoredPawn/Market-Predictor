"""Shared Streamlit pieces: cached data, ticker pickers, formatting, callouts."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from engine import data as D
from engine import universe as U

POPULAR = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "KO", "SPY", "QQQ", "VTI"]


# ------------------------------------------------------------------ cached data


@st.cache_data(ttl=3 * 3600, show_spinner="Downloading price history…", max_entries=200)
def get_prices(symbols: tuple[str, ...], years: int) -> D.PriceData:
    return D.fetch_prices(list(symbols), years)


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=500)
def get_metadata(symbol: str) -> dict | None:
    return D.fetch_metadata(symbol)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_risk_free() -> float:
    return D.fetch_risk_free_rate()


def resolve(raw: str) -> U.TickerCheck:
    """Check a ticker offline first, then online for tickers we don't know."""
    check = U.check_ticker(raw)
    if check.ok and check.needs_online_check:
        with st.spinner(f"Checking {check.symbol}…"):
            return U.check_metadata(check.symbol, get_metadata(check.symbol))
    return check


# ------------------------------------------------------------------ ticker pickers


def default_ticker(fallback: str = "AAPL") -> str:
    """Use ?t=XYZ from the address bar when present (links from the home page)."""
    t = st.query_params.get("t")
    return U.normalize_ticker(t) if t else fallback


def ticker_select(label: str, key: str, default: str = "AAPL", help: str | None = None) -> str | None:
    labels = U.option_labels()
    default_label = U.option_label(default)
    if default_label not in labels:
        default_label = U.option_label("AAPL")
    choice = st.selectbox(
        label,
        labels,
        index=labels.index(default_label),
        key=key,
        accept_new_options=True,
        placeholder="Type a company name or ticker…",
        help=help or "Search ~3,900 US companies and popular US stock ETFs by name or ticker. "
        "You can also type any US ticker that isn't listed.",
    )
    if not choice:
        st.info("Pick a stock to begin.", icon=":material/search:")
        return None
    check = resolve(choice)
    if not check.ok:
        st.error(check.reason, icon=":material/block:")
        return None
    return check.symbol


def ticker_multiselect(label: str, key: str, defaults: list[str], max_selections: int = 15) -> list[str]:
    labels = U.option_labels()
    default_labels = [U.option_label(d) for d in defaults if U.option_label(d) in labels]
    choices = st.multiselect(
        label,
        labels,
        default=default_labels,
        key=key,
        accept_new_options=True,
        max_selections=max_selections,
        placeholder="Add a company name or ticker…",
        help="Search by company name or ticker. US companies and US stock ETFs only.",
    )
    symbols: list[str] = []
    for c in choices:
        check = resolve(c)
        if check.ok:
            if check.symbol not in symbols:
                symbols.append(check.symbol)
        else:
            st.warning(check.reason, icon=":material/block:")
    return symbols


# ------------------------------------------------------------------ formatting


def money(x: float, decimals: int = 0) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"


def money_short(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    a = abs(x)
    sign = "−" if x < 0 else ""
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div * 10 or (a >= div and suffix != "K"):
            return f"{sign}${a / div:,.1f}{suffix}"
    return f"{sign}${a:,.0f}"


def pct(x: float, decimals: int = 1, signed: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    s = f"{x * 100:+.{decimals}f}%" if signed else f"{x * 100:.{decimals}f}%"
    return s.replace("-", "−")


def delta(x: float, decimals: int = 0, suffix: str = "") -> str:
    """Signed percentage for st.metric deltas (ASCII minus, so the arrow and color are right)."""
    if x is None or not math.isfinite(x):
        return ""
    return f"{x * 100:+.{decimals}f}%{suffix}"


def chance(p: float) -> str:
    """0.034 -> 'about 3%' ; 0.0004 -> 'less than 1%'."""
    if p < 0.01:
        return "less than 1%"
    if p > 0.99:
        return "more than 99%"
    return f"about {round(p * 100)}%"


def odds_in_words(p: float) -> str:
    """0.2 -> '1 in 5'."""
    if p <= 0:
        return "almost never"
    if p >= 0.995:
        return "almost always"
    n = round(1 / p) if p < 0.5 else None
    return f"about 1 in {n}" if n else f"about {round(p * 10)} in 10"


# ------------------------------------------------------------------ layout pieces


def page_header(icon: str, title: str, intro: str):
    st.markdown(f"# {icon} {title}")
    st.markdown(f"<p class='mp-intro'>{intro}</p>", unsafe_allow_html=True)


def headline(text: str):
    """The plain-English answer, shown big at the top of the results."""
    st.markdown(f"<div class='mp-headline'>{text}</div>", unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None):
    """Section heading with the site's indigo-to-pink accent bar."""
    st.markdown(f"<h3 class='mp-h'>{title}</h3>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def card(key: str, **kwargs):
    """A bordered box styled like the cards on armoredpawn.github.io (gradient, shadow, hover)."""
    return st.container(border=True, key=f"card-{key}", **kwargs)


def tag(text: str):
    """Small uppercase accent label above a card title."""
    st.markdown(f"<span class='mp-tag'>{text}</span>", unsafe_allow_html=True)


def see_numbers(df: pd.DataFrame, key: str, column_config: dict | None = None, label: str = "See the numbers"):
    with st.expander(label, icon=":material/table:"):
        st.dataframe(df, column_config=column_config, key=key)


def learn_more(title: str = "How does this work? (the math)"):
    return st.expander(title, icon=":material/school:")


def data_error(exc: Exception):
    st.error(str(exc), icon=":material/cloud_off:")
    st.stop()


def disclaimer():
    st.divider()
    st.caption(
        "**Educational tool, not financial advice.** Forecasts and simulations are built from past prices, "
        "and the past does not guarantee the future. Prices come from Yahoo Finance via the open-source "
        "`yfinance` library, are adjusted for splits and dividends, and may be delayed or occasionally wrong. "
        "Covers US-based companies and US stock ETFs only."
    )


# Styling that mirrors armoredpawn.github.io/About (deep navy, indigo + pink accents, Inter,
# gradient cards with soft shadows). Colors and fonts themselves live in .streamlit/config.toml.
CSS = """
<style>
:root {
  --mp-bg: #07090f;
  --mp-surface: #10141f;
  --mp-surface-top: #151a28;
  --mp-ink: #eceef5;
  --mp-ink-soft: #98a1b5;
  --mp-line: #1f2535;
  --mp-accent: #8f95ff;
  --mp-accent-2: #f472b6;
  --mp-shadow: 0 1px 2px rgb(0 0 0 / 0.5), 0 12px 32px rgb(0 0 0 / 0.45);
}

/* A soft wash of the two accents at the top of every page */
[data-testid="stApp"] {
  background:
    radial-gradient(56rem 30rem at 10% -8rem, color-mix(in srgb, var(--mp-accent) 22%, transparent), transparent 70%) no-repeat,
    radial-gradient(40rem 26rem at 95% -6rem, color-mix(in srgb, var(--mp-accent-2) 15%, transparent), transparent 70%) no-repeat,
    var(--mp-bg);
}

/* Frosted top navigation bar */
[data-testid="stHeader"] {
  background: color-mix(in srgb, var(--mp-bg) 85%, transparent);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--mp-line);
}
[data-testid="stTopNavLink"] { font-weight: 600; }

h1, h2, h3 { letter-spacing: -0.02em; }

/* Cards: gradient surface, soft shadow, accent border on hover */
[class*="st-key-card-"], [data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--mp-surface-top), var(--mp-surface));
  box-shadow: var(--mp-shadow);
  transition: border-color 0.15s ease, transform 0.15s ease;
}
[class*="st-key-card-"]:hover { border-color: color-mix(in srgb, var(--mp-accent) 35%, var(--mp-line)) !important; }
[class*="st-key-card-tool"]:hover { transform: translateY(-2px); border-color: var(--mp-accent) !important; }
[data-testid="stMetric"] { border-radius: 16px; }

/* Primary buttons wear the indigo-to-pink gradient, like the avatar ring */
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--mp-accent), var(--mp-accent-2));
  border: none;
  color: #07090f;
  font-weight: 600;
}
[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); color: #07090f; }
[data-testid="stBaseButton-primary"] p { color: #07090f; font-weight: 600; }

/* Section headings with the gradient accent bar */
.mp-h {
  display: flex; align-items: center; gap: 10px;
  font-size: 1.4rem !important; font-weight: 700; letter-spacing: -0.01em;
  margin: 1.6rem 0 0.2rem 0 !important; padding: 0 !important;
}
.mp-h::before {
  content: ""; flex: none; width: 4px; height: 1.1em; border-radius: 2px;
  background: linear-gradient(var(--mp-accent), var(--mp-accent-2));
}

.mp-tag {
  color: var(--mp-accent); font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.04em; text-transform: uppercase;
}

.mp-intro { font-size: 1.1rem; color: var(--mp-ink-soft); margin-top: -0.6rem; max-width: 60ch; }

/* The plain-English answer: a card with a glowing accent edge */
.mp-headline {
  position: relative;
  font-size: 1.25rem; line-height: 1.6; padding: 1.2rem 1.4rem 1.2rem 1.6rem;
  border-radius: 16px; margin: 0.4rem 0 1.2rem 0;
  background: linear-gradient(180deg, var(--mp-surface-top), var(--mp-surface));
  border: 1px solid color-mix(in srgb, var(--mp-accent) 35%, var(--mp-line));
  box-shadow: var(--mp-shadow);
  overflow: hidden;
}
.mp-headline::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: linear-gradient(var(--mp-accent), var(--mp-accent-2));
}
.mp-headline b { font-weight: 700; color: #ffffff; }

.mp-hero { padding: 0 0 0.4rem 0; margin-top: -1.5rem; }
.mp-hero h1 { font-size: clamp(2.2rem, 5vw, 3.4rem); line-height: 1.1; margin-bottom: 0.4rem; letter-spacing: -0.02em; }
.mp-hero h1 .mp-grad {
  background: linear-gradient(90deg, var(--mp-accent), var(--mp-accent-2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.mp-hero p { font-size: clamp(1.05rem, 2.2vw, 1.2rem); color: var(--mp-ink-soft); max-width: 60ch; }
.mp-card-q { font-size: 1rem; color: var(--mp-ink-soft); min-height: 3.2rem; margin-bottom: 0.4rem; }
.mp-big { font-size: 2.6rem; font-weight: 700; line-height: 1.1; color: var(--mp-accent); }
.mp-muted { color: var(--mp-ink-soft); }
</style>
"""
