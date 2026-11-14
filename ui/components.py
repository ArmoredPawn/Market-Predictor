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
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


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


CSS = """
<style>
.mp-intro { font-size: 1.15rem; opacity: 0.8; margin-top: -0.6rem; max-width: 52rem; }
.mp-headline {
  font-size: 1.3rem; line-height: 1.55; padding: 1.1rem 1.3rem; border-radius: 0.75rem;
  border: 1px solid rgba(128,128,128,0.25); background: rgba(42,120,214,0.07); margin: 0.4rem 0 1.2rem 0;
}
.mp-headline b { font-weight: 700; }
.mp-hero { padding: 0 0 0.4rem 0; margin-top: -1.5rem; }
.mp-hero h1 { font-size: clamp(2.2rem, 5vw, 3.4rem); line-height: 1.1; margin-bottom: 0.4rem; }
.mp-hero p { font-size: clamp(1.05rem, 2.2vw, 1.3rem); opacity: 0.8; max-width: 46rem; }
.mp-card-q { font-size: 1.05rem; font-style: italic; opacity: 0.85; min-height: 3.3rem; margin-bottom: 0.4rem; }
.mp-big { font-size: 2.6rem; font-weight: 700; line-height: 1.1; }
.mp-muted { opacity: 0.7; }
</style>
"""
