"""The investable universe: US-based companies and US stock ETFs only.

A ticker is accepted when either
  1. it is in our bundled list of US securities (data/us_tickers.csv), or
  2. it is not in any list we have (for example a brand-new IPO), *and* Yahoo
     Finance reports it trades on a major US exchange, in US dollars, as a
     stock or ETF.

Known foreign companies (data/non_us_tickers.csv) are always rejected with an
explanation, even though some of them trade on US exchanges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
US_FILE = DATA_DIR / "us_tickers.csv"
FOREIGN_FILE = DATA_DIR / "non_us_tickers.csv"

# Yahoo Finance exchange codes for the main US exchanges.
US_EXCHANGE_CODES = frozenset({"NMS", "NGM", "NCM", "NAS", "NYQ", "NYS", "ASE", "PCX", "BTS", "CBO"})
ALLOWED_INSTRUMENTS = frozenset({"EQUITY", "ETF"})
LABEL_SEP = " · "
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}(-[A-Z])?$")


@dataclass(frozen=True)
class TickerCheck:
    """Result of checking whether a user's ticker is allowed."""

    ok: bool
    symbol: str
    name: str = ""
    reason: str = ""
    needs_online_check: bool = False


@lru_cache(maxsize=1)
def load_universe() -> pd.DataFrame:
    df = pd.read_csv(US_FILE, keep_default_na=False, na_values=[""])
    df["symbol"] = df["symbol"].astype(str)
    return df.set_index("symbol", drop=False)


@lru_cache(maxsize=1)
def load_foreign() -> pd.DataFrame:
    if not FOREIGN_FILE.exists():
        return pd.DataFrame(columns=["symbol", "name", "country"]).set_index("symbol", drop=False)
    df = pd.read_csv(FOREIGN_FILE, keep_default_na=False)
    return df.set_index("symbol", drop=False)


def normalize_ticker(raw: str) -> str:
    """Turn what a person typed into a Yahoo Finance symbol.

    >>> normalize_ticker(" brk.b ")
    'BRK-B'
    >>> normalize_ticker("$aapl")
    'AAPL'
    """
    s = str(raw).strip().upper()
    if LABEL_SEP in s:  # a dropdown label like "AAPL · Apple Inc."
        s = s.split(LABEL_SEP, 1)[0].strip()
    s = s.lstrip("$").replace("/", "-").replace(".", "-").replace(" ", "")
    return s


def option_label(symbol: str) -> str:
    """Dropdown label, e.g. 'AAPL · Apple Inc.'"""
    uni = load_universe()
    if symbol in uni.index:
        return f"{symbol}{LABEL_SEP}{uni.at[symbol, 'name']}"
    return symbol


@lru_cache(maxsize=1)
def option_labels() -> tuple[str, ...]:
    uni = load_universe()
    return tuple(f"{s}{LABEL_SEP}{n}" for s, n in zip(uni["symbol"], uni["name"]))


def display_name(symbol: str) -> str:
    """Name for sentences: 'SPDR S&P 500 ETF (the 500 largest…)' -> 'SPDR S&P 500 ETF'."""
    uni = load_universe()
    if symbol in uni.index:
        return re.sub(r"\s*\(.*?\)\s*$", "", str(uni.at[symbol, "name"])).strip()
    return symbol


def check_ticker(raw: str) -> TickerCheck:
    """Offline check. `needs_online_check=True` means: unknown, verify with metadata."""
    symbol = normalize_ticker(raw)
    if not symbol:
        return TickerCheck(False, symbol, reason="Please enter a ticker symbol.")
    uni = load_universe()
    if symbol in uni.index:
        return TickerCheck(True, symbol, name=str(uni.at[symbol, "name"]))
    foreign = load_foreign()
    if symbol in foreign.index:
        row = foreign.loc[symbol]
        return TickerCheck(
            False,
            symbol,
            name=str(row["name"]),
            reason=f"{symbol} is {row['name']}, a company based in {row['country']}. "
            "This site only covers US companies.",
        )
    # Exchange suffixes like ".TO" (Toronto) or ".L" (London) became "-TO"/"-L".
    if not _TICKER_RE.match(symbol):
        return TickerCheck(
            False,
            symbol,
            reason=f"“{raw}” doesn't look like a US ticker. US tickers are 1–5 letters, like AAPL or BRK.B.",
        )
    return TickerCheck(True, symbol, needs_online_check=True)


def check_metadata(symbol: str, meta: dict | None) -> TickerCheck:
    """Online check for tickers not in our lists, using Yahoo's price metadata."""
    if not meta:
        return TickerCheck(False, symbol, reason=f"We couldn't find any price data for {symbol}. Check the spelling?")
    name = meta.get("longName") or meta.get("shortName") or symbol
    currency = meta.get("currency")
    exchange = meta.get("exchangeName")
    instrument = meta.get("instrumentType")
    if instrument not in ALLOWED_INSTRUMENTS:
        kind = (instrument or "unknown").lower()
        return TickerCheck(False, symbol, name, f"{symbol} is a {kind}, not a stock or ETF.")
    if currency != "USD" or exchange not in US_EXCHANGE_CODES:
        where = meta.get("fullExchangeName") or exchange or "a non-US market"
        return TickerCheck(False, symbol, name, f"{symbol} ({name}) trades on {where}. This site only covers US stocks.")
    return TickerCheck(True, symbol, name)
