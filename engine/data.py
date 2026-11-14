"""Downloading daily price history from Yahoo Finance (via the yfinance library).

Prices are *adjusted*: dividends and stock splits are folded in, so the
percentage change between two days is the true total return an investor got.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

BENCHMARK = "SPY"  # S&P 500 ETF, our yardstick for "the market"
DEFAULT_RISK_FREE = 0.04


class DataUnavailable(RuntimeError):
    """Raised with a message that is safe to show to a website visitor."""


@dataclass
class PriceData:
    close: pd.DataFrame  # adjusted close, one column per ticker
    volume: pd.DataFrame
    missing: list[str] = field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def _strip_tz(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def fetch_prices(symbols: list[str] | tuple[str, ...], years: int) -> PriceData:
    """Daily adjusted closes and volumes for up to ~25 tickers."""
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise DataUnavailable("No tickers were given.")
    try:
        raw = yf.download(
            symbols,
            period=f"{int(years)}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:  # network errors, rate limits, ...
        log.exception("yfinance download failed")
        raise DataUnavailable(
            "We couldn't reach the price data service right now. Please try again in a minute."
        ) from exc
    if raw is None or raw.empty:
        raise DataUnavailable("No price data came back. The data service may be busy — please try again shortly.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy() if "Volume" in raw.columns.get_level_values(0) else close * np.nan
    else:  # very old yfinance: single ticker, flat columns
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": symbols[0]})

    close.index = _strip_tz(close.index)
    volume.index = _strip_tz(volume.index)
    close = close.reindex(columns=symbols)
    volume = volume.reindex(columns=symbols)

    missing = [s for s in symbols if close[s].dropna().shape[0] < 2]
    keep = [s for s in symbols if s not in missing]
    close = close[keep].dropna(how="all").sort_index()
    volume = volume[keep].reindex(close.index)
    close = close[~close.index.duplicated(keep="last")]
    volume = volume[~volume.index.duplicated(keep="last")]
    if close.empty:
        raise DataUnavailable("We couldn't find price data for: " + ", ".join(symbols) + ".")
    return PriceData(close=close, volume=volume, missing=missing)


def fetch_metadata(symbol: str) -> dict | None:
    """Yahoo's description of a ticker (exchange, currency, type, name)."""
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        return dict(t.history_metadata or {})
    except Exception:
        log.exception("metadata lookup failed for %s", symbol)
        return None


def fetch_risk_free_rate() -> float:
    """Current 13-week US Treasury bill yield, as a decimal (0.041 = 4.1%)."""
    try:
        hist = yf.Ticker("^IRX").history(period="1mo")
        value = float(hist["Close"].dropna().iloc[-1]) / 100.0
        if 0.0 <= value < 0.25:
            return value
    except Exception:
        log.exception("risk-free rate lookup failed")
    return DEFAULT_RISK_FREE
