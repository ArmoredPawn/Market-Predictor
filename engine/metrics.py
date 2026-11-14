"""Performance and risk measures shared by every tool.

Conventions
-----------
* "returns" are simple daily returns: r_t = P_t / P_{t-1} - 1
* "log returns" are ln(P_t / P_{t-1}); they add up over time, which makes
  them convenient for simulation.
* Annualizing uses 252 trading days per year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import TRADING_DAYS


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices / prices.shift(1)).iloc[1:]


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate: the steady yearly rate that turns start into end."""
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (len(equity) - 1) / TRADING_DAYS
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def annual_volatility(returns: pd.Series) -> float:
    """Standard deviation of daily returns, scaled to a year (multiply by √252)."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """(average yearly return − risk-free rate) / yearly volatility."""
    vol = annual_volatility(returns)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return float((returns.mean() * TRADING_DAYS - risk_free) / vol)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """How far below its previous peak the value is on each day (0 to −1)."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """The worst peak-to-bottom fall, e.g. −0.34 means a 34% drop at the worst point."""
    return float(drawdown_series(equity).min()) if len(equity) else float("nan")


@dataclass(frozen=True)
class PerformanceSummary:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    best_year: float
    worst_year: float


def summarize(equity: pd.Series, risk_free: float = 0.0) -> PerformanceSummary:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    yearly = equity.resample("YE").last().pct_change().dropna() if isinstance(equity.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    return PerformanceSummary(
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else float("nan"),
        cagr=cagr(equity),
        volatility=annual_volatility(rets),
        sharpe=sharpe_ratio(rets, risk_free),
        max_drawdown=max_drawdown(equity),
        best_year=float(yearly.max()) if len(yearly) else float("nan"),
        worst_year=float(yearly.min()) if len(yearly) else float("nan"),
    )
