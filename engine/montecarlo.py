"""Monte Carlo simulation: "What might my investment be worth in N years?"

We can't know the future, but we can generate thousands of *plausible* futures
that behave statistically like the stock's past, then look at the spread of
outcomes.

Two models
----------
1. Geometric Brownian Motion ("smooth" model). Each day's log return is drawn
   from a bell curve:  ln(S_{t+1}/S_t) ~ Normal(m, s²).  This is the model behind
   the Black–Scholes option-pricing formula. It is clean but underestimates
   crashes, because real returns have fatter tails than a bell curve.

2. Stationary block bootstrap ("history replay" model). We replay real past
   days in random order, in short runs ("blocks") averaging `block_days` days,
   so calm and stormy stretches stay together (Politis & Romano, 1994). This
   keeps the real fat tails and volatility clustering.

The return assumption
---------------------
A stock that soared in the past will not necessarily keep soaring. So the
*average* growth rate is a separate, explicit assumption: either the stock's
own history, or a number the user chooses. Only the day-to-day *wiggle*
(volatility and shape) always comes from history. Concretely, the daily
log returns are shifted so their mean is ln(1 + g) / 252, where g is the
chosen typical (median) yearly growth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import TRADING_DAYS

PERCENTILES = (5, 25, 50, 75, 95)
DAYS_PER_MONTH = 21


@dataclass(frozen=True)
class SimulationConfig:
    start_value: float
    years: int
    n_paths: int = 2000
    monthly_contribution: float = 0.0
    model: str = "bootstrap"  # "bootstrap" or "gbm"
    annual_growth: float | None = None  # None = use the stock's own history
    block_days: int = 20
    seed: int | None = 7
    n_sample_paths: int = 30


@dataclass
class SimulationResult:
    config: SimulationConfig
    percentiles: np.ndarray  # shape (steps + 1, len(PERCENTILES))
    sample_paths: np.ndarray  # shape (n_sample_paths, steps + 1)
    final_values: np.ndarray  # shape (n_paths,)
    max_drawdowns: np.ndarray  # worst peak-to-trough fall in each path (≤ 0)
    invested: np.ndarray  # cumulative money put in, shape (steps + 1,)
    annual_growth_used: float  # typical (median) yearly growth assumed
    annual_volatility: float

    @property
    def steps(self) -> int:
        return self.percentiles.shape[0] - 1

    @property
    def total_invested(self) -> float:
        return float(self.invested[-1])

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.final_values, p))

    def prob_loss(self) -> float:
        """Chance of ending with less money than you put in."""
        return float(np.mean(self.final_values < self.total_invested))

    def prob_at_least(self, goal: float) -> float:
        return float(np.mean(self.final_values >= goal))

    def value_at_risk(self, level: float = 0.95) -> float:
        """Loss (vs. money put in) that is only exceeded in (1 − level) of futures."""
        return max(0.0, self.total_invested - float(np.quantile(self.final_values, 1 - level)))

    def expected_shortfall(self, level: float = 0.95) -> float:
        """Average loss in the worst (1 − level) of futures (a.k.a. CVaR)."""
        cutoff = np.quantile(self.final_values, 1 - level)
        tail = self.final_values[self.final_values <= cutoff]
        return max(0.0, self.total_invested - float(tail.mean())) if tail.size else 0.0

    def median_annual_return(self) -> float:
        """Typical yearly growth of the money (lump-sum only; ignores contributions)."""
        years = self.steps / TRADING_DAYS
        start = self.config.start_value
        if start <= 0 or years <= 0 or self.config.monthly_contribution:
            return float("nan")
        return float((self.percentile(50) / start) ** (1 / years) - 1)


def historical_growth(log_rets: np.ndarray) -> float:
    """Typical yearly growth implied by history: exp(252 · mean log return) − 1."""
    return float(np.expm1(np.mean(log_rets) * TRADING_DAYS))


def simulate(log_rets: np.ndarray | pd.Series, config: SimulationConfig) -> SimulationResult:
    """Run the simulation. `log_rets` are the stock's historical daily log returns."""
    hist = np.asarray(pd.Series(log_rets).dropna(), dtype=float)
    if hist.size < 60:
        raise ValueError("Need at least 60 days of price history to simulate.")
    if config.start_value < 0 or config.monthly_contribution < 0:
        raise ValueError("Amounts can't be negative.")
    if config.start_value == 0 and config.monthly_contribution == 0:
        raise ValueError("Enter a starting amount or a monthly contribution.")

    rng = np.random.default_rng(config.seed)
    steps = int(round(config.years * TRADING_DAYS))
    n = int(config.n_paths)
    n_hist = hist.size

    hist_mean = hist.mean()
    sigma = hist.std(ddof=1)
    growth = historical_growth(hist) if config.annual_growth is None else float(config.annual_growth)
    target_mean = np.log1p(growth) / TRADING_DAYS
    shifted = hist - hist_mean + target_mean  # same wiggles, chosen average

    values = np.full(n, float(config.start_value))
    peaks = values.copy()
    worst_dd = np.zeros(n)
    k = min(config.n_sample_paths, n)
    pct = np.empty((steps + 1, len(PERCENTILES)))
    samples = np.empty((k, steps + 1))
    invested = np.empty(steps + 1)
    pct[0] = values[0]
    samples[:, 0] = values[0]
    invested[0] = config.start_value

    idx = rng.integers(0, n_hist, size=n)  # bootstrap position for each path
    p_jump = 1.0 / max(1, config.block_days)
    total_in = float(config.start_value)

    for t in range(1, steps + 1):
        if config.model == "gbm":
            r = rng.normal(target_mean, sigma, size=n)
        else:
            jump = rng.random(n) < p_jump
            idx = np.where(jump, rng.integers(0, n_hist, size=n), (idx + 1) % n_hist)
            r = shifted[idx]
        values *= np.exp(r)
        if config.monthly_contribution and t % DAYS_PER_MONTH == 0:
            values += config.monthly_contribution
            total_in += config.monthly_contribution
            peaks += config.monthly_contribution  # new money isn't a "gain" to fall from
        np.maximum(peaks, values, out=peaks)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peaks > 0, values / peaks - 1.0, 0.0)
        np.minimum(worst_dd, dd, out=worst_dd)
        pct[t] = np.percentile(values, PERCENTILES)
        samples[:, t] = values[:k]
        invested[t] = total_in

    return SimulationResult(
        config=config,
        percentiles=pct,
        sample_paths=samples,
        final_values=values,
        max_drawdowns=worst_dd,
        invested=invested,
        annual_growth_used=growth,
        annual_volatility=float(sigma * np.sqrt(TRADING_DAYS)),
    )


def historical_windows(prices: pd.Series, days: int) -> pd.Series:
    """Growth multiple over every past window of `days` trading days (reality check)."""
    prices = prices.dropna()
    if len(prices) <= days:
        return pd.Series(dtype=float)
    return (prices.shift(-days) / prices).dropna()
