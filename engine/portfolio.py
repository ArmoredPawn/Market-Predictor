"""Portfolio optimization: "How should I split my money between these stocks?"

Modern Portfolio Theory (Harry Markowitz, 1952; Nobel Prize 1990)
------------------------------------------------------------------
For weights w (fractions of your money, summing to 1):

    expected yearly return   R(w) = wᵀ μ
    yearly risk (volatility) σ(w) = √(wᵀ Σ w)

where μ is the vector of expected returns and Σ the covariance matrix (how
much each pair of stocks moves together). Because stocks don't move in
lock-step, a mix can have *less* risk than any single stock in it. That is
diversification, and Σ is how we measure it.

The "efficient frontier" is the set of mixes with the lowest possible risk for
each level of return. Strategies offered:

  max_sharpe   best return per unit of risk:  maximize (R − r_f) / σ
  min_vol      the least bumpy mix possible:   minimize σ
  risk_parity  every stock contributes the same share of total risk
  equal        1/N in each (a surprisingly hard benchmark to beat)

Making it robust
----------------
Raw historical averages are very noisy estimates of future returns, and the
optimizer would happily bet everything on last decade's winner. We therefore:
  * shrink the covariance matrix with Ledoit–Wolf (2004),
  * optionally shrink each stock's expected return halfway toward the group
    average ("cautious" estimates),
  * cap the weight of any single stock,
  * and test everything out-of-sample with a walk-forward backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from engine import TRADING_DAYS

OBJECTIVES = ("max_sharpe", "min_vol", "risk_parity", "equal")


@dataclass(frozen=True)
class Estimates:
    mu: pd.Series  # expected yearly returns
    cov: pd.DataFrame  # yearly covariance matrix
    risk_free: float

    @property
    def symbols(self) -> list[str]:
        return list(self.mu.index)

    @property
    def vols(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.cov.values)), index=self.mu.index)

    def corr(self) -> pd.DataFrame:
        v = self.vols.values
        return pd.DataFrame(self.cov.values / np.outer(v, v), index=self.mu.index, columns=self.mu.index)


def estimate(prices: pd.DataFrame, risk_free: float, cautious: bool = True, shrink: float = 0.5) -> Estimates:
    rets = prices.pct_change().dropna(how="any")
    if len(rets) < 60:
        raise ValueError("Need at least 60 days of overlapping price history.")
    mu = rets.mean() * TRADING_DAYS
    if cautious and len(mu) > 1:
        mu = (1 - shrink) * mu + shrink * mu.mean()
    lw = LedoitWolf().fit(rets.values)
    cov = pd.DataFrame(lw.covariance_ * TRADING_DAYS, index=prices.columns, columns=prices.columns)
    return Estimates(mu=mu, cov=cov, risk_free=risk_free)


def portfolio_return(w: np.ndarray, est: Estimates) -> float:
    return float(w @ est.mu.values)


def portfolio_vol(w: np.ndarray, est: Estimates) -> float:
    return float(np.sqrt(max(w @ est.cov.values @ w, 0.0)))


def portfolio_sharpe(w: np.ndarray, est: Estimates) -> float:
    vol = portfolio_vol(w, est)
    return (portfolio_return(w, est) - est.risk_free) / vol if vol > 0 else float("nan")


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of total risk coming from each holding (sums to 1)."""
    var = float(w @ cov @ w)
    return (w * (cov @ w)) / var if var > 0 else np.full_like(w, 1 / len(w))


def effective_cap(n: int, max_weight: float) -> float:
    """A cap below 1/N is impossible (weights must sum to 1), so lift it."""
    return float(min(1.0, max(max_weight, 1.0 / n + 1e-9)))


def _greedy_max_return(mu: np.ndarray, cap: float) -> np.ndarray:
    """Highest-return portfolio under the cap: fill the best stocks first."""
    w = np.zeros_like(mu)
    left = 1.0
    for i in np.argsort(-mu):
        w[i] = min(cap, left)
        left -= w[i]
        if left <= 1e-12:
            break
    return w


def _solve(fun, n: int, cap: float, starts: list[np.ndarray], extra_constraints=()) -> np.ndarray:
    bounds = [(0.0, cap)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *extra_constraints]
    best, best_val = None, np.inf
    for x0 in starts:
        res = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        w = np.clip(res.x, 0, cap)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        val = fun(w)
        if np.isfinite(val) and val < best_val:
            best, best_val = w, val
    if best is None:
        best = np.full(n, 1.0 / n)
    return best


def optimize(est: Estimates, objective: str, max_weight: float = 1.0) -> np.ndarray:
    n = len(est.mu)
    if n == 1:
        return np.array([1.0])
    cap = effective_cap(n, max_weight)
    mu, cov = est.mu.values, est.cov.values
    equal = np.full(n, 1.0 / n)

    if objective == "equal":
        return equal
    if objective == "min_vol":
        return _solve(lambda w: w @ cov @ w, n, cap, [equal])
    if objective == "risk_parity":
        def rp(w):
            rc = risk_contributions(w, cov)
            return float(np.sum((rc - 1.0 / n) ** 2)) * 1e3
        inv_vol = 1 / np.sqrt(np.diag(cov))
        x0 = np.minimum(inv_vol / inv_vol.sum(), cap)
        return _solve(rp, n, cap, [x0 / x0.sum(), equal])
    if objective == "max_sharpe":
        if np.max(mu) <= est.risk_free:  # no stock beats cash: least-risk mix
            return optimize(est, "min_vol", max_weight)
        def neg_sharpe(w):
            vol = np.sqrt(max(w @ cov @ w, 1e-16))
            return -(w @ mu - est.risk_free) / vol
        greedy = _greedy_max_return(mu, cap)
        minvol = _solve(lambda w: w @ cov @ w, n, cap, [equal])
        return _solve(neg_sharpe, n, cap, [equal, greedy, minvol, (equal + greedy) / 2])
    raise ValueError(f"Unknown objective: {objective}")


def efficient_frontier(est: Estimates, max_weight: float = 1.0, points: int = 40) -> pd.DataFrame:
    """Lowest-risk portfolio for each target return, from min-risk to max-return."""
    n = len(est.mu)
    cap = effective_cap(n, max_weight)
    mu, cov = est.mu.values, est.cov.values
    w_min = optimize(est, "min_vol", max_weight)
    r_lo = float(w_min @ mu)
    r_hi = float(_greedy_max_return(mu, cap) @ mu)
    rows = []
    prev = w_min
    for target in np.linspace(r_lo, r_hi, points):
        cons = ({"type": "eq", "fun": lambda w, t=target: w @ mu - t},)
        w = _solve(lambda w: w @ cov @ w, n, cap, [prev], cons)
        if abs(w @ mu - target) > 1e-3:
            continue
        prev = w
        rows.append({"ret": float(w @ mu), "vol": portfolio_vol(w, est), "weights": w})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sharpe"] = (df["ret"] - est.risk_free) / df["vol"]
    return df


def random_portfolios(est: Estimates, n: int = 3000, seed: int = 1) -> pd.DataFrame:
    """Random long-only mixes, to show the cloud of possible portfolios."""
    rng = np.random.default_rng(seed)
    k = len(est.mu)
    w = rng.dirichlet(np.full(k, 0.7), size=n)
    rets = w @ est.mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, est.cov.values, w))
    return pd.DataFrame({"ret": rets, "vol": vols, "sharpe": (rets - est.risk_free) / vols})


def backtest(
    prices: pd.DataFrame,
    objective: str,
    risk_free: float,
    max_weight: float = 1.0,
    cautious: bool = True,
    lookback: int = 504,
    rebalance_every: int = 63,
    cost: float = 0.001,
) -> tuple[pd.Series, pd.DataFrame]:
    """Walk-forward test: at each rebalance date, look *only* at the past
    `lookback` days, pick weights, then hold them (letting them drift) until
    the next rebalance. Trading costs are charged on every change.

    Returns the growth of $1 and the history of chosen weights.
    """
    prices = prices.dropna(how="any")
    rets = prices.pct_change().fillna(0.0).values
    dates = prices.index
    n_days, n = rets.shape
    if n_days <= lookback + rebalance_every:
        raise ValueError("Not enough history for a walk-forward test.")

    value = 1.0
    w = np.zeros(n)
    equity = [value]
    eq_dates = [dates[lookback]]
    weight_log = []
    for i in range(lookback, n_days - 1):
        if (i - lookback) % rebalance_every == 0:
            window = prices.iloc[i - lookback : i + 1]
            est = estimate(window, risk_free, cautious=cautious)
            target = optimize(est, objective, max_weight)
            value *= 1.0 - cost * float(np.abs(target - w).sum())
            w = target
            weight_log.append(pd.Series(target, index=prices.columns, name=dates[i]))
        growth = 1.0 + rets[i + 1]
        port_growth = float(w @ growth)
        value *= port_growth
        w = w * growth / port_growth  # weights drift with prices
        equity.append(value)
        eq_dates.append(dates[i + 1])
    return pd.Series(equity, index=eq_dates), pd.DataFrame(weight_log)
