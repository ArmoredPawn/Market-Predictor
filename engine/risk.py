"""Risk forecasting: "How bumpy will the next month be?"

Unlike direction, *volatility* (the size of price swings) is genuinely
predictable, because calm and stormy periods tend to cluster: a wild week is
usually followed by more wild days. This is one of the most reliable findings
in finance (Mandelbrot 1963; Engle's ARCH model, Nobel Prize 2003).

We use the HAR model (Heterogeneous AutoRegressive, Corsi 2009), which
predicts next month's volatility from volatility over the past week, month,
quarter and year:

    log σ_next = β0 + β1·log σ_week + β2·log σ_month + β3·log σ_quarter
               + β4·log σ_year + β5·(recent fall) + β6·log σ_market

Working with logs keeps forecasts positive and tames extreme values. The
"recent fall" term captures the *leverage effect*: prices that just dropped
tend to get more volatile.

The forecast becomes an expected price range. If yearly volatility is σ, a
typical h-day move is σ·√(h/252); about 2 in 3 outcomes should land within
±1 typical move, and about 19 in 20 within ±2 (if returns were bell-shaped).
We measure how often that really happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from engine import TRADING_DAYS


def _realized_vol(lr: pd.Series, window: int) -> pd.Series:
    return np.sqrt(TRADING_DAYS * (lr**2).rolling(window).mean())


def har_features(close: pd.Series, market: pd.Series | None = None) -> pd.DataFrame:
    lr = np.log(close / close.shift(1))
    f = pd.DataFrame(index=close.index)
    for name, w in (("week", 5), ("month", 21), ("quarter", 63), ("year", 252)):
        f[f"log_vol_{name}"] = np.log(_realized_vol(lr, w).clip(lower=1e-4))
    f["recent_fall"] = np.log(close / close.shift(21)).clip(upper=0)
    if market is not None:
        m = market.reindex(close.index).ffill()
        mlr = np.log(m / m.shift(1))
        f["log_vol_market"] = np.log(_realized_vol(mlr, 21).clip(lower=1e-4))
    return f.replace([np.inf, -np.inf], np.nan)


def forward_vol(close: pd.Series, horizon: int) -> pd.Series:
    """Realized yearly volatility over the *next* `horizon` days (the answer we try to predict)."""
    lr = np.log(close / close.shift(1))
    return _realized_vol(lr, horizon).shift(-horizon)


@dataclass
class RiskBacktest:
    table: pd.DataFrame  # forecast, naive, realized, fwd_return
    horizon: int
    mae_model: float
    mae_naive: float
    r2: float
    coverage_1: float  # share of periods whose move stayed within ±1 forecast move
    coverage_2: float

    @property
    def improvement(self) -> float:
        """How much smaller the model's typical error is than the naive guess's."""
        return 1 - self.mae_model / self.mae_naive if self.mae_naive > 0 else float("nan")


@dataclass
class RiskForecast:
    vol: float  # forecast yearly volatility
    horizon: int
    price: float
    as_of: pd.Timestamp
    last_month_vol: float
    long_run_vol: float

    @property
    def typical_move(self) -> float:
        return self.vol * np.sqrt(self.horizon / TRADING_DAYS)

    def price_range(self, n_moves: float) -> tuple[float, float]:
        m = self.typical_move * n_moves
        return self.price * float(np.exp(-m)), self.price * float(np.exp(m))


def _prepare(close: pd.Series, market: pd.Series | None, horizon: int) -> pd.DataFrame:
    X = har_features(close, market)
    data = X.copy()
    data["target"] = np.log(forward_vol(close, horizon))
    data["naive"] = np.exp(X["log_vol_month"])
    data["fwd_return"] = np.log(close.shift(-horizon) / close)
    return data.dropna(subset=list(X.columns))


def backtest(
    close: pd.Series,
    market: pd.Series | None = None,
    horizon: int = 21,
    min_train: int = 756,
    retrain_every: int = 21,
) -> RiskBacktest:
    data = _prepare(close, market, horizon)
    cols = [c for c in data.columns if c not in ("target", "naive", "fwd_return")]
    n = len(data)
    if n < min_train + 126:
        raise ValueError("Not enough price history to test risk forecasts (need about 4 years).")
    pred = pd.Series(np.nan, index=data.index)
    for start in range(min_train, n, retrain_every):
        stop = min(start + retrain_every, n)
        train = data.iloc[: max(0, start - horizon + 1)].dropna(subset=["target"])
        model = LinearRegression().fit(train[cols], train["target"])
        pred.iloc[start:stop] = model.predict(data[cols].iloc[start:stop])

    out = data.iloc[min_train:].copy()
    out["forecast"] = np.exp(pred.iloc[min_train:])
    out["realized"] = np.exp(out["target"])
    scored = out.dropna(subset=["realized"])
    mae_model = float((scored["forecast"] - scored["realized"]).abs().mean())
    mae_naive = float((scored["naive"] - scored["realized"]).abs().mean())
    resid = scored["target"] - np.log(scored["forecast"])
    r2 = float(1 - (resid**2).sum() / ((scored["target"] - scored["target"].mean()) ** 2).sum())

    sample = scored.iloc[::horizon]  # non-overlapping periods
    move = sample["forecast"] * np.sqrt(horizon / TRADING_DAYS)
    cov1 = float((sample["fwd_return"].abs() <= move).mean())
    cov2 = float((sample["fwd_return"].abs() <= 2 * move).mean())
    table = out[["forecast", "naive", "realized", "fwd_return"]]
    return RiskBacktest(table, horizon, mae_model, mae_naive, r2, cov1, cov2)


def forecast(close: pd.Series, market: pd.Series | None = None, horizon: int = 21) -> RiskForecast:
    data = _prepare(close, market, horizon)
    cols = [c for c in data.columns if c not in ("target", "naive", "fwd_return")]
    train = data.dropna(subset=["target"])
    model = LinearRegression().fit(train[cols], train["target"])
    latest = data[cols].iloc[[-1]]
    vol = float(np.exp(model.predict(latest)[0]))
    lr = np.log(close / close.shift(1)).dropna()
    return RiskForecast(
        vol=vol,
        horizon=horizon,
        price=float(close.iloc[-1]),
        as_of=data.index[-1],
        last_month_vol=float(np.exp(data["log_vol_month"].iloc[-1])),
        long_run_vol=float(lr.std() * np.sqrt(TRADING_DAYS)),
    )
