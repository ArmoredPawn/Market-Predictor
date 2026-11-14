"""Machine-learning direction forecasts: "Will this stock be higher in a week?"

Honest by design
----------------
Predicting stock prices is extremely hard: prices already reflect what
millions of investors know (the "efficient market hypothesis"). Most models
that look brilliant are fooling themselves through *look-ahead bias*: they were
accidentally trained on information from the future. This module prevents
that and then measures whether the model beats simple guessing:

1. Features at day t use only prices up to and including day t's close.
2. Walk-forward testing: the model is retrained every `retrain_every` days
   on the past only, and scored on the following days it has never seen.
3. Embargo: when predicting h days ahead, the last h training rows are dropped
   because their answers are not yet known on the prediction date.
4. Baseline: stocks rise on ~53% of days, so "always guess up" is right more
   than half the time. A model is only useful if it beats *that*.
5. Luck check: a one-sided binomial test asks how often pure guessing at the
   baseline rate would score this well. For h > 1 we use every h-th
   prediction only, because overlapping windows aren't independent.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from engine import TRADING_DAYS
from engine.metrics import cagr, max_drawdown, summarize
from engine.zoo import ZOO, has_probabilities

HORIZONS = {1: "the next trading day", 5: "1 week (5 trading days)", 21: "1 month (21 trading days)"}

FEATURE_INFO: dict[str, str] = {
    "ret_1": "Price change over the last day",
    "ret_5": "Price change over the last week",
    "ret_10": "Price change over the last 2 weeks",
    "ret_21": "Price change over the last month",
    "ret_63": "Price change over the last 3 months",
    "vol_10": "How jumpy the price was over the last 2 weeks",
    "vol_21": "How jumpy the price was over the last month",
    "vol_ratio": "Is it jumpier than usual lately?",
    "rsi_14": "RSI: has it been rising or falling more often lately?",
    "macd_hist": "MACD: is short-term momentum speeding up or slowing?",
    "dist_sma_20": "Price vs. its 20-day average",
    "dist_sma_50": "Price vs. its 50-day average",
    "dist_sma_200": "Price vs. its 200-day average (long-term trend)",
    "zscore_20": "How unusual today's price is vs. the last month",
    "dist_high_252": "How far below its 1-year high it is",
    "volume_trend": "Is trading activity higher than usual?",
    "mkt_ret_5": "Whole market's change over the last week",
    "mkt_ret_21": "Whole market's change over the last month",
    "mkt_dist_sma_200": "Is the whole market above its long-term trend?",
    "mkt_vol_21": "How jumpy the whole market has been",
    "rel_strength_63": "Did it beat the market over the last 3 months?",
}


# ---------------------------------------------------------------- features

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder, 1978), from 0 to 100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def build_features(close: pd.Series, volume: pd.Series | None = None, market: pd.Series | None = None) -> pd.DataFrame:
    """Every feature on day t uses data up to and including day t only."""
    c = close.astype(float)
    lr = np.log(c / c.shift(1))
    f = pd.DataFrame(index=c.index)
    for d in (1, 5, 10, 21, 63):
        f[f"ret_{d}"] = np.log(c / c.shift(d))
    f["vol_10"] = lr.rolling(10).std() * np.sqrt(TRADING_DAYS)
    f["vol_21"] = lr.rolling(21).std() * np.sqrt(TRADING_DAYS)
    f["vol_ratio"] = lr.rolling(10).std() / lr.rolling(63).std()
    f["rsi_14"] = rsi(c, 14) / 100.0
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    for w in (20, 50, 200):
        f[f"dist_sma_{w}"] = c / c.rolling(w).mean() - 1
    f["zscore_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    f["dist_high_252"] = c / c.rolling(252, min_periods=126).max() - 1
    if volume is not None and volume.notna().sum() > 100 and (volume > 0).mean() > 0.9:
        v = volume.reindex(c.index).replace(0, np.nan)
        f["volume_trend"] = np.log(v.rolling(5).mean() / v.rolling(63).mean())
    if market is not None:
        m = market.reindex(c.index).ffill()
        mlr = np.log(m / m.shift(1))
        f["mkt_ret_5"] = np.log(m / m.shift(5))
        f["mkt_ret_21"] = np.log(m / m.shift(21))
        f["mkt_dist_sma_200"] = m / m.rolling(200).mean() - 1
        f["mkt_vol_21"] = mlr.rolling(21).std() * np.sqrt(TRADING_DAYS)
        f["rel_strength_63"] = f["ret_63"] - np.log(m / m.shift(63))
    return f.replace([np.inf, -np.inf], np.nan)


def make_labels(close: pd.Series, horizon: int) -> pd.Series:
    """1 if the price is higher `horizon` days later, 0 if not, NaN if unknown yet."""
    future = close.shift(-horizon)
    y = (future > close).astype(float)
    return y.where(future.notna())


# ---------------------------------------------------------------- models

# Short list for the single-model view; the tournament uses every model in the zoo.
MODEL_NAMES = {k: ZOO[k].name + (" (recommended)" if k == "ensemble" else "") for k in
               ["ensemble", "logistic", "extra_trees", "hgb", "forest_shallow", "knn_100", "mlp_small", "svm_rbf"]}


def make_model(key: str):
    if key not in ZOO:
        raise ValueError(f"Unknown model: {key}")
    return ZOO[key].make()


def _prob_up(model, X: pd.DataFrame) -> np.ndarray:
    """Probability (or a 0–1 score) that the price goes up."""
    if has_probabilities(model):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        return proba[:, classes.index(1.0)] if 1.0 in classes else np.zeros(len(X))
    if len(model.classes_) < 2:
        return np.full(len(X), float(model.classes_[0]))
    score = np.asarray(model.decision_function(X), dtype=float)  # > 0 means "up"
    return 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))


# ---------------------------------------------------------------- evaluation

@dataclass
class WalkForwardResult:
    horizon: int
    model_name: str
    predictions: pd.DataFrame  # columns: prob, label, next_ret, position
    accuracy: float
    base_rate: float  # share of test days that actually went up
    naive_accuracy: float  # accuracy of always guessing the more common outcome
    auc: float
    n_tests: int
    n_independent: int
    p_value: float
    calibration: pd.DataFrame
    strategy_equity: pd.Series
    buyhold_equity: pd.Series
    time_in_market: float
    trades: int
    seconds: float = 0.0  # time spent training and predicting
    probabilistic: bool = True  # False when the model outputs scores, not probabilities

    @property
    def edge(self) -> float:
        return self.accuracy - self.naive_accuracy

    @property
    def verdict(self) -> str:
        """'edge', 'weak' or 'none' — the plain-English bottom line."""
        if self.p_value < 0.05 and self.edge > 0.01:
            return "edge"
        if self.edge > 0 and self.p_value < 0.25:
            return "weak"
        return "none"


def _calibration(prob: pd.Series, label: pd.Series) -> pd.DataFrame:
    edges = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0001]
    names = ["under 40%", "40–45%", "45–50%", "50–55%", "55–60%", "over 60%"]
    bins = pd.cut(prob, edges, labels=names, right=False)
    g = pd.DataFrame({"bin": bins, "prob": prob, "label": label}).groupby("bin", observed=False)
    return pd.DataFrame({"times": g.size(), "model_said": g["prob"].mean(), "actually_rose": g["label"].mean()})


def _prepare(features: pd.DataFrame, close: pd.Series, horizon: int, min_train: int) -> pd.DataFrame:
    data = features.dropna().copy()
    data["label"] = make_labels(close, horizon).reindex(data.index)
    data["next_ret"] = (close.shift(-1) / close - 1).reindex(data.index)
    if len(data) < min_train + 126:
        years = (min_train + 126 + 210) / TRADING_DAYS
        raise ValueError(f"Not enough price history for honest testing (need about {years:.0f} years).")
    return data


def walk_forward_probs(
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int,
    model_key: str,
    min_train: int = 756,
    retrain_every: int = 252,
    progress=None,
) -> tuple[pd.Series, bool]:
    """Out-of-sample probabilities for every test day, plus whether they're real probabilities."""
    data = _prepare(features, close, horizon, min_train)
    cols = list(features.columns)
    X_all = data[cols]
    n = len(data)
    probs = pd.Series(np.nan, index=data.index)
    starts = list(range(min_train, n, retrain_every))
    probabilistic = True
    for i, start in enumerate(starts):
        if progress is not None:
            progress(i / len(starts))
        stop = min(start + retrain_every, n)
        train = data.iloc[: max(0, start - horizon + 1)]  # embargo: answers known by `start`
        train = train[train["label"].notna()]
        model = make_model(model_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[cols], train["label"])
            probs.iloc[start:stop] = _prob_up(model, X_all.iloc[start:stop])
        probabilistic = has_probabilities(model)
    return probs.iloc[min_train:], probabilistic


def evaluate(
    probs: pd.Series,
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int,
    model_name: str,
    threshold: float = 0.5,
    cost: float = 0.0005,
    seconds: float = 0.0,
    probabilistic: bool = True,
) -> WalkForwardResult:
    """Score out-of-sample probabilities: accuracy, luck check, calibration, trading test."""
    data = _prepare(features, close, horizon, 0)
    test = data.loc[probs.index].copy()
    test["prob"] = probs
    scored = test[test["label"].notna()]
    pred_up = scored["prob"] > 0.5
    correct = pred_up == (scored["label"] == 1.0)
    accuracy = float(correct.mean())
    base_rate = float(scored["label"].mean())
    naive = max(base_rate, 1 - base_rate)
    try:
        auc = float(roc_auc_score(scored["label"], scored["prob"]))
    except ValueError:
        auc = float("nan")

    independent = correct.iloc[::horizon]
    k, m = int(independent.sum()), int(len(independent))
    p_value = float(binomtest(k, m, p=min(naive, 0.999), alternative="greater").pvalue) if m else 1.0

    # Trading rule: hold the stock tomorrow if the model says "up" with prob > threshold.
    trade = test[test["next_ret"].notna()].copy()
    trade["position"] = (trade["prob"] > threshold).astype(float)
    changes = trade["position"].diff().abs().fillna(trade["position"].iloc[0])
    strat_rets = trade["position"] * trade["next_ret"] - cost * changes
    # Equity is stamped on the day the return is *earned* (the next trading day).
    earn_dates = close.index[close.index.get_indexer(trade.index) + 1]
    start_date = trade.index[0]
    strategy_equity = pd.concat([pd.Series([1.0], index=[start_date]),
                                 pd.Series((1 + strat_rets).cumprod().values, index=earn_dates)])
    buyhold_equity = pd.concat([pd.Series([1.0], index=[start_date]),
                                pd.Series((1 + trade["next_ret"]).cumprod().values, index=earn_dates)])

    test["position"] = (test["prob"] > threshold).astype(float)
    return WalkForwardResult(
        horizon=horizon,
        model_name=model_name,
        predictions=test[["prob", "label", "next_ret", "position"]],
        accuracy=accuracy,
        base_rate=base_rate,
        naive_accuracy=naive,
        auc=auc,
        n_tests=int(len(scored)),
        n_independent=m,
        p_value=p_value,
        calibration=_calibration(scored["prob"], scored["label"]),
        strategy_equity=strategy_equity,
        buyhold_equity=buyhold_equity,
        time_in_market=float(trade["position"].mean()),
        trades=int(changes.sum()),
        seconds=seconds,
        probabilistic=probabilistic,
    )


def walk_forward(
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int,
    model_name: str = "ensemble",
    min_train: int = 756,
    retrain_every: int = 252,
    threshold: float = 0.5,
    cost: float = 0.0005,
    progress=None,
) -> WalkForwardResult:
    """Retrain on the past every `retrain_every` days; score on the unseen days after.

    `progress`, if given, is called with a number from 0 to 1 as training proceeds.
    """
    t0 = time.perf_counter()
    probs, probabilistic = walk_forward_probs(features, close, horizon, model_name, min_train, retrain_every, progress)
    return evaluate(probs, features, close, horizon, model_name, threshold, cost,
                    time.perf_counter() - t0, probabilistic)


# ---------------------------------------------------------------- tournament

CROWD_KEY = "crowd"
CROWD_NAME = "Everyone votes (average of all models)"


def _run_one(key, features, close, horizon, min_train, retrain_every):
    try:
        return key, walk_forward(features, close, horizon, key, min_train, retrain_every)
    except Exception as exc:  # one broken model must not sink the tournament
        return key, exc


def run_tournament(
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int,
    keys: list[str] | None = None,
    min_train: int = 756,
    retrain_every: int = 252,
    n_jobs: int = -1,
    on_result=None,
) -> dict[str, WalkForwardResult]:
    """Run the same honest walk-forward test for every model, in parallel.

    `on_result(key, result_or_exception, n_done, n_total)` is called as each model finishes.
    Adds a bonus entry, "Everyone votes", that averages all learning models' probabilities.
    """
    keys = list(keys or ZOO)
    teams = [k for k in keys if ZOO[k].members]
    trained = sorted([k for k in keys if not ZOO[k].members], key=lambda k: -ZOO[k].cost)  # slowest first
    trained += [m for t in teams for m in ZOO[t].members if m not in trained]
    total = len(trained) + len(teams)
    results: dict[str, WalkForwardResult] = {}
    jobs = (delayed(_run_one)(k, features, close, horizon, min_train, retrain_every) for k in trained)
    runner = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
    for i, (key, res) in enumerate(runner(jobs), start=1):
        if isinstance(res, WalkForwardResult):
            results[key] = res
        if on_result is not None:
            on_result(key, res, i, total)

    # A soft-voting team's prediction is exactly the average of its members' probabilities
    # (each member is trained on the same data), so teams cost nothing extra here.
    for j, key in enumerate(teams, start=len(trained) + 1):
        members = [results[m] for m in ZOO[key].members if m in results]
        if len(members) == len(ZOO[key].members):
            avg = pd.concat([r.predictions["prob"] for r in members], axis=1).mean(axis=1)
            results[key] = evaluate(avg, features, close, horizon, key, seconds=sum(r.seconds for r in members))
        if on_result is not None:
            on_result(key, results.get(key), j, total)

    learners = [results[k] for k in keys if k in results and not ZOO[k].baseline and not ZOO[k].members]
    if len(learners) >= 2:
        crowd = pd.concat([r.predictions["prob"] for r in learners], axis=1).mean(axis=1)
        results[CROWD_KEY] = evaluate(crowd, features, close, horizon, CROWD_KEY)
    return results


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm–Bonferroni: corrects p-values for having tried many models.

    With m models, the smallest p-value is multiplied by m, the next by m − 1,
    and so on (never letting an adjusted value drop below an earlier one).
    """
    p = p_values.dropna().sort_values()
    m = len(p)
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p.values))
    return pd.Series(adjusted, index=p.index).reindex(p_values.index)


def leaderboard(results: dict[str, WalkForwardResult]) -> pd.DataFrame:
    """One row per model, best first, with luck-corrected verdicts."""
    rows = []
    for key, r in results.items():
        spec = ZOO.get(key)
        rows.append({
            "key": key,
            "name": spec.name if spec else CROWD_NAME,
            "family": spec.family if spec else "Teams",
            "baseline": bool(spec.baseline) if spec else False,
            "accuracy": r.accuracy,
            "edge": r.edge,
            "auc": r.auc,
            "p_value": r.p_value,
            "growth": cagr(r.strategy_equity),
            "max_drawdown": max_drawdown(r.strategy_equity),
            "time_in_market": r.time_in_market,
            "seconds": r.seconds,
        })
    df = pd.DataFrame(rows).set_index("key")
    # "Always guess the usual direction" *is* the benchmark, so it isn't counted as a try.
    tried = df.index != "always_up"
    df["p_adjusted"] = np.nan
    df.loc[tried, "p_adjusted"] = holm_adjust(df.loc[tried, "p_value"])
    df["verdict"] = np.select(
        [(df["p_adjusted"] < 0.05) & (df["edge"] > 0.01), (df["p_value"] < 0.05) & (df["edge"] > 0)],
        ["Real edge", "Looks good, likely luck"],
        default="No edge",
    )
    if "always_up" in df.index:
        df.loc["always_up", "verdict"] = "Benchmark"
    df = df.sort_values(["accuracy", "auc"], ascending=False)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def luck_band(result: WalkForwardResult, z: float = 1.96) -> tuple[float, float]:
    """Accuracy range a no-skill model matching the benchmark lands in ~95% of the time."""
    p, n = result.naive_accuracy, max(result.n_independent, 1)
    half = z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


@dataclass
class Forecast:
    prob_up: float
    as_of: pd.Timestamp
    horizon: int
    importance: pd.Series  # feature -> importance (higher = model relies on it more)


def latest_forecast(features: pd.DataFrame, close: pd.Series, horizon: int, model_name: str = "ensemble") -> Forecast:
    """Train on all history with known answers, then forecast from the latest day."""
    labels = make_labels(close, horizon)
    data = features.dropna().copy()
    data["label"] = labels.reindex(data.index)
    train = data[data["label"].notna()]
    model = make_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[features.columns], train["label"])
    latest = data[features.columns].iloc[[-1]]
    prob = float(_prob_up(model, latest)[0])

    recent = train.iloc[-504:]
    imp = permutation_importance(
        model, recent[features.columns], recent["label"], scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=1
    )
    importance = pd.Series(imp.importances_mean, index=features.columns).clip(lower=0).sort_values(ascending=False)
    return Forecast(prob_up=prob, as_of=data.index[-1], horizon=horizon, importance=importance)


def strategy_summaries(result: WalkForwardResult) -> tuple:
    return summarize(result.strategy_equity), summarize(result.buyhold_equity)
