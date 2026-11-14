"""Tests for the math engine. They use synthetic prices, so no internet is needed."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from engine import metrics, montecarlo, portfolio, predictor, risk, universe

# ------------------------------------------------------------------ universe


@pytest.mark.parametrize(
    "raw, expected",
    [("aapl", "AAPL"), (" brk.b ", "BRK-B"), ("BRK/B", "BRK-B"), ("$spy", "SPY"), ("AAPL · Apple Inc.", "AAPL")],
)
def test_normalize_ticker(raw, expected):
    assert universe.normalize_ticker(raw) == expected


def test_us_company_accepted():
    check = universe.check_ticker("AAPL")
    assert check.ok and not check.needs_online_check and "Apple" in check.name


def test_us_etf_accepted():
    assert universe.check_ticker("SPY").ok


def test_foreign_company_rejected_with_reason():
    check = universe.check_ticker("BABA")
    assert not check.ok and "China" in check.reason


def test_foreign_exchange_suffix_rejected():
    assert not universe.check_ticker("SHOP.TO").ok


def test_unknown_ticker_needs_online_check():
    check = universe.check_ticker("ZZZQ")
    assert check.ok and check.needs_online_check


def test_metadata_check():
    us = {"currency": "USD", "exchangeName": "NMS", "instrumentType": "EQUITY", "longName": "New Co"}
    assert universe.check_metadata("NEWC", us).ok
    assert not universe.check_metadata("X", {**us, "currency": "CAD", "exchangeName": "TOR"}).ok
    assert not universe.check_metadata("X", {**us, "instrumentType": "CRYPTOCURRENCY"}).ok
    assert not universe.check_metadata("X", None).ok


# ------------------------------------------------------------------ metrics


def test_cagr_and_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(np.linspace(100, 110, 253), index=idx)
    assert metrics.cagr(eq) == pytest.approx(0.10, abs=1e-9)
    dd = pd.Series([100, 120, 60, 90, 130], index=idx[:5], dtype=float)
    assert metrics.max_drawdown(dd) == pytest.approx(-0.5)


# ------------------------------------------------------------------ monte carlo


def test_simulation_median_matches_growth_assumption(prices):
    lr = metrics.log_returns(prices)
    for model in ("gbm", "bootstrap"):
        cfg = montecarlo.SimulationConfig(10_000, years=5, n_paths=8000, model=model, annual_growth=0.07, seed=1)
        res = montecarlo.simulate(lr, cfg)
        assert res.median_annual_return() == pytest.approx(0.07, abs=0.012), model


def test_simulation_contributions_and_shapes(prices):
    lr = metrics.log_returns(prices)
    cfg = montecarlo.SimulationConfig(1000, years=2, n_paths=500, monthly_contribution=100, seed=3)
    res = montecarlo.simulate(lr, cfg)
    assert res.total_invested == pytest.approx(1000 + 100 * (504 // 21))
    assert res.percentiles.shape == (505, len(montecarlo.PERCENTILES))
    assert np.all(np.diff(res.percentiles, axis=1) >= 0)  # 5th <= 25th <= ... <= 95th
    assert (res.max_drawdowns <= 0).all() and 0 <= res.prob_loss() <= 1
    assert res.expected_shortfall() >= res.value_at_risk() >= 0


def test_simulation_is_reproducible(prices):
    lr = metrics.log_returns(prices)
    cfg = montecarlo.SimulationConfig(100, years=1, n_paths=200, seed=42)
    a, b = montecarlo.simulate(lr, cfg), montecarlo.simulate(lr, cfg)
    assert np.array_equal(a.final_values, b.final_values)


def test_simulation_rejects_bad_input(prices):
    lr = metrics.log_returns(prices)
    with pytest.raises(ValueError):
        montecarlo.simulate(lr, montecarlo.SimulationConfig(0, years=1))
    with pytest.raises(ValueError):
        montecarlo.simulate(lr.iloc[:10], montecarlo.SimulationConfig(100, years=1))


# ------------------------------------------------------------------ portfolio


@pytest.mark.parametrize("objective", portfolio.OBJECTIVES)
def test_weights_are_valid(panel, objective):
    est = portfolio.estimate(panel, risk_free=0.03)
    w = portfolio.optimize(est, objective, max_weight=0.4)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all() and (w <= 0.4 + 1e-6).all()


def test_min_vol_beats_every_other_mix(panel):
    est = portfolio.estimate(panel, risk_free=0.03)
    v_min = portfolio.portfolio_vol(portfolio.optimize(est, "min_vol"), est)
    for obj in ("equal", "max_sharpe", "risk_parity"):
        assert v_min <= portfolio.portfolio_vol(portfolio.optimize(est, obj), est) + 1e-6


def test_max_sharpe_beats_frontier_and_random(panel):
    est = portfolio.estimate(panel, risk_free=0.03)
    best = portfolio.portfolio_sharpe(portfolio.optimize(est, "max_sharpe"), est)
    assert best >= portfolio.efficient_frontier(est)["sharpe"].max() - 1e-3
    assert best >= portfolio.random_portfolios(est, 2000)["sharpe"].max() - 1e-3


def test_risk_parity_equalizes_risk(panel):
    est = portfolio.estimate(panel, risk_free=0.03)
    rc = portfolio.risk_contributions(portfolio.optimize(est, "risk_parity"), est.cov.values)
    assert np.allclose(rc, 0.25, atol=0.01)


def test_cap_below_one_over_n_is_lifted(panel):
    est = portfolio.estimate(panel, risk_free=0.03)
    w = portfolio.optimize(est, "max_sharpe", max_weight=0.1)  # 4 stocks * 10% < 100%
    assert w.sum() == pytest.approx(1.0)


def test_backtest_does_not_peek_at_the_future(panel):
    eq1, w1 = portfolio.backtest(panel, "max_sharpe", 0.03, lookback=504, rebalance_every=63)
    future = panel.copy()
    future.iloc[700:] *= np.linspace(1, 3, len(panel) - 700)[:, None] ** np.arange(1, 5)  # rewrite the future
    eq2, w2 = portfolio.backtest(future, "max_sharpe", 0.03, lookback=504, rebalance_every=63)
    early = w1.index[w1.index < panel.index[700]]
    pd.testing.assert_frame_equal(w1.loc[early], w2.loc[early])
    assert eq1.loc[: panel.index[699]].equals(eq2.loc[: panel.index[699]])


# ------------------------------------------------------------------ predictor


def test_features_do_not_peek_at_the_future(prices):
    market = make_prices(seed=99)
    f1 = predictor.build_features(prices, None, market)
    changed = prices.copy()
    changed.iloc[1000:] *= 1.5
    f2 = predictor.build_features(changed, None, market)
    pd.testing.assert_frame_equal(f1.iloc[:1000], f2.iloc[:1000])


def test_labels():
    s = pd.Series([1.0, 2.0, 1.5, 3.0], index=pd.bdate_range("2020-01-01", periods=4))
    y = predictor.make_labels(s, 1)
    assert y.iloc[:3].tolist() == [1.0, 0.0, 1.0] and np.isnan(y.iloc[3])


def test_walk_forward_finds_no_edge_in_random_walk():
    prices = make_prices(n=1800, seed=5)
    res = predictor.walk_forward(predictor.build_features(prices), prices, 1, "logistic", min_train=504)
    assert res.verdict != "edge"
    assert abs(res.edge) < 0.05


def test_walk_forward_finds_real_edge():
    prices = make_prices(n=1800, seed=6, mu=0.0, phi=0.35)  # strong, real momentum
    res = predictor.walk_forward(predictor.build_features(prices), prices, 1, "logistic", min_train=504)
    assert res.verdict == "edge" and res.accuracy > 0.58


def test_latest_forecast_runs(prices):
    fc = predictor.latest_forecast(predictor.build_features(prices), prices, 5, "logistic")
    assert 0 <= fc.prob_up <= 1 and fc.as_of == prices.index[-1]


# ------------------------------------------------------------------ risk


def _clustered_prices(n=2000, seed=0):
    """Prices with GARCH-style volatility clustering (calm and stormy spells)."""
    rng = np.random.default_rng(seed)
    var, r = 0.0001, np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = 0.000004 + 0.08 * r[t] ** 2 + 0.9 * var
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-01", periods=n))


def test_risk_model_beats_naive_on_clustered_volatility():
    bt = risk.backtest(_clustered_prices())
    assert bt.mae_model < bt.mae_naive
    assert 0.5 < bt.coverage_1 < 0.85 and bt.coverage_2 > 0.85


def test_risk_forecast_range_is_ordered():
    fc = risk.forecast(_clustered_prices())
    lo1, hi1 = fc.price_range(1)
    lo2, hi2 = fc.price_range(2)
    assert lo2 < lo1 < fc.price < hi1 < hi2 and fc.vol > 0


# ------------------------------------------------------------------ model zoo & tournament

from engine.zoo import FAMILIES, LEARNING_MODELS, ZOO  # noqa: E402


def test_zoo_is_big_and_well_formed():
    assert len(LEARNING_MODELS) >= 35
    assert {s.family for s in ZOO.values()} <= set(FAMILIES)
    for spec in ZOO.values():
        assert spec.name and spec.blurb.endswith((".", "!", "?", "'.", ".'"))
        for m in spec.members:
            assert m in ZOO


@pytest.mark.parametrize("key", list(ZOO))
def test_every_model_trains_and_predicts(key):
    prices = make_prices(n=900, seed=3)
    f = predictor.build_features(prices, None, make_prices(n=900, seed=4)).dropna()
    y = predictor.make_labels(prices, 5).reindex(f.index)
    train, test = f.iloc[:500][y.iloc[:500].notna()], f.iloc[500:520]
    model = predictor.make_model(key)
    model.fit(train, y.loc[train.index])
    p = predictor._prob_up(model, test)
    assert p.shape == (20,) and np.all((p >= 0) & (p <= 1))


def test_holm_adjust_matches_hand_calculation():
    adj = predictor.holm_adjust(pd.Series({"a": 0.01, "b": 0.04, "c": 0.03}))
    assert adj.to_dict() == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_tournament_teams_and_crowd():
    prices = make_prices(n=1500, seed=8)
    f = predictor.build_features(prices)
    keys = ["always_up", "momentum_rule", "logistic", "gnb", "ensemble"]
    res = predictor.run_tournament(f, prices, 5, keys=keys, min_train=504, n_jobs=1)
    assert set(keys) | {"extra_trees", "hgb", predictor.CROWD_KEY} <= set(res)
    members = pd.concat([res[m].predictions["prob"] for m in ("logistic", "extra_trees", "hgb")], axis=1).mean(axis=1)
    pd.testing.assert_series_equal(res["ensemble"].predictions["prob"], members, check_names=False)
    lb = predictor.leaderboard(res)
    assert lb.loc["always_up", "verdict"] == "Benchmark"
    assert list(lb["rank"]) == list(range(1, len(lb) + 1))
    assert lb["accuracy"].is_monotonic_decreasing


def test_lucky_winner_is_not_called_real():
    # 45 pure-noise "models": the best raw p-value looks good, but Holm should reject it.
    rng = np.random.default_rng(0)
    p = pd.Series(rng.uniform(0, 1, 45))
    p.iloc[0] = 0.02
    assert predictor.holm_adjust(p).iloc[0] > 0.05
