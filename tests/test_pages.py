"""Run every page of the website end-to-end, offline, with synthetic prices.

Streamlit's AppTest executes the real scripts. We swap the internet data
functions for fake ones so the tests are fast and never flaky.
"""

from pathlib import Path

import pandas as pd
import pytest
from conftest import make_prices
from streamlit.testing.v1 import AppTest

from engine import data as D
from engine import predictor as P

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def fake_fetch_prices(symbols, years):
    n = min(int(years) * 252, 15 * 252)
    close = pd.DataFrame({s: make_prices(n=n, seed=sum(map(ord, s)), mu=0.0004) for s in symbols})
    volume = close * 0 + 1_000_000
    return D.PriceData(close=close, volume=volume, missing=[])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(D, "fetch_prices", fake_fetch_prices)
    monkeypatch.setattr(D, "fetch_risk_free_rate", lambda: 0.04)
    monkeypatch.setattr(D, "fetch_metadata", lambda s: {"currency": "USD", "exchangeName": "NMS", "instrumentType": "EQUITY"})
    real = P.run_tournament
    small = ["always_up", "coin_flip", "momentum_rule", "logistic", "gnb", "tree_3", "ensemble"]
    monkeypatch.setattr(P, "run_tournament", lambda *a, **k: real(*a, **{**k, "keys": small, "n_jobs": 1}))


def run_page(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    return at


def assert_clean(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_home():
    at = run_page()
    assert_clean(at)
    assert any("Understand any US stock" in m.value for m in at.markdown)


def test_simulator():
    at = run_page("app_pages/monte_carlo.py")
    assert_clean(at)
    assert [m.label for m in at.metric][:3] == ["Typical outcome", "Bad case", "Great case"]


def test_portfolio():
    at = run_page("app_pages/portfolio.py")
    assert_clean(at)
    assert any("mp-headline" in m.value for m in at.markdown)


def test_prediction_lab_direction_and_risk():
    at = run_page("app_pages/predictor.py")
    assert_clean(at)
    at.button(key="pr_run").click().run()
    assert_clean(at)
    labels = [m.label for m in at.metric]
    assert "Model was right" in labels and any(l.startswith("Forecast") for l in labels)


def test_algorithm_showdown():
    at = run_page("app_pages/predictor.py")
    at.button(key="pr_show_btn").click().run()
    assert_clean(at)
    labels = [m.label for m in at.metric]
    assert "Methods tested" in labels and "Real edge after luck check" in labels
    assert any("prediction methods" in m.value for m in at.markdown)


def test_learn():
    assert_clean(run_page("app_pages/learn.py"))
