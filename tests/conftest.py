import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_prices(n=1500, mu=0.0004, sigma=0.015, seed=0, start=100.0, phi=0.0):
    """Synthetic daily prices. `phi` adds day-to-day momentum (autocorrelation)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    r += mu
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.Series(start * np.exp(np.cumsum(r)), index=idx)


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def panel():
    cols = {}
    for i, (mu, sig) in enumerate([(0.0006, 0.02), (0.0003, 0.01), (0.0004, 0.015), (0.0001, 0.012)]):
        cols[f"S{i}"] = make_prices(n=1300, mu=mu, sigma=sig, seed=10 + i)
    return pd.DataFrame(cols)
