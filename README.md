# Market Predictor

**Honest, math-powered tools for US stocks, explained in plain English.**

Market Predictor is a free website with three tools that answer real questions people have about
investing, using real math, from Nobel Prize-winning portfolio theory to machine learning. Every model is
tested on data it has never seen, and the site says so plainly when a model doesn't work.

| Tool | The question it answers | The math |
|---|---|---|
| **Future Simulator** | "If I invest $10,000, what could it be worth in 10 years?" | Monte Carlo simulation (block bootstrap and geometric Brownian motion), percentiles, Value at Risk, expected shortfall, drawdowns |
| **Portfolio Builder** | "How should I split my money between a few stocks?" | Markowitz mean-variance optimization, Sharpe ratio, efficient frontier, risk parity, Ledoit–Wolf covariance shrinkage, walk-forward backtest |
| **Prediction Lab** | "Can AI predict if this stock goes up, and how bumpy will it be?" | A 45-method showdown (rules of thumb, linear models, naive Bayes, nearest neighbors, SVMs and kernel methods, trees and forests, boosting, neural networks, ensembles), all with walk-forward validation and embargo, binomial tests with Holm–Bonferroni correction for trying many models, AUC, calibration; HAR volatility model |

## What makes it different

- **No look-ahead bias.** Features, labels and backtests only ever use past data, and the test suite proves
  it by rewriting future prices and checking that past decisions don't change.
- **Always compared with a baseline.** The direction model has to beat "always guess up", and the risk model
  has to beat "next month will be like last month". A binomial "luck check" asks whether the edge is real.
- **Honest results.** For most stocks the direction model does *not* beat guessing, and the site says so.
  Volatility, on the other hand, really is predictable: the HAR model cuts forecast error by roughly 15–20%.
- **Plain English first.** Every result starts with a one-sentence answer; the math is one click away.
- **US stocks only.** About 3,900 US-based companies (NASDAQ, NYSE, NYSE American) plus popular US stock
  ETFs. Foreign companies listed in the US, like Alibaba or Toyota, are filtered out.
- **Accessible charts.** Colorblind-checked palette, light and dark modes, works on phones, and a
  "See the numbers" table under the charts.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501.

## Deploy it for free

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Pick this repository, branch `main`, main file `streamlit_app.py`, and deploy.

The app redeploys automatically on every push.

## Project layout

```
streamlit_app.py        entry point and navigation
app_pages/              one file per page (home, simulator, portfolio, predictor, learn)
ui/                     shared widgets, formatting and chart styling
engine/                 the math, plain Python with no Streamlit (tested separately)
  universe.py           which tickers are allowed (US only) and why
  data.py               price downloads (Yahoo Finance via yfinance)
  metrics.py            returns, volatility, Sharpe ratio, drawdowns
  montecarlo.py         simulation engine
  portfolio.py          optimization, efficient frontier, walk-forward backtest
  predictor.py          ML features, walk-forward testing, the tournament and luck correction
  zoo.py                the 45 prediction methods
  risk.py               HAR volatility forecasting
data/                   US ticker list (refresh with scripts/build_ticker_list.py)
tests/                  engine tests + offline end-to-end page tests
```

## Tests

```bash
pip install pytest
pytest
```

All tests run offline using synthetic prices.

## Data sources

- Prices: Yahoo Finance via the open-source [`yfinance`](https://github.com/ranaroussi/yfinance) library
  (adjusted for splits and dividends; may be delayed).
- Ticker list: NASDAQ, NYSE and AMEX listings via
  [US-Stock-Symbols](https://github.com/rreichel3/US-Stock-Symbols), filtered to US-based companies.
- Risk-free rate: the 13-week US Treasury bill yield (^IRX).

## Disclaimer

This is an educational project, not financial advice. Results come from past prices and simple
assumptions; the past does not guarantee the future.
