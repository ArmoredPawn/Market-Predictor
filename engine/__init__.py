"""Market Predictor's math engine.

Everything in this package is plain Python (no Streamlit), so it can be tested
and reused on its own. The website in `app_pages/` is a thin layer on top.

Modules
-------
universe    which tickers are allowed (US companies and US stock ETFs only)
data        downloading price history
metrics     return, risk and drawdown measures shared by every tool
montecarlo  simulating thousands of possible futures for an investment
portfolio   mean-variance portfolio optimization and honest backtesting
predictor   machine-learning direction forecasts with walk-forward testing
"""

TRADING_DAYS = 252
