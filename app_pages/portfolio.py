import numpy as np
import pandas as pd
import streamlit as st

from engine import TRADING_DAYS, metrics
from engine import portfolio as pf
from engine.data import BENCHMARK, DataUnavailable
from engine.universe import display_name
from ui import charts
from ui import components as ui

STRATEGIES = {
    "max_sharpe": "Best return for the risk",
    "min_vol": "Smoothest ride",
    "risk_parity": "Balanced risk",
    "equal": "Equal split",
}
STRATEGY_HELP = {
    "max_sharpe": "Aims for the most return per unit of risk (the highest Sharpe ratio).",
    "min_vol": "Aims for the smallest ups and downs possible with these stocks.",
    "risk_parity": "Each stock contributes the same amount of risk, so no single stock dominates.",
    "equal": "The same amount in each stock. Simple, and surprisingly hard to beat.",
}
DEFAULTS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "COST", "KO"]


@st.cache_data(show_spinner="Testing the strategy on past data…", max_entries=64)
def run_backtests(symbols: tuple[str, ...], years: int, objective: str, cap: float, cautious: bool, rf: float):
    prices = ui.get_prices(symbols + (BENCHMARK,), years).close
    aligned = prices[list(symbols)].dropna()
    out = {}
    for obj in dict.fromkeys([objective, "equal"]):
        out[obj], _ = pf.backtest(aligned, obj, rf, max_weight=cap, cautious=cautious)
    bench = prices[BENCHMARK].reindex(out[objective].index).ffill()
    return out, bench / bench.iloc[0]


ui.page_header(":material/pie_chart:", "Portfolio Builder",
               "How should you split your money between several stocks? We use Nobel Prize-winning portfolio "
               "math to find mixes that balance growth against risk, then test them honestly on the past.")

with st.container(border=True):
    symbols = ui.ticker_multiselect("Your stocks (pick 2 to 15)", key="pf_tickers", defaults=DEFAULTS)
    c1, c2, c3 = st.columns([1.1, 2.6, 1.3])
    with c1:
        amount = st.number_input("Amount to invest ($)", min_value=100, max_value=100_000_000, value=10_000,
                                 step=1_000, key="pf_amount")
    with c2:
        objective = st.segmented_control(
            "What matters most to you?", options=list(STRATEGIES), format_func=STRATEGIES.get,
            default="max_sharpe", required=True, key="pf_objective",
            help=" · ".join(f"{STRATEGIES[k]}: {v}" for k, v in STRATEGY_HELP.items()),
        )
    with c3:
        cap_pct = st.slider("Max in any one stock", 10, 100, 35, step=5, format="%d%%", key="pf_cap",
                            help="Keeps you diversified. Lower = more spread out.")
    with st.expander("More options", icon=":material/tune:"):
        a1, a2, a3 = st.columns(3)
        with a1:
            years = st.select_slider("History to learn from", options=[3, 5, 10, 15], value=10,
                                     format_func=lambda y: f"{y} years", key="pf_years",
                                     help="Longer history = more stable estimates and a longer honest test.")
        with a2:
            cautious = st.toggle("Cautious return estimates (recommended)", value=True, key="pf_cautious",
                                 help="Blends each stock's past return halfway toward the group's average. Past "
                                      "returns are a noisy guide to the future, and this stops the math from going "
                                      "all-in on last decade's winner.")
        with a3:
            live_rf = ui.get_risk_free()
            rf_pct = st.number_input("Safe interest rate (%/yr)", min_value=0.0, max_value=15.0,
                                     value=round(live_rf * 100, 2), step=0.25, key="pf_rf",
                                     help="What you'd earn risk-free in US Treasury bills. "
                                          f"Today's 13-week T-bill yield is {live_rf * 100:.2f}%.")

if len(symbols) < 2:
    st.info("Pick at least two stocks to build a portfolio.", icon=":material/add_circle:")
    ui.disclaimer()
    st.stop()

cap = cap_pct / 100
rf = rf_pct / 100
try:
    pdata = ui.get_prices(tuple(symbols) + (BENCHMARK,), years)
except DataUnavailable as exc:
    ui.data_error(exc)
missing = [s for s in symbols if s not in pdata.close.columns]
if missing:
    st.warning("No price data for: " + ", ".join(missing) + ". Leaving them out.", icon=":material/warning:")
symbols = [s for s in symbols if s in pdata.close.columns]
if len(symbols) < 2:
    st.stop()

aligned = pdata.close[symbols].dropna()
first_dates = pdata.close[symbols].apply(lambda s: s.first_valid_index())
youngest = first_dates.idxmax()
if first_dates.max() > pdata.close.index[0] + pd.Timedelta(days=30):
    st.caption(f":material/info: {youngest} only has prices from {first_dates.max():%B %Y}, so everything uses data "
               f"from then on ({len(aligned) / TRADING_DAYS:.1f} years).")
if len(aligned) < TRADING_DAYS:
    st.error("These stocks share less than a year of price history, which is too little to work with.",
             icon=":material/history:")
    st.stop()

est = pf.estimate(aligned, rf, cautious=cautious)
weights = {k: pf.optimize(est, k, cap) for k in STRATEGIES}
w = weights[objective]
w_s = pd.Series(w, index=symbols)
last_price = aligned.iloc[-1]

# ------------------------------------------------------------------ headline + allocation
held = w_s[w_s >= 0.005].sort_values(ascending=False)
top = ", ".join(f"<b>{ui.money(amount * v)}</b> in {k}" for k, v in held.head(3).items())
dropped = [s for s in symbols if w_s[s] < 0.005]
exp_ret, exp_vol = pf.portfolio_return(w, est), pf.portfolio_vol(w, est)
ui.headline(
    f"With the <b>{STRATEGIES[objective]}</b> approach, put {top}"
    f"{' and the rest across ' + str(len(held) - 3) + ' more' if len(held) > 3 else ''}. "
    f"Based on the past, that mix has an estimated return of about <b>{ui.pct(exp_ret, 0)} a year</b>, "
    f"with typical yearly ups and downs of <b>±{ui.pct(exp_vol, 0)}</b>."
)
if dropped:
    st.caption(f":material/info: This approach leaves out {', '.join(dropped)}: they didn't improve the mix enough.")

left, right = st.columns([1.15, 1], gap="large")
with left:
    ui.section("Your suggested split")
    labels = {s: f"{ui.money(amount * v)} · {ui.pct(v, 0)}" for s, v in w_s.items()}
    charts.show(charts.hbar(w_s[w_s >= 0.005], labels=labels, x_format=".0%"), key="pf_alloc")
with right:
    ui.section("Compare the four approaches", "Estimated from past prices. Higher \"return per risk\" is better.")
    rows = []
    for k, wk in weights.items():
        rows.append({
            "Approach": STRATEGIES[k] + ("  ← yours" if k == objective else ""),
            "Return / yr": pf.portfolio_return(wk, est),
            "Ups & downs / yr": pf.portfolio_vol(wk, est),
            "Return per risk": pf.portfolio_sharpe(wk, est),
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("Approach"), key="pf_compare",
        column_config={
            "Return / yr": st.column_config.NumberColumn(format="percent"),
            "Ups & downs / yr": st.column_config.NumberColumn(format="percent"),
            "Return per risk": st.column_config.NumberColumn(format="%.2f",
                                                             help="Sharpe ratio: (return − safe rate) ÷ ups & downs"),
        },
    )
    st.caption(STRATEGY_HELP[objective])

shares = pd.DataFrame({
    "Company": [display_name(s) for s in symbols],
    "Share of money": w_s.values,
    "Dollars": amount * w_s.values,
    "Latest price": last_price.values,
    "≈ Shares to buy": np.floor(amount * w_s.values / last_price.values),
}, index=symbols)
ui.see_numbers(shares.sort_values("Share of money", ascending=False), key="pf_shares", label="See the shopping list",
               column_config={"Share of money": st.column_config.NumberColumn(format="percent"),
                              "Dollars": st.column_config.NumberColumn(format="dollar"),
                              "Latest price": st.column_config.NumberColumn(format="dollar")})

# ------------------------------------------------------------------ frontier
ui.section("The efficient frontier: every possible mix",
           "Each gray dot is a random mix of your stocks. The blue curve is the best possible return for each "
           "level of risk; nothing above it is possible. Up and to the left is better.")
frontier = pf.efficient_frontier(est, cap)
cloud = pf.random_portfolios(est, 2500)
stocks_pts = pd.DataFrame({"vol": est.vols, "ret": est.mu})
strat_pts = pd.DataFrame(
    {"vol": [pf.portfolio_vol(weights[k], est) for k in STRATEGIES], "ret": [pf.portfolio_return(weights[k], est) for k in STRATEGIES]},
    index=[STRATEGIES[k] for k in STRATEGIES],
)
charts.show(charts.frontier_chart(cloud, frontier, stocks_pts, strat_pts, STRATEGIES[objective]), key="pf_frontier")
st.caption(f"Notice that most mixes have less risk than the riskiest single stock, and some have less risk than "
           f"*any* single stock. That's diversification at work. (With a {cap_pct}% cap per stock, some mixes "
           "of the gray cloud aren't allowed.)")

# ------------------------------------------------------------------ correlation + risk contribution
left, right = st.columns(2, gap="large")
corr = est.corr()
with left:
    ui.section("Which stocks move together?",
               "1.0 means two stocks always move the same way; 0 means unrelated. Low numbers diversify best.")
    charts.show(charts.correlation_heatmap(corr), key="pf_corr")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
    if len(pairs):
        lo_pair, hi_pair = pairs.idxmin(), pairs.idxmax()
        st.markdown(f"- **Best diversifiers:** {lo_pair[0]} & {lo_pair[1]} ({pairs.min():.2f})\n"
                    f"- **Most alike:** {hi_pair[0]} & {hi_pair[1]} ({pairs.max():.2f})")
with right:
    ui.section("Where your risk comes from",
               "A stock can hold a small share of your money but cause a big share of the ups and downs.")
    rc = pf.risk_contributions(w, est.cov.values)
    rc_df = pd.DataFrame({"Share of money": w, "Share of risk": rc}, index=symbols)
    rc_df = rc_df[(rc_df["Share of money"] >= 0.005)].sort_values("Share of money", ascending=False)
    charts.show(charts.grouped_hbar(rc_df), key="pf_rc")

# ------------------------------------------------------------------ honest backtest
ui.section("Honest test: how would this approach really have done?",
           "We replay history. Every 3 months, the approach only looks at the 2 years before that date, picks "
           "its mix, and holds it. It never sees the future. We charge 0.1% trading costs, and compare against "
           "an equal split and the S&P 500.")
lookback = 504
if len(aligned) <= lookback + 126:
    st.info("Not enough shared history for an honest test. Choose more history under **More options**, or "
            "stocks that have traded for longer.", icon=":material/history:")
else:
    tests, bench = run_backtests(tuple(symbols), years, objective, cap, cautious, rf)
    series = {f"{STRATEGIES[objective]} (yours)": tests[objective] * amount}
    if objective != "equal":
        series["Equal split"] = tests["equal"] * amount
    series["S&P 500 (SPY)"] = bench * amount
    charts.show(charts.line_chart(series, f"Value of {ui.money(amount)}"), key="pf_backtest")

    rows = {}
    for name, s in series.items():
        summ = metrics.summarize(s, rf)
        rows[name] = {"Ended with": s.iloc[-1], "Growth / yr": summ.cagr, "Ups & downs / yr": summ.volatility,
                      "Worst drop": summ.max_drawdown, "Return per risk": summ.sharpe}
    table = pd.DataFrame(rows).T
    st.dataframe(pd.DataFrame({
        "Ended with": table["Ended with"].map(ui.money),
        "Growth / yr": table["Growth / yr"].map(ui.pct),
        "Ups & downs / yr": table["Ups & downs / yr"].map(ui.pct),
        "Worst drop": table["Worst drop"].map(ui.pct),
        "Return per risk": table["Return per risk"].map(lambda v: f"{v:.2f}"),
    }), key="pf_bt_table")
    mine = table.iloc[0]
    spy = table.loc["S&P 500 (SPY)"]
    start_year = tests[objective].index[0].year
    beat_spy = mine["Growth / yr"] > spy["Growth / yr"]
    smoother = mine["Ups & downs / yr"] < spy["Ups & downs / yr"]
    verdict = (
        f"Since {start_year}, this approach grew **{ui.pct(mine['Growth / yr'])} a year** versus "
        f"**{ui.pct(spy['Growth / yr'])}** for the S&P 500, with "
        f"{'smaller' if smoother else 'bigger'} ups and downs ({ui.pct(mine['Ups & downs / yr'], 0)} vs. "
        f"{ui.pct(spy['Ups & downs / yr'], 0)}). Its worst drop was {ui.pct(-mine['Worst drop'], 0)}."
    )
    (st.success if beat_spy else st.info)(verdict, icon=":material/fact_check:")
    if objective != "equal" and table.loc["Equal split", "Growth / yr"] > mine["Growth / yr"]:
        st.markdown(
            f":material/lightbulb: **Interesting:** the simple equal split did better still "
            f"({ui.pct(table.loc['Equal split', 'Growth / yr'])} a year). That's a famous finding: in a 2009 study, "
            "DeMiguel, Garlappi and Uppal showed that splitting money equally often beats fancy optimization "
            "out-of-sample, because the optimizer's inputs are estimated from noisy data."
        )
    st.caption("A good past result doesn't guarantee a good future one, especially when you picked the stocks "
               "knowing how they did. Still, this test is far more honest than the estimates above, which are "
               "computed with hindsight.")

# ------------------------------------------------------------------ learn
with ui.learn_more():
    st.markdown("**Modern Portfolio Theory** (Harry Markowitz, 1952; Nobel Prize 1990). For a mix with weights "
                "*w* (the fraction of money in each stock):")
    st.latex(r"\text{return} = w^\top \mu \qquad \text{risk} = \sqrt{w^\top \Sigma\, w}")
    st.markdown(
        """
Here μ holds each stock's expected yearly return and Σ is the **covariance matrix**, which records how
every pair of stocks moves together. Because stocks don't move in lock-step, the risk of a mix is less
than the average risk of its parts. That's **diversification**, the only "free lunch" in investing.

**The approaches.**
- *Best return for the risk* maximizes the **Sharpe ratio** (William Sharpe, Nobel Prize 1990):
"""
    )
    st.latex(r"\max_w \ \frac{w^\top\mu - r_f}{\sqrt{w^\top\Sigma w}} \quad \text{s.t. } \textstyle\sum w_i = 1,\ 0 \le w_i \le \text{cap}")
    st.markdown(
        """
- *Smoothest ride* minimizes risk.
- *Balanced risk* (risk parity) makes each stock's **risk contribution** $w_i(\\Sigma w)_i / w^\\top\\Sigma w$ equal.
- *Equal split* puts 1/N in each stock.

The optimizer is SciPy's SLSQP (sequential quadratic programming) with several starting points.

**Making the math trustworthy.** Plain historical averages are very noisy, so the textbook method tends to
bet everything on whatever did best recently. We (1) shrink the covariance matrix with the **Ledoit–Wolf**
estimator, (2) optionally pull each return estimate halfway toward the group average, (3) cap any single
holding, and (4) judge everything with the **walk-forward test**, which only ever uses past data.

**Limits.** The estimates assume the future resembles the past. Correlations tend to rise in crashes,
just when you need diversification most.
"""
    )

ui.disclaimer()
