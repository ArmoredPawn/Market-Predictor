import numpy as np
import pandas as pd
import streamlit as st

from engine import TRADING_DAYS, metrics
from engine import montecarlo as mc
from engine.data import DataUnavailable
from engine.universe import display_name
from ui import charts
from ui import components as ui

GROWTH_CHOICES = {
    "cautious": "Cautious 5%",
    "moderate": "Moderate 8%",
    "history": "Its past",
    "custom": "Custom",
}


@st.cache_data(show_spinner="Simulating thousands of futures…", max_entries=64)
def run_simulation(symbol: str, history_years: int, start: float, monthly: float, years: int,
                   growth: float | None, model: str, n_paths: int, seed: int):
    prices = ui.get_prices((symbol,), history_years).close[symbol].dropna()
    cfg = mc.SimulationConfig(start_value=start, years=years, n_paths=n_paths, monthly_contribution=monthly,
                              model=model, annual_growth=growth, seed=seed)
    return mc.simulate(metrics.log_returns(prices), cfg), prices


ui.page_header(":material/stacked_line_chart:", "Future Simulator",
               "What could an investment be worth in the future? We simulate thousands of possible futures "
               "based on how the stock has really behaved, so you can see the good, the typical and the bad.")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([2.4, 1.2, 1.2, 1.4])
    with c1:
        symbol = ui.ticker_select("Stock or ETF", key="mc_ticker", default=ui.default_ticker("SPY"))
    with c2:
        start = st.number_input("Invest today ($)", min_value=0, max_value=100_000_000, value=10_000, step=1_000,
                                key="mc_start")
    with c3:
        monthly = st.number_input("Add every month ($)", min_value=0, max_value=1_000_000, value=0, step=100,
                                  key="mc_monthly", help="Regular contributions, like saving part of each paycheck.")
    with c4:
        years = st.slider("Years from now", 1, 30, 10, key="mc_years")

    g1, g2 = st.columns([3, 1.4], vertical_alignment="bottom")
    with g1:
        growth_key = st.segmented_control(
            "How fast should we assume it grows on average?",
            options=list(GROWTH_CHOICES), format_func=GROWTH_CHOICES.get, default="moderate", required=True,
            key="mc_growth",
            help="The single biggest assumption. Past winners don't always keep winning, so 'Moderate' "
                 "(close to the long-run US stock market average after a cautious haircut) is a sensible "
                 "default. The ups and downs always come from the stock's real history.",
        )
    with g2:
        custom_growth = st.number_input("Custom growth (%/yr)", min_value=-20.0, max_value=40.0, value=8.0, step=0.5,
                                        key="mc_custom", disabled=growth_key != "custom")

    with st.expander("More options", icon=":material/tune:"):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            model = st.radio("Simulation style", ["bootstrap", "gbm"], key="mc_model",
                             format_func={"bootstrap": "Replay real history (recommended)",
                                          "gbm": "Smooth bell curve (classic)"}.get,
                             help="'Replay real history' reshuffles real past days, so crashes as bad as real "
                                  "ones can happen. The classic model (geometric Brownian motion) is smoother "
                                  "and tends to underestimate crashes.")
        with a2:
            history_years = st.select_slider("History to learn from", options=[3, 5, 10, 15, 20], value=10,
                                             format_func=lambda y: f"{y} years", key="mc_hist")
        with a3:
            n_paths = st.select_slider("Number of futures", options=[1000, 2000, 5000, 10000], value=5000,
                                       key="mc_paths", help="More futures = smoother results, a bit slower.")
        with a4:
            goal = st.number_input("Goal amount ($, optional)", min_value=0, value=0, step=5_000, key="mc_goal",
                                   help="We'll tell you the chance of reaching it.")
            seed = st.number_input("Random seed", min_value=0, value=7, step=1, key="mc_seed",
                                   help="Same seed = same results. Change it to see a different set of futures.")

if not symbol:
    ui.disclaimer()
    st.stop()
if start == 0 and monthly == 0:
    st.info("Enter an amount to invest today or every month.", icon=":material/payments:")
    st.stop()

growth = {"cautious": 0.05, "moderate": 0.08, "history": None, "custom": custom_growth / 100}[growth_key]
try:
    result, prices = run_simulation(symbol, history_years, float(start), float(monthly), years, growth, model,
                                    n_paths, int(seed))
except DataUnavailable as exc:
    ui.data_error(exc)
except ValueError as exc:
    st.error(f"{exc} {symbol} may be too new to simulate.", icon=":material/history:")
    st.stop()

name = display_name(symbol)
hist_years_real = len(prices) / TRADING_DAYS
hist_growth = mc.historical_growth(metrics.log_returns(prices).values)
invested = result.total_invested
p5, p25, p50, p75, p95 = (result.percentile(p) for p in mc.PERCENTILES)
contrib_text = f" and add {ui.money(monthly)} every month" if monthly else ""

# ------------------------------------------------------------------ headline
ui.headline(
    f"If you invest <b>{ui.money(start)}</b> in <b>{name} ({symbol})</b>{contrib_text}, "
    f"after <b>{years} year{'s' if years > 1 else ''}</b> (putting in {ui.money(invested)} in total) "
    f"the typical outcome is about <b>{ui.money(p50)}</b>. "
    f"In 9 out of 10 simulated futures you'd end up between <b>{ui.money(p5)}</b> and <b>{ui.money(p95)}</b>."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Typical outcome", ui.money_short(p50), ui.delta(p50 / invested - 1, suffix=" vs. money put in"),
          help="The median: half of the simulated futures end higher, half lower.", border=True)
m2.metric("Bad case", ui.money_short(p5), ui.delta(p5 / invested - 1),
          help="Only 1 in 20 futures ends lower than this (the 5th percentile).", border=True)
m3.metric("Great case", ui.money_short(p95), ui.delta(p95 / invested - 1),
          help="Only 1 in 20 futures ends higher than this (the 95th percentile).", border=True)
if goal:
    m4.metric("Chance of reaching your goal", ui.chance(result.prob_at_least(goal)).capitalize(),
              help=f"Share of simulated futures that end at {ui.money(goal)} or more.", border=True)
else:
    m4.metric("Chance of losing money", ui.chance(result.prob_loss()).capitalize(),
              help="Share of simulated futures that end below the total amount you put in.", border=True)

growth_note = {
    "history": f"Assumes it keeps growing like its own past: about **{ui.pct(hist_growth)}** a year "
               f"(from the last {hist_years_real:.0f} years). Be careful: past winners often slow down.",
    "custom": f"Assumes typical growth of **{ui.pct(growth)}** a year.",
}.get(growth_key, f"Assumes typical growth of **{ui.pct(growth, 0)}** a year. For comparison, {symbol} actually "
                  f"grew about **{ui.pct(hist_growth)}** a year over the last {hist_years_real:.0f} years.")
st.caption(f":material/info: {growth_note} The size of the ups and downs ({ui.pct(result.annual_volatility, 0)} "
           f"a year) comes from {symbol}'s real history.")

# ------------------------------------------------------------------ fan chart
ui.section("The range of possible futures",
           "The darker the band, the more likely. The line in the middle is the typical future.")
show_paths = st.toggle("Show 30 individual simulated futures", value=False, key="mc_paths_toggle",
                       help="Each thin line is one possible future. Notice how bumpy every single one is.")
future_dates = pd.bdate_range(prices.index[-1], periods=result.steps + 1)
charts.show(charts.fan_chart(future_dates, result.percentiles, result.invested,
                             result.sample_paths if show_paths else None), key="mc_fan")

yearly_rows = []
for y in range(1, years + 1):
    t = min(y * TRADING_DAYS, result.steps)
    row = result.percentiles[t]
    yearly_rows.append({"Year": y, "Money put in": result.invested[t], "Bad case (5%)": row[0],
                        "Below typical (25%)": row[1], "Typical (median)": row[2],
                        "Above typical (75%)": row[3], "Great case (95%)": row[4]})
money_cols = {c: st.column_config.NumberColumn(c, format="dollar") for c in yearly_rows[0] if c != "Year"}
ui.see_numbers(pd.DataFrame(yearly_rows).set_index("Year"), key="mc_table", column_config=money_cols)

# ------------------------------------------------------------------ outcomes + bumps
left, right = st.columns([1.35, 1], gap="large")
with left:
    ui.section("Where the futures ended up", f"Each bar counts how many of the {n_paths:,} futures ended at that value.")
    charts.show(charts.outcome_histogram(result.final_values, invested, p50), key="mc_outcomes")
with right:
    ui.section("Expect bumps along the way")
    dd_typ = float(np.median(result.max_drawdowns))
    dd_bad = float(np.percentile(result.max_drawdowns, 10))
    st.markdown(
        f"Even in good futures, the ride is rarely smooth. In a **typical** future, at some point your investment "
        f"is **{ui.pct(-dd_typ, 0)} below** its previous high. In **1 in 10** futures the worst drop is "
        f"**{ui.pct(-dd_bad, 0)} or more**."
    )
    st.markdown(
        "Many people sell in a panic during drops like these and lock in their losses. Knowing they're "
        "normal makes it easier to stay calm."
    )
    var95, es95 = result.value_at_risk(0.95), result.expected_shortfall(0.95)
    st.markdown("**Risk in numbers**")
    st.markdown(
        f"- **Value at Risk (95%)**: {ui.money(var95)} — losing more than this (vs. money put in) happens in only "
        f"1 in 20 futures.\n"
        f"- **Expected shortfall**: {ui.money(es95)} — the average loss in those worst 1-in-20 futures.\n"
        f"- **Chance of ending below what you put in**: {ui.chance(result.prob_loss())}."
    )

# ------------------------------------------------------------------ reality check
window_days = years * TRADING_DAYS
full_hist = ui.get_prices((symbol,), 20).close[symbol].dropna()
windows = mc.historical_windows(full_hist, window_days)
ui.section("Reality check: what really happened in the past")
if len(windows) >= 60:
    lo, med, hi = windows.min(), windows.median(), windows.max()
    first, last = windows.index[0].year, windows.index[-1].year
    st.markdown(
        f"Looking at every real **{years}-year** stretch for {symbol} that started between {first} and {last} "
        f"({len(windows):,} overlapping periods), $1 invested at the start turned into between "
        f"**{ui.money(lo, 2)}** and **{ui.money(hi, 2)}**, and typically **{ui.money(med, 2)}**. "
        f"It lost money in **{ui.pct(float((windows < 1).mean()), 0)}** of those periods."
    )
    st.caption("This is only one stock's history, and it overlaps a lot, so treat it as a sanity check, not a forecast.")
else:
    st.markdown(f"There isn't enough history to look at many real {years}-year periods for {symbol}. "
                "Try a shorter time frame for this check.")

# ------------------------------------------------------------------ learn
with ui.learn_more():
    st.markdown(
        """
**Monte Carlo simulation** means answering a question by running a random experiment many times. Here,
each "future" is a sequence of daily returns, one for every trading day (252 per year).

**Replay real history (block bootstrap).** We build each future by stitching together randomly chosen
runs of *real* past days (about 20 days at a time on average), so calm and stormy stretches stay
together. Real crashes can show up in the simulation, which a bell curve tends to miss.

**Smooth bell curve (geometric Brownian motion).** Each day's log return is drawn from a normal
distribution. This is the model behind the famous Black–Scholes option-pricing formula:
"""
    )
    st.latex(r"S_{t+1} = S_t \cdot e^{\,r_t}, \qquad r_t \sim \mathcal{N}(m,\ s^2)")
    st.markdown(
        """
**Separating the trend from the wiggles.** The size and shape of the daily wiggles always come from the
stock's own history. The *average* growth is your choice, because it's the hardest thing to know. We shift
every daily log return so that the typical (median) yearly growth equals your assumption *g*:
"""
    )
    st.latex(r"\tilde r_t = r_t - \bar r + \frac{\ln(1+g)}{252}")
    st.markdown(
        """
**Reading the results.** The *median* is the middle outcome. The *5th percentile* means 5% of futures
end lower. *Value at Risk* is the loss you'd exceed only 5% of the time; *expected shortfall* (CVaR) is
the average loss in those worst cases, which is what banks use to plan for bad years.

**Limits.** No model knows about future events: new products, recessions, scandals. Simulations show
what is *plausible* if the future wiggles like the past, not what *will* happen.
"""
    )

ui.disclaimer()
