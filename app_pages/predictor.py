import numpy as np
import pandas as pd
import streamlit as st

from engine import TRADING_DAYS, metrics, risk
from engine import predictor as pr
from engine.data import BENCHMARK, DataUnavailable
from engine.universe import display_name
from engine.zoo import FAMILIES, ZOO
from ui import charts
from ui import components as ui

HISTORY_YEARS = 15
DIRECTION_HORIZONS = {1: "Tomorrow", 5: "In 1 week", 21: "In 1 month"}
RISK_HORIZONS = {5: "Next week", 21: "Next month", 63: "Next 3 months"}


def load(symbol: str):
    symbols = (symbol,) if symbol == BENCHMARK else (symbol, BENCHMARK)
    pdata = ui.get_prices(symbols, HISTORY_YEARS)
    close = pdata.close[symbol].dropna()
    volume = pdata.volume[symbol].reindex(close.index)
    market = None if symbol == BENCHMARK else pdata.close[BENCHMARK].reindex(close.index)
    return close, volume, market


def features_for(symbol: str):
    close, volume, market = load(symbol)
    return pr.build_features(close, volume, market), close


@st.cache_data(show_spinner=False, max_entries=48)
def run_direction(symbol: str, horizon: int, model: str):
    features, close = features_for(symbol)
    cached = tournament_store().get((symbol, horizon, close.index[-1]))
    result = cached[model] if cached and model in cached else pr.walk_forward(features, close, horizon, model)
    forecast = pr.latest_forecast(features, close, horizon, model)
    return result, forecast


@st.cache_resource
def tournament_store() -> dict:
    """Finished tournaments, shared by every visitor: {(symbol, horizon, last price date): results}."""
    return {}


def run_showdown(symbol: str, horizon: int) -> dict:
    """Run (or fetch) the tournament, showing a live leaderboard while it runs."""
    features, close = features_for(symbol)
    key = (symbol, horizon, close.index[-1])
    store = tournament_store()
    if key in store:
        return store[key]

    progress = st.progress(0.0, text=f"Training {len(ZOO)} algorithms on {HISTORY_YEARS} years of {symbol} prices… "
                                     "about a minute. The slowest ones go first.")
    live = st.empty()
    done: dict[str, float] = {}

    def on_result(model_key, res, i, total):
        if isinstance(res, pr.WalkForwardResult):
            done[model_key] = res.accuracy
        leader = max(done, key=done.get) if done else None
        text = f"{i} of {total} algorithms tested"
        if leader:
            text += f" · current leader: {ZOO[leader].name} ({ui.pct(done[leader])})"
        progress.progress(i / total, text=text)
        if done:
            board = pd.DataFrame({"Algorithm": [ZOO[k].name for k in done], "Right": list(done.values())})
            board = board.sort_values("Right", ascending=False).head(8).reset_index(drop=True)
            board.index += 1
            live.dataframe(board, column_config={"Right": st.column_config.ProgressColumn(
                "Right", min_value=0.4, max_value=0.7, format="percent")})

    results = pr.run_tournament(features, close, horizon, on_result=on_result)
    progress.empty()
    live.empty()
    if len(store) > 200:  # keep memory bounded
        store.pop(next(iter(store)))
    store[key] = results
    return results


@st.cache_data(show_spinner="Forecasting risk…", max_entries=64)
def run_risk(symbol: str, horizon: int):
    close, _, market = load(symbol)
    return risk.backtest(close, market, horizon), risk.forecast(close, market, horizon), close


ui.page_header(":material/model_training:", "Prediction Lab",
               "Can machine learning predict a stock? We train real models, test them honestly on years of data "
               "they've never seen, and tell you plainly whether they work.")

with ui.card("inputs"):
    c1, c2 = st.columns([2.2, 1.4], vertical_alignment="bottom")
    with c1:
        symbol = ui.ticker_select("Stock or ETF", key="pr_ticker", default=ui.default_ticker("AAPL"))
    with c2:
        horizon = st.segmented_control("Predict whether the price is higher…", options=list(DIRECTION_HORIZONS),
                                       format_func=DIRECTION_HORIZONS.get, default=5, required=True, key="pr_h")

if not symbol:
    ui.disclaimer()
    st.stop()
name = display_name(symbol)
when = DIRECTION_HORIZONS[horizon].lower()

tab_show, tab_dir, tab_risk = st.tabs([
    f":material/trophy: Algorithm showdown ({len(ZOO)} models)",
    ":material/swap_vert: One model up close",
    ":material/waves: How bumpy will it be?",
])

# ================================================================== showdown
with tab_show:
    store_key = None
    try:
        _, close_now = features_for(symbol)
        store_key = (symbol, horizon, close_now.index[-1])
    except DataUnavailable as exc:
        ui.data_error(exc)
    ready = store_key in tournament_store()
    started = st.session_state.get("pr_show") == (symbol, horizon)

    if not ready and not started:
        st.markdown(
            f"""
Which kind of machine learning is best at predicting whether **{name}** will be higher **{when}**?
We'll put **{len(ZOO)} algorithms** head to head, from one-line rules of thumb to neural networks, in
{len(FAMILIES)} families. Every one gets the same clues, the same {HISTORY_YEARS} years of prices, and the same honest test
on years it has never seen.

Then comes the part most websites skip. **When you try {len(ZOO)} models, some will look good by pure luck.**
We correct for that, so a lucky winner can't fool you.
"""
        )
        if st.button(f"Start the showdown", type="primary", icon=":material/trophy:", key="pr_show_btn"):
            st.session_state["pr_show"] = (symbol, horizon)
            st.rerun()
        st.caption("Takes about a minute the first time. After that, results are saved for everyone for the rest of the day.")
    else:
        try:
            results = run_showdown(symbol, horizon)
        except ValueError as exc:
            st.error(f"{exc} {symbol} may be too new.", icon=":material/history:")
            st.stop()
        lb = pr.leaderboard(results)
        bench = results["always_up"] if "always_up" in results else next(iter(results.values()))
        band = pr.luck_band(bench)
        contenders = lb[lb.index != "always_up"]
        learners = contenders[~contenders["baseline"]]
        best_key = learners.index[0]
        best = learners.iloc[0]
        n_beat = int((contenders["edge"] > 0).sum())
        n_real = int((contenders["verdict"] == "Real edge").sum())
        n_lucky = int((contenders["verdict"] == "Looks good, likely luck").sum())
        guess = "up" if bench.base_rate >= 0.5 else "down"
        years_tested = bench.n_tests / TRADING_DAYS

        # ---------------------------------------------------------- verdict
        story = (
            f"We tested <b>{len(contenders)} prediction methods</b> on {years_tested:.0f} years of {symbol} prices they "
            f"had never seen. The best learning algorithm, <b>{best['name']}</b>, was right "
            f"<b>{ui.pct(best['accuracy'])}</b> of the time"
            + (f" (it learned to say \"{guess}\" almost every time, rediscovering the simplest strategy)"
               if best["time_in_market"] > 0.97 or best["time_in_market"] < 0.03 else "")
            + f". Simply guessing \"{guess}\" every time was right <b>{ui.pct(bench.naive_accuracy)}</b>. "
        )
        if n_real:
            story += (f"<b>{n_real} method{'s' if n_real > 1 else ''} still beat{'' if n_real > 1 else 's'} the benchmark "
                      f"after correcting for luck</b>, a rare result worth a closer look.")
        else:
            story += (f"<b>{n_beat}</b> of {len(contenders)} beat the benchmark at all, and after correcting for the luck of "
                      f"trying so many, <b>none has a real edge</b>.")
        ui.headline(story)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Methods tested", f"{len(contenders)}", border=True,
                  help=f"{len(ZOO) - 1} algorithms and rules, plus 'Everyone votes', which averages all of them.")
        m2.metric("Beat the benchmark", f"{n_beat}", border=True,
                  help=f"Right more often than always guessing \"{guess}\" ({ui.pct(bench.naive_accuracy)}).")
        m3.metric("Real edge after luck check", f"{n_real}", border=True,
                  help="Still significant after the Holm–Bonferroni correction for trying many models (5% level).")
        m4.metric("Best accuracy", ui.pct(contenders["accuracy"].max()), border=True,
                  help=f"Achieved by {contenders.iloc[0]['name']}.")
        if n_lucky:
            st.caption(f":material/casino: {n_lucky} method{'s' if n_lucky > 1 else ''} passed the usual luck check on "
                       "its own but failed once we accounted for trying so many. That's the trap this page protects you from.")

        # ---------------------------------------------------------- ranking chart
        ui.section("Every method, ranked",
                   "Each dot is one method's share of correct predictions. Dots right of the dotted line beat "
                   "always guessing the usual direction. The gray zone is where a method with no skill at all "
                   "lands 95% of the time, so dots inside it prove nothing. Purple = learning algorithm, gray = simple "
                   "rule, pink = best learning algorithm.")
        charts.show(charts.model_ranking(lb, bench.naive_accuracy, band, highlight=best_key), key="pr_rank")

        # ---------------------------------------------------------- families
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            ui.section("Which family did best?", "The best and the average method in each family.")
            fam = contenders.groupby("family")["accuracy"].agg(["max", "mean"])
            fam = fam.reindex([f for f in FAMILIES if f in fam.index]).sort_values("max", ascending=False)
            fam.columns = ["Best in family", "Family average"]
            charts.show(charts.family_bars(fam, bench.naive_accuracy), key="pr_fam")
        with right:
            ui.section("What this tells you")
            simple_best = lb[lb["baseline"] & (lb.index != "always_up")]["accuracy"].max()
            complex_fams = ["Neural networks", "Boosting", "Kernel methods"]
            complex_best = contenders[contenders["family"].isin(complex_fams)]["accuracy"].max()
            st.markdown(
                f"- **Fancier isn't better.** The best neural network, boosting or kernel method scored "
                f"{ui.pct(complex_best)}, versus {ui.pct(simple_best)} for the best one-line rule of thumb.\n"
                f"- **Teamwork helps a little.** Averaging many models (\"Everyone votes\") scored "
                f"{ui.pct(lb.loc[pr.CROWD_KEY, 'accuracy']) if pr.CROWD_KEY in lb.index else '—'}: steadier than most "
                "single models, but still no magic.\n"
                f"- **Hard to beat the obvious.** {symbol} rose in {ui.pct(bench.base_rate, 0)} of these periods, "
                f"so just betting on \"{guess}\" is a tough benchmark.\n"
                f"- **Luck is everywhere.** If {len(contenders)} methods were all pure guessing, there would still be a "
                f"{ui.pct(1 - 0.95 ** len(contenders), 0)} chance that at least one passes a normal luck check."
            )

        # ---------------------------------------------------------- leaderboard
        ui.section("The full leaderboard", "Click a column header to sort. Filter by family below.")
        fams = st.pills("Families", list(FAMILIES), selection_mode="multi", default=list(FAMILIES),
                        key="pr_fam_filter", label_visibility="collapsed")
        view = lb[lb["family"].isin(fams or FAMILIES)]
        table = pd.DataFrame({
            "Rank": view["rank"],
            "Method": view["name"],
            "Family": view["family"],
            "Right": view["accuracy"],
            "vs. benchmark": view["edge"].map(lambda e: f"{e * 100:+.1f} pts"),
            "AUC": view["auc"],
            "Luck-checked verdict": view["verdict"],
            "Growth / yr if traded": view["growth"],
            "Worst drop": view["max_drawdown"],
            "Training time": view["seconds"].map(lambda s: f"{s:.1f} s" if s else "—"),
        }).set_index("Rank")
        st.dataframe(table, key="pr_lb", height=min(36 * (len(table) + 1) + 4, 900), column_config={
            "Right": st.column_config.ProgressColumn("Right", min_value=0.4, max_value=0.7, format="percent",
                                                     help="Share of test periods where it called the direction correctly."),
            "AUC": st.column_config.NumberColumn(format="%.3f", help="0.5 = no better than random ordering; "
                                                 "1.0 = perfect. Measures ranking skill, separately from accuracy."),
            "Growth / yr if traded": st.column_config.NumberColumn(format="percent",
                                                                   help=f"Own {symbol} only when the method says 'up'."),
            "Worst drop": st.column_config.NumberColumn(format="percent"),
        })
        st.caption(f"For comparison: just holding {symbol} grew {ui.pct(metrics.cagr(bench.buyhold_equity))} a year "
                   f"over the same period, with a worst drop of {ui.pct(metrics.max_drawdown(bench.buyhold_equity))}.")

        # ---------------------------------------------------------- inspect one
        ui.section("Inspect any method")
        options = list(lb.index)
        pick = st.selectbox("Method", options, index=options.index(best_key), key="pr_pick",
                            format_func=lambda k: f"#{lb.loc[k, 'rank']} · {lb.loc[k, 'name']}")
        res = results[pick]
        spec = ZOO.get(pick)
        st.markdown(f"**{lb.loc[pick, 'name']}** · {lb.loc[pick, 'family']}. "
                    + (spec.blurb if spec else "Averages the predictions of every learning algorithm above."))
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Right", ui.pct(res.accuracy), ui.delta(res.edge, 1, " vs. benchmark"), border=True)
        i2.metric("AUC", f"{res.auc:.3f}", border=True)
        i3.metric("Luck-checked verdict", lb.loc[pick, "verdict"], border=True,
                  help=f"Raw p = {res.p_value:.3f}; after correcting for {len(contenders)} tries: "
                       f"{lb.loc[pick, 'p_adjusted']:.3f}." if pick != "always_up" else None)
        i4.metric("Time invested if traded", ui.pct(res.time_in_market, 0), border=True)
        charts.show(charts.line_chart({f"Following {lb.loc[pick, 'name']}": res.strategy_equity * 10_000,
                                       f"Just holding {symbol}": res.buyhold_equity * 10_000},
                                      "Value of $10,000", height=340), key="pr_pick_eq")

        with ui.learn_more("Meet all the algorithms"):
            st.dataframe(pd.DataFrame(
                [{"Method": s.name, "Family": s.family, "How it works": s.blurb} for s in ZOO.values()]
            ), key="pr_zoo", hide_index=True)
        with ui.learn_more("Why trying many models is dangerous (and how we fix it)"):
            st.markdown(
                f"""
Imagine {len(contenders)} people each flipping a coin 100 times. Someone will probably get 60 heads, and it would
be silly to call them a "skilled flipper". Testing many prediction methods works the same way: the best one looks
good partly *because* it was the luckiest. Statisticians call this **data snooping** or the **multiple comparisons
problem**, and it's a big reason flashy trading strategies fail in real life.

If each method had a 5% chance of passing a luck check by accident, the chance that *at least one* of {len(contenders)}
passes is
"""
            )
            st.latex(rf"1 - 0.95^{{{len(contenders)}}} \approx {1 - 0.95 ** len(contenders):.0%}")
            st.markdown(
                """
The **Holm–Bonferroni correction** fixes this. Sort the methods' p-values from smallest to largest, then
multiply the smallest by *m* (the number of methods), the next by *m* − 1, and so on. A method only counts as a
**real edge** if its corrected p-value stays below 0.05 *and* it beats the benchmark by more than 1 percentage point.

**The gray "luck zone"** on the chart is the range where a no-skill method lands 95% of the time:
"""
            )
            st.latex(r"p_0 \pm 1.96\sqrt{\frac{p_0(1-p_0)}{n}}")
            st.markdown(
                "where *p₀* is the benchmark's accuracy and *n* is the number of independent test periods. "
                "**AUC** (area under the ROC curve) measures whether a method ranks up-days above down-days, "
                "regardless of where it draws the line. 0.5 means no skill."
            )

# ================================================================== one model
with tab_dir:
    zoo_keys = [k for k in ZOO]
    c2, c3 = st.columns([3, 1], vertical_alignment="bottom")
    with c2:
        model = st.selectbox("Model", zoo_keys, index=zoo_keys.index("ensemble"), key="pr_model",
                             format_func=lambda k: f"{ZOO[k].name} · {ZOO[k].family}",
                             help="Any of the algorithms from the showdown. 'Team of 3' averages logistic regression, "
                                  "an extra-trees forest and gradient boosting.")
    with c3:
        run_key = (symbol, horizon, model)
        if st.button("Train & test", type="primary", icon=":material/play_arrow:", key="pr_run", width="stretch"):
            st.session_state["pr_ran"] = run_key

    if st.session_state.get("pr_ran") != run_key:
        st.markdown(
            f"""
Press **Train & test** to teach a model to predict whether **{name}** will be higher
**{DIRECTION_HORIZONS[horizon].lower()}**. Here's what happens:

1. We download up to {HISTORY_YEARS} years of prices and compute {len(pr.FEATURE_INFO)} clues, such as recent momentum,
   how jumpy the price is, and what the whole market is doing.
2. The model learns from the first 3 years, then predicts the next year **without seeing it**. Then it
   learns from everything up to that point and predicts the following year, and so on to today.
3. We compare its record against simply guessing "up" every time, and check whether it could just be luck.

This takes about 10–30 seconds the first time.
"""
        )
    else:
        try:
            with st.spinner(f"Training and testing on {HISTORY_YEARS} years of {symbol} prices…"):
                result, forecast = run_direction(symbol, horizon, model)
        except DataUnavailable as exc:
            ui.data_error(exc)
        except ValueError as exc:
            st.error(f"{exc} {symbol} may be too new.", icon=":material/history:")
            st.stop()

        when = DIRECTION_HORIZONS[horizon].lower()
        years_tested = result.n_tests / TRADING_DAYS
        acc, naive = result.accuracy, result.naive_accuracy
        guess = "up" if result.base_rate >= 0.5 else "down"

        # ---------------------------------------------------------- verdict
        if result.verdict == "edge":
            st.success(
                f"**This model shows a real but small edge for {symbol}.** Over {years_tested:.1f} years of honest "
                f"testing it was right **{ui.pct(acc)}** of the time, versus **{ui.pct(naive)}** for always "
                f"guessing \"{guess}\". Pure luck would rarely do this well (p = {result.p_value:.3f}). Small "
                "edges can still vanish in the future, so treat it with care.",
                icon=":material/verified:",
            )
        elif result.verdict == "weak":
            st.warning(
                f"**Slightly better than guessing, but it could easily be luck.** Over {years_tested:.1f} years of "
                f"testing the model was right **{ui.pct(acc)}** of the time, versus **{ui.pct(naive)}** for always "
                f"guessing \"{guess}\". That gap isn't big enough to trust (p = {result.p_value:.2f}).",
                icon=":material/help:",
            )
        else:
            st.warning(
                f"**No reliable edge.** Over {years_tested:.1f} years of honest testing, the model was right "
                f"**{ui.pct(acc)}** of the time, but simply guessing \"{guess}\" every time was right "
                f"**{ui.pct(naive)}** of the time. For {symbol}, it can't beat the simplest possible strategy. "
                "This is the normal result: if price patterns were easy to predict, traders would already have "
                "used them up.",
                icon=":material/do_not_disturb_on:",
            )

        # ---------------------------------------------------------- forecast + record
        left, right = st.columns([1, 1.4], gap="large")
        with left:
            with ui.card("lean"):
                st.markdown(f"**The model's current lean** · based on prices up to {forecast.as_of:%b %d, %Y}")
                prob = forecast.prob_up
                lean = "higher" if prob >= 0.5 else "lower"
                st.markdown(f"<div class='mp-big'>{ui.pct(max(prob, 1 - prob), 0)}</div>", unsafe_allow_html=True)
                st.markdown(f"chance that {symbol} is **{lean}** {when}")
                st.progress(float(prob), text=f"{ui.pct(prob, 0)} up · {ui.pct(1 - prob, 0)} down")
                if result.verdict == "none":
                    st.caption("Because this model has no proven edge for this stock, treat this like a coin flip. "
                               "We show it only so you can see what the model thinks.")
                else:
                    st.caption("Even a model with an edge is wrong often. Never bet money you can't afford to lose.")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Model was right", ui.pct(acc), ui.delta(result.edge, 1, " vs. guessing"),
                      delta_color="normal", border=True, help="Share of test days where it called the direction correctly.")
            m2.metric(f'Always guessing "{guess}"', ui.pct(naive), border=True,
                      help=f"{symbol} went {guess} on {ui.pct(max(result.base_rate, 1 - result.base_rate))} of test days.")
            m3, m4 = st.columns(2)
            luck = ("Unlikely to be luck" if result.p_value < 0.05 else
                    "Might be luck" if result.p_value < 0.25 else "Could easily be luck")
            m3.metric(f"Luck check (p = {result.p_value:.2f})", luck, border=True,
                      help="A binomial test: how often would random guessing at the baseline rate score at "
                           "least this well? Below 0.05 is the usual bar for 'probably not luck'. Uses "
                           f"{result.n_independent:,} non-overlapping predictions.")
            m4.metric(f"Days tested ({years_tested:.1f} years)", f"{result.n_tests:,}", border=True,
                      help="Every one of these days was predicted by a model that had never seen it.")

        # ---------------------------------------------------------- trading test
        ui.section("If you had traded on it",
                   f"Rule: own {symbol} on days the model expects it to be higher {when} (above 50%), otherwise "
                   "hold cash. We charge 0.05% per trade.")
        strat, hold = result.strategy_equity * 10_000, result.buyhold_equity * 10_000
        charts.show(charts.line_chart({"Following the model": strat, f"Just holding {symbol}": hold},
                                      "Value of $10,000"), key="pr_equity")
        s_sum, h_sum = metrics.summarize(strat), metrics.summarize(hold)
        st.dataframe(pd.DataFrame({
            "Ended with": [ui.money(strat.iloc[-1]), ui.money(hold.iloc[-1])],
            "Growth / yr": [ui.pct(s_sum.cagr), ui.pct(h_sum.cagr)],
            "Worst drop": [ui.pct(s_sum.max_drawdown), ui.pct(h_sum.max_drawdown)],
            "Time invested": [ui.pct(result.time_in_market, 0), "100%"],
            "Trades": [f"{result.trades:,}", "1"],
        }, index=["Following the model", f"Just holding {symbol}"]), key="pr_eq_table")

        # ---------------------------------------------------------- over time + calibration
        left, right = st.columns(2, gap="large")
        with left:
            ui.section("Is it getting better or worse?", "Accuracy over the previous 12 months, at each point in time.")
            preds = result.predictions.dropna(subset=["label"])
            correct = ((preds["prob"] > 0.5) == (preds["label"] == 1)).astype(float)
            up_share = preds["label"].rolling(TRADING_DAYS).mean()
            rolling = {"Model": correct.rolling(TRADING_DAYS).mean(),
                       "Always guessing the usual direction": np.maximum(up_share, 1 - up_share)}
            charts.show(charts.line_chart(rolling, "Share correct", y_format=".0%", height=340, reference=0.5),
                        key="pr_rolling")
        with right:
            ui.section("Can you trust its confidence?",
                       "When the model said a certain chance, how often did the stock really rise? A trustworthy "
                       "model's pairs of bars match.")
            cal = result.calibration[result.calibration["times"] >= 15]
            cal_df = cal[["model_said", "actually_rose"]].rename(
                columns={"model_said": "Model's average chance", "actually_rose": "Actually went up"})
            charts.show(charts.grouped_vbar(cal_df, y_title="Chance of going up", hover_extra=cal["times"]),
                        key="pr_cal")

        ui.section("What the model pays attention to",
                   "How much the model's skill drops when each clue is scrambled (permutation importance, "
                   "measured on the last 2 years of training data).")
        imp = forecast.importance.head(8)
        if imp.sum() > 0:
            imp = imp / imp.sum()
            imp.index = [pr.FEATURE_INFO.get(k, k) for k in imp.index]
            charts.show(charts.hbar(imp, x_format=".0%", color_index=0, x_title="Share of the model's attention"),
                        key="pr_imp")
        else:
            st.caption("No clue stood out: the model barely relies on any of them, which fits a stock that's hard to predict.")

        ui.see_numbers(result.predictions.rename(columns={"prob": "Chance up (model)", "label": "Went up?",
                                                          "next_ret": "Next-day return", "position": "Owned?"}),
                       key="pr_preds", label="See every prediction")

        with ui.learn_more():
            st.markdown(
                f"""
**The question.** For each day *t*, will the price *h* trading days later be higher? (h = {horizon} here.)
That makes it a **classification** problem with answer 1 (up) or 0 (down).

**The clues (features).** All computed only from prices up to day *t*: returns over 1 day to 3 months,
volatility, **RSI**, **MACD**, distance from 20/50/200-day moving averages, distance from the 1-year
high, trading volume, and the whole market's (S&P 500's) trend and volatility.

**The models.**
- *Logistic regression* turns a weighted sum of the clues into a probability:
"""
            )
            st.latex(r"P(\text{up}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_k x_k)}}")
            st.markdown(
                """
- *Forests* average many decision trees, each trained on random subsets of the data and clues.
- *Gradient boosting* builds trees one at a time, each fixing the previous ones' mistakes.
- *Teams* average several models' probabilities ("soft voting").

The **Algorithm showdown** tab describes every method. All are deliberately restrained (shallow trees, strong
regularization), because complex models memorize noise.

**Honest testing (walk-forward).** Train on years 1–3, test on year 4, retrain including year 4, test on
year 5, and so on. When predicting *h* days ahead, we also drop the last *h* training
days, because their answers aren't known yet on the prediction date (an **embargo**). Mixing up past and
future here is called **look-ahead bias**, and it's the #1 reason amateur stock models look amazing on
paper and fail in real life.

**The luck check** is a one-sided binomial test. If you guessed at the baseline rate *p₀* on *n*
independent days, how likely is it to get *k* or more right by chance?
"""
            )
            st.latex(r"\text{p-value} = P(X \ge k), \quad X \sim \text{Binomial}(n,\ p_0)")
            st.markdown(
                "For predictions more than 1 day ahead, neighbouring windows overlap, so we only use every *h*-th "
                "prediction for this test."
            )

# ================================================================== risk
with tab_risk:
    rh = st.segmented_control("Forecast the ups and downs for…", options=list(RISK_HORIZONS),
                              format_func=RISK_HORIZONS.get, default=21, required=True, key="pr_rh")
    try:
        bt, fc, close = run_risk(symbol, rh)
    except DataUnavailable as exc:
        ui.data_error(exc)
    except ValueError as exc:
        st.error(f"{exc} {symbol} may be too new.", icon=":material/history:")
        st.stop()

    period = RISK_HORIZONS[rh].lower()
    lo1, hi1 = fc.price_range(1)
    lo2, hi2 = fc.price_range(2)
    mood = ("calmer than usual" if fc.vol < 0.85 * fc.long_run_vol else
            "stormier than usual" if fc.vol > 1.15 * fc.long_run_vol else "about as bumpy as usual")
    ui.headline(
        f"Over the <b>{period}</b>, {symbol} will most likely move about <b>±{ui.pct(fc.typical_move)}</b> "
        f"from today's {ui.money(fc.price, 2)}. There's roughly a <b>2-in-3</b> chance it ends between "
        f"<b>{ui.money(lo1, 2)}</b> and <b>{ui.money(hi1, 2)}</b>, and about a <b>19-in-20</b> chance between "
        f"{ui.money(lo2, 2)} and {ui.money(hi2, 2)}. That's <b>{mood}</b> for {symbol}."
    )
    if bt.improvement > 0.03:
        st.success(
            f"**Unlike direction, risk is genuinely predictable.** In honest testing on {symbol}, this forecast's "
            f"typical error was **{ui.pct(bt.improvement, 0)} smaller** than the simple guess \"the {period} will "
            f"be like the last month\". The 2-in-3 range really caught the outcome **{ui.pct(bt.coverage_1, 0)}** "
            f"of the time (goal: 67%), and the 19-in-20 range **{ui.pct(bt.coverage_2, 0)}** (goal: 95%).",
            icon=":material/verified:",
        )
    else:
        st.info(
            f"For {symbol}, this risk forecast was only about as good as assuming the {period} will be like the "
            f"last month (difference: {ui.pct(bt.improvement, 0, signed=True)}). The ranges still show the "
            "typical size of moves.",
            icon=":material/info:",
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast: ups & downs per year", ui.pct(fc.vol, 0), border=True,
              help="Volatility: the typical size of price swings, scaled to a year.")
    m2.metric("Last month", ui.pct(fc.last_month_vol, 0), border=True)
    m3.metric(f"{symbol}'s long-run normal", ui.pct(fc.long_run_vol, 0), border=True,
              help=f"Over the last {len(close) / TRADING_DAYS:.0f} years.")

    ui.section("Where the price could be", "Recent prices, then the forecast range. Darker band = 2-in-3 chance.")
    hist = close.iloc[-126:]
    fdates = pd.bdate_range(close.index[-1], periods=rh + 1)
    steps = np.sqrt(np.arange(rh + 1) / TRADING_DAYS) * fc.vol
    cone = [fc.price * np.exp(k * steps) for k in (-1, 1, -2, 2)]
    charts.show(charts.price_cone(hist, fdates, cone[0], cone[1], cone[2], cone[3]), key="pr_cone")

    ui.section("Track record: forecast vs. what really happened",
               f"Each point is a forecast made on that day for the {period} after it.")
    tbl = bt.table.dropna(subset=["realized"])
    charts.show(charts.line_chart({"Forecast": tbl["forecast"], "What actually happened": tbl["realized"]},
                                  "Ups & downs per year (volatility)", y_format=".0%", height=360), key="pr_vol_bt")
    ui.see_numbers(bt.table.rename(columns={"forecast": "Forecast", "naive": "Simple guess (last month)",
                                            "realized": "Actual", "fwd_return": "Actual move (log)"}),
                   key="pr_vol_table",
                   column_config={c: st.column_config.NumberColumn(format="percent")
                                  for c in ["Forecast", "Simple guess (last month)", "Actual", "Actual move (log)"]})

    with ui.learn_more():
        st.markdown(
            """
**Why risk is predictable.** Calm and stormy periods cluster: a wild week is usually followed by more
wild days. This "volatility clustering" was noticed by Benoit Mandelbrot in 1963, and modelling it earned
Robert Engle the 2003 Nobel Prize.

**The HAR model** (Corsi, 2009) predicts future volatility from volatility measured over the past week,
month, quarter and year. Different kinds of traders act on different time scales, and this captures all of them:
"""
        )
        st.latex(r"\log\sigma_{\text{next}} = \beta_0 + \beta_1\log\sigma_{\text{week}} + \beta_2\log\sigma_{\text{month}} "
                 r"+ \beta_3\log\sigma_{\text{quarter}} + \beta_4\log\sigma_{\text{year}} + \beta_5\,\text{fall} + \beta_6\log\sigma_{\text{market}}")
        st.markdown(
            """
"fall" is how much the price dropped over the last month. Falling prices tend to bring more volatility
(the **leverage effect**). The model is fit by ordinary least squares and tested walk-forward, just like
the direction models.

**From volatility to a price range.** With yearly volatility σ, a typical move over *h* trading days is
"""
        )
        st.latex(r"\text{move} = \sigma\sqrt{h/252}, \qquad \text{range} = P_0 \cdot e^{\pm k\cdot\text{move}}")
        st.markdown(
            "With k = 1 we'd expect about 68% of outcomes inside, and with k = 2 about 95%, if returns followed "
            "a bell curve. Real returns have fatter tails, which is why we measure the true hit rate."
        )

ui.disclaimer()
