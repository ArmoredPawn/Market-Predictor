import streamlit as st

from ui import components as ui

ui.page_header(":material/school:", "Learn",
               "The key ideas behind every tool on this site, in plain English. About a 5-minute read.")

IDEAS = [
    ("Stock", "A small piece of ownership in a company. If the company grows and earns more, its stock "
              "usually becomes more valuable over time, though not in a straight line."),
    ("ETF (exchange-traded fund)", "A basket of many stocks you can buy as one. SPY holds the 500 biggest US "
                                   "companies, so buying it is like buying a slice of the whole US market."),
    ("Return", "How much an investment gained or lost, as a percentage. +10% means $100 became $110."),
    ("Compound growth", "Growth on top of growth. At 8% a year, $10,000 becomes about $21,600 in 10 years and "
                        "about $100,600 in 30 years. Time matters more than almost anything else."),
    ("Volatility (ups & downs)", "How much a price typically swings. A volatility of 30% a year means a typical "
                                "year could easily be +30% or −30% from the average. Higher volatility = bumpier "
                                "ride."),
    ("Drawdown", "A fall from a previous high. If your investment went from $10,000 to $7,000, that's a 30% "
                 "drawdown. Even great stocks have had drawdowns of 50% or more."),
    ("Diversification", "Not putting all your eggs in one basket. Because stocks don't all fall at the same time, "
                        "a mix is usually less bumpy than its parts, without giving up much return."),
    ("Correlation", "How much two stocks move together, from −1 (opposite) through 0 (unrelated) to +1 (always "
                    "the same way). Low correlation is what makes diversification work."),
    ("Sharpe ratio (return per risk)", "Return above a safe interest rate, divided by volatility. It answers: "
                                       "how much am I being paid for each unit of risk I take?"),
    ("Monte Carlo simulation", "Answering a question by running a random experiment thousands of times. Instead "
                               "of one guess about the future, you see the whole range of possibilities."),
    ("Percentile", "The 5th percentile is the value that only 5% of outcomes fall below. The 50th percentile "
                   "(the median) is the middle outcome."),
    ("Machine learning", "Computer programs that find patterns in examples. Here: past price clues and whether "
                         "the stock went up or down afterwards."),
    ("Overfitting", "When a model memorizes the past instead of learning something real. It looks brilliant on "
                    "old data and fails on new data. The #1 trap in stock prediction."),
    ("Look-ahead bias", "Accidentally letting a model see information from the future while testing it. It makes "
                        "results look far better than they really are. Our tests are built to prevent it."),
    ("p-value (luck check)", "If something was pure luck, how often would we see a result this good? A p-value "
                             "below 0.05 (1 in 20) is the usual bar for 'probably not luck'."),
]

ui.section("Key ideas")
left, right = st.columns(2, gap="large")
for i, (term, text) in enumerate(IDEAS):
    with (left if i % 2 == 0 else right):
        st.markdown(f"**{term}.** {text}")

ui.section("Questions people ask")
with st.expander("Can this website predict the stock market?", icon=":material/help:", expanded=True):
    st.markdown(
        """
Not reliably, and neither can anyone else consistently. Stock prices already reflect what millions of
investors know, so short-term moves are mostly noise. That's the **efficient market hypothesis**
(Eugene Fama, Nobel Prize 2013). The Prediction Lab lets you see this for yourself: for most stocks, the
models don't beat simple guessing once tested honestly.

What *is* predictable is **risk**, meaning how big the swings will be, because calm and stormy periods
cluster. The Prediction Lab's risk forecasts beat a simple guess by about 15–20% for most stocks.
"""
    )
with st.expander("So what's the point of the math?", icon=":material/help:"):
    st.markdown(
        """
Math is excellent at the questions that actually matter for most people:
- **What range of outcomes should I expect?** (Future Simulator)
- **How do I spread my money to take less risk for the same return?** (Portfolio Builder)
- **How bumpy will the next month likely be?** (Prediction Lab → risk)
- **Is this "amazing" strategy real, or did it just get lucky?** (every honest test on this site)
"""
    )
with st.expander("Why only US stocks?", icon=":material/help:"):
    st.markdown(
        """
Focusing on one market keeps the data clean and comparable: every price is in US dollars, follows the
same trading calendar, and is compared with the same benchmark (the S&P 500). We include US-based
companies listed on the NASDAQ, NYSE and NYSE American, plus popular ETFs that hold US stocks. Foreign
companies that also trade in the US (like Toyota or Alibaba) are filtered out.
"""
    )
with st.expander("Where does the data come from?", icon=":material/help:"):
    st.markdown(
        """
- **Prices:** Yahoo Finance, through the open-source `yfinance` library. Prices are adjusted for stock
  splits and dividends, so they show an investor's true total return. They may be delayed.
- **Company list:** NASDAQ, NYSE and AMEX listings, filtered to US-based companies.
- **Safe interest rate:** the current 13-week US Treasury bill yield.
"""
    )
with st.expander("Is this financial advice?", icon=":material/help:"):
    st.markdown(
        """
No. This is an educational tool. The results come from past prices and simple assumptions, and they
can't account for news, company events or the economy. Before investing real money, consider talking
to a qualified, fee-only financial adviser, and never invest money you can't afford to lose.
"""
    )

ui.section("A simple plan most experts agree on")
st.markdown(
    """
1. **Start early and invest regularly.** Compound growth needs time more than anything.
2. **Diversify.** A broad, low-cost index fund (like one tracking the S&P 500 or the total US market)
   holds hundreds of companies at once.
3. **Expect drops and don't panic-sell.** 20–30% falls happen every few years. Selling during them is how
   most people lose money.
4. **Keep costs low.** Fees and frequent trading quietly eat returns.
5. **Be skeptical of anyone who claims they can predict prices**, especially if they're selling something.
"""
)

ui.disclaimer()
