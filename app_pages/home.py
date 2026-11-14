import streamlit as st

from ui import components as ui

st.markdown(
    """
<div class="mp-hero">
  <h1>Understand any US stock in minutes.</h1>
  <p>Free tools that use real math, from Nobel Prize-winning portfolio theory to machine learning, explained in
  plain English. No finance degree needed, and no hype: we test every model honestly and tell you when it
  doesn't work.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**Start here:** pick a stock, then choose a tool.")
    symbol = ui.ticker_select("Stock or ETF", key="home_ticker", default="AAPL") or "AAPL"
    st.caption("Not sure? Try **SPY**, which tracks the 500 biggest US companies, or a company you know, like "
               "Apple, Costco or Nike.")

TOOLS = [
    {
        "page": "app_pages/monte_carlo.py",
        "icon": ":material/stacked_line_chart:",
        "title": "Future Simulator",
        "question": "“If I invest $10,000, what could it be worth in 10 years?”",
        "points": ["Thousands of possible futures, built from real history",
                   "Good, typical and bad cases in dollars",
                   "Chance of losing money or reaching a goal"],
        "cta": f"Simulate {symbol}",
    },
    {
        "page": "app_pages/portfolio.py",
        "icon": ":material/pie_chart:",
        "title": "Portfolio Builder",
        "question": "“How should I split my money between a few stocks?”",
        "points": ["Optimal mixes with Markowitz portfolio theory",
                   "See which stocks move together",
                   "Honest test against the S&P 500"],
        "cta": "Build a portfolio",
    },
    {
        "page": "app_pages/predictor.py",
        "icon": ":material/model_training:",
        "title": "Prediction Lab",
        "question": "“Can AI predict if this stock goes up — and how bumpy will it be?”",
        "points": ["45 algorithms go head to head, from rules of thumb to neural networks",
                   "Tested on unseen years, with a luck check",
                   "Risk forecasts that really work"],
        "cta": f"Test predictions for {symbol}",
    },
]

cols = st.columns(3, gap="medium")
for i, (col, tool) in enumerate(zip(cols, TOOLS)):
    with col, st.container(border=True, height="stretch"):
        st.markdown(f"### {tool['icon']} {tool['title']}")
        st.markdown(f"<div class='mp-card-q'>{tool['question']}</div>", unsafe_allow_html=True)
        if st.button(tool["cta"], key=f"home_go_{i}", type="primary" if i == 0 else "secondary",
                     icon=":material/arrow_forward:", icon_position="right", width="stretch"):
            params = {"t": symbol} if tool["page"] != "app_pages/portfolio.py" else None
            st.switch_page(tool["page"], query_params=params)
        st.markdown("\n".join(f"- {p}" for p in tool["points"]))

st.markdown("## Why trust this site?")
a, b, c = st.columns(3, gap="large")
with a:
    st.markdown("#### :material/fact_check: Tested honestly")
    st.markdown("Every model is tested on years of data it never saw, and compared with the simplest possible "
                "strategy. If it can't beat that, we say so, in big letters.")
with b:
    st.markdown("#### :material/translate: Plain English first")
    st.markdown("Every result starts with a one-sentence answer. Charts show ranges, not false certainty. "
                "The math is there too, one click away, if you want it.")
with c:
    st.markdown("#### :material/flag: US stocks only")
    st.markdown("About 3,900 US-based companies and popular US stock ETFs, sorted biggest first. "
                "Foreign companies are filtered out, even ones that trade in New York.")

st.markdown("## The one thing to remember")
st.info(
    "**Nobody can reliably predict next week's stock price, including us.** What math *can* do is show you "
    "the range of what might happen, how much risk you're taking, and how to spread it out. That's what this "
    "site is for.",
    icon=":material/lightbulb:",
)
st.page_link("app_pages/learn.py", label="New to investing? Learn the key ideas in 5 minutes",
             icon=":material/school:")

ui.disclaimer()
