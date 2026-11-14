"""Market Predictor: honest, math-powered tools for US stocks.

Run locally:   streamlit run streamlit_app.py
"""

import streamlit as st

from ui.components import CSS

st.set_page_config(
    page_title="Market Predictor · Honest math for US stocks",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Market Predictor: free, honest, math-powered tools for US stocks. Educational only, not "
                 "financial advice.",
    },
)
st.html(CSS)

pages = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/monte_carlo.py", title="Future Simulator", icon=":material/stacked_line_chart:",
            url_path="simulator"),
    st.Page("app_pages/portfolio.py", title="Portfolio Builder", icon=":material/pie_chart:", url_path="portfolio"),
    st.Page("app_pages/predictor.py", title="Prediction Lab", icon=":material/model_training:", url_path="predict"),
    st.Page("app_pages/learn.py", title="Learn", icon=":material/school:", url_path="learn"),
]
st.navigation(pages, position="top").run()
