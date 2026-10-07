"""Streamlit UI for the AI Stock Market Assistant (run: streamlit run app.py)."""
import os

import streamlit as st

from graph import run_assistant

st.set_page_config(page_title="AI Stock Market Assistant", page_icon="📈")
st.title("📈 AI Stock Market Assistant")
st.caption("LangChain + LangGraph multi-agent analysis with news RAG. "
           "Recommendations are simulated and not financial advice.")

missing = [k for k in ("GROQ_API_KEY", "GROQ_MODEL", "NEWS_API_KEY") if not os.getenv(k)]
if missing:
    st.error(f"Missing configuration: {', '.join(missing)}. Set them in `.env` or the host's secrets.")
    st.stop()

with st.form("query"):
    ticker = st.text_input("Stock ticker", value="AAPL")
    question = st.text_area("Question (optional)", placeholder="e.g. What recent news could affect this stock?")
    submitted = st.form_submit_button("Analyse")

if submitted:
    if not ticker.strip():
        st.warning("Enter a ticker.")
        st.stop()
    try:
        with st.spinner("Running agents: stock analysis, news RAG, insights, recommendation..."):
            result = run_assistant(ticker, question)
    except Exception as exc:  # LLM / network failures
        st.error(f"Analysis failed: {exc}")
        st.stop()

    if result.get("error"):
        st.error(result["error"])
        st.stop()

    data = result["stock_data"]
    st.subheader(f"Stock analysis - {data['ticker']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"${data['last_close']}", f"{data['change_1d_pct']}% 1d")
    c2.metric("1 month", f"{data['change_1m_pct']}%")
    c3.metric("6 months", f"{data['change_6m_pct']}%")
    c4.metric("Daily volatility", f"{data['daily_volatility_pct']}%")
    st.write(f"**Trend:** {data['trend']}")
    with st.expander("All stock metrics"):
        st.json(data)

    st.subheader("Retrieved financial news (RAG)")
    st.text(result["news_context"])

    st.subheader("Insights")
    st.write(result["insights"])

    st.subheader("Recommendation (simulated)")
    show = {"BUY": st.success, "SELL": st.error, "HOLD": st.warning}[result["recommendation"]]
    show(f"**{result['recommendation']}**")
    st.write(result["justification"])
