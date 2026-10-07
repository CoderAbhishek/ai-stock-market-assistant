"""Session 3: LangGraph multi-agent workflow built on the components in core.py."""
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core import (build_analysis_chain, build_news_store, get_financial_news,
                  get_llm, get_stock_data, retrieve_news_context)

DEFAULT_QUESTION = "What recent news could affect this stock's price?"


# ---------- Shared state ----------
class State(TypedDict, total=False):
    ticker: str
    question: str
    stock_data: dict
    news_context: str
    insights: str
    recommendation: str   # BUY | SELL | HOLD
    justification: str
    error: str


class Recommendation(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="Simulated recommendation")
    justification: str = Field(description="2-3 sentences citing only the provided data")


recommendation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You produce SIMULATED, educational stock recommendations (not financial advice). "
               "Choose exactly one of BUY, SELL or HOLD. Base the decision and justification only "
               "on the stock data, news context and insights provided; do not add outside facts. "
               "Choose HOLD when the evidence is mixed or thin."),
    ("human", "Stock data:\n{stock_data}\n\nNews context:\n{news_context}\n\n"
              "Insights:\n{insights}\n\nGive the recommendation for {ticker}."),
])


# ---------- Agent nodes ----------
def stock_analysis_agent(state: State) -> dict:
    """Fetch price data and add a rule-based trend label."""
    data = get_stock_data.invoke({"ticker": state["ticker"]})
    if "error" in data:
        return {"stock_data": data, "error": data["error"]}
    sma50 = data["sma_50"]
    if sma50 is None:
        data["trend"] = "unknown (insufficient history)"
    elif data["last_close"] > data["sma_20"] > sma50:
        data["trend"] = "uptrend (close > SMA20 > SMA50)"
    elif data["last_close"] < data["sma_20"] < sma50:
        data["trend"] = "downtrend (close < SMA20 < SMA50)"
    else:
        data["trend"] = "mixed (no clear SMA alignment)"
    return {"stock_data": data}


def news_rag_agent(state: State) -> dict:
    """NewsAPI -> vector store -> similarity retrieval."""
    articles = get_financial_news.invoke({"query": f"{state['ticker']} stock"})
    store = build_news_store(articles)
    return {"news_context": retrieve_news_context(store, state["question"])}


def insight_agent(state: State) -> dict:
    chain = build_analysis_chain()
    return {"insights": chain.invoke({"stock_data": state["stock_data"],
                                      "news_context": state["news_context"],
                                      "question": state["question"]})}


def recommendation_agent(state: State) -> dict:
    chain = recommendation_prompt | get_llm().with_structured_output(Recommendation)
    rec = chain.invoke({"ticker": state["ticker"], "stock_data": state["stock_data"],
                        "news_context": state["news_context"], "insights": state["insights"]})
    return {"recommendation": rec.action, "justification": rec.justification}


def fallback_node(state: State) -> dict:
    return {"error": f"Cannot analyse '{state['ticker']}': {state['error']}"}


# ---------- Conditional routing ----------
def route_after_stock_analysis(state: State) -> Literal["news_rag", "fallback"]:
    """Continue only if valid price data was retrieved."""
    return "fallback" if state.get("error") else "news_rag"


# ---------- Graph ----------
def build_graph():
    g = StateGraph(State)
    g.add_node("stock_analysis", stock_analysis_agent)
    g.add_node("news_rag", news_rag_agent)
    g.add_node("insight", insight_agent)
    g.add_node("recommendation", recommendation_agent)
    g.add_node("fallback", fallback_node)

    g.add_edge(START, "stock_analysis")
    g.add_conditional_edges("stock_analysis", route_after_stock_analysis)
    g.add_edge("news_rag", "insight")
    g.add_edge("insight", "recommendation")
    g.add_edge("recommendation", END)
    g.add_edge("fallback", END)
    return g.compile()


def run_assistant(ticker: str, question: str = "") -> State:
    return build_graph().invoke({"ticker": ticker.strip().upper(),
                                 "question": question.strip() or DEFAULT_QUESTION})


if __name__ == "__main__":
    app = build_graph()
    print(app.get_graph().draw_mermaid())

    for t in ("AAPL", "NOTATICKERXYZ"):
        print(f"\n===== {t} =====")
        result = run_assistant(t)
        for k, v in result.items():
            print(f"--- {k} ---\n{v}")
