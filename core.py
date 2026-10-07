"""Session 2 core components: Groq LLM, stock data, news, minimal RAG, analysis chain."""
import os
import re
from datetime import date, timedelta

import requests
import yfinance as yf
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# ---------- LLM ----------
def get_llm() -> ChatGroq:
    """Groq chat model; GROQ_API_KEY is read from the environment by ChatGroq."""
    return ChatGroq(model=os.environ["GROQ_MODEL"], temperature=0.2)


# ---------- Stock data ----------
@tool
def get_stock_data(ticker: str) -> dict:
    """Fetch 6 months of daily prices for a ticker and return summary metrics.
    Returns {"error": ...} if the ticker is invalid or data is unavailable."""
    ticker = ticker.strip().upper()
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
    except Exception as exc:
        return {"ticker": ticker, "error": f"Price download failed: {exc}"}
    if hist.empty or len(hist) < 20:
        return {"ticker": ticker, "error": "No (or too little) price data - check the ticker."}

    close = hist["Close"]
    last = float(close.iloc[-1])
    ret = lambda n: round((last / float(close.iloc[-n - 1]) - 1) * 100, 2)  # % over last n sessions
    return {
        "ticker": ticker,
        "last_close": round(last, 2),
        "change_1d_pct": ret(1),
        "change_5d_pct": ret(5),
        "change_1m_pct": ret(21) if len(close) > 21 else None,
        "change_6m_pct": round((last / float(close.iloc[0]) - 1) * 100, 2),
        "sma_20": round(float(close.tail(20).mean()), 2),
        "sma_50": round(float(close.tail(50).mean()), 2) if len(close) >= 50 else None,
        "high_6m": round(float(hist["High"].max()), 2),
        "low_6m": round(float(hist["Low"].min()), 2),
        "daily_volatility_pct": round(float(close.pct_change().std() * 100), 2),
        "avg_volume": int(hist["Volume"].mean()),
        "n_sessions": len(close),
    }


# ---------- Financial news ----------
@tool
def get_financial_news(query: str, days: int = 14) -> list[dict]:
    """Fetch recent English news articles from NewsAPI for a company name or ticker.
    Returns a list of {title, description, content, source, published_at, url}; empty list on failure."""
    key = os.getenv("NEWS_API_KEY")
    if not key:
        print("NEWS_API_KEY not set")
        return []
    params = {
        "q": query,
        "from": (date.today() - timedelta(days=days)).isoformat(),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
    }
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params,
                            headers={"X-Api-Key": key}, timeout=15)
        data = resp.json()
    except Exception as exc:
        print(f"NewsAPI request failed: {exc}")
        return []
    if data.get("status") != "ok":
        print(f"NewsAPI error: {data.get('code')} - {data.get('message')}")
        return []
    return [
        {
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "content": a.get("content") or "",
            "source": (a.get("source") or {}).get("name", ""),
            "published_at": a.get("publishedAt", ""),
            "url": a.get("url", ""),
        }
        for a in data.get("articles", [])
        if a.get("title") and a.get("title") != "[Removed]"
    ]


# ---------- Minimal RAG ----------
class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings wrapper over fastembed (local ONNX model, no API key)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


def build_news_store(articles: list[dict]) -> InMemoryVectorStore | None:
    """articles -> Documents -> chunks -> FastEmbed embeddings -> InMemoryVectorStore."""
    docs = []
    for a in articles:
        body = re.sub(r"\s*\[\+\d+ chars\]", "", a["content"])  # NewsAPI truncation marker
        if body[:60] in a["description"]:  # content often just repeats the description
            body = ""
        docs.append(Document(
            page_content=f"{a['title']}. {a['description']} {body}".strip(),
            metadata={k: a[k] for k in ("title", "source", "published_at", "url")},
        ))
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    if not chunks:
        return None
    return InMemoryVectorStore.from_documents(chunks, FastEmbedEmbeddings())


def retrieve_news_context(store: InMemoryVectorStore | None, question: str, k: int = 4) -> str:
    """Similarity search; returns the top chunks formatted as LLM-ready context."""
    if store is None:
        return "No relevant news found."
    hits = store.similarity_search(question, k=k)
    return "\n\n".join(
        f"[{d.metadata['source']}, {d.metadata['published_at'][:10]}] {d.page_content}" for d in hits
    )


# ---------- LangChain prompt + chain ----------
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a careful equity analyst. Use only the data provided; do not add "
               "facts, events or risks that are not supported by it. "
               "If the news context is thin or irrelevant, say so."),
    ("human", "Stock data:\n{stock_data}\n\nRetrieved news context:\n{news_context}\n\n"
              "User request: {question}\n\n"
              "Write a concise analysis (max 150 words): price trend, key news themes, risks."),
])


def build_analysis_chain():
    """prompt | Groq LLM | string parser."""
    return analysis_prompt | get_llm() | StrOutputParser()


if __name__ == "__main__":  # smoke test of every component
    print("LLM:", get_llm().invoke("Reply with the single word: ready").content)

    stock = get_stock_data.invoke({"ticker": "AAPL"})
    print("Stock:", stock)
    print("Invalid ticker:", get_stock_data.invoke({"ticker": "NOTATICKERXYZ"}))

    news = get_financial_news.invoke({"query": "Apple AAPL stock"})
    print(f"News: {len(news)} articles; first: {news[0]['title'] if news else None}")

    question = "What recent news could affect Apple's stock price?"
    store = build_news_store(news)
    context = retrieve_news_context(store, question)
    print("RAG context:\n", context)

    print("Analysis:\n", build_analysis_chain().invoke(
        {"stock_data": stock, "news_context": context, "question": question}))
