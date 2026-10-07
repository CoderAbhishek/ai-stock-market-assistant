# AI Stock Market Assistant

AI-powered stock market assistant built with LangChain and LangGraph.

The application analyses stock data, retrieves relevant financial news using RAG, generates LLM-based insights, and provides simulated Buy/Sell/Hold recommendations. Recommendations are educational only and not financial advice.

**Live app:** https://ai-stock-market-assistant.streamlit.app/

## Workflow

```
START → stock_analysis ──(valid data?)──► news_rag → insight → recommendation → END
                       └──(no)─────────► fallback ───────────────────────────► END
```

| Node | Role |
|---|---|
| `stock_analysis` | Price data from yfinance, rule-based trend label |
| `news_rag` | NewsAPI → text splitter → FastEmbed → `InMemoryVectorStore` → retrieval |
| `insight` | LangChain prompt + Groq LLM over stock data and retrieved news |
| `recommendation` | Structured output: exactly one of BUY / SELL / HOLD plus justification |
| `fallback` | Error path for invalid tickers |

## Files

- `core.py` – Groq LLM, stock-data tool, NewsAPI tool, RAG, analysis chain
- `graph.py` – LangGraph state, agent nodes, conditional routing, compiled graph
- `app.py` – Streamlit UI

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in the values
streamlit run app.py
```

Required environment variables (see `.env.example`):

- `GROQ_API_KEY` – Groq API key
- `GROQ_MODEL` – Groq model name, e.g. `openai/gpt-oss-20b`
- `NEWS_API_KEY` – NewsAPI key

## Deploy (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. On share.streamlit.io choose **Create app**, select this repo, branch `main`, main file `app.py`.
3. Under **Advanced settings** pick Python 3.12 and paste the three variables above as secrets in TOML form (`GROQ_API_KEY = "..."`). Never commit them.
