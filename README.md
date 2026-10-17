# Multi-Agent Market Analyst

Generates a structured investment memo for a ticker by running parallel **Bull** and **Bear**
agents over a hybrid-RAG index of SEC 10-K filings, grounded with validated live financial metrics.

## Architecture

```
            ┌──────────────┐
 ticker ──► │ fetch_data   │  EDGAR 10-K + market metrics (Pydantic-validated)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ retrieve     │  hybrid RAG: vector + BM25 → fuse (RRF) → rerank
            └──┬────────┬──┘
               ▼        ▼          (parallel fan-out)
          ┌──────┐  ┌──────┐
          │ bull │  │ bear │
          └──┬───┘  └───┬──┘
             └────┬─────┘          (fan-in)
                  ▼
            ┌──────────────┐
            │ synthesize   │  → InvestmentMemo (structured output)
            └──────────────┘
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your Gemini key + SEC user agent
python -m market_analyst.main AAPL
pytest
```

## Build order (suggested)

Each TODO is tagged with its step number — search for `TODO(1)`, `TODO(2)`, …

1. `schemas.py` – finish the Pydantic validators (already partly done — run `pytest`)
2. `data/market.py` – pull metrics from yfinance into `FinancialMetrics`
3. `data/sec.py` – ticker → CIK → latest 10-K text from EDGAR
4. `rag/chunking.py` – split filing text into overlapping chunks
5. `rag/retriever.py` – vector search, BM25, reciprocal rank fusion, reranking
6. `agents/*.py` – write the prompts; agents return structured `Thesis` objects
7. `graph.py` – already wired; add timing + caching to hit the <15s goal

The graph runs end-to-end right now with stub data (`python -m market_analyst.main AAPL --stub`),
so you can watch the flow before any real component exists.
