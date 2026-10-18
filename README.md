# Multi-Agent Market Analyst

Generates a structured investment memo for a stock ticker. Parallel **Bull** and **Bear**
agents argue opposite sides using a hybrid-RAG index over the company's latest SEC 10-K
filing, grounded with validated live market metrics. A **Synthesizer** agent then weighs both
theses and writes the final memo.

Built with [LangGraph](https://github.com/langchain-ai/langgraph), Google Gemini (chat +
embeddings), FAISS, BM25 and a cross-encoder reranker.

## Architecture

```
            ┌──────────────┐
 ticker ──► │ fetch_data   │  yfinance metrics ‖ EDGAR 10-K text (fetched concurrently)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ retrieve     │  hybrid RAG: vector (FAISS) + BM25 → fuse (RRF) → rerank
            └──┬────────┬──┘
               ▼        ▼          (parallel fan-out)
          ┌──────┐  ┌──────┐
          │ bull │  │ bear │       each returns a structured Thesis
          └──┬───┘  └───┬──┘
             └────┬─────┘          (fan-in)
                  ▼
            ┌──────────────┐
            │ synthesize   │  → InvestmentMemo (structured output)
            └──────────────┘
```

### How it works

1. **Fetch** – `data/market.py` pulls price, market cap, P/E, gross margin and debt/equity
   from yfinance into a Pydantic `FinancialMetrics` model. `data/sec.py` maps the ticker to
   its CIK, finds the most recent 10-K on EDGAR, strips the HTML (including inline-XBRL
   headers) and caches the plain text.
2. **Chunk** – `rag/chunking.py` splits the filing into ~1,200-character overlapping chunks,
   ending on sentence boundaries, and tags each with its 10-K section (e.g. `Item 1A.`),
   skipping the table of contents.
3. **Retrieve** – `rag/retriever.py` runs each agent's query through dense vector search
   (Gemini embeddings in a FAISS cosine index) and BM25, merges the two rankings with
   reciprocal rank fusion, then reorders the top 25 with a cross-encoder and keeps the top 8.
4. **Argue** – the Bull and Bear agents receive the metrics plus their own retrieved chunks
   and return a structured `Thesis` (summary, 2–6 key points, cited chunk ids, confidence).
5. **Synthesize** – a portfolio-manager agent weighs both theses and emits an
   `InvestmentMemo` with a `buy` / `hold` / `sell` recommendation. The `key_metrics` field is
   overwritten with the deterministic metrics block, so a hallucinated number can't survive.

### Guardrails

- **Validated inputs** – `FinancialMetrics` normalizes tickers, rejects non-positive prices
  and out-of-range margins, and coerces implausible P/E values (< 0 or > 1000) and negative
  debt/equity to `None`, which prompts render as `n/a` instead of a fake number.
- **Structured outputs** – every LLM call uses `with_structured_output`, so agents return
  Pydantic objects rather than free text.
- **Citations** – chunks are numbered in the prompt so agents can cite the evidence they used.

## Project layout

```
market_analyst/
├── main.py            CLI entry point
├── graph.py           LangGraph wiring, per-step timings, stub mode
├── state.py           shared graph state (AnalystState)
├── schemas.py         Pydantic models: FinancialMetrics, Chunk, Thesis, InvestmentMemo
├── config.py          settings loaded from .env
├── stubs.py           canned data for running without network/API calls
├── data/
│   ├── market.py      yfinance → FinancialMetrics
│   └── sec.py         EDGAR ticker → CIK → latest 10-K text (cached)
├── rag/
│   ├── chunking.py    section-aware overlapping chunker
│   └── retriever.py   HybridRetriever: FAISS + BM25 + RRF + cross-encoder
└── agents/
    ├── llm.py         Gemini chat model factory, context formatting
    ├── bull.py        bull query + prompt
    ├── bear.py        bear query + prompt
    └── synthesizer.py memo prompt
tests/                 pytest suite (schemas, graph, RRF)
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env`:

| Variable          | Required | Default                | Purpose                                                       |
| ----------------- | -------- | ---------------------- | ------------------------------------------------------------- |
| `GOOGLE_API_KEY`  | yes      | –                      | Gemini API key for the chat model and embeddings              |
| `SEC_USER_AGENT`  | yes      | –                      | Name + email; [required by EDGAR](https://www.sec.gov/os/accessing-edgar-data) |
| `LLM_MODEL`       | no       | `gemini-2.5-flash`     | Chat model used by all three agents                           |
| `EMBEDDING_MODEL` | no       | `gemini-embedding-001` | Embedding model for vector search                             |
| `CACHE_DIR`       | no       | `.cache`               | Where 10-K text and embeddings are cached                     |

Chunk size, overlap, top-k and the reranker model are set in `market_analyst/config.py`.

## Usage

```bash
# Full run: live data, real LLM calls
python -m market_analyst.main AAPL

# Stub run: canned data, no network or API key needed
python -m market_analyst.main AAPL --stub
```

The CLI prints the memo as JSON, followed by per-step timings and the total runtime
(the target is under 15 seconds).

```json
{
  "ticker": "AAPL",
  "recommendation": "hold",
  "summary": "...",
  "bull_case": ["..."],
  "bear_case": ["..."],
  "key_metrics": "Ticker: AAPL\nPrice: $190.00\n...",
  "risks": ["..."]
}
```

**First-run notes:** the cross-encoder reranker downloads ~90 MB from Hugging Face, and
embedding a full 10-K takes a while. Both the filing text and its embeddings are cached in
`CACHE_DIR` per ticker, so repeat runs are much faster. Delete the cache to force a refresh
(e.g. after a new 10-K is filed).

## Tests

```bash
pytest
```

The suite covers schema validation, reciprocal rank fusion, and an end-to-end stub run of
the graph. It makes no network calls.

## Roadmap

Remaining `TODO`s in the code, tagged by area:

- **Prompts (`TODO(6)`)** – sharpen the bull/bear prompts (require a citation per key point,
  forbid numbers not present in the metrics or context, calibrate confidence, weight
  `Item 1A` chunks for the bear) and define an explainable rule for the synthesizer's
  recommendation.
- **Metrics** – `operating_margin` is in the schema but not yet populated from yfinance.
- **Performance** – reach the < 15s goal on cold runs; the retriever is rebuilt on every
  invocation and the reranker model is reloaded each time.
- **Error handling** – `AnalystState.error` exists but nodes don't yet catch failures and
  surface them cleanly.
