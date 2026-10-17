"""Shared LangGraph state. Bull and bear write to different keys, so the
parallel branches never conflict and no reducer is needed."""
from typing import Optional, TypedDict

from .schemas import Chunk, FinancialMetrics, InvestmentMemo, Thesis


class AnalystState(TypedDict, total=False):
    ticker: str
    stub: bool                      # True = canned data, no network/LLM calls
    metrics: FinancialMetrics
    filing_text: str
    bull_context: list[Chunk]
    bear_context: list[Chunk]
    bull_thesis: Thesis
    bear_thesis: Thesis
    memo: InvestmentMemo
    timings: dict[str, float]
    error: Optional[str]
