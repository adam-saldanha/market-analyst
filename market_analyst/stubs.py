"""Canned data so the whole graph runs before any real component exists."""
from .schemas import Chunk, FinancialMetrics, InvestmentMemo, Thesis


def stub_metrics(ticker: str) -> FinancialMetrics:
    return FinancialMetrics(
        ticker=ticker, price=190.0, market_cap=2.9e12, pe_ratio=29.5,
        gross_margin=0.45, operating_margin=0.30, debt_to_equity=1.5,
    )


STUB_FILING = (
    "Item 1. Business. The Company designs consumer devices and sells services. "
    "Services revenue grew year over year driven by subscriptions. "
    "Item 1A. Risk Factors. The Company depends on third-party manufacturers "
    "concentrated in a single region. Regulatory scrutiny of app store fees is increasing. "
    "Item 7. MD&A. Gross margin expanded due to a favorable services mix."
)


def stub_chunks() -> list[Chunk]:
    return [
        Chunk(id=0, text="Services revenue grew year over year driven by subscriptions.", section="Item 1"),
        Chunk(id=1, text="Depends on third-party manufacturers concentrated in a single region.", section="Item 1A"),
        Chunk(id=2, text="Regulatory scrutiny of app store fees is increasing.", section="Item 1A"),
        Chunk(id=3, text="Gross margin expanded due to a favorable services mix.", section="Item 7"),
    ]


def stub_thesis(stance: str) -> Thesis:
    if stance == "bull":
        return Thesis(stance="bull", summary="High-margin services growth supports the valuation.",
                      key_points=["Services growing", "Margin expansion"], cited_chunk_ids=[0, 3], confidence=0.6)
    return Thesis(stance="bear", summary="Supply-chain and regulatory risk are underpriced.",
                  key_points=["Manufacturing concentration", "App store regulation"], cited_chunk_ids=[1, 2], confidence=0.5)


def stub_memo(ticker: str, bull: Thesis, bear: Thesis, metrics: FinancialMetrics) -> InvestmentMemo:
    return InvestmentMemo(
        ticker=ticker, recommendation="hold",
        summary=f"Bull ({bull.confidence:.0%}) and bear ({bear.confidence:.0%}) cases roughly balance.",
        bull_case=bull.key_points, bear_case=bear.key_points,
        key_metrics=metrics.to_prompt(), risks=bear.key_points,
    )
