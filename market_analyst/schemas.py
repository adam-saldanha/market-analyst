"""Pydantic models: the contract between data, agents and the final memo.

Validating numbers here is what stops the LLM from seeing (or inventing) junk values.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class FinancialMetrics(BaseModel):
    ticker: str
    price: float = Field(gt=0)
    market_cap: Optional[float] = Field(default=None, ge=0)
    pe_ratio: Optional[float] = None
    gross_margin: Optional[float] = None       # fraction, e.g. 0.44
    operating_margin: Optional[float] = None   # fraction
    debt_to_equity: Optional[float] = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not v or len(v) > 10:
            raise ValueError("ticker must be 1-10 characters")
        return v

    @field_validator("gross_margin", "operating_margin")
    @classmethod
    def margin_is_fraction(cls, v: Optional[float]) -> Optional[float]:
        # Margins can be negative (losses) but never beyond ±100%.
        if v is not None and not -1.0 <= v <= 1.0:
            raise ValueError("margin must be a fraction between -1 and 1")
        return v

    # TODO(1): add a validator for pe_ratio. Negative or absurd P/E (e.g. > 1000)
    #   is usually a data artifact — decide: reject it, or coerce it to None?
    #   Then make tests/test_schemas.py::test_pe_ratio_outlier pass.

    # TODO(1): add a validator for debt_to_equity. yfinance reports it as a
    #   percentage (150.0 = 1.5x). Normalize to a ratio so prompts stay consistent.

    @field_validator("pe_ratio")
    @classmethod
    def pe_is_sane(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0 <= v <= 1000:
            return None
        return v

    @field_validator("debt_to_equity")
    @classmethod
    def debt_to_equity(cls, v:Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            return None
        return v

    def to_prompt(self) -> str:
        """Deterministic text block injected into agent prompts."""
        def fmt(x, pct=False):
            if x is None:
                return "n/a"
            return f"{x:.1%}" if pct else f"{x:,.2f}"

        return (
            f"Ticker: {self.ticker}\n"
            f"Price: ${fmt(self.price)}\n"
            f"Market cap: ${fmt(self.market_cap)}\n"
            f"P/E: {fmt(self.pe_ratio)}\n"
            f"Gross margin: {fmt(self.gross_margin, pct=True)}\n"
            f"Operating margin: {fmt(self.operating_margin, pct=True)}\n"
            f"Debt/Equity: {fmt(self.debt_to_equity)}"
        )


class Chunk(BaseModel):
    id: int
    text: str
    section: Optional[str] = None   # e.g. "Item 1A. Risk Factors"


class Thesis(BaseModel):
    stance: Literal["bull", "bear"]
    summary: str
    key_points: list[str] = Field(min_length=2, max_length=6)
    cited_chunk_ids: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class InvestmentMemo(BaseModel):
    ticker: str
    recommendation: Literal["buy", "hold", "sell"]
    summary: str
    bull_case: list[str]
    bear_case: list[str]
    key_metrics: str
    risks: list[str]
