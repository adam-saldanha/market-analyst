"""Live market metrics → validated FinancialMetrics."""
from ..schemas import FinancialMetrics
import yfinance as yf


def fetch_metrics(ticker: str) -> FinancialMetrics:
    """Pull price and fundamentals for `ticker`.

    TODO(2): implement with yfinance:
        import yfinance as yf
        info = yf.Ticker(ticker).info
    Useful keys: currentPrice, marketCap, trailingPE, grossMargins,
    operatingMargins, debtToEquity. Any key can be missing — pass None
    rather than 0 so the prompt shows "n/a" instead of a fake number.
    Let Pydantic raise on bad data; graph.py turns that into a clean error.
    """
    info = yf.Ticker(ticker).info
    dte = info.get("debtToEquity")
    return FinancialMetrics(
        ticker=ticker,
        price=info.get("currentPrice"),
        market_cap=info.get("marketCap"),
        pe_ratio=info.get("trailingPE"),
        gross_margin=info.get("grossMargins"),
        debt_to_equity=dte/100 if dte is not None else None,
    )

