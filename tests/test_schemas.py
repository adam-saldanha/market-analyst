import pytest
from pydantic import ValidationError

from market_analyst.schemas import FinancialMetrics, Thesis


def make(**overrides):
    base = dict(ticker="aapl", price=190.0, pe_ratio=29.5, gross_margin=0.45, debt_to_equity=1.5)
    return FinancialMetrics(**{**base, **overrides})


def test_ticker_is_normalized():
    assert make().ticker == "AAPL"


def test_price_must_be_positive():
    with pytest.raises(ValidationError):
        make(price=0)


def test_margin_must_be_fraction():
    with pytest.raises(ValidationError):
        make(gross_margin=45.0)   # percent passed by mistake


def test_prompt_shows_na_for_missing():
    assert "P/E: n/a" in make(pe_ratio=None).to_prompt()


def test_thesis_needs_two_points():
    with pytest.raises(ValidationError):
        Thesis(stance="bull", summary="x", key_points=["one"], confidence=0.5)


def test_pe_ratio_outlier():
    # Decide your rule; this test assumes outliers become None.
    assert make(pe_ratio=-5).pe_ratio is None
    assert make(pe_ratio=5000).pe_ratio is None
