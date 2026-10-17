from ..schemas import FinancialMetrics, InvestmentMemo, Thesis
from .llm import get_llm

# TODO(6): define how the recommendation is decided (e.g. weigh confidence,
#   valuation vs. margins) so results are explainable, not vibes.
SYNTH_PROMPT = """You are a portfolio manager. Weigh the two theses and write an investment memo.
Copy key_metrics verbatim from METRICS — do not alter any number.

METRICS:
{metrics}

BULL THESIS:
{bull}

BEAR THESIS:
{bear}
"""


def run_synthesizer(ticker: str, metrics: FinancialMetrics, bull: Thesis, bear: Thesis) -> InvestmentMemo:
    llm = get_llm(temperature=0).with_structured_output(InvestmentMemo)
    memo = llm.invoke(SYNTH_PROMPT.format(
        metrics=metrics.to_prompt(), bull=bull.model_dump_json(indent=2), bear=bear.model_dump_json(indent=2),
    ))
    # Overwrite with the deterministic block so a hallucinated number can't survive.
    return memo.model_copy(update={"ticker": ticker, "key_metrics": metrics.to_prompt()})
