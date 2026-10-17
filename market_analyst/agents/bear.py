from ..schemas import Chunk, FinancialMetrics, Thesis
from .llm import format_context, get_llm

BEAR_QUERY = "risk factors, competition, regulation, litigation, debt, declining margins, customer concentration"

# TODO(6): sharpen this prompt (see bull.py). Consider weighting Item 1A chunks.
BEAR_PROMPT = """You are a short-seller making the strongest honest BEAR case.
Use only the METRICS and CONTEXT below. Cite chunk ids you relied on.

METRICS:
{metrics}

CONTEXT:
{context}
"""


def run_bear(metrics: FinancialMetrics, chunks: list[Chunk]) -> Thesis:
    llm = get_llm().with_structured_output(Thesis)
    thesis = llm.invoke(BEAR_PROMPT.format(metrics=metrics.to_prompt(), context=format_context(chunks)))
    return thesis.model_copy(update={"stance": "bear"})
