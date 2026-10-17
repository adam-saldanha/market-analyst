from ..schemas import Chunk, FinancialMetrics, Thesis
from .llm import format_context, get_llm

BULL_QUERY = "revenue growth, competitive advantages, margin expansion, new products, capital returns"

# TODO(6): sharpen this prompt. Ideas: require each key point to cite a chunk id,
#   forbid numbers that aren't in METRICS or CONTEXT, calibrate confidence.
BULL_PROMPT = """You are a buy-side analyst making the strongest honest BULL case.
Use only the METRICS and CONTEXT below. Cite chunk ids you relied on.

METRICS:
{metrics}

CONTEXT:
{context}
"""


def run_bull(metrics: FinancialMetrics, chunks: list[Chunk]) -> Thesis:
    llm = get_llm().with_structured_output(Thesis)
    thesis = llm.invoke(BULL_PROMPT.format(metrics=metrics.to_prompt(), context=format_context(chunks)))
    return thesis.model_copy(update={"stance": "bull"})
