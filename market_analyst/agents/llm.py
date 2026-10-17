"""One place to build the chat model, so swapping providers is a one-line change."""
from ..config import LLM_MODEL


def get_llm(temperature: float = 0.2):
    from langchain_google_genai import ChatGoogleGenerativeAI  # imported lazily so --stub needs no key
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=temperature)


def format_context(chunks) -> str:
    """Number chunks by id so agents can cite them in cited_chunk_ids."""
    return "\n\n".join(f"[{c.id}] ({c.section or 'n/a'}) {c.text}" for c in chunks)
