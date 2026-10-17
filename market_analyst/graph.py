"""LangGraph wiring: fetch → retrieve → (bull ‖ bear) → synthesize."""
import time
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, START, StateGraph

from . import stubs
from .state import AnalystState


def _timed(name, state, fn):
    t0 = time.perf_counter()
    out = fn()
    timings = dict(state.get("timings", {}))
    timings[name] = round(time.perf_counter() - t0, 3)
    out["timings"] = timings
    return out


def fetch_data(state: AnalystState) -> dict:
    def run():
        t = state["ticker"]
        if state.get("stub"):
            return {"metrics": stubs.stub_metrics(t), "filing_text": stubs.STUB_FILING}
        from .data.market import fetch_metrics
        from .data.sec import fetch_10k_text
        # Network-bound and independent, so fetch both at once.
        with ThreadPoolExecutor(max_workers=2) as pool:
            m, f = pool.submit(fetch_metrics, t), pool.submit(fetch_10k_text, t)
            return {"metrics": m.result(), "filing_text": f.result()}
    return _timed("fetch_data", state, run)


def retrieve(state: AnalystState) -> dict:
    def run():
        if state.get("stub"):
            chunks = stubs.stub_chunks()
            return {"bull_context": [chunks[0], chunks[3]], "bear_context": [chunks[1], chunks[2]]}
        from .agents.bear import BEAR_QUERY
        from .agents.bull import BULL_QUERY
        from .rag.chunking import chunk_text
        from .rag.retriever import HybridRetriever
        retriever = HybridRetriever(chunk_text(state["filing_text"]), cache_key=state["ticker"])
        return {"bull_context": retriever.retrieve(BULL_QUERY), "bear_context": retriever.retrieve(BEAR_QUERY)}
    return _timed("retrieve", state, run)


def bull(state: AnalystState) -> dict:
    if state.get("stub"):
        return {"bull_thesis": stubs.stub_thesis("bull")}
    from .agents.bull import run_bull
    return {"bull_thesis": run_bull(state["metrics"], state["bull_context"])}


def bear(state: AnalystState) -> dict:
    if state.get("stub"):
        return {"bear_thesis": stubs.stub_thesis("bear")}
    from .agents.bear import run_bear
    return {"bear_thesis": run_bear(state["metrics"], state["bear_context"])}


def synthesize(state: AnalystState) -> dict:
    def run():
        args = (state["ticker"], state["bull_thesis"], state["bear_thesis"], state["metrics"])
        if state.get("stub"):
            return {"memo": stubs.stub_memo(*args)}
        from .agents.synthesizer import run_synthesizer
        return {"memo": run_synthesizer(state["ticker"], state["metrics"], state["bull_thesis"], state["bear_thesis"])}
    return _timed("synthesize", state, run)


def build_graph():
    g = StateGraph(AnalystState)
    for name, fn in [("fetch_data", fetch_data), ("retrieve", retrieve), ("bull", bull),
                     ("bear", bear), ("synthesize", synthesize)]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_data")
    g.add_edge("fetch_data", "retrieve")
    # Fan-out: two edges from one node run bull and bear in the same superstep.
    g.add_edge("retrieve", "bull")
    g.add_edge("retrieve", "bear")
    # Fan-in: synthesize waits for both branches.
    g.add_edge(["bull", "bear"], "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()
