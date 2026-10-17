from market_analyst.graph import build_graph
from market_analyst.rag.retriever import reciprocal_rank_fusion


def test_stub_graph_runs_end_to_end():
    result = build_graph().invoke({"ticker": "AAPL", "stub": True, "timings": {}})
    memo = result["memo"]
    assert memo.ticker == "AAPL"
    assert memo.recommendation in {"buy", "hold", "sell"}
    assert result["bull_thesis"].stance == "bull"
    assert result["bear_thesis"].stance == "bear"


def test_rrf_rewards_agreement():
    # id 2 is ranked well by both lists, so it should win.
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 4, 1]])[0] == 2
