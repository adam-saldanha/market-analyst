"""Hybrid retrieval: dense vectors + BM25, fused with RRF, then cross-encoder rerank.

Why hybrid: embeddings catch paraphrases ("supply chain risk" ≈ "manufacturing
concentration"); BM25 catches exact terms embeddings blur (tickers, "Item 1A",
specific product names). Fusion gets both; reranking fixes the final order.
"""
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ..config import TOP_K, CACHE_DIR, EMBEDDING_MODEL, RERANKER_MODEL
from ..schemas import Chunk


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge several ranked lists of chunk ids. Done for you — RRF only needs
    ranks, not scores, so it sidesteps BM25 and cosine being on different scales."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], cache_key: str | None = None):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}

        # 5a: dense vectors in a cosine-similarity index
        self.embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vecs = self._load_or_embed(cache_key)
        faiss.normalize_L2(vecs)                       # normalized + inner product = cosine
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

        # 5b: keyword index
        self.bm25 = BM25Okapi([c.text.lower().split() for c in chunks])

        # 5c: cross-encoder (downloads ~90 MB on first use)
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def _load_or_embed(self, cache_key: str | None) -> np.ndarray:
        """Embedding is the slow, rate-limited step, so cache it per ticker + model."""
        path = None
        if cache_key:
            model = EMBEDDING_MODEL.replace("/", "_")
            path = CACHE_DIR / f"{cache_key.upper()}_{model}_emb.npy"
            if path.exists():
                vecs = np.load(path)
                if len(vecs) == len(self.chunks):      # stale if the chunking changed
                    return vecs

        texts = [c.text for c in self.chunks]
        vecs = []
        for i in range(0, len(texts), 50):             # batch to stay under rate limits
            vecs.extend(self.embedder.embed_documents(texts[i:i + 50]))
        vecs = np.array(vecs, dtype="float32")

        if path:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, vecs)
        return vecs

    def vector_search(self, query: str, n: int = 25) -> list[int]:
        q = np.array([self.embedder.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        _, idx = self.index.search(q, min(n, len(self.chunks)))
        return [self.chunks[i].id for i in idx[0] if i != -1]

    def bm25_search(self, query: str, n: int = 25) -> list[int]:
        scores = self.bm25.get_scores(query.lower().split())
        return [self.chunks[i].id for i in np.argsort(scores)[::-1][:n]]

    def rerank(self, query: str, ids: list[int], top_k: int) -> list[int]:
        scores = self.reranker.predict([(query, self.by_id[i].text) for i in ids])
        ranked = sorted(zip(ids, scores), key=lambda p: p[1], reverse=True)
        return [i for i, _ in ranked[:top_k]]

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        fused = reciprocal_rank_fusion([self.vector_search(query), self.bm25_search(query)])
        return [self.by_id[i] for i in self.rerank(query, fused[:25], top_k)]
