from __future__ import annotations

from src.retrieval.embedder import TextEmbedder
from src.retrieval.faiss_store import FaissStore
from src.retrieval.reranker import lexical_rerank


class Retriever:
    def __init__(self, embedder: TextEmbedder, store: FaissStore, use_rerank: bool = False):
        self.embedder = embedder
        self.store = store
        self.use_rerank = use_rerank

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        qvec = self.embedder.encode([query])
        docs = self.store.search(qvec, top_k=top_k * (2 if self.use_rerank else 1))
        if self.use_rerank:
            docs = lexical_rerank(query, docs, top_k=top_k)
        return docs[:top_k]
