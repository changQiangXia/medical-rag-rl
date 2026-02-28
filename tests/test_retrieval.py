from __future__ import annotations

import numpy as np

from src.retrieval.faiss_store import FaissStore


def test_faiss_store_search() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = ["a", "b"]
    texts = ["alpha", "beta"]
    metadata = [{}, {}]

    store = FaissStore(normalize=True)
    store.build(vectors=vectors, ids=ids, texts=texts, metadata=metadata)
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "a"
