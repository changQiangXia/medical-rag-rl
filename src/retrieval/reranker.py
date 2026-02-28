from __future__ import annotations


def lexical_rerank(query: str, docs: list[dict], top_k: int | None = None) -> list[dict]:
    q_tokens = set(query.lower().split())

    def score(doc: dict) -> float:
        tokens = set(doc.get("text", "").lower().split())
        return float(len(q_tokens & tokens))

    ranked = sorted(docs, key=score, reverse=True)
    if top_k is not None:
        return ranked[:top_k]
    return ranked
