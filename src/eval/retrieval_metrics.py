from __future__ import annotations


def recall_at_k(retrieved_ids: list[list[str]], gold_ids: list[set[str]], k: int = 5) -> float:
    if not retrieved_ids:
        return 0.0
    hits = 0
    for preds, gold in zip(retrieved_ids, gold_ids):
        topk = preds[:k]
        if any(pid in gold for pid in topk):
            hits += 1
    return hits / len(retrieved_ids)


def mean_reciprocal_rank(retrieved_ids: list[list[str]], gold_ids: list[set[str]]) -> float:
    if not retrieved_ids:
        return 0.0
    rr_sum = 0.0
    for preds, gold in zip(retrieved_ids, gold_ids):
        rr = 0.0
        for rank, pid in enumerate(preds, start=1):
            if pid in gold:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return rr_sum / len(retrieved_ids)
