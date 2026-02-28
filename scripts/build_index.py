#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io_utils import read_jsonl
from src.retrieval.embedder import TextEmbedder
from src.retrieval.faiss_store import FaissStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--rag-config", default=str(ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    rag_cfg = load_config(args.rag_config)

    input_path = Path(data_cfg["processed_jsonl_dir"]) / f"{args.split}.jsonl"
    if not input_path.exists():
        print(f"Missing JSONL: {input_path}. Run prepare_pubmed.py first.")
        return 1

    rows = list(read_jsonl(input_path))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print("No rows to index.")
        return 1

    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    metadata = [r.get("metadata", {}) | {"paper_id": r.get("paper_id", "")} for r in rows]

    emb_cfg = rag_cfg["embedding"]
    device = None
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = None

    embedder = TextEmbedder(
        model_name_or_path=emb_cfg["model_name_or_path"],
        normalize=bool(emb_cfg.get("normalize", True)),
        device=device,
    )

    if embedder.backend == "tfidf":
        embedder.fit(texts)

    vectors = embedder.encode(texts, batch_size=int(args.batch_size))
    store = FaissStore(normalize=bool(emb_cfg.get("normalize", True)))
    store.build(vectors=vectors, ids=ids, texts=texts, metadata=metadata)

    out_dir = Path(rag_cfg["retrieval"]["index_dir"]) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save(out_dir)

    if embedder.backend == "tfidf":
        with (out_dir / "tfidf_vectorizer.pkl").open("wb") as f:
            pickle.dump(embedder.vectorizer, f)

    metrics = {
        "split": args.split,
        "count": len(rows),
        "dim": int(vectors.shape[1]),
        "embedder_backend": embedder.backend,
        "sample_query": "",
        "sample_top1_id": "",
        "sample_top1_score": 0.0,
    }

    query = random.choice(texts)
    qvec = embedder.encode([query])
    top = store.search(qvec, top_k=1)
    if top:
        metrics["sample_query"] = query[:120]
        metrics["sample_top1_id"] = top[0]["id"]
        metrics["sample_top1_score"] = round(float(top[0]["score"]), 6)

    metric_path = out_dir / "index_metrics.json"
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] indexed {len(rows)} rows -> {out_dir}")
    print(f"[OK] metrics: {metric_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
