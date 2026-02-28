#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io_utils import read_jsonl
from src.rag.generator import Generator
from src.rag.pipeline import RAGPipeline
from src.retrieval.embedder import TextEmbedder
from src.retrieval.faiss_store import FaissStore
from src.retrieval.retriever import Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-config", default=str(ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--query", default="")
    parser.add_argument("--query-file", default="")
    parser.add_argument("--query-jsonl", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--name", default="base_rag")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def load_query_rows(args: argparse.Namespace) -> list[dict]:
    if args.query_jsonl:
        rows = []
        for i, row in enumerate(read_jsonl(args.query_jsonl)):
            q = row.get("query", "") or row.get("instruction", "")
            ref = row.get("reference", "") or row.get("output", "")
            if not str(q).strip():
                continue
            rows.append(
                {
                    "id": row.get("id", row.get("metadata", {}).get("sample_id", str(i))),
                    "query": str(q).strip(),
                    "reference": str(ref).strip(),
                }
            )
        if args.max_samples > 0:
            rows = rows[: args.max_samples]
        return rows

    if args.query:
        return [{"id": "q0", "query": args.query.strip(), "reference": ""}]

    if args.query_file:
        lines = Path(args.query_file).read_text(encoding="utf-8").splitlines()
        rows = [{"id": f"q{i}", "query": x.strip(), "reference": ""} for i, x in enumerate(lines) if x.strip()]
        if args.max_samples > 0:
            rows = rows[: args.max_samples]
        return rows

    return [
        {
            "id": "q0",
            "query": "What was the efficacy of low-dose prednisolone in knee osteoarthritis?",
            "reference": "",
        },
        {
            "id": "q1",
            "query": "How is emotional eating linked to mood in the trial results?",
            "reference": "",
        },
    ]


def main() -> int:
    args = parse_args()
    rag_cfg = load_config(args.rag_config)

    index_dir = Path(rag_cfg["retrieval"]["index_dir"]) / args.split
    if not index_dir.exists():
        print(f"Missing index: {index_dir}. Run build_index.py first.")
        return 1

    store = FaissStore.load(index_dir)
    emb_cfg = rag_cfg["embedding"]
    embedder = TextEmbedder(emb_cfg["model_name_or_path"], normalize=bool(emb_cfg.get("normalize", True)))

    # Compatibility: old index may be built with hashing (4096d).
    if store.vectors is not None and embedder.backend != "hashing":
        try:
            probe_dim = int(embedder.encode(["dimension probe"]).shape[1])
            index_dim = int(store.vectors.shape[1])
            if probe_dim != index_dim:
                print(f"[WARN] embedding dim mismatch query={probe_dim} index={index_dim}, fallback to hashing backend")
                embedder = TextEmbedder("__hashing__", normalize=bool(emb_cfg.get("normalize", True)))
        except Exception:
            print("[WARN] failed to probe embedding dim, fallback to hashing backend")
            embedder = TextEmbedder("__hashing__", normalize=bool(emb_cfg.get("normalize", True)))

    if embedder.backend == "tfidf":
        pkl = index_dir / "tfidf_vectorizer.pkl"
        if not pkl.exists():
            print(f"Missing TF-IDF vectorizer: {pkl}")
            return 1
        with pkl.open("rb") as f:
            embedder.vectorizer = pickle.load(f)
    retriever = Retriever(embedder=embedder, store=store, use_rerank=True)

    gen_cfg = rag_cfg["generation"]
    generator = Generator(
        model_name_or_path=gen_cfg["model_name_or_path"],
        device="cuda",
        adapter_name_or_path=args.adapter.strip(),
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=int(rag_cfg["retrieval"].get("top_k", 5)),
    )

    query_rows = load_query_rows(args)
    if not query_rows:
        print("No queries to run.")
        return 1

    outputs = []
    pbar = tqdm(query_rows, total=len(query_rows), desc=f"RAG Eval {args.name}", unit="sample")
    for item in pbar:
        out = pipeline.answer(
            item["query"],
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 192)),
            temperature=float(gen_cfg.get("temperature", 0.2)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.1)),
            no_repeat_ngram_size=int(gen_cfg.get("no_repeat_ngram_size", 4)),
        )
        out["id"] = item.get("id", "")
        out["reference"] = item.get("reference", "")
        out["model_name"] = args.name
        out["adapter"] = args.adapter.strip()
        outputs.append(out)
        pbar.set_postfix({"done": len(outputs)})
    pbar.close()

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_dir = ROOT / "outputs" / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.name}_{args.split}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] outputs written: {out_path}")
    print(f"[OK] samples={len(outputs)} adapter={'none' if not args.adapter else args.adapter}")
    if outputs:
        print(json.dumps(outputs[0], ensure_ascii=False, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
