from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissStore:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self.vectors: np.ndarray | None = None
        self.use_faiss = False
        self.index = None

        try:
            import faiss  # type: ignore

            self.faiss = faiss
            self.use_faiss = True
        except Exception:
            self.faiss = None

    def build(self, vectors: np.ndarray, ids: list[str], texts: list[str], metadata: list[dict]) -> None:
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            vectors = vectors / norms

        self.ids = ids
        self.texts = texts
        self.metadata = metadata
        self.vectors = vectors.astype(np.float32)

        if self.use_faiss:
            dim = self.vectors.shape[1]
            self.index = self.faiss.IndexFlatIP(dim)
            self.index.add(self.vectors)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        q = query_vec.astype(np.float32)
        if self.normalize:
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        if self.use_faiss and self.index is not None:
            scores, idxs = self.index.search(q, top_k)
            pairs = list(zip(idxs[0].tolist(), scores[0].tolist()))
        else:
            sims = np.dot(self.vectors, q[0])
            idxs = np.argsort(-sims)[:top_k]
            pairs = [(int(i), float(sims[i])) for i in idxs]

        results: list[dict] = []
        for idx, score in pairs:
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append(
                {
                    "id": self.ids[idx],
                    "score": score,
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                }
            )
        return results

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.use_faiss and self.index is not None:
            self.faiss.write_index(self.index, str(out / "index.faiss"))
        elif self.vectors is not None:
            np.save(out / "index.npy", self.vectors)

        with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
            for i, doc_id in enumerate(self.ids):
                row = {
                    "id": doc_id,
                    "text": self.texts[i],
                    "metadata": self.metadata[i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        meta = {
            "normalize": self.normalize,
            "use_faiss": self.use_faiss,
            "count": len(self.ids),
            "dim": int(self.vectors.shape[1]) if self.vectors is not None else 0,
        }
        with (out / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, out_dir: str | Path) -> "FaissStore":
        out = Path(out_dir)
        with (out / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        store = cls(normalize=meta.get("normalize", True))
        store.use_faiss = bool(meta.get("use_faiss", False)) and store.use_faiss

        store.ids = []
        store.texts = []
        store.metadata = []
        with (out / "docs.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                store.ids.append(row["id"])
                store.texts.append(row["text"])
                store.metadata.append(row.get("metadata", {}))

        if store.use_faiss and (out / "index.faiss").exists():
            store.index = store.faiss.read_index(str(out / "index.faiss"))
        elif (out / "index.npy").exists():
            store.vectors = np.load(out / "index.npy")

        return store
