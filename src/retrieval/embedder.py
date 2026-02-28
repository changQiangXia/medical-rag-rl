from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


class TextEmbedder:
    def __init__(self, model_name_or_path: str, normalize: bool = True, device: str | None = None):
        self.model_name_or_path = model_name_or_path
        self.normalize = normalize
        self.device = device
        self.backend = "hashing"
        self.model = None
        self.dim = 4096

        if model_name_or_path == "__hashing__":
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name_or_path, device=device)
            self.backend = "sentence_transformers"
        except Exception:
            self.model = None

    def fit(self, texts: Iterable[str]) -> None:
        _ = texts
        return None

    def _hash_tokens(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h, "little") % self.dim
            vec[idx] += 1.0
        return vec

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.backend == "sentence_transformers":
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
            return vectors.astype(np.float32)

        _ = batch_size
        mat = np.vstack([self._hash_tokens(t) for t in texts]).astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            mat = mat / norms
        return mat
