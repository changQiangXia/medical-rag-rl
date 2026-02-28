from __future__ import annotations

from collections import Counter
from statistics import mean

from src.data.chunker import chunk_words
from src.data.schema import AbstractRecord


def build_chunked_rows(
    records: list[AbstractRecord],
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    max_chars: int,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    section_counter: Counter[str] = Counter()
    lengths: list[int] = []

    for rec in records:
        full_text = " ".join(s.text for s in rec.sentences).strip()
        if len(full_text) < min_chars or len(full_text) > max_chars:
            continue

        for sent in rec.sentences:
            section_counter[sent.section] += 1

        for idx, chunk in enumerate(chunk_words(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            chunk_len = len(chunk.split())
            lengths.append(chunk_len)
            rows.append(
                {
                    "id": f"{rec.split}:{rec.paper_id}:{idx}",
                    "paper_id": rec.paper_id,
                    "split": rec.split,
                    "chunk_id": idx,
                    "text": chunk,
                    "sections": sorted(set(s.section for s in rec.sentences)),
                    "metadata": {
                        "num_sentences": len(rec.sentences),
                        "chunk_words": chunk_len,
                    },
                }
            )

    stats = {
        "num_records": len(records),
        "num_rows": len(rows),
        "avg_chunk_words": round(mean(lengths), 2) if lengths else 0.0,
        "section_coverage": dict(section_counter),
    }
    return rows, stats
