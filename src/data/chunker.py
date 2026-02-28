from __future__ import annotations


def chunk_words(text: str, chunk_size: int = 160, chunk_overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(words):
        end = min(start + chunk_size, len(words))
        segment = " ".join(words[start:end]).strip()
        if segment:
            chunks.append(segment)
        if end == len(words):
            break
        start += step
    return chunks
