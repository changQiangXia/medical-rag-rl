from __future__ import annotations

import re
from collections import Counter


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    cjk_tokens = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    en_tokens = [w.lower() for w in WORD_RE.findall(text)]
    return cjk_tokens + en_tokens


def token_f1(pred: str, ref: str) -> float:
    p = _tokenize(pred)
    r = _tokenize(ref)
    if not p or not r:
        return 0.0
    p_count = Counter(p)
    r_count = Counter(r)
    inter = sum((p_count & r_count).values())
    if inter == 0:
        return 0.0
    precision = inter / len(p)
    recall = inter / len(r)
    return 2 * precision * recall / (precision + recall)
