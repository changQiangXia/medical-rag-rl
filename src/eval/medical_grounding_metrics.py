from __future__ import annotations

import re


DOC_RE = re.compile(r"\[Doc\s*(\d+)\]", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _tokenize(text: str) -> set[str]:
    cjk_tokens = {ch for ch in text if "\u4e00" <= ch <= "\u9fff"}
    en_tokens = {w.lower() for w in WORD_RE.findall(text)}
    return cjk_tokens | en_tokens


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(1, len(text))


def doc_tag_repeat_ratio(response: str) -> float:
    cites = DOC_RE.findall(response or "")
    if not cites:
        return 0.0
    unique = set(cites)
    return (len(cites) - len(unique)) / len(cites)


def _strip_doc_tags(response: str) -> str:
    return DOC_RE.sub(" ", response or "").strip()


def citation_consistency(response: str, num_docs: int) -> float:
    if not response or num_docs <= 0:
        return 0.0
    cited_ids = [int(x) for x in DOC_RE.findall(response)]
    if not cited_ids:
        return 0.0

    unique_cited = set(cited_ids)
    valid_unique = {x for x in unique_cited if 1 <= x <= num_docs}
    precision = len(valid_unique) / max(1, len(unique_cited))
    repeat_penalty = doc_tag_repeat_ratio(response)
    return max(0.0, precision * (1.0 - 0.5 * repeat_penalty))


def evidence_hit_rate(response: str, docs: list[dict]) -> float:
    if not response or not docs:
        return 0.0

    cited_ids = {int(x) for x in DOC_RE.findall(response)}
    valid_cited = {x for x in cited_ids if 1 <= x <= len(docs)}
    if not valid_cited:
        return 0.0

    content = _strip_doc_tags(response)
    if len(content) < 20:
        return 0.0

    resp_tokens = _tokenize(content)
    if not resp_tokens:
        return 0.0

    evidence_text = " ".join(str(d.get("text", "")) for d in docs)
    evidence_tokens = _tokenize(evidence_text)
    if not evidence_tokens:
        return 0.0

    # Cross-lingual fallback: Chinese answer over English evidence has near-zero lexical overlap.
    if _cjk_ratio(content) >= 0.10 and _cjk_ratio(evidence_text) < 0.02:
        return 1.0

    overlap = len(resp_tokens & evidence_tokens) / max(1, len(resp_tokens))
    return 1.0 if overlap >= 0.08 else 0.0


def safety_refusal_rate(response: str) -> float:
    text = response.lower()
    triggers = [
        "insufficient evidence",
        "cannot answer safely",
        "not enough evidence",
        "证据不足",
        "无法安全回答",
        "信息不足",
    ]
    return 1.0 if any(t in text for t in triggers) else 0.0
