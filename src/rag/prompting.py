from __future__ import annotations


def _is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def build_rag_prompt(query: str, docs: list[dict]) -> str:
    evidence_blocks = []
    for i, d in enumerate(docs, start=1):
        evidence_blocks.append(f"[Doc {i}] {d.get('text', '')}")

    evidence = "\n".join(evidence_blocks)
    if not evidence:
        evidence = "(No evidence retrieved)"

    if _is_cjk(query):
        lang_hint = "Please answer in Chinese."
    else:
        lang_hint = "Please answer in the same language as the question."

    return (
        "You are a medical assistant. Answer using only the retrieved evidence.\n"
        f"{lang_hint}\n"
        "If evidence is insufficient, explicitly state that evidence is insufficient.\n"
        "Avoid repeating citation tags.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved Evidence:\n{evidence}\n\n"
        "Answer:\n"
    )
