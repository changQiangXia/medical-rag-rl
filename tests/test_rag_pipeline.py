from __future__ import annotations

from src.rag.pipeline import RAGPipeline


class DummyRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return [{"id": "x", "text": "evidence", "metadata": {}}]


class DummyGenerator:
    def translate_query_for_retrieval(self, query: str) -> str:
        return query

    def generate(self, prompt: str, docs: list[dict], **kwargs) -> str:
        return "answer [Doc 1]"


def test_pipeline_answer() -> None:
    pipeline = RAGPipeline(retriever=DummyRetriever(), generator=DummyGenerator(), top_k=1)
    out = pipeline.answer("test")
    assert "response" in out
    assert "docs" in out
    assert out["response"].endswith("[Doc 1]")
