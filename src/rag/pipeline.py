from __future__ import annotations

from src.rag.generator import Generator
from src.rag.prompting import build_rag_prompt
from src.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(
        self,
        query: str,
        max_new_tokens: int = 192,
        temperature: float = 0.2,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 4,
    ) -> dict:
        retrieval_query = self.generator.translate_query_for_retrieval(query)
        docs = self.retriever.retrieve(retrieval_query, top_k=self.top_k)
        prompt = build_rag_prompt(query, docs)
        response = self.generator.generate(
            prompt,
            docs=docs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return {
            "query": query,
            "response": response,
            "docs": docs,
        }
